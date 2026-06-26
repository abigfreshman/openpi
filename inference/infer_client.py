# -*- coding: utf-8 -*-
"""
机器人端 WebSocket 客户端
- 通过 WebSocket 连接云端推理服务
- 发送: 相机图像(base64 JPEG) + 机器人状态(16维)
- 接收: action_chunk (N x 16 的动作序列)
"""
import math
import time
import base64
import threading
import json

import numpy as np
import cv2
import websockets
import asyncio
import tomli

from xrocs.common.data_type import BaseData, Joints
from xrocs.core.config_loader import ConfigLoader
from xrocs.core.station_loader import StationLoader
from xrocs.entity.camera.camera_loader import CameraLoader
from xrocs.entity.hand.hand_loader import HandLoader
from xrocs.utils.logger.logger_loader import logger

# ================== 配置区 ==================
CLOUD_WS_URL = "ws://<CLOUD_IP>:18000/ws/inference"
CAMERA_WARMUP_SECS = 8.0
CHUNK_EXEC_STEPS = 15
CONTROL_DT = 0.05          # 20Hz 控制频率
MAX_STEP_RAD = 0.05         # 单步最大关节变化量 (rad)
PRINT_EVERY = 5
CONFIG_PATH = "/home/ubuntu/Documents/configuration.toml"

ARM_DIM = 8                 # 单臂维度: 7关节 + 1夹爪
DUAL_DIM = ARM_DIM * 2      # 双臂总维度 = 16

# 左臂 7 关节限位 (弧度)
LEFT_ARM_LIMITS = [
    (math.radians(-170), math.radians(170)),
    (math.radians(-15),  math.radians(150)),
    (math.radians(-170), math.radians(170)),
    (math.radians(-150), math.radians(15)),
    (math.radians(-170), math.radians(170)),
    (math.radians(-45),  math.radians(60)),
    (math.radians(-95),  math.radians(75)),
]

# 右臂 7 关节限位 (弧度)
RIGHT_ARM_LIMITS = [
    (math.radians(-170), math.radians(170)),
    (math.radians(-150), math.radians(15)),
    (math.radians(-170), math.radians(170)),
    (math.radians(-150), math.radians(15)),
    (math.radians(-170), math.radians(170)),
    (math.radians(-45),  math.radians(60)),
    (math.radians(-75),  math.radians(95)),
]
# =============================================


class TianYiRobotClient:
    """
    机器人端 WebSocket 客户端。
    职责:
      1. 初始化硬件 (双臂、夹爪、相机)
      2. 通过 WebSocket 发送观测 (images + state)
      3. 接收 action_chunk 并逐步执行
    """

    CAMERA_KEYS = ("head_image","left_image", "right_image")

    def __init__(self, config_path: str = CONFIG_PATH, task_name: str = "default"):
        with open(config_path, "rb") as f:
            self.cfg = tomli.load(f)

        # 1) Station + Robot
        cfg_loader = ConfigLoader(config_path)
        cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()

        robot_handle = self.robot_station.get_robot_handle()["robot"]
        self._arm_ctrler = robot_handle.dual_arm_ctrler
        logger.success("Station + DualArmController 初始化完成")

        # 2) 夹爪
        hand_cfgs = self.cfg.get("hand", {})
        hand_loader = HandLoader()
        self._hands = hand_loader.instantiate_hands(hand_cfgs)
        for hand in self._hands.values():
            hand.connect()
        logger.success("双夹爪初始化完成")

        # 3) 相机
        camera_cfgs = self.cfg.get("camera", {})
        camera_loader = CameraLoader()
        self._cameras = camera_loader.instantiate_cameras(camera_cfgs)
        logger.success(f"相机初始化完成: {list(self._cameras.keys())}")

        # 4) 任务 & 状态
        self.current_task = task_name
        self._home_joints = self.cfg.get("robot", {}).get("arm", {}).get("home", {}).get("robot")
        self._last_arm_target: np.ndarray | None = None
        self._total_frames_sent = 0
        self._chunk_count = 0

        # 5) 图像后台线程
        self._latest_images: dict[str, np.ndarray] = {}
        self._image_lock = threading.Lock()
        self._image_thread_running = False
        self._image_thread: threading.Thread | None = None

    # -------------------- 硬件工具方法 --------------------
    @staticmethod
    def clip_action(action: np.ndarray) -> np.ndarray:
        action = np.array(action, dtype=np.float64)
        for i, (lo, hi) in enumerate(LEFT_ARM_LIMITS):
            action[i] = np.clip(action[i], lo, hi)
        for i, (lo, hi) in enumerate(RIGHT_ARM_LIMITS):
            action[8 + i] = np.clip(action[8 + i], lo, hi)
        action[7] = np.clip(action[7], 0.0, 1.0)
        action[15] = np.clip(action[15], 0.0, 1.0)
        return action

    def _slew_limit(self, q_prev: np.ndarray, q_target: np.ndarray) -> np.ndarray:
        dq = np.clip(q_target - q_prev, -MAX_STEP_RAD, MAX_STEP_RAD)
        return q_prev + dq

    def _send_hand(self, left_pos: float, right_pos: float) -> None:
        if "left" in self._hands:
            left_joints = np.array([left_pos * 0.5] * 5 + [1.0], dtype=np.float64)
            self._hands["left"].set_target_joint(Joints(left_joints, num_of_dofs=6))
        if "right" in self._hands:
            right_joints = np.array([right_pos * 0.5] * 5 + [1.0], dtype=np.float64)
            self._hands["right"].set_target_joint(Joints(right_joints, num_of_dofs=6))

    # -------------------- 状态获取 --------------------
    def _get_arm_state(self) -> np.ndarray:
        joints = self._arm_ctrler.get_current_joint()
        return joints.get_radian_ndarray()

    def _get_hand_state(self) -> tuple[float, float]:
        left_pos, right_pos = 0.0, 0.0
        if "left" in self._hands:
            lj = self._hands["left"].get_current_joint()
            if lj is not None:
                left_joints = lj.get_radian_ndarray()
                left_pos = float(np.mean(left_joints[:5]) / 0.5) if len(left_joints) >= 5 else float(left_joints[0])
        if "right" in self._hands:
            rj = self._hands["right"].get_current_joint()
            if rj is not None:
                right_joints = rj.get_radian_ndarray()
                right_pos = float(np.mean(right_joints[:5]) / 0.5) if len(right_joints) >= 5 else float(right_joints[0])
        return left_pos, right_pos

    def _get_full_state(self) -> np.ndarray:
        arm = self._get_arm_state()
        left_hand, right_hand = self._get_hand_state()
        return np.concatenate([arm[:7], [left_hand], arm[7:14], [right_hand]])

    # -------------------- 图像 --------------------
    def _image_reader_thread(self):
        while self._image_thread_running:
            try:
                images = {}
                for name, cam in self._cameras.items():
                    rgb, _ = cam.read()
                    if rgb is not None and rgb.size > 0:
                        images[name] = rgb
                if images:
                    with self._image_lock:
                        self._latest_images = images
                time.sleep(0.01)
            except Exception as e:
                logger.warning(f"图像读取线程错误: {e}")
                time.sleep(0.1)

    def _start_image_thread(self):
        if not self._image_thread_running:
            self._image_thread_running = True
            self._image_thread = threading.Thread(target=self._image_reader_thread, daemon=True)
            self._image_thread.start()

    def _stop_image_thread(self):
        self._image_thread_running = False
        if self._image_thread:
            self._image_thread.join(timeout=2.0)

    def _get_latest_images(self) -> dict[str, np.ndarray]:
        with self._image_lock:
            return self._latest_images.copy()

    def _encode_image_b64(self, img: np.ndarray, quality: int = 90) -> str:
        if img.dtype != np.uint8:
            img = np.clip(img.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        bgr = img[:, :, ::-1]  # RGB -> BGR for cv2
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # -------------------- 控制执行 --------------------
    def prepare(self) -> None:
        if self._home_joints and len(self._home_joints) >= 14:
            home_joint = Joints(self._home_joints[:14], num_of_dofs=14)
            self._arm_ctrler.reach_target_joint(home_joint)
            logger.success("双臂归位完成")
        else:
            logger.warning("未找到 home 配置，跳过归位")
        self._send_hand(0.0, 0.0)
        time.sleep(1.0)
        self._last_arm_target = self._get_arm_state().copy()

    def _execute_single_step(self, q_abs: np.ndarray) -> None:
        q_abs = self.clip_action(q_abs)
        arm_target = np.concatenate([q_abs[:7], q_abs[8:15]])
        if self._last_arm_target is not None:
            arm_target = self._slew_limit(self._last_arm_target, arm_target)
        self._arm_ctrler.set_cmd_pos(arm_target, timeout=0.0)
        self._last_arm_target = arm_target.copy()
        self._send_hand(float(q_abs[7]), float(q_abs[15]))

    # -------------------- WebSocket 主循环 --------------------
    async def run(self):
        """主循环: 连接云端 WebSocket，发送观测，接收并执行 action_chunk。"""
        logger.info(f"正在连接云端推理服务: {CLOUD_WS_URL}")

        self._start_image_thread()

        # 等待相机预热
        t0 = time.time()
        while time.time() - t0 < CAMERA_WARMUP_SECS:
            if self._get_latest_images():
                break
            await asyncio.sleep(0.05)

        async with websockets.connect(
            CLOUD_WS_URL,
            max_size=50 * 1024 * 1024,  # 50MB，足够传输图像
            ping_interval=20,
            ping_timeout=60,
        ) as ws:
            logger.success("WebSocket 连接已建立")

            # 发送初始握手: 告知任务名
            await ws.send(json.dumps({"type": "init", "task": self.current_task}))
            ack = json.loads(await ws.recv())
            if ack.get("type") != "ready":
                raise RuntimeError(f"握手失败: {ack}")
            logger.success(f"云端就绪, 任务: {self.current_task}")

            # 推理循环
            while True:
                try:
                    # 1. 获取最新观测
                    images = self._get_latest_images()
                    if len(images) < len(self.CAMERA_KEYS):
                        await asyncio.sleep(0.02)
                        continue

                    state = self._get_full_state().tolist()
                    images_b64 = {
                        name: self._encode_image_b64(images[name])
                        for name in self.CAMERA_KEYS
                    }

                    # 2. 发送观测
                    msg = json.dumps({
                        "type": "obs",
                        "state": state,
                        "images": images_b64,
                    })
                    await ws.send(msg)

                    # 3. 等待接收 action_chunk
                    resp_raw = await ws.recv()
                    resp = json.loads(resp_raw)

                    if resp.get("type") == "done":
                        logger.info("云端返回任务完成信号")
                        break

                    if resp.get("type") != "action":
                        logger.warning(f"未知响应类型: {resp.get('type')}")
                        continue

                    action_chunk = np.array(resp["actions"], dtype=np.float32)
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk[None, :]

                    # 4. 执行 action_chunk
                    self._chunk_count += 1
                    steps_to_run = min(CHUNK_EXEC_STEPS, len(action_chunk))
                    logger.info(f"[#{self._chunk_count}] 收到 chunk(len={len(action_chunk)}), 执行 {steps_to_run} 步")

                    for i in range(steps_to_run):
                        self._execute_single_step(action_chunk[i])
                        self._total_frames_sent += 1
                        if (i + 1) % PRINT_EVERY == 0:
                            fb = self._get_full_state()
                            logger.info(f"  step {self._total_frames_sent}: state={fb[:4]}...")
                        await asyncio.sleep(CONTROL_DT)

                except websockets.ConnectionClosed:
                    logger.error("WebSocket 连接断开")
                    break
                except KeyboardInterrupt:
                    logger.info("用户中断")
                    break
                except Exception as e:
                    logger.error(f"循环错误: {type(e).__name__}: {e}")
                    await asyncio.sleep(0.05)

        self._stop_image_thread()
        self.shutdown()

    def shutdown(self) -> None:
        self._stop_image_thread()
        try:
            self.robot_station.shutdown()
        except Exception as e:
            logger.error(f"关闭机器人站出错: {e}")
        logger.info("TianYiRobotClient shutdown complete.")


# -------------------- 入口 --------------------
if __name__ == "__main__":
    task_name = "Put the clothes into the washing machine"
    print(f"任务: {task_name}")

    robot = TianYiRobotClient(config_path=CONFIG_PATH, task_name=task_name)
    robot.prepare()

    try:
        asyncio.run(robot.run())
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        robot.shutdown()
