# -*- coding: utf-8 -*-
"""
云端 WebSocket 推理服务器
- 接收机器人发送的观测 (images + state)
- 执行 OpenPI 模型推理
- 返回 action_chunk
"""
import json
import base64
import asyncio
import time

import numpy as np
from PIL import Image
import io

import websockets
from websockets.server import serve

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# ================== 配置区 ==================
WS_HOST = "0.0.0.0"
WS_PORT = 18000
MODEL_CONFIG = "pi0_tianyi"
CHECKPOINT_PATH = "/media/leon/output/pi0/clothes_washing/pi0_tianyi/clothes_washing_32_ft/30000"

# 图像 key 映射: 机器人端相机名 -> 模型输入 key
# 根据你的模型训练配置调整
IMAGE_KEY_MAP = {
    # "state": "observation/state",
    "head_image": "observation/image",
    "left_image": "observation/left_wrist_image",
    "right_image": "observation/right_wrist_image",
}
# =============================================


class InferenceServer:
    """
    云端推理服务器。
    职责:
      1. 加载 OpenPI 模型
      2. 通过 WebSocket 接收观测数据
      3. 执行推理，返回 action_chunk
    """

    def __init__(self):
        print(f"[Server] 正在加载模型: {MODEL_CONFIG}")
        config = _config.get_config(MODEL_CONFIG)
        checkpoint_dir = download.maybe_download(CHECKPOINT_PATH)
        self.policy = policy_config.create_trained_policy(config, checkpoint_dir)
        print("[Server] 模型加载完成")

        self._current_task: str = "Put the clothes into the washing machine"
        self._inference_count: int = 0

    def _decode_image(self, b64_str: str) -> np.ndarray:
        """将 base64 JPEG 解码为 numpy RGB 数组。"""
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(img)

    def _build_inputs(self, state: list, images_b64: dict) -> dict:
        """
        构建模型推理输入。
        根据你的模型训练数据格式调整此方法。
        """
        inputs = {}

        # 图像
        for robot_key, b64_str in images_b64.items():
            model_key = IMAGE_KEY_MAP.get(robot_key)
            if model_key:
                inputs[model_key] = self._decode_image(b64_str)

        # 状态 (根据模型需要的 key 调整)
        inputs["observation/state"] = np.array(state, dtype=np.float32)

        # 任务 prompt
        inputs["prompt"] = self._current_task

        return inputs

    def _infer(self, state: list, images_b64: dict) -> np.ndarray:
        """执行一次推理，返回 action_chunk (N, action_dim)。"""
        inputs = self._build_inputs(state, images_b64)
        result = self.policy.infer(inputs)
        action_chunk = result["actions"]  # (N, action_dim)
        return np.array(action_chunk, dtype=np.float32)

    async def handle_connection(self, websocket):
        """处理单个 WebSocket 连接。"""
        client_addr = websocket.remote_address
        print(f"[Server] 新连接: {client_addr}")

        try:
            # 1. 握手: 接收 init 消息
            init_raw = await websocket.recv()
            init_msg = json.loads(init_raw)

            if init_msg.get("type") != "init":
                await websocket.send(json.dumps({"type": "error", "msg": "Expected init message"}))
                return

            self._current_task = init_msg.get("task", "default")
            print(f"[Server] 任务设置为: {self._current_task}")

            # 回复 ready
            await websocket.send(json.dumps({"type": "ready"}))

            # 2. 推理循环
            while True:
                try:
                    raw = await websocket.recv()
                    msg = json.loads(raw)

                    if msg.get("type") == "stop":
                        print(f"[Server] 客户端请求停止")
                        await websocket.send(json.dumps({"type": "done"}))
                        break

                    if msg.get("type") != "obs":
                        continue

                    state = msg["state"]
                    images_b64 = msg["images"]

                    # 执行推理
                    t0 = time.time()
                    action_chunk = self._infer(state, images_b64)
                    infer_time = time.time() - t0

                    self._inference_count += 1
                    print(f"[Server] 推理 #{self._inference_count}: "
                          f"耗时={infer_time:.3f}s, chunk_shape={action_chunk.shape}")

                    # 发送 action_chunk
                    resp = json.dumps({
                        "type": "action",
                        "actions": action_chunk.tolist(),
                    })
                    await websocket.send(resp)

                except websockets.ConnectionClosed:
                    print(f"[Server] 连接断开: {client_addr}")
                    break
                except Exception as e:
                    print(f"[Server] 推理错误: {type(e).__name__}: {e}")
                    error_resp = json.dumps({"type": "error", "msg": str(e)})
                    try:
                        await websocket.send(error_resp)
                    except:
                        break

        except Exception as e:
            print(f"[Server] 连接处理错误: {e}")
        finally:
            print(f"[Server] 连接关闭: {client_addr}")

    async def start(self):
        """启动 WebSocket 服务器。"""
        print(f"[Server] 启动 WebSocket 服务: ws://{WS_HOST}:{WS_PORT}/ws/inference")
        async with serve(
            self.handle_connection,
            WS_HOST,
            WS_PORT,
            max_size=50 * 1024 * 1024,  # 50MB
            ping_interval=20,
            ping_timeout=60,
        ):
            await asyncio.Future()  # 永久运行


# -------------------- 入口 --------------------
if __name__ == "__main__":
    server = InferenceServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n[Server] 服务器已停止")
