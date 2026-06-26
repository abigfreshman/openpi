#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天逸机器人增强版代理服务器

新增功能：
  1. 异步推理队列：推理和执行解耦，保证控制频率
  2. Web 可视化：实时监控相机画面、关节状态、推理结果
  3. WebSocket 推送：实时数据流

用法：
  python proxy_server_enhanced.py --inference-url http://127.0.0.1:18001 --port 18000
"""

import argparse
import asyncio
import json
import os
import time
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(title="TianYi Proxy Server Enhanced", version="2.0.0")

# ============================================================================
# 全局状态
# ============================================================================

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:18001")

current_task: Optional[str] = None
task_completed: bool = False
task_start_time: Optional[float] = None

# 异步推理队列
inference_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
latest_result: Optional[Dict[str, Any]] = None

# WebSocket 连接池
active_websockets: List[WebSocket] = []

# 异步 HTTP 客户端
http_client: Optional[httpx.AsyncClient] = None

def now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串"""
    return datetime.now(timezone.utc).isoformat()

# ============================================================================
# 任务完成检测器
# ============================================================================

class CompletionDetector:
    """
    基于轨迹的任务完成检测器。

    检测策略：
    1. 夹爪相位检测：检测 闭合 → 打开 循环（物体释放）
    2. 动作收敛：模型输出趋向于零（模型认为任务完成）
    3. 关节状态收敛：物理确认（机械臂停止运动）
    """

    # 可调参数
    GRIPPER_CLOSE_THRESHOLD = 0.5  # 夹爪闭合阈值（0~1 范围）
    STATE_CONVERGENCE_WINDOW = 10  # 状态收敛窗口
    STATE_CONVERGENCE_THRESHOLD = 0.003  # 状态收敛阈值
    ACTION_CONVERGENCE_WINDOW = 3  # 动作收敛窗口
    ACTION_CONVERGENCE_THRESHOLD = 0.02  # 动作收敛阈值
    MIN_STEPS_BEFORE_CHECK = 30  # 最小步数（之前不检查完成）
    MAX_STEPS = 1000  # 最大步数（超时）

    def __init__(self, control_mode: str = "dual"):
        """
        参数：
            control_mode: "left", "right", "dual"
                决定监控哪个手臂的夹爪
        """
        self.control_mode = control_mode.strip().lower()
        self.reset()

    def reset(self):
        """重置检测器状态"""
        self.step_count = 0
        self.gripper_phase = "idle"  # idle → closed → released
        self.gripper_was_closed = False
        self.gripper_release_step: Optional[int] = None

        self._state_buffer: deque = deque(maxlen=50)
        self._action_buffer: deque = deque(maxlen=50)
        self._gripper_buffer: deque = deque(maxlen=50)

        self.latest_images: Dict[str, str] = {}
        self._result: Dict[str, Any] = {"triggered": False}

    def get_result(self) -> Dict[str, Any]:
        """获取最新的检测结果"""
        return self._result

    def _extract_gripper(self, state: np.ndarray) -> float:
        """
        提取夹爪值。

        对于 16 维状态：
        - left gripper: state[7]
        - right gripper: state[15]
        - dual: 取两者的平均值
        """
        if len(state) == 16:
            if self.control_mode == "left":
                return float(state[7])
            elif self.control_mode == "right":
                return float(state[15])
            else:  # dual
                return float((state[7] + state[15]) / 2.0)
        else:
            # 单臂或其他维度，取最后一个
            return float(state[-1])

    def _extract_arm_joints(self, state: np.ndarray) -> np.ndarray:
        """
        提取手臂关节（不包括夹爪）。

        对于 16 维状态：
        - left arm: state[0:7]
        - right arm: state[8:15]
        - dual: 拼接两者
        """
        if len(state) == 16:
            if self.control_mode == "left":
                return state[0:7]
            elif self.control_mode == "right":
                return state[8:15]
            else:  # dual
                return np.concatenate([state[0:7], state[8:15]])
        else:
            # 单臂或其他维度，去掉最后一个（夹爪）
            return state[:-1]

    def step(
        self,
        state: np.ndarray,
        action_pred: np.ndarray,
        images: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        处理一步观测和动作预测。

        参数：
            state: 机器人状态 [16]
            action_pred: 模型预测的动作 [chunk_size, 16] 或 [16]
            images: 可选的相机图像 {name: base64}

        返回：
            {"triggered": bool, "reason": str, "confidence": float, ...}
        """
        self.step_count += 1

        if images:
            self.latest_images = images

        # 展平动作（如果是 2D）
        if action_pred.ndim > 1:
            action_flat = action_pred[0]  # 取第一步
        else:
            action_flat = action_pred

        # 提取夹爪值
        gripper_val = self._extract_gripper(state)

        # 提取手臂关节
        arm_state = self._extract_arm_joints(state)
        arm_action = self._extract_arm_joints(action_flat)

        # 计算动作差值（模型期望的运动）
        if len(arm_action) == len(arm_state):
            action_delta = np.abs(arm_action - arm_state)
        else:
            action_delta = np.zeros_like(arm_state)

        # 存储到缓冲区
        self._state_buffer.append(arm_state.copy())
        self._action_buffer.append(arm_action.copy())
        self._gripper_buffer.append(gripper_val)

        # 预热阶段
        if self.step_count < self.MIN_STEPS_BEFORE_CHECK:
            return self._make_result(False, "warming up", 0.0)

        # 超时
        if self.step_count >= self.MAX_STEPS:
            return self._make_result(
                True,
                f"timeout after {self.MAX_STEPS} steps",
                0.5,
                strategy="timeout"
            )

        # --- 信号 1: 夹爪相位检测 ---
        gripper_closed = gripper_val > self.GRIPPER_CLOSE_THRESHOLD

        # 检测闭合
        if gripper_closed and not self.gripper_was_closed:
            self.gripper_phase = "closed"
            self.gripper_was_closed = True
            print(f"[DETECT] 夹爪闭合 at step {self.step_count} (value={gripper_val:.3f})")

        # 检测释放
        if not gripper_closed and self.gripper_was_closed and self.gripper_phase == "closed":
            self.gripper_phase = "released"
            self.gripper_release_step = self.step_count
            print(f"[DETECT] 夹爪释放 at step {self.step_count} (value={gripper_val:.3f})")

        # 门控：只有在夹爪释放后才检查收敛
        if self.gripper_phase != "released":
            return self._make_result(False, f"gripper_phase={self.gripper_phase}", 0.0)

        # 释放后需要等待几步
        steps_since_release = self.step_count - (self.gripper_release_step or self.step_count)
        if steps_since_release < 5:
            return self._make_result(False, "just released, waiting", 0.1)

        # --- 信号 2: 动作收敛 ---
        action_converged = self._check_action_convergence()

        # --- 信号 3: 状态收敛 ---
        state_converged, state_score = self._check_state_convergence()

        # --- 融合决策 ---
        convergence_score = 0.0
        if action_converged:
            convergence_score += 0.4
        if state_converged:
            convergence_score += 0.4
        if steps_since_release > 15:
            convergence_score += 0.2

        # 决策
        if action_converged and state_converged:
            return self._make_result(
                True,
                f"gripper released at step {self.gripper_release_step}, "
                f"action+state converged (score={convergence_score:.2f})",
                confidence=min(1.0, convergence_score),
                strategy="trajectory",
                convergence_score=convergence_score,
            )

        if convergence_score >= 0.6:
            return self._make_result(
                True,
                f"gripper released, partial convergence (score={convergence_score:.2f})",
                confidence=convergence_score,
                strategy="trajectory",
                convergence_score=convergence_score,
            )

        return self._make_result(
            False,
            f"released but not converged (action={action_converged}, state={state_converged}, "
            f"score={convergence_score:.2f})",
            convergence_score,
            convergence_score=convergence_score,
        )

    def _check_action_convergence(self) -> bool:
        """检查动作是否收敛（模型认为任务完成）"""
        if len(self._action_buffer) < self.ACTION_CONVERGENCE_WINDOW:
            return False
        if len(self._state_buffer) < self.ACTION_CONVERGENCE_WINDOW:
            return False

        recent_actions = list(self._action_buffer)[-self.ACTION_CONVERGENCE_WINDOW:]
        recent_states = list(self._state_buffer)[-self.ACTION_CONVERGENCE_WINDOW:]

        deltas = [np.linalg.norm(a - s) for a, s in zip(recent_actions, recent_states)]
        return all(d < self.ACTION_CONVERGENCE_THRESHOLD for d in deltas)

    def _check_state_convergence(self) -> tuple:
        """检查状态是否收敛（机械臂停止运动）"""
        if len(self._state_buffer) < self.STATE_CONVERGENCE_WINDOW + 1:
            return False, 0.0

        recent = list(self._state_buffer)[-(self.STATE_CONVERGENCE_WINDOW + 1):]
        deltas = []
        for i in range(1, len(recent)):
            d = np.linalg.norm(recent[i] - recent[i - 1])
            deltas.append(d)

        avg_delta = np.mean(deltas)
        converged = all(d < self.STATE_CONVERGENCE_THRESHOLD for d in deltas)
        return converged, float(avg_delta)

    def _make_result(
        self,
        triggered: bool,
        reason: str,
        confidence: float,
        strategy: str = "trajectory",
        **kwargs
    ) -> Dict[str, Any]:
        """构造检测结果"""
        result = {
            "triggered": triggered,
            "reason": reason,
            "confidence": confidence,
            "strategy": strategy,
            "step": self.step_count,
            "gripper_phase": self.gripper_phase,
            **kwargs,
        }
        self._result = result
        return result

# 全局检测器
detector: Optional[CompletionDetector] = None

# ============================================================================
# 数据模型
# ============================================================================

class InferencePayload(BaseModel):
    """推理请求 payload"""
    images: Dict[str, str]
    state: List[float]
    task: Optional[str] = None

class TaskPayload(BaseModel):
    """任务设置 payload"""
    instruction: str

class CompletionStatus(BaseModel):
    """任务完成状态"""
    task: Optional[str]
    completed: bool
    strategy: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    elapsed_s: float = 0.0
    inference_count: int = 0
    gripper_phase: str = "idle"

# ============================================================================
# 异步推理处理
# ============================================================================

async def process_inference(payload: InferencePayload) -> Dict[str, Any]:
    """
    实际推理逻辑：转发到推理服务器 + 任务完成检测
    """
    global task_completed

    task = payload.task or current_task
    trace_id = f"proxy_{int(time.time()*1000)}"

    # 如果任务已完成，返回 done 信号
    if task_completed:
        print(f"[PROXY] 任务已完成，返回 done 信号 (step {detector.step_count if detector else 0})")
        return {
            "status": "done",
            "trace_id": trace_id,
            "task": task,
            "completed": True,
            "completion_reason": detector.get_result().get("reason", "") if detector else "",
            "action_pred": [[0.0] * 16] * 15,  # 返回零动作
        }

    # 转发到真实推理服务器
    forward_headers = {"Content-Type": "application/json; charset=utf-8"}
    if task:
        forward_headers["X-Task-Name"] = urllib.parse.quote(str(task))
    forward_headers["X-Trace-Id"] = trace_id

    try:
        resp = await http_client.post(
            f"{INFERENCE_URL}/inference",
            json=payload.dict(),
            headers=forward_headers,
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"[ERROR] 推理代理失败: {e}")
        return {
            "status": "error",
            "detail": str(e),
            "action_pred": [[0.0] * 16] * 15,
        }

    # 喂给完成检测器
    if detector:
        state = np.array(payload.state, dtype=np.float32)
        action_pred = np.array(result.get("action_pred", []), dtype=np.float32)

        detection = detector.step(state, action_pred, images=payload.images)

        if detection["triggered"]:
            print(f"[COMPLETION] step={detector.step_count} reason={detection['reason']}")
            task_completed = True
            result["completed"] = True
            result["completion_reason"] = detection["reason"]
        else:
            result["completed"] = False

        result["monitor"] = {
            "step": detector.step_count,
            "gripper_phase": detector.gripper_phase,
            "convergence_score": detection.get("convergence_score", 0.0),
        }

    return result

async def inference_worker():
    """后台推理工作线程：从队列中取出请求并处理"""
    global latest_result
    print("[WORKER] 推理工作线程已启动")

    while True:
        try:
            payload = await inference_queue.get()
            latest_result = await process_inference(payload)
            inference_queue.task_done()
        except Exception as e:
            print(f"[WORKER ERROR] {e}")
            await asyncio.sleep(0.1)

async def broadcast_to_websockets(data: Dict[str, Any]):
    """广播数据到所有 WebSocket 连接"""
    if not active_websockets:
        return

    message = json.dumps(data)
    disconnected = []

    for ws in active_websockets:
        try:
            await ws.send_text(message)
        except Exception as e:
            print(f"[WS] 发送失败: {e}")
            disconnected.append(ws)

    # 清理断开的连接
    for ws in disconnected:
        if ws in active_websockets:
            active_websockets.remove(ws)

# ============================================================================
# API 端点
# ============================================================================

@app.post("/task")
async def set_task(payload: TaskPayload):
    """设置当前任务"""
    global current_task, task_completed, task_start_time

    old = current_task
    current_task = payload.instruction
    task_completed = False
    task_start_time = time.time()

    if detector:
        detector.reset()

    print(f"[TASK] 切换: \"{old}\" -> \"{current_task}\"")
    return {"status": "ok", "previous": old, "current": current_task}

@app.get("/task")
async def get_task():
    """获取当前任务"""
    return {"current": current_task}

@app.get("/status", response_model=CompletionStatus)
async def get_status():
    """查询任务完成状态"""
    elapsed = time.time() - task_start_time if task_start_time else 0.0
    result = detector.get_result() if detector else {}

    return CompletionStatus(
        task=current_task,
        completed=task_completed,
        strategy=result.get("strategy") if task_completed else None,
        confidence=result.get("confidence", 0.0),
        reason=result.get("reason", ""),
        elapsed_s=round(elapsed, 1),
        inference_count=detector.step_count if detector else 0,
        gripper_phase=detector.gripper_phase if detector else "idle",
    )

@app.post("/inference")
async def inference(
    payload: InferencePayload,
    x_task_name: Optional[str] = Header(default=None, alias="X-Task-Name"),
    x_trace_id: Optional[str] = Header(default=None, alias="X-Trace-Id"),
):
    """
    推理端点：异步推理队列 + 立即返回上一次结果
    """
    global latest_result

    # 将请求加入队列（非阻塞）
    try:
        inference_queue.put_nowait(payload)
    except asyncio.QueueFull:
        # 队列满则丢弃
        pass

    # 首次请求，同步等待
    if latest_result is None:
        latest_result = await process_inference(payload)

    # 广播到所有 WebSocket 连接
    detection = detector.get_result() if detector else {}
    await broadcast_to_websockets({
        "timestamp": now_iso(),
        "images": payload.images,
        "state": payload.state,
        "action_pred": latest_result.get("action_pred", []),
        "task": payload.task or current_task or "",
        "completion_status": {
            "triggered": detection.get("triggered", False),
            "reason": detection.get("reason"),
            "gripper_phase": detection.get("gripper_phase", "unknown"),
            "action_magnitude": detection.get("convergence_score", 0.0),
        }
    })

    return JSONResponse(latest_result)

@app.get("/frame")
async def get_frame():
    """获取最新的相机帧"""
    if not detector or not detector.latest_images:
        return {"error": "no frames yet"}
    return {"images": detector.latest_images, "timestamp": now_iso()}

@app.post("/reset")
async def reset():
    """重置所有状态"""
    global current_task, task_completed, task_start_time, latest_result

    current_task = None
    task_completed = False
    task_start_time = None
    latest_result = None

    if detector:
        detector.reset()

    # 清空队列
    while not inference_queue.empty():
        try:
            inference_queue.get_nowait()
            inference_queue.task_done()
        except asyncio.QueueEmpty:
            break

    print("[RESET] 状态已重置")
    return {"status": "ok"}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

# ============================================================================
# WebSocket 端点
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点：实时推送数据"""
    await websocket.accept()
    active_websockets.append(websocket)
    print(f"[WS] 新连接，当前连接数: {len(active_websockets)}")

    try:
        while True:
            # 保持连接，接收心跳
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"[WS] 连接断开")
    except Exception as e:
        print(f"[WS ERROR] {e}")
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        print(f"[WS] 连接已移除，当前连接数: {len(active_websockets)}")

# ============================================================================
# 静态文件和主页
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回 Web 可视化页面"""
    web_dir = "/media/terryxu/TianYi-Production/web"
    index_path = os.path.join(web_dir, "index.html")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(
            content=f"<h1>Web 界面未找到</h1><p>请确保 {index_path} 存在</p>",
            status_code=404
        )

# 挂载静态文件目录
web_dir = "/media/terryxu/TianYi-Production/web"
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

# ============================================================================
# 生命周期事件
# ============================================================================

@app.on_event("startup")
async def startup():
    """启动时初始化"""
    global http_client

    # 创建异步 HTTP 客户端
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    )

    # 启动推理工作线程
    asyncio.create_task(inference_worker())

    print("[STARTUP] 异步推理工作线程已启动")

@app.on_event("shutdown")
async def shutdown():
    """关闭时清理"""
    global http_client

    if http_client:
        await http_client.aclose()

    print("[SHUTDOWN] 清理完成")

# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="天逸机器人增强版代理服务器 - 异步推理 + Web 可视化"
    )
    parser.add_argument("--inference-url", default="http://127.0.0.1:18001",
                        help="真实推理服务器 URL")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--control-mode", default="dual",
                        choices=["left", "right", "dual"],
                        help="监控哪个手臂的夹爪 (默认: dual)")

    args = parser.parse_args()

    global INFERENCE_URL, detector
    INFERENCE_URL = args.inference_url
    detector = CompletionDetector(control_mode=args.control_mode)

    print("=" * 60)
    print(f"[天逸增强版代理服务器] 代理 → {INFERENCE_URL}")
    print(f"[天逸增强版代理服务器] 监听 {args.host}:{args.port}")
    print(f"[天逸增强版代理服务器] 控制模式: {args.control_mode}")
    print(f"[天逸增强版代理服务器] 检测策略: trajectory (gripper + convergence)")
    print(f"[天逸增强版代理服务器] Web 界面: http://{args.host}:{args.port}")
    print(f"[天逸增强版代理服务器] WebSocket: ws://{args.host}:{args.port}/ws")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
