# 伪流式 (Pseudo-Streaming) Progressive 响应接入计划

## 背景

当前架构：客户端发送音频流 → VAD 断句 → 断句完成后整段送 ASR (1.7B) → ITN → 以 `status=1, msgtype=sentence` 推送结果。

**问题**：VAD 在等待停顿阈值期间（最长 0.7s，语音越长停顿阈值越短直到 0.35s），客户端看不到任何中间结果，用户体验差。

**目标**：在 VAD 尚未触发断句期间，每隔 0.6s 将**最近 0.6s 的语音片段**送入 **0.6B 小模型**进行快速推理，每次推理结果**立即**以 `status=1, msgtype=progressive` 响应给客户端，作为中间"预览"结果。当 VAD 最终断句并由 1.7B 模型得出正式结果后，再以 `msgtype=sentence` 推送最终结果覆盖。

**关键设计决策**：
- **片段模式**：每次只送最近 ~0.6s 的音频片段（非累积全量），控制推理延迟
- **独立 segId**：progressive 使用独立的 segId 序号空间（Option A），与 sentence 的 segId 互不干扰
- **VAD 断句时清空**：VAD 触发断句后，progressive 缓冲区清空、segId 重置，重新开始累积
- **同容器双 vLLM 实例**：1.7B 和 0.6B 在同一容器内各启动一个 vLLM 子进程，启动参数与 1.7B 一致，仅端口和模型名/路径不同

---

## 已确认的设计决策

### 1. 音频送入范围：片段模式

每次 progressive 仅送入自上次发送以来累积的 ~0.6s 语音片段，**不是**从 speech_start 到当前的全量音频。

原因：控制 0.6B 推理延迟，避免长语音段（最长 30s）导致推理时间超过 step 间隔。

### 2. segId 编号：独立序号空间（Option A）

progressive 使用 `_progressive_seg_id` 独立计数器（0, 1, 2, ...），与 sentence 的 `seg_id` 完全分离。客户端通过 `msgtype=progressive` 区分中间结果和最终结果。每次 VAD 断句后 segId 重置为 0。

### 3. 0.6B 模型部署：同容器独立 vLLM 子进程

0.6B 模型与 1.7B 模型在**同一容器**内各启动一个 vLLM 子进程，共享同一 NPU 设备。两个 vLLM 实例使用相同的启动参数（`--tensor-parallel-size 1 --max-model-len 32768 --gpu-memory-utilization 0.6 --dtype bfloat16`），仅端口和模型名/路径不同。

### 4. 推理频率控制：跳过策略

上一次 progressive 未返回时，跳过本次发送。避免推理堆积。

### 5. VAD 断句时取消：是

VAD 断句时取消所有进行中的 progressive 任务，避免 sentence 发送后收到过时的 progressive。

---

## 整体流程

```
Client → [audio frame] → Server
                            ↓
                      VAD.feed_audio()
                            ↓
                    ┌───────┴───────┐
                    │ 未断句         │ 断句触发
                    ↓               ↓
             ┌─── 累积音频到         取消 progressive
             │    progressive 缓冲   清空 progressive 缓冲
             ↓                      ↓
         距离上次发送 >= 0.6s?    1.7B ASR + ITN
             ↓                      ↓
        0.6B 推理（仅最新片段）  status=1, msgtype=sentence
             ↓                      → Client
    status=1, msgtype=progressive
             → Client
```

时序示意：

```
时间线 →

音频帧:  |--0.6s--|--0.6s--|--0.6s--|--0.6s--|--VAD断句--|

Progressive:    P0        P1        P2        P3
响应:        prog(0)   prog(1)   prog(2)   prog(3)    sentence(0)
                                                          ↑
                                                    1.7B 最终结果
                                            （progressive 缓冲清空，segId 重置为 0）
```

---

## 需要修改的文件清单

| # | 文件 | 改动范围 | 改动说明 |
|---|------|---------|---------|
| 1 | `src/core/config.py` | 新增 8 个配置项 | Progressive 功能开关 + ASR 端点 + vLLM 子进程启动参数 |
| 2 | `src/services/asr_service.py` | 中等 | `ASRService` 支持自定义 `api_base` / `model_name`，所有方法使用实例变量 |
| 3 | `src/services/vad_service.py` | 微小 | 新增 `in_speech`、`speech_start_sample` 两个只读属性 |
| 4 | `src/api/session.py` | 中等 | 新增 progressive 状态字段和管理方法；`close()` 增加 progressive 清理 |
| 5 | `src/api/websocket.py` | **核心改动** | `_handle_audio_frame` 增加 progressive 调度；新增 `_process_progressive`；`_handle_end_frame` 增加 progressive 清理 |
| 6 | `main.py` | **大改** | 双 vLLM 子进程管理 + 双健康监测；lifespan 中条件初始化 progressive ASR 服务 |
| 7 | `src/models/schemas.py` | **无改动** | `msgtype` 已是 str 类型 |
| 8 | `Dockerfile` | 微小 | 暴露 0.6B vLLM 内部端口（可选，调试用） |
| 9 | `tmgenius-docker-800i-a2-v2-single-server.yaml` | 小改 | 新增 0.6B 模型权重挂载 + 环境变量 + 端口映射 |

---

## 各文件详细改动

### 1. `src/core/config.py` — 新增配置

```python
# ---- 伪流式 Progressive 功能开关 ----
PROGRESSIVE_ENABLED: bool = os.getenv("PROGRESSIVE_ENABLED", "false").lower() == "true"
"""是否启用伪流式 progressive 推理。"""

PROGRESSIVE_STEP: float = float(os.getenv("PROGRESSIVE_STEP", "0.6"))
"""progressive 推理步进间隔（秒）。"""

# ---- Progressive vLLM 子进程启动参数（与 1.7B 参数一致，仅端口和模型不同）----
PROGRESSIVE_VLLM_PORT: int = int(os.getenv("PROGRESSIVE_VLLM_PORT", "15003"))
"""0.6B vLLM 子进程监听端口。"""

PROGRESSIVE_VLLM_MODEL_PATH: str = os.getenv(
    "PROGRESSIVE_VLLM_MODEL_PATH", "/weights/Qwen3-ASR-0.6B"
)
"""0.6B 模型权重路径。"""

PROGRESSIVE_MODEL_NAME: str = os.getenv("PROGRESSIVE_MODEL_NAME", "Qwen3-ASR-0.6B")
"""0.6B 模型 served-model-name。"""

# ---- Progressive ASR 客户端 ----
PROGRESSIVE_API_BASE: str = os.getenv(
    "PROGRESSIVE_API_BASE", f"http://127.0.0.1:{PROGRESSIVE_VLLM_PORT}/v1"
)
"""0.6B vLLM 的 API 地址，默认根据 PROGRESSIVE_VLLM_PORT 自动拼接。"""

PROGRESSIVE_HEALTH_CHECK_INTERVAL: float = float(
    os.getenv("PROGRESSIVE_HEALTH_CHECK_INTERVAL", "5")
)
"""0.6B vLLM 健康检查间隔（秒）。"""

PROGRESSIVE_STARTUP_TIMEOUT: int = int(
    os.getenv("PROGRESSIVE_STARTUP_TIMEOUT", "1200")
)
"""0.6B vLLM 启动超时（秒），0.6B 模型小，设置较短。"""
```

### 2. `src/services/asr_service.py` — 支持自定义端点

当前 `ASRService` 所有方法都硬编码使用 `settings.VLLM_API_BASE` 和 `settings.VLLM_MODEL_NAME`。改造为构造时可传入自定义值，**所有方法统一使用实例变量**：

```python
class ASRService:
    def __init__(self, api_base: str | None = None, model_name: str | None = None) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._api_base = api_base or settings.VLLM_API_BASE
        self._model_name = model_name or settings.VLLM_MODEL_NAME

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(timeout=60.0, proxy=None, trust_env=False)
        logger.info(
            "ASR service started, endpoint: %s, model: %s",
            self._api_base, self._model_name,
        )

    async def recognize(self, audio_int16, sr=16000, context=""):
        ...
        url = f"{self._api_base}/chat/completions"
        payload = {"model": self._model_name, "messages": messages}
        ...

    async def is_available(self) -> bool:
        if self._client is None:
            return False
        try:
            resp = await self._client.get(f"{self._api_base}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
```

### 3. `src/services/vad_service.py` — 暴露语音状态

新增两个只读属性，供 progressive 调度和 bg/ed 计算使用：

```python
class TenVADSession:
    ...

    @property
    def in_speech(self) -> bool:
        """当前是否处于语音段中（供 progressive 调度使用）。"""
        return self._in_speech

    @property
    def speech_start_sample(self) -> int:
        """当前语音段起始采样位置（供 progressive bg/ed 计算）。"""
        return self._speech_start_sample
```

### 4. `src/api/session.py` — Progressive 状态管理

在 `ASRSession.__init__` 中新增：

```python
# ---- 伪流式 Progressive 状态 ----
self._progressive_seg_id: int = 0                      # progressive 独立 segId 计数器
self._progressive_task: asyncio.Task | None = None      # 当前进行中的 progressive 任务
self._progressive_audio_frames: list[np.ndarray] = []   # 自上次发送以来累积的语音帧（片段模式，非全量）
self._progressive_total_samples: int = 0                # 缓冲区内累计采样数
self._progressive_start_sample: int = 0                 # 缓冲区首帧对应的全局采样偏移
self._progressive_last_send_time: float = 0.0           # 上一次发起 progressive 的时间（0 = 未开始）
```

新增方法：

```python
def append_progressive_audio(self, frame: np.ndarray, frame_start_sample: int) -> None:
    """追加一帧到 progressive 片段缓冲。

    当缓冲区为空时（新语音段开始或上次发送后重置），自动记录起始时间和采样位置，
    解决首次 progressive 因 _progressive_last_send_time=0 立即触发的问题。
    """
    if not self._progressive_audio_frames:
        self._progressive_last_send_time = time.monotonic()
        self._progressive_start_sample = frame_start_sample
    self._progressive_audio_frames.append(frame)
    self._progressive_total_samples += len(frame)

def pop_progressive_audio(self) -> tuple[np.ndarray, int, int]:
    """取出当前累积的片段音频，返回 (audio_int16, start_sample, end_sample)，并重置缓冲区。"""
    if not self._progressive_audio_frames:
        return np.array([], dtype=np.int16), 0, 0
    audio = np.concatenate(self._progressive_audio_frames)
    start_sample = self._progressive_start_sample
    end_sample = start_sample + self._progressive_total_samples
    self._progressive_audio_frames = []
    self._progressive_total_samples = 0
    self._progressive_start_sample = 0
    self._progressive_last_send_time = time.monotonic()
    return audio, start_sample, end_sample

def next_progressive_seg_id(self) -> int:
    """获取并递增 progressive 独立 segId。"""
    current = self._progressive_seg_id
    self._progressive_seg_id += 1
    return current

def cancel_progressive(self) -> None:
    """取消进行中的 progressive 任务，清空片段缓冲区，重置 segId。

    VAD 断句时和连接关闭时调用，确保新语音段从头开始。
    """
    if self._progressive_task is not None and not self._progressive_task.done():
        self._progressive_task.cancel()
    self._progressive_task = None
    self._progressive_audio_frames = []
    self._progressive_total_samples = 0
    self._progressive_start_sample = 0
    self._progressive_last_send_time = 0.0
    self._progressive_seg_id = 0
```

修改 `close()` 方法，增加 progressive 清理：

```python
def close(self) -> None:
    """释放资源：取消后台任务 + 取消 progressive + 从 VAD 注销 + 销毁 Opus 解码器。"""
    self.cancel_pending_asr()
    self.cancel_progressive()  # ← 新增
    self.vad.close()
    if self._opus_decoder is not None:
        self._opus_decoder.close()
        self._opus_decoder = None
```

### 5. `src/api/websocket.py` — 核心改动

#### 5.1 新增 progressive 服务实例变量

```python
# 服务实例（由 main.py 生命周期管理器初始化）
asr_service: ASRService = ASRService()
itn_pool: ITNPool = ITNPool()
progressive_asr_service: ASRService | None = None  # 仅 PROGRESSIVE_ENABLED 时初始化
```

#### 5.2 修改 `_handle_audio_frame` — 增加 progressive 调度

在现有 VAD 处理之后，增加 progressive 调度逻辑：

```python
async def _handle_audio_frame(websocket, session, msg):
    # ... 原有音频解码 + VAD 处理 ...

    segments = await session.vad.feed_audio(pcm_int16)

    # ... 原有音频到达延时诊断 ...

    # 对每个触发的语音段，启动后台 ASR+ITN 任务
    for seg in segments:
        session.cancel_progressive()    # ← 新增：断句时取消 progressive + 清空缓冲 + 重置 segId
        task = asyncio.create_task(_process_segment(websocket, session, seg))
        session.track_asr_task(task)

    # ---- 新增：伪流式 progressive 调度 ----
    if (settings.PROGRESSIVE_ENABLED
            and progressive_asr_service is not None
            and not segments                           # 本帧未触发断句
            and session.vad.in_speech):                # 当前在语音段中
        # 计算本帧在全局样本流中的起始位置
        frame_start_sample = session._accumulated_audio_samples - len(pcm_int16)
        session.append_progressive_audio(pcm_int16, frame_start_sample)

        now = time.monotonic()
        elapsed = now - session._progressive_last_send_time
        task_idle = (session._progressive_task is None
                     or session._progressive_task.done())

        if elapsed >= settings.PROGRESSIVE_STEP and task_idle:
            audio, start_sample, end_sample = session.pop_progressive_audio()
            if len(audio) > 0:
                session._progressive_task = asyncio.create_task(
                    _process_progressive(websocket, session, audio, start_sample, end_sample)
                )
```

#### 5.3 新增 `_process_progressive` 函数

```python
async def _process_progressive(
    websocket: WebSocket,
    session: ASRSession,
    audio_int16: np.ndarray,
    start_sample: int,
    end_sample: int,
) -> None:
    """0.6B 模型 progressive 推理，结果立即回复客户端。

    不走 push_result_in_order 排序，直接通过 send_lock 互斥发送。
    不做 ITN 后处理（中间预览结果无需精确标点）。
    """
    seg_id = session.next_progressive_seg_id()
    audio_ms = len(audio_int16) / 16.0

    try:
        raw_text = await progressive_asr_service.recognize(
            audio_int16, sr=16000, context=session.hotword_context
        )

        bg_ms = samples_to_ms(start_sample)
        ed_ms = samples_to_ms(end_sample)

        result = ResultPayload(
            segId=seg_id,
            bg=bg_ms,
            ed=ed_ms,
            msgtype="progressive",
            ws=[WSItem(cw=[CWItem(w=raw_text, wp="n")])],
        )

        response = ServerMessage(
            header=ResponseHeader(
                code=0,
                message="progressive",
                sid=session.sid,
                traceId=session.trace_id,
                status=1,
            ),
            payload=ResponsePayloadWrapper(result=result),
        )

        async with session.send_lock:
            await websocket.send_text(response.model_dump_json())

        logger.debug(
            "Progressive sent: seg_id=%d, text=%s, audio=%.0fms, pos=[%d-%d]ms",
            seg_id, raw_text, audio_ms, bg_ms, ed_ms,
        )

    except asyncio.CancelledError:
        logger.debug("Progressive task cancelled: sid=%s, seg_id=%d", session.sid, seg_id)
    except (WebSocketDisconnect, ClientDisconnected):
        logger.debug("Client disconnected during progressive send: sid=%s, seg_id=%d", session.sid, seg_id)
    except Exception as exc:
        logger.warning("Progressive ASR failed: sid=%s, error=%s", session.sid, exc)
```

#### 5.4 修改 `_handle_end_frame` — 增加 progressive 清理

在函数开头增加 progressive 取消，防止终态 status=2 之后收到过时的 progressive：

```python
async def _handle_end_frame(websocket: WebSocket, session: ASRSession) -> None:
    """处理结束帧：刷空 VAD 缓冲区，等待所有 ASR 任务完成，推送终态。"""
    session.set_closing()

    # ← 新增：取消进行中的 progressive，防止在 status=2 之后发送
    session.cancel_progressive()

    # 强制刷出残余音频
    seg = session.vad.flush()
    if seg is not None:
        task = asyncio.create_task(
            _process_segment(websocket, session, seg, is_final=True)
        )
        session.track_asr_task(task)

    # 等待所有后台 ASR 任务完成
    await session.wait_pending_asr()

    # 发送终态 (status=2)
    ...
```

### 6. `main.py` — 双 vLLM 子进程管理

这是本次改动最大的部分。需要将当前的单一 `VLLMManager` 改造为支持两个 vLLM 实例。

#### 6.1 扩展 VLLMManager 支持具名实例

```python
import dataclasses
from typing import ClassVar

@dataclasses.dataclass
class VLLMInstance:
    """单个 vLLM 子进程的配置和状态。"""
    port: int
    model_path: str
    model_name: str
    process: subprocess.Popen | None = None

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/models"

    def build_cmd(self, tensor_parallel_size: int, max_model_len: int,
                  gpu_memory_utilization: float, extra_args: str) -> list[str]:
        """构建 vllm serve 命令行，与 1.7B 使用完全相同的参数模板。"""
        cmd = [
            "vllm", "serve", self.model_path,
            "--served-model-name", self.model_name,
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--port", str(self.port),
        ]
        if extra_args:
            cmd.extend(extra_args.split())
        return cmd


class VLLMManager:
    """vLLM 子进程生命周期管理（支持 1.7B + 0.6B 双实例）。"""

    primary: ClassVar[VLLMInstance | None] = None    # 1.7B
    progressive: ClassVar[VLLMInstance | None] = None  # 0.6B

    @classmethod
    def _start_instance(cls, instance: VLLMInstance, name: str = "vLLM") -> bool:
        """启动单个 vLLM 实例，阻塞等待就绪。

        启动日志格式与现有 1.7B VLLMManager.start() 保持一致：
          - Starting vLLM: <完整命令行>
          - vLLM started, PID: <pid>
          - Waiting for vLLM at <url> (timeout=<n>s)...
          - vLLM is ready (took <n>s)
        """
        if instance.process is not None and instance.process.poll() is None:
            logger.info("vLLM %s process already running: port=%d", name, instance.port)
            return True

        cmd = instance.build_cmd(
            settings.VLLM_TENSOR_PARALLEL_SIZE,
            settings.VLLM_MAX_MODEL_LEN,
            settings.VLLM_GPU_MEMORY_UTILIZATION,
            settings.VLLM_EXTRA_ARGS,
        )
        logger.info("Starting vLLM (%s): %s", name, " ".join(cmd))

        try:
            instance.process = subprocess.Popen(cmd, preexec_fn=os.setsid)
            logger.info("vLLM (%s) started, PID: %d", name, instance.process.pid)
        except Exception:
            logger.exception("Failed to start vLLM (%s)", name)
            return False

        timeout = (settings.PROGRESSIVE_STARTUP_TIMEOUT
                   if instance.port == settings.PROGRESSIVE_VLLM_PORT
                   else settings.VLLM_STARTUP_TIMEOUT)
        health_url = instance.health_url()
        deadline = time.monotonic() + timeout

        logger.info("Waiting for vLLM (%s) at %s (timeout=%ds)...", name, health_url, timeout)
        while time.monotonic() < deadline:
            if instance.process.poll() is not None:
                logger.error("vLLM (%s) exited with code %d", name, instance.process.returncode)
                return False
            try:
                r = httpx.get(health_url, timeout=2)
                if r.status_code == 200:
                    elapsed = timeout - (deadline - time.monotonic())
                    logger.info("vLLM (%s) is ready (took %.1fs)", name, elapsed)
                    return True
            except Exception:
                pass
            time.sleep(2)

        logger.error("vLLM (%s) did not become ready within %ds", name, timeout)
        cls._stop_instance(instance)
        return False

    @classmethod
    def _stop_instance(cls, instance: VLLMInstance, name: str = "vLLM") -> None:
        """关闭单个 vLLM 实例及其进程组。"""
        if instance.process is None:
            return
        try:
            pgid = os.getpgid(instance.process.pid)
        except OSError:
            instance.process = None
            return
        logger.info("Stopping vLLM (%s) process group (PGID %d)...", name, pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
        try:
            instance.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM (%s) did not stop, killing...", name)
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
            instance.process.wait()
        instance.process = None
        logger.info("vLLM (%s) stopped", name)

    @classmethod
    def start_all(cls) -> bool:
        """启动所有 vLLM 实例。先启动 1.7B，再启动 0.6B。任一失败均返回 False。"""
        # 1. 启动 1.7B（必须成功）
        cls.primary = VLLMInstance(
            port=settings.VLLM_PORT,
            model_path=settings.VLLM_MODEL_PATH,
            model_name=settings.VLLM_MODEL_NAME,
        )
        if not cls._start_instance(cls.primary, name="1.7B"):
            return False

        # 2. 条件启动 0.6B（必须成功，失败则整个服务退出触发容器重启）
        if settings.PROGRESSIVE_ENABLED:
            if not settings.PROGRESSIVE_VLLM_MODEL_PATH:
                logger.critical("PROGRESSIVE_ENABLED=true but PROGRESSIVE_VLLM_MODEL_PATH is empty")
                return False
            cls.progressive = VLLMInstance(
                port=settings.PROGRESSIVE_VLLM_PORT,
                model_path=settings.PROGRESSIVE_VLLM_MODEL_PATH,
                model_name=settings.PROGRESSIVE_MODEL_NAME,
            )
            if not cls._start_instance(cls.progressive, name="0.6B"):
                logger.critical("Progressive vLLM (0.6B) failed to start")
                return False
            logger.info("Progressive vLLM (0.6B) started successfully")

        return True

    @classmethod
    def stop_all(cls) -> None:
        """关闭所有 vLLM 实例。"""
        for inst in (cls.progressive, cls.primary):
            if inst is not None:
                cls._stop_instance(inst)
        cls.primary = None
        cls.progressive = None

    @classmethod
    def is_alive(cls, instance: VLLMInstance | None) -> bool:
        """检查指定 vLLM 实例是否存活。"""
        if instance is None:
            return False
        return instance.process is not None and instance.process.poll() is None
```

#### 6.2 扩展健康监测支持双实例

```python
MAX_HEALTH_FAILURES = 3

async def _health_monitor(instance: VLLMInstance, name: str) -> None:
    """后台持续监测单个 vLLM 实例健康，连续多次失败后触发优雅关闭。"""
    interval = settings.VLLM_HEALTH_CHECK_INTERVAL
    consecutive_failures = 0

    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            await asyncio.sleep(interval)

            if not VLLMManager.is_alive(instance):
                logger.critical("vLLM %s process died, triggering shutdown", name)
                if _shutdown_event:
                    _shutdown_event.set()
                return

            try:
                r = await client.get(instance.health_url(), headers={"Connection": "close"})
                if r.status_code == 200:
                    if consecutive_failures > 0:
                        logger.info("vLLM %s health recovered after %d failures", name, consecutive_failures)
                    consecutive_failures = 0
                    continue
                else:
                    logger.warning("vLLM %s health check returned %d", name, r.status_code)
            except Exception:
                logger.warning("vLLM %s health check request failed", name, exc_info=True)

            consecutive_failures += 1
            logger.warning("vLLM %s health check failed (%d/%d)", name, consecutive_failures, MAX_HEALTH_FAILURES)
            if consecutive_failures >= MAX_HEALTH_FAILURES:
                logger.critical("vLLM %s health check failed %d consecutive times, triggering shutdown", name, MAX_HEALTH_FAILURES)
                if _shutdown_event:
                    _shutdown_event.set()
                return
```

#### 6.3 更新 lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    # ---- 启动 ----
    logger.info("Starting ASR service on %s:%d", settings.WS_HOST, settings.WS_PORT)

    # 1. 启动 vLLM（1.7B + 可选 0.6B），同步阻塞通过线程池执行
    ok = await asyncio.to_thread(VLLMManager.start_all)
    if not ok:
        logger.critical("vLLM start failed, exiting")
        sys.exit(1)

    # 2. 启动健康监测（1.7B）
    primary_monitor = asyncio.create_task(
        _health_monitor(VLLMManager.primary, "1.7B")
    )

    # 3. 启动 0.6B 健康监测（PROGRESSIVE_ENABLED 时必定存在，因为启动失败已 exit）
    progressive_monitor = None
    if VLLMManager.progressive is not None:
        progressive_monitor = asyncio.create_task(
            _health_monitor(VLLMManager.progressive, "0.6B")
        )

    # 4. shutdown 监听
    async def _watch_shutdown():
        await _shutdown_event.wait()
        logger.critical("Shutdown event received, stopping server...")
        os.kill(os.getpid(), signal.SIGTERM)

    shutdown_watcher = asyncio.create_task(_watch_shutdown())

    # 5. ITN 多进程池
    itn_pool.start()
    logger.info("ITN pool ready: %d workers", itn_pool.num_workers)

    # 6. ASR HTTP 客户端（1.7B）
    await asr_service.startup()

    # 7. Progressive ASR 客户端（0.6B，PROGRESSIVE_ENABLED 时必定存在）
    if settings.PROGRESSIVE_ENABLED:
        ws_module.progressive_asr_service = ASRService(
            api_base=settings.PROGRESSIVE_API_BASE,
            model_name=settings.PROGRESSIVE_MODEL_NAME,
        )
        await ws_module.progressive_asr_service.startup()
        logger.info(
            "Progressive ASR service started: api=%s, model=%s",
            settings.PROGRESSIVE_API_BASE,
            settings.PROGRESSIVE_MODEL_NAME,
        )

    logger.info("All services initialized")

    yield

    # ---- 关闭 ----
    logger.info("Shutting down ASR service...")
    for t in (primary_monitor, progressive_monitor, shutdown_watcher):
        if t is not None:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
    await asr_service.shutdown()
    if ws_module.progressive_asr_service is not None:
        await ws_module.progressive_asr_service.shutdown()
    itn_pool.shutdown()
    VLLMManager.stop_all()
    logger.info("Shutdown complete")
```

### 7. `src/models/schemas.py` — 无改动

`msgtype` 字段已是 `str` 类型，`"progressive"` 可直接作为值使用。

---

### 8. Dockerfile — 微小改动

镜像构建无需改动（两个模型的权重都通过 Docker Volume 挂载，不入镜像）。仅需确保 0.6B 的 healthcheck 端口可用。当前 Dockerfile 不 EXPOSE 任何端口（端口映射在 k8s/docker-compose 层处理），因此 **Dockerfile 无需修改**。

### 9. K8s 部署配置 — `tmgenius-docker-800i-a2-v2-single-server.yaml`

```yaml
  tmgenius-asr-qwen:
    image: tmgenius/vllm-ascend:v0.19.1rc1
    container_name: tmgenius-asr-qwen
    restart: unless-stopped

    ports:
      - "8856:8856"    # ASR WebSocket
      - "15002:15002"  # 1.7B vLLM
      - "15003:15003"  # 0.6B vLLM（新增，调试用）

    volumes:
      # ... 原有 volumes ...
      - ./weights/Qwen3-ASR-1.7B:/weights/Qwen3-ASR-1.7B:ro
      - ./weights/Qwen3-ASR-0.6B:/weights/Qwen3-ASR-0.6B:ro  # ← 新增

    environment:
      # ... 原有 env vars ...

      # ---- Progressive 伪流式（新增）----
      PROGRESSIVE_ENABLED: "true"
      PROGRESSIVE_STEP: "0.6"
      PROGRESSIVE_VLLM_PORT: "15003"
      PROGRESSIVE_VLLM_MODEL_PATH: "/weights/Qwen3-ASR-0.6B"
      PROGRESSIVE_MODEL_NAME: "Qwen3-ASR-0.6B"
      PROGRESSIVE_HEALTH_CHECK_INTERVAL: "5"
      PROGRESSIVE_STARTUP_TIMEOUT: "1200"

    # devices, healthcheck, logging 等保持不变
    ...
```

---

## NPU 显存预算

单张 Ascend 800I A2 拥有 64GB HBM。两个模型显存估算：

| 模型 | 参数量 | 估算显存（bf16） |
|------|--------|------------------|
| Qwen3-ASR-1.7B | 1.7B | ~3.4 GB（权重）+ KV Cache |
| Qwen3-ASR-0.6B | 0.6B | ~1.2 GB（权重）+ KV Cache |
| **合计** | **2.3B** | **~5 GB + KV Cache × 2** |

当前 `gpu-memory-utilization=0.6`，即每个实例最多使用 64 × 0.6 = 38.4 GB。两个实例各设 0.6 会争抢显存。**建议 0.6B 使用更低的 `gpu-memory-utilization`（如 0.2）**，或改为两个实例各自保守设置（0.6 + 0.2 = 相对安全，且实际权重只有 ~1.2GB）。

> ⚠️ **注意**：当前实现中两个 vLLM 实例共用 `VLLM_GPU_MEMORY_UTILIZATION` 配置（第 6.1 节 `build_cmd` 方法）。如需为 0.6B 单独设置，需新增 `PROGRESSIVE_GPU_MEMORY_UTILIZATION` 配置项。建议 v1 先使用相同值，观察实际显存使用后再决定是否需要独立配置。

---

## 与原始测试脚本的差异

| | `test/pseudo_streaming_asr.py` | 服务端集成 |
|---|---|---|
| 音频来源 | 整段 WAV 文件 | 实时 WebSocket 流 |
| 切片方式 | 按 step 机械切片 | 按 VAD in_speech 状态动态累积 |
| 步进触发 | 固定间隔 `asyncio.sleep(step)` | 基于音频帧到达 + wall-clock 计时器 |
| 断句感知 | 无 | VAD 断句时自动清空重置 |
| 音频编码 | float32 → WAV → base64 | int16 PCM → WAV → base64（复用 `_encode_audio_to_data_url`） |
| ASR 客户端 | OpenAI SDK (`AsyncOpenAI`) | httpx 直接调用（`ASRService.recognize`） |

---

## 风险评估

### 高风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **NPU 显存不足** | 0.6B + 1.7B 同时运行可能 OOM | 两个模型权重合计仅 ~4.6GB；必要时为 0.6B 设置更低 `gpu-memory-utilization` |
| **双 vLLM 启动时间过长** | 容器启动耗时增加 | 0.6B 模型小，启动快（~30s）；1.7B 启动不受影响 |
| **progressive 推理堆积** | 0.6B 推理慢于 0.6s 间隔时请求积压 | "上一个未完成则跳过"策略；监控 progressive 延迟 |
| **send_lock 竞争加剧** | progressive 每 0.6s 发送占用 send_lock，可能延迟 sentence 推送 | VAD 断句时 cancel progressive 保证 sentence 优先；send_lock 持有时间极短（仅 send_text） |

### 中风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **0.6B vLLM 启动失败** | progressive 功能不可用，整个服务退出 | 启动失败 → `sys.exit(1)` → 容器自动重启（`restart: unless-stopped`），与 1.7B 失败策略一致 |
| **progressive 与 sentence 交错** | 客户端收到 progressive 后紧接 sentence | VAD 断句时 cancel；`_handle_end_frame` 时 cancel；利用 `asyncio.Lock` 互斥 |
| **片段模式识别准确率** | 0.6B 仅看到 0.6s 片段，缺少上下文 | 预期行为，"预览"级别准确率可接受 |
| **progressive 频繁 base64 编码** | 每 0.6s 编码 ~19KB 音频为 data URL | CPU 操作开销 < 1ms，可接受 |

### 低风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **0.6B 模型精度低** | progressive 预览文字不准确 | 预期行为，仅作预览 |
| **功能开关失效** | `PROGRESSIVE_ENABLED=false` 时残留逻辑执行 | config + 代码入口双重检查 |
| **segId 独立空间导致客户端无法关联** | 客户端不知道 progressive 对应哪个 sentence | 客户端可通过时间戳 (bg/ed) 关联 |
| **两个 vLLM 子进程管理复杂度** | 僵尸进程、资源泄漏 | VLLMManager 统一使用 `os.setsid` + `os.killpg` 模式，已在 1.7B 上验证 |

---

## 验证方案

1. **单元测试**：测试 `session.py` 新增的 progressive 状态管理方法（append/pop/cancel/reset）
2. **集成测试**：通过 WebSocket 连接验证：
   - 每隔 ~0.6s 收到 `msgtype=progressive` 的中间结果
   - progressive 的 `segId` 从 0 开始独立递增
   - VAD 断句后收到 `msgtype=sentence` 的最终结果
   - 新语音段开始时 progressive `segId` 重置为 0
   - `status=2` 终态正常发送，且之后无 progressive 消息
   - `PROGRESSIVE_ENABLED=false` 时行为与改动前完全一致
3. **NPU 显存验证**：启动后通过 `npu-smi info` 检查显存使用，确认两个模型同时加载不 OOM
4. **压测**：多连接并发场景下验证 progressive 不会导致延迟恶化或内存泄漏
