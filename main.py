"""
ASR 实时流式转录服务入口。

启动方式：
    python main.py
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from typing import ClassVar

import httpx

from fastapi import FastAPI

from src.core.config import settings

# ---- 将 config.py 的 WS Ping 配置注入环境变量，供 uvicorn CLI 启动时自动读取 ----
os.environ.setdefault("UVICORN_WS_PING_INTERVAL", str(int(settings.WS_PING_INTERVAL)))
os.environ.setdefault("UVICORN_WS_PING_TIMEOUT", str(int(settings.WS_PING_TIMEOUT)))

from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.api.websocket import asr_service, itn_pool, router as ws_router
import src.api.websocket as ws_module
from src.core.logging import get_logger, setup_logging
from src.services.asr_service import ASRService

setup_logging()
logger = get_logger(__name__)


class VLLMInstance:
    """单个 vLLM 子进程的配置和状态。"""

    def __init__(self, port: int, model_path: str, model_name: str) -> None:
        self.port = port
        self.model_path = model_path
        self.model_name = model_name
        self.process: subprocess.Popen | None = None

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/models"

    def build_cmd(
        self,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        extra_args: str,
    ) -> list[str]:
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
    def _start_instance(cls, instance: VLLMInstance, name: str) -> bool:
        """启动单个 vLLM 实例，阻塞等待就绪。

        启动日志格式与原有 1.7B 单实例保持一致。
        """
        if instance.process is not None and instance.process.poll() is None:
            logger.info("vLLM (%s) process already running: port=%d", name, instance.port)
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
        cls._stop_instance(instance, name)
        return False

    @classmethod
    def _stop_instance(cls, instance: VLLMInstance, name: str) -> None:
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
        if not cls._start_instance(cls.primary, "1.7B"):
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
            if not cls._start_instance(cls.progressive, "0.6B"):
                logger.critical("Progressive vLLM (0.6B) failed to start")
                return False
            logger.info("Progressive vLLM (0.6B) started successfully")

        return True

    @classmethod
    def stop_all(cls) -> None:
        """关闭所有 vLLM 实例。"""
        for inst, name in ((cls.progressive, "0.6B"), (cls.primary, "1.7B")):
            if inst is not None:
                cls._stop_instance(inst, name)
        cls.primary = None
        cls.progressive = None

    @classmethod
    def is_alive(cls, instance: VLLMInstance | None) -> bool:
        """检查指定 vLLM 实例是否存活。"""
        if instance is None:
            return False
        return instance.process is not None and instance.process.poll() is None


# 用于通知 lifespan 进行优雅关闭的事件
_shutdown_event: asyncio.Event | None = None

MAX_HEALTH_FAILURES = 3  # 连续失败次数阈值


async def _health_monitor(instance: VLLMInstance, name: str) -> None:
    """后台持续监测单个 vLLM 实例健康，连续多次失败后通知优雅关闭。"""
    health_url = instance.health_url()
    interval = settings.VLLM_HEALTH_CHECK_INTERVAL
    consecutive_failures = 0

    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            await asyncio.sleep(interval)

            # 进程级检查
            if not VLLMManager.is_alive(instance):
                logger.critical("vLLM (%s) process died, triggering shutdown", name)
                if _shutdown_event:
                    _shutdown_event.set()
                return

            # HTTP 级检查
            try:
                r = await client.get(health_url, headers={"Connection": "close"})
                if r.status_code == 200:
                    if consecutive_failures > 0:
                        logger.info("vLLM (%s) health recovered after %d failures", name, consecutive_failures)
                    consecutive_failures = 0
                    continue
                else:
                    logger.warning("vLLM (%s) health check returned %d", name, r.status_code)
            except Exception:
                logger.warning("vLLM (%s) health check request failed", name, exc_info=True)

            consecutive_failures += 1
            logger.warning(
                "vLLM (%s) health check failed (%d/%d)",
                name, consecutive_failures, MAX_HEALTH_FAILURES,
            )
            if consecutive_failures >= MAX_HEALTH_FAILURES:
                logger.critical(
                    "vLLM (%s) health check failed %d consecutive times, triggering shutdown",
                    name, MAX_HEALTH_FAILURES,
                )
                if _shutdown_event:
                    _shutdown_event.set()
                return


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动/关闭资源。"""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    # ---- 启动 ----
    logger.info(
        "Starting ASR service on %s:%d (max_conn=%d, ping_interval=%.0f, ping_timeout=%.0f)",
        settings.WS_HOST,
        settings.WS_PORT,
        settings.MAX_CONNECTIONS,
        settings.WS_PING_INTERVAL,
        settings.WS_PING_TIMEOUT,
    )

    # 1. 启动 vLLM（1.7B → 0.6B，同步阻塞跑在线程池）
    ok = await asyncio.to_thread(VLLMManager.start_all)
    if not ok:
        logger.critical("vLLM start failed, exiting")
        sys.exit(1)

    # 2. 启动健康监测（1.7B）
    monitors: list[asyncio.Task] = []
    assert VLLMManager.primary is not None
    monitors.append(asyncio.create_task(_health_monitor(VLLMManager.primary, "1.7B")))

    # 3. 启动 0.6B 健康监测（PROGRESSIVE_ENABLED 时必定存在，启动失败已 exit）
    if VLLMManager.progressive is not None:
        monitors.append(asyncio.create_task(_health_monitor(VLLMManager.progressive, "0.6B")))

    # 4. shutdown 监听（健康检查失败时触发优雅退出）
    async def _watch_shutdown():
        await _shutdown_event.wait()
        logger.critical("Shutdown event received, stopping server...")
        os.kill(os.getpid(), signal.SIGTERM)

    shutdown_watcher = asyncio.create_task(_watch_shutdown())

    # 5. ITN 多进程池（eager init）
    itn_pool.start()
    logger.info("ITN pool ready: %d workers", itn_pool.num_workers)

    # 6. ASR HTTP 客户端（1.7B）
    await asr_service.startup()

    # 7. Progressive ASR 客户端（0.6B）
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
    shutdown_watcher.cancel()
    for t in monitors:
        t.cancel()
    for t in monitors + [shutdown_watcher]:
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


app = FastAPI(
    title="ASR Real-time Streaming Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ws_router)
app.include_router(health_router)
app.include_router(metrics_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.WS_HOST,
        port=settings.WS_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        ws_ping_interval=settings.WS_PING_INTERVAL,
        ws_ping_timeout=settings.WS_PING_TIMEOUT,
    )
