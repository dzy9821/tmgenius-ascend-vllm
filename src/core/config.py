"""
统一配置管理 —— 所有字段均可通过同名环境变量覆盖，代码内提供默认值。
"""

import os


class Settings:
    """服务全局配置。未来通过 docker-compose environment 段注入。"""

    # ---- 服务参数 ----
    WS_HOST: str = os.getenv("WS_HOST", "0.0.0.0")
    WS_PORT: int = int(os.getenv("WS_PORT", "8856"))
    MAX_CONNECTIONS: int = int(os.getenv("MAX_CONNECTIONS", "64"))
    HANDSHAKE_TIMEOUT: int = int(os.getenv("HANDSHAKE_TIMEOUT", "5"))
    WS_PING_INTERVAL: float = float(os.getenv("WS_PING_INTERVAL", "5"))
    WS_PING_TIMEOUT: float = float(os.getenv("WS_PING_TIMEOUT", "20"))

    # ---- ITN 多进程池 ----
    ITN_WORKERS: int = int(os.getenv("ITN_WORKERS", "8"))

    # ---- vLLM ----
    VLLM_PORT: int = int(os.getenv("VLLM_PORT", "15002"))
    VLLM_API_BASE: str = os.getenv(
        "VLLM_API_BASE", f"http://127.0.0.1:{VLLM_PORT}/v1"
    )
    VLLM_MODEL_NAME: str = os.getenv("VLLM_MODEL_NAME", "Qwen3-ASR-1.7B")
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "EMPTY")
    VLLM_MODEL_PATH: str = os.getenv(
        "VLLM_MODEL_PATH", "/weights/Qwen3-ASR-1.7B"
    )
    VLLM_TENSOR_PARALLEL_SIZE: int = int(
        os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")
    )
    VLLM_MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "32768"))
    VLLM_GPU_MEMORY_UTILIZATION: float = float(
        os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.6")
    )
    VLLM_EXTRA_ARGS: str = os.getenv("VLLM_EXTRA_ARGS", "")
    VLLM_STARTUP_TIMEOUT: int = int(os.getenv("VLLM_STARTUP_TIMEOUT", "3000"))
    """vLLM 启动超时（秒）。"""
    VLLM_HEALTH_CHECK_INTERVAL: float = float(
        os.getenv("VLLM_HEALTH_CHECK_INTERVAL", "5")
    )
    """vLLM 健康检查间隔（秒）。"""

    # ---- NPU ----
    ASCEND_RT_VISIBLE_DEVICES: str = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0")

    # ---- 热词 ----
    HOTWORDS: str = os.getenv("HOTWORDS", "")  # 逗号分隔，如 "张三丰,武当山,太极拳"

    # ---- 日志 ----
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MP_QUEUE_LOG_INTERVAL_SEC: float = float(
        os.getenv("MP_QUEUE_LOG_INTERVAL_SEC", "10")
    )

    # ---- ASR 音频填充 ----
    ASR_PAD_FRAMES: int = int(os.getenv("ASR_PAD_FRAMES", "5"))
    """送给 ASR 时首尾各附加的真实音频上下文帧数（帧长 = VAD_HOP_SIZE samples），替代静默填充。"""

    # ---- 伪流式 Progressive ----
    PROGRESSIVE_ENABLED: bool = os.getenv("PROGRESSIVE_ENABLED", "false").lower() == "true"
    """是否启用伪流式 progressive 推理。"""

    PROGRESSIVE_STEP: float = float(os.getenv("PROGRESSIVE_STEP", "0.6"))
    """progressive 推理步进间隔（秒）。"""

    # Progressive vLLM 子进程启动参数（与 1.7B 参数一致，仅端口和模型不同）
    PROGRESSIVE_VLLM_PORT: int = int(os.getenv("PROGRESSIVE_VLLM_PORT", "15003"))
    """0.6B vLLM 子进程监听端口。"""

    PROGRESSIVE_VLLM_MODEL_PATH: str = os.getenv(
        "PROGRESSIVE_VLLM_MODEL_PATH", "/weights/Qwen3-ASR-0.6B"
    )
    """0.6B 模型权重路径。"""

    PROGRESSIVE_MODEL_NAME: str = os.getenv("PROGRESSIVE_MODEL_NAME", "Qwen3-ASR-0.6B")
    """0.6B 模型 served-model-name。"""

    PROGRESSIVE_API_BASE: str = os.getenv(
        "PROGRESSIVE_API_BASE",
        f"http://127.0.0.1:{PROGRESSIVE_VLLM_PORT}/v1",
    )
    """0.6B vLLM 的 API 地址。"""

    PROGRESSIVE_HEALTH_CHECK_INTERVAL: float = float(
        os.getenv("PROGRESSIVE_HEALTH_CHECK_INTERVAL", "5")
    )
    """0.6B vLLM 健康检查间隔（秒）。"""

    PROGRESSIVE_STARTUP_TIMEOUT: int = int(
        os.getenv("PROGRESSIVE_STARTUP_TIMEOUT", "1200")
    )
    """0.6B vLLM 启动超时（秒）。"""

    PROGRESSIVE_TENSOR_PARALLEL_SIZE: int = int(
        os.getenv("PROGRESSIVE_TENSOR_PARALLEL_SIZE", "1")
    )
    """0.6B vLLM 张量并行数。"""

    PROGRESSIVE_MAX_MODEL_LEN: int = int(
        os.getenv("PROGRESSIVE_MAX_MODEL_LEN", "32768")
    )
    """0.6B vLLM 最大模型长度。"""

    PROGRESSIVE_GPU_MEMORY_UTILIZATION: float = float(
        os.getenv("PROGRESSIVE_GPU_MEMORY_UTILIZATION", "0.3")
    )
    """0.6B vLLM GPU 显存利用率。"""

    PROGRESSIVE_EXTRA_ARGS: str = os.getenv("PROGRESSIVE_EXTRA_ARGS", "")
    """0.6B vLLM 额外启动参数。"""

    # ---- VAD 动态断句阈值 ----
    VAD_HOP_SIZE: int = int(os.getenv("VAD_HOP_SIZE", "640"))
    """VAD 帧长（采样数），16kHz 下 640 = 40ms，对齐客户端发送间隔。"""
    VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.4"))
    """VAD 语音概率阈值 [0.0, 1.0]，>= 此值判定为语音帧。"""
    VAD_PAUSE_MAX: float = float(os.getenv("VAD_PAUSE_MAX", "0.7"))
    """累积语音 0s 时所需停顿（秒）。"""
    VAD_PAUSE_MIN: float = float(os.getenv("VAD_PAUSE_MIN", "0.35"))
    """累积语音 >= DYNAMIC_RANGE_END 时所需停顿（秒）。"""
    VAD_DYNAMIC_RANGE_END: float = float(os.getenv("VAD_DYNAMIC_RANGE_END", "20.0"))
    """动态线性区间终点（秒），超过此值使用 VAD_PAUSE_MIN。"""
    VAD_MIN_SPEECH: float = float(os.getenv("VAD_MIN_SPEECH", "0.5"))
    """短音频抑制门限（秒），不足则不转发。"""
    VAD_MAX_SPEECH: float = float(os.getenv("VAD_MAX_SPEECH", "30.0"))
    """长音频强制触发门限（秒），立即转发。"""


# 全局单例
settings = Settings()
