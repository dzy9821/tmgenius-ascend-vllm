"""
异步 ASR 调用服务 —— 通过 httpx 调用 vLLM OpenAI 兼容接口。
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
from typing import Optional

import httpx
import numpy as np

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ASRError(Exception):
    """ASR 推理异常，携带 vLLM HTTP 状态码。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ASRService:
    """异步 ASR 推理服务，封装对 vLLM 的 HTTP 调用。"""

    def __init__(self, api_base: str | None = None, model_name: str | None = None) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._api_base = api_base or settings.VLLM_API_BASE
        self._model_name = model_name or settings.VLLM_MODEL_NAME

    async def startup(self) -> None:
        """初始化 HTTP 客户端（应用启动时调用）。"""
        # 显式禁用代理，等同于 unset http_proxy/https_proxy
        self._client = httpx.AsyncClient(timeout=60.0, proxy=None, trust_env=False)
        logger.info(
            "ASR service started, endpoint: %s, model: %s",
            self._api_base,
            self._model_name,
        )

    async def shutdown(self) -> None:
        """关闭 HTTP 客户端。"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def recognize(
        self,
        audio_int16: np.ndarray,
        sr: int = 16000,
        context: str = "",
    ) -> str:
        """
        将音频片段发送至 vLLM 进行 ASR 推理。

        Args:
            audio_int16: int16 PCM 音频
            sr: 采样率
            context: 热词/系统提示词

        Returns:
            清洗后的识别文本
        """
        assert self._client is not None, "ASRService not started"

        # CPU 密集型编码操作移出事件循环，避免阻塞 pong 响应
        loop = asyncio.get_running_loop()
        data_url = await loop.run_in_executor(
            None, _encode_audio_to_data_url, audio_int16, sr
        )

        messages: list[dict] = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": data_url}},
                ],
            }
        )

        url = f"{self._api_base}/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": messages,
        }
        headers = {}
        if settings.VLLM_API_KEY and settings.VLLM_API_KEY != "EMPTY":
            headers["Authorization"] = f"Bearer {settings.VLLM_API_KEY}"

        # 重试逻辑：连接错误重试，HTTP 错误码不重试（避免加剧 vLLM 负载）
        max_retries = 3
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await self._client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return _clean_asr_output(content if isinstance(content, str) else str(content))
            except httpx.HTTPStatusError as exc:
                raise ASRError(
                    f"vLLM returned {exc.response.status_code}",
                    status_code=exc.response.status_code,
                ) from exc
            except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                wait = 1.0 * (2 ** attempt)  # 1s, 2s, 4s
                logger.warning(
                    "ASR vLLM request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, wait, exc,
                )
                await asyncio.sleep(wait)

        raise last_exc  # type: ignore[misc]

    async def is_available(self) -> bool:
        """检查 vLLM 服务是否可达。"""
        if self._client is None:
            return False
        try:
            resp = await self._client.get(f"{self._api_base}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


# ---- 工具函数（复用 pipeline_common.py 的逻辑） ----


def _encode_audio_to_data_url(audio_int16: np.ndarray, sr: int) -> str:
    """将 int16 音频编码为 data:audio/wav;base64,... 格式。"""
    import soundfile as sf

    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    buf = io.BytesIO()
    sf.write(buf, audio_f32, sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def _clean_asr_output(text: str) -> str:
    """清洗 ASR 模型输出，移除 <asr_text> 等标记。"""
    text = text.strip()
    if "<asr_text>" not in text:
        return text
    parts = re.split(r"(?:language\s+[^\s<]+)?<asr_text>", text, flags=re.IGNORECASE)
    return "".join(part.strip() for part in parts if part.strip()).strip()


def build_hotword_context(hotwords: Optional[str]) -> str:
    """将客户端传入的热词构建为系统提示词。"""
    if not hotwords or not hotwords.strip():
        return ""
    words = [w.strip() for w in re.split(r"[,\n;，；、|]+", hotwords) if w.strip()]
    if not words:
        return ""
    return f"热词：{'、'.join(words)}"
