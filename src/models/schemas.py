"""
WebSocket 请求/响应 Pydantic 模型 —— 严格映射设计文档 §3 接口定义。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ============================================================
# 请求结构
# ============================================================


class RequestHeader(BaseModel):
    """客户端消息头。"""

    traceId: str
    appId: Optional[str] = None
    bizId: str
    status: int = Field(..., description="0=握手帧, 1=音频帧, 2=结束帧")
    resIdList: Optional[list[str]] = None


class AudioPayload(BaseModel):
    audio: str = Field(..., description="Base64 编码的音频数据")
    encoding: Optional[str] = Field(None, description="音频编码格式，不传或为 None 时按 PCM 16k/16bit 处理，opus 时按 Opus 编码处理")


class TextPayload(BaseModel):
    text: Optional[str] = Field(None, description="热词，用于提升识别准确率")


class RequestPayload(BaseModel):
    audio: Optional[AudioPayload] = None
    text: Optional[TextPayload] = None


class ClientMessage(BaseModel):
    """客户端发送的完整 JSON 消息。"""

    header: RequestHeader
    payload: Optional[RequestPayload] = None


# ============================================================
# 响应结构
# ============================================================


class ResponseHeader(BaseModel):
    """服务端消息头。"""

    code: int = 0
    message: str = "success"
    sid: str = Field(..., description="会话唯一标识，格式 AST_XXXX")
    traceId: str = ""
    status: int = Field(..., description="0=开始, 1=识别中, 2=结束")


class CWItem(BaseModel):
    """词汇级识别结果。"""

    w: str = Field(..., description="识别词汇")
    wp: str = Field("n", description="词性标记")
    wb: int = Field(0, description="词汇起始时间（厘秒 cs）")
    we: int = Field(0, description="词汇结束时间（厘秒 cs）")
    rl: int = Field(0, description="角色识别标记（当前不支持，默认 0）")
    sc: str = Field("0.00", description="分数")
    sf: int = Field(0, description="标志位")
    wc: str = Field("0.00", description="词置信度")


class WSItem(BaseModel):
    """词段。"""

    bg: int = Field(0, description="词段起始时间（厘秒 cs）")
    cw: list[CWItem] = Field(default_factory=list)


class ResultPayload(BaseModel):
    """识别结果体。"""

    segId: int = Field(0, description="段序号")
    bg: int = Field(0, description="句子起始时间（毫秒 ms）")
    ed: int = Field(0, description="句子结束时间（毫秒 ms）")
    msgtype: str = "sentence"
    ws: list[WSItem] = Field(default_factory=list)


class ResponsePayloadWrapper(BaseModel):
    result: ResultPayload


class ServerMessage(BaseModel):
    """服务端推送的完整 JSON 消息。"""

    header: ResponseHeader
    payload: ResponsePayloadWrapper
