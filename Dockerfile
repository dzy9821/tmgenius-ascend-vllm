# 耦合 vLLM-Ascend v0.19.1rc1 目录布局（connections.py 补丁路径）
ARG BASE_IMAGE=quay.io/ascend/vllm-ascend:v0.19.1rc1
FROM ${BASE_IMAGE}

ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

# ---- 1. 运行时系统库 + healthcheck ----
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopus0 libsndfile1 curl && \
    rm -rf /var/lib/apt/lists/*

# ---- 2. OpenFst（需真实 ARM64 libfst*.so 置于 3rd-party/openfst/lib/）----
COPY 3rd-party/openfst1.8.3/lib/* /usr/local/lib/
COPY 3rd-party/openfst1.8.3/include/ /usr/local/include/fst/
RUN echo /usr/local/lib > /etc/ld.so.conf.d/openfst.conf && ldconfig

# ---- 3. Python 依赖（先 COPY 慢变文件，最大化构建缓存）----
COPY 3rd-party/*.whl /tmp/

# ITN: pynini + WeTextProcessing
RUN pip install /tmp/*.whl

# Qwen3-ASR 音频处理（必须，否则 vLLM 处理 audio_url 返回 400）
RUN pip install --no-cache-dir --no-deps 'qwen-asr[vllm]'

# 项目依赖（torch/vllm 已内置，不要重装以免破坏兼容）
RUN pip install --no-cache-dir \
    librosa \
    "torchaudio>=2.0.0" \
    "fastapi>=0.115.0" \
    "websockets>=12.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.5.0" \
    "numpy==1.26.4" \
    "httpx>=0.27.0" \
    "prometheus-client>=0.21.0" \
    "soundfile>=0.12.0"

# ---- 4. 应用代码（变更频繁，置于依赖层之后）----
WORKDIR /app
COPY main.py .
COPY src/ ./src/
COPY weights/ ./weights/

# ---- 5. VAD 路径（ARM aarch64），替换 C 库内部的相对路径 onnx_model ----
RUN ln -sf weights/vad/ten-vad/onnx_model /app/onnx_model
ENV LD_LIBRARY_PATH=/app/weights/vad/ten-vad/lib/Linux/aarch64:/usr/local/lib

# ---- 6. vLLM 补丁（路径依赖 BASE_IMAGE 版本）----
COPY src/connections.py /vllm-workspace/vllm/vllm/connections.py
COPY src/translation_proxy.py /workspace/translation_proxy.py
COPY healthcheck-vl-7b.sh /healthcheck-vl-7b.sh
RUN chmod +x /healthcheck-vl-7b.sh