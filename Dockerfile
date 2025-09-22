# ====== PyTorch 2.6 + CUDA 12.4 ======
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG USE_TUNA=1
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates tzdata curl wget git \
      ffmpeg libsndfile1 tini && \
    rm -rf /var/lib/apt/lists/*

# 国内构建可走清华源
RUN if [ "$USE_TUNA" = "1" ]; then \
      python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
      python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn; \
    fi

RUN python -m pip install --upgrade pip && \
    python - <<'PY'
import torch
print(">>> Base torch:", torch.__version__)
assert torch.__version__.startswith("2.6"), "Base image torch must be 2.6.x"
PY

WORKDIR /workspace/app
COPY constraints.txt ./constraints.txt
RUN python -m pip install --no-cache-dir -r constraints.txt

COPY requirements.txt ./requirements.txt
RUN if [ -s requirements.txt ]; then \
      python -m pip install --no-cache-dir -r requirements.txt; \
    fi

# 把你的代码打进镜像（包含 asr_worker.py / modules 等）
COPY . /workspace/app

ENV HOST=0.0.0.0 \
    PORT=8000 \
    ASR_ENGINE=whisperx \
    DIARIZE=auto

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/healthz" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
EXPOSE 8000
CMD ["bash", "-lc", "uvicorn asr_worker:app --host ${HOST} --port ${PORT} --workers 1"]