ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000 \
    REPO_DIR=/workspace/app \
    REPO_URL=https://github.com/felixliu83/asr-worker.git \
    REPO_REF=main \
    WATCHGOD_FORCE_POLLING=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    bash -euxo pipefail -c '\
      export DEBIAN_FRONTEND=noninteractive; \
      # 1) 尝试官方源 + 重试
      apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 update || true; \
      # 2) 若 update 或 install 失败，自动切换到日本/Cloudflare 镜像再试
      install_pkgs() { \
        apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 install -y --no-install-recommends "$@" || return 1; \
      }; \
      if ! install_pkgs git ffmpeg ca-certificates curl tini; then \
        echo "[apt] switch to JP mirror"; \
        sed -i "s|http://archive.ubuntu.com/ubuntu/|http://jp.archive.ubuntu.com/ubuntu/|g" /etc/apt/sources.list; \
        apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 update || true; \
        if ! install_pkgs git ffmpeg ca-certificates curl tini; then \
          echo "[apt] switch to Cloudflare mirror"; \
          sed -i "s|http://[^ ]*/ubuntu/|http://mirror.cloudflare.com/ubuntu/|g" /etc/apt/sources.list; \
          apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 update; \
          install_pkgs git ffmpeg ca-certificates curl tini; \
        fi; \
      fi; \
      # 清理缓存（保留 apt 缓存 mount 会自动复用）
      apt-get clean; \
      rm -rf /var/lib/apt/lists/* \
    '

WORKDIR /workspace

# 依赖固定（只装依赖，不打代码）
COPY constraints.txt requirements.txt /tmp/dep/
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/dep/constraints.txt && \
    python -m pip install --no-cache-dir -r /tmp/dep/requirements.txt

# 入口：启动时拉/更新代码 + uvicorn --reload
RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -euo pipefail' \
': "${REPO_URL:=https://github.com/felixliu83/asr-worker.git}"' \
': "${REPO_REF:=main}"' \
': "${REPO_DIR:=/workspace/app}"' \
': "${UVICORN_HOST:=0.0.0.0}"' \
': "${UVICORN_PORT:=8000}"' \
': "${WATCHGOD_FORCE_POLLING:=1}"' \
'echo "[dev] repo=$REPO_URL ref=$REPO_REF dir=$REPO_DIR"' \
'mkdir -p "$REPO_DIR"' \
'' \
'# 初次 clone：不用浅克隆，改用 blob 过滤，保留完整引用结构' \
'if [ ! -d "$REPO_DIR/.git" ]; then' \
'  echo "[dev] clone (no shallow, blob filter)";' \
'  git clone --filter=blob:none --tags "$REPO_URL" "$REPO_DIR"' \
'fi' \
'' \
'# 保证能抓到所有远端分支，并清理陈旧引用' \
'git -C "$REPO_DIR" config --replace-all remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"' \
'git -C "$REPO_DIR" fetch origin --tags --prune' \
'' \
'# 分支优先（存在 origin/$REPO_REF 就切分支；否则尝试 tag）' \
'if git -C "$REPO_DIR" ls-remote --heads origin "$REPO_REF" | grep -q "$REPO_REF"; then' \
'  echo "[dev] switch to branch $REPO_REF";' \
'  git -C "$REPO_DIR" switch -C "$REPO_REF" --track "origin/$REPO_REF" 2>/dev/null || \ ' \
'  git -C "$REPO_DIR" checkout -B "$REPO_REF" "origin/$REPO_REF"' \
'  git -C "$REPO_DIR" reset -q --hard "origin/$REPO_REF"' \
'else' \
'  echo "[dev] $REPO_REF not a branch; try tag";' \
'  git -C "$REPO_DIR" checkout -q "tags/$REPO_REF" || { echo "[dev] cannot find tag $REPO_REF"; exit 1; }' \
'fi' \
'' \
'export PYTHONPATH="$REPO_DIR"' \
'export WATCHGOD_FORCE_POLLING' \
'' \
'# 简单自检：能 import asr_worker:app' \
'python3 - <<PY || { echo "[dev] import failed"; exit 1; }' \
'import importlib, os, sys' \
'sys.path.insert(0, os.environ.get("REPO_DIR","/workspace/app"))' \
'm = importlib.import_module("asr_worker")' \
'assert hasattr(m,"app"), "asr_worker.app not found"' \
'print("[dev] import ok")' \
'PY' \
'' \
'echo "[dev] start uvicorn --reload"' \
'exec uvicorn --app-dir "$REPO_DIR" asr_worker:app --host "$UVICORN_HOST" --port "$UVICORN_PORT" --reload' \
> /usr/local/bin/dev-entrypoint.sh && chmod +x /usr/local/bin/dev-entrypoint.sh

ENTRYPOINT ["/usr/bin/tini","-s","--"]
EXPOSE 8000
CMD ["/usr/local/bin/dev-entrypoint.sh"]
