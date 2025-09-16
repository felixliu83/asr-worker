#!/usr/bin/env bash
set -euo pipefail

: "${GIT_URL:?need GIT_URL}"
: "${REPO_DIR:=/workspace/app}"
: "${WORKDIR:=/workspace}"

mkdir -p "$REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[bootstrap] clone repo..."
  git clone --depth 1 "$GIT_URL" "$REPO_DIR"
else
  echo "[bootstrap] pull repo..."
  git -C "$REPO_DIR" pull --rebase
fi

cd "$REPO_DIR"

# Python venv（放到网络卷，持久化依赖）
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip

if [ -f requirements.txt ]; then
  echo "[bootstrap] install requirements..."
  # GPU CUDA 12 的 torch 源
  pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
fi

# 运行（热重载），应用可在仓库里的 asr_worker.py
exec uvicorn asr_worker:app --host 0.0.0.0 --port 8000 --reload