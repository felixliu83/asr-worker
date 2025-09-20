#!/usr/bin/env bash
set -euo pipefail

# -------- 基础环境 --------
: "${WORKDIR:=/workspace}"
: "${REPO_DIR:=/workspace/app}"
: "${GIT_URL:?need GIT_URL env}"
: "${PORT:=8000}"

: "${PIP_TIMEOUT:=60}"
: "${PIP_RETRIES:=3}"

echo "[bootstrap] base info"
uname -a || true
nvidia-smi || echo "[bootstrap] nvidia-smi not available (CPU or driver not mounted)"
echo "[bootstrap] ENV: WORKDIR=$WORKDIR REPO_DIR=$REPO_DIR PORT=$PORT"
echo "[bootstrap] ENV: PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL:-<unset>}"

echo "[bootstrap] ensure tools: git/ffmpeg/python3/pip/curl"
which git >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*)
which ffmpeg >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*)
which python3 >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*)
which curl >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*)

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# -------- 拉代码（带重试）--------
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[bootstrap] clone repo..."
  for i in $(seq 1 3); do
    if git clone --depth=1 "$GIT_URL" "$REPO_DIR"; then
      break
    fi
    echo "[bootstrap] git clone failed (attempt $i), retrying..."
    sleep 2
  done
else
  echo "[bootstrap] pull repo..."
  git -C "$REPO_DIR" fetch --all || true
  git -C "$REPO_DIR" pull --rebase || true
fi

cd "$REPO_DIR"

# -------- pip 安装（带重试/自适应 CUDA）--------
if [ -f requirements.txt ]; then
  echo "[bootstrap] python/pip versions"
  python3 -V || true
  python3 -m pip -V || true

  echo "[bootstrap] upgrade pip"
  python3 -m pip install --no-cache-dir --timeout "$PIP_TIMEOUT" --retries "$PIP_RETRIES" --upgrade pip

  PIP_ARGS=(--no-cache-dir --timeout "$PIP_TIMEOUT" --retries "$PIP_RETRIES")

  if [ -n "${PIP_EXTRA_INDEX_URL:-}" ]; then
    echo "[bootstrap] use PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL"
    PIP_ARGS+=(--extra-index-url "$PIP_EXTRA_INDEX_URL")
  else
    echo "[bootstrap] detect cuda availability for pip extra index"
    # 注意：heredoc 的 if，**这里不要在行尾写 then**，then 放在 PY 结束行之后
    if python3 - <<'PY' 2>/dev/null
import sys
try:
    import torch  # 可能未安装
    print("torch_imported=True")
except Exception:
    print("torch_imported=False")
PY
    then
      echo "[bootstrap] torch module importable; skip extra index auto inject"
    else
      if [ "${DEVICE:-auto}" = "auto" ] || [ "${DEVICE:-auto}" = "cuda" ]; then
        echo "[bootstrap] DEVICE=$DEVICE, inject cu124 extra index for torch wheels"
        PIP_ARGS+=(--extra-index-url "https://download.pytorch.org/whl/cu124")
      fi
    fi
  fi

  echo "[bootstrap] pip install -r requirements.txt"
  for i in $(seq 1 3); do
    if python3 -m pip install "${PIP_ARGS[@]}" -r requirements.txt; then
      break
    fi
    echo "[bootstrap] pip install failed (attempt $i), retry..."
    sleep 3
  done
fi

#-----version-------
echo "[bootstrap] write version file"
TAG=$(git -C "$REPO_DIR" describe --tags --always 2>/dev/null || true)
COMMIT=$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=${TAG:-$COMMIT}
printf '%s\n' "$VERSION" > /workspace/VERSION
echo "[bootstrap] VERSION=$VERSION"


# -------- 运行前自检 --------
echo "[bootstrap] runtime quick check"
python3 - <<'PY' || true
import glob
try:
    import torch
    print("[check] torch.version:", torch.__version__)
    print("[check] torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("[check] cuda device:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("[check] cuda device error:", e)
except Exception as e:
    print("[check] torch import failed:", e)

libs = sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libcudnn*so*"))
print("[check] cudnn libs sample:", libs[:3])
try:
    import ctranslate2
    print("[check] ctranslate2 imported OK")
except Exception as e:
    print("[check] ctranslate2 import failed:", e)
PY

# -------- 启动服务 --------
echo "[bootstrap] start uvicorn on :$PORT ..."
exec uvicorn asr_worker:app --host 0.0.0.0 --port "$PORT" --reload


