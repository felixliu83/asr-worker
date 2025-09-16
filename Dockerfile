# CUDA 12 运行时，适配 RunPod GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/workspace \
    REPO_DIR=/workspace/app

WORKDIR /workspace

# 系统依赖（git/ffmpeg/python），带重试与 --fix-missing
RUN set -eux; \
    sed -i 's|http://archive.ubuntu.com|http://mirror.azure.cn|g' /etc/apt/sources.list; \
    sed -i 's|http://security.ubuntu.com|http://mirrors.ustc.edu.cn|g' /etc/apt/sources.list; \
    for i in 1 2 3; do \
      apt-get update && \
      apt-get install -y --no-install-recommends \
        git ffmpeg python3 python3-pip python3-venv ca-certificates curl \
      && break || { \
        echo "APT attempt $i failed. Retrying..."; \
        apt-get -y --fix-broken install || true; \
        apt-get -y --fix-missing update || true; \
        sleep 5; \
      }; \
    done; \
    rm -rf /var/lib/apt/lists/*

# 放一个引导脚本：启动时 clone/pull + 安装依赖 + 运行（热重载）
COPY bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh

EXPOSE 8000
CMD ["/usr/local/bin/bootstrap.sh"]