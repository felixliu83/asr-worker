# 使用 CUDA 12.4 + cuDNN 9（与 faster-whisper 的 CUDA 依赖匹配）
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/workspace \
    REPO_DIR=/workspace/app \
    # 给将来装 torch cu124 用（bootstrap 中 pip 会继承这个 env）
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

WORKDIR /workspace

# 换源到清华，并加重试/超时
RUN set -eux; \
    cp -f /etc/apt/sources.list /etc/apt/sources.list.bak; \
    sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list; \
    sed -i 's|http://security.ubuntu.com/ubuntu/|http://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list; \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git ffmpeg python3 python3-pip python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# 启动脚本（拉代码 + pip 安装 + 起服务）
COPY bootstrap.sh /usr/local/bin/bootstrap.sh
RUN chmod +x /usr/local/bin/bootstrap.sh

# 预创建挂载/产物目录
RUN mkdir -p /workspace /workspace/app /workspace/data/uploads /workspace/data/results

EXPOSE 8000

# 容器主进程常驻：bootstrap 里运行 uvicorn
CMD ["/usr/local/bin/bootstrap.sh"]