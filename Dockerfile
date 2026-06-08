# ---------- 基础镜像 ----------
# CUDA 12.6 runtime + cuDNN 8
# FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu22.04
# FROM nvidia/cuda:12.6.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04


# ---------- 基础环境 ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# 系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN python3 -m pip install --upgrade pip

# ---------- 安装 PyTorch GPU 版本 ----------
RUN pip install torch torchvision torchaudio --index-url https://pypi.org/simple

# ---------- 安装 uv ----------
RUN pip install uv

# ---------- 复制项目 ----------
COPY . /workspace

ENV UV_PYTHON=python3.11

RUN uv python install 3.11 && \
    uv python pin 3.11


# ---------- 安装项目依赖 ----------
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync \
    && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# ---------- 可选 Jupyter ----------
RUN pip install jupyterlab

# ---------- 默认进入 bash ----------
CMD ["/bin/bash"]
