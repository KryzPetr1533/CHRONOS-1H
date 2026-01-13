# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- Build args ----
ARG UID=1000
ARG GID=1000
ARG NB_USER=appuser
ARG NB_GROUP=appuser

# Pick your CUDA wheel channel + torch version
# Examples from PyTorch docs: cu126 / cu128 / cu129 (and cpu).  :contentReference[oaicite:1]{index=1}
ARG TORCH_CUDA_CHANNEL=cu128
ARG TORCH_VERSION=2.8.0

# Minimal OS deps (tini + runtime libs)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        tini \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only requirements first for better layer caching
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch CUDA wheel (separate layer; biggest dependency)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip \
    && python -m pip install --prefer-binary \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL} \
        torch==${TORCH_VERSION} \
    && python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)  # should be non-None for CUDA builds
PY

# Install the rest of your deps + JupyterLab
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --prefer-binary -r /tmp/requirements.txt \
    && python -m pip install --prefer-binary jupyterlab

ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Create non-root user (after installs)
RUN groupadd -g ${GID} ${NB_GROUP} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${NB_USER}

USER ${NB_USER}

EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
