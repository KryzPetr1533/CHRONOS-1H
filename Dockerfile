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

# Pick your CUDA wheel channel + torch version.
# For a CPU-only build on a Mac/arm64 use: --build-arg TORCH_CUDA_CHANNEL=cpu
# For CUDA on the training server use cu128 (default).
ARG TORCH_CUDA_CHANNEL=cu128
ARG TORCH_VERSION=2.8.0

# Minimal OS deps (tini + runtime libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only requirements first for better layer caching
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch wheel (separate layer; biggest dependency)
RUN python -m pip install -U pip \
    && python -m pip install --prefer-binary \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL} \
        torch==${TORCH_VERSION} \
    && python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda)"

# Install project deps + JupyterLab
RUN python -m pip install --prefer-binary -r /tmp/requirements.txt \
    && python -m pip install --prefer-binary jupyterlab

ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Create non-root user (after installs, avoids permission issues on bind mounts)
# --force-badname / --non-unique: GID may already exist in the base image (e.g. GID 20 = dialout)
RUN groupadd --gid ${GID} --force ${NB_GROUP} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${NB_USER} \
    || useradd -m -u ${UID} -s /bin/bash ${NB_USER}

USER ${NB_USER}

EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
