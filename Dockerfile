# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user (override UID/GID at build if needed)
ARG UID=1000
ARG GID=1000
ARG NB_USER=appuser
ARG NB_GROUP=appuser

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create group/user
RUN groupadd -g ${GID} ${NB_GROUP} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${NB_USER}

WORKDIR /workspace

# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt /tmp/requirements.txt

# Install Python deps (Jupyter added here for notebook use)
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && pip install jupyterlab

ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Switch to non-root user for runtime
USER ${NB_USER}

EXPOSE 8888

# Tini for proper signal handling, default to bash (we override via Makefile)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
