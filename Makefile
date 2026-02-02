# ------- Config -------
IMAGE     ?= btcusdt-dev:latest
CONTAINER ?= btcusdt-dev
PORT      ?= 8888

# GPU controls:
#   GPU=all        (default) expose all GPUs
#   GPU=0          expose only GPU 0
#   GPU=none       disable GPU (or set GPU= to empty)
GPU ?= all

# Optional: increase shared memory for PyTorch DataLoader / multiprocessing
# SHM ?= 1g
SHM ?=

# Resolve host UID/GID to avoid permission issues when writing to bind mounts
UID := $(shell id -u 2>/dev/null || echo 1000)
GID := $(shell id -g 2>/dev/null || echo 1000)

# Absolute working directory on host (the dir you call `make` from)
WORKDIR_ABS := $(shell pwd)

# GPU flags (no-op if GPU is empty or "none")
ifeq ($(GPU),)
  DOCKER_GPU_FLAGS :=
else ifeq ($(GPU),none)
  DOCKER_GPU_FLAGS :=
else
  DOCKER_GPU_FLAGS := --gpus $(GPU)
endif

# Optional shm flag
ifneq ($(SHM),)
  DOCKER_SHM_FLAG := --shm-size=$(SHM)
else
  DOCKER_SHM_FLAG :=
endif

# Common docker run flags:
# - bind mount the current working directory at the same path inside the container
# - set working dir to that path
# - run processes as the host UID:GID to avoid file permission conflicts
# - enable GPU if configured
DOCKER_RUN_BASE = docker run --rm -it --name $(CONTAINER) \
	$(DOCKER_GPU_FLAGS) \
	$(DOCKER_SHM_FLAG) \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
	-p $(PORT):8888 \
	-v "$(WORKDIR_ABS)":"$(WORKDIR_ABS)" \
	-w "$(WORKDIR_ABS)" \
	-u $(UID):$(GID) \
	-e HOME="$(WORKDIR_ABS)" \
	-e TZ=UTC

# ------- Targets -------
.PHONY: help build rebuild bash start exec attach stop rm logs jupyter jupyter-secure prune cuda-check nvidia-smi

help:
	@echo "Targets:"
	@echo "  build           Build image"
	@echo "  rebuild         Rebuild with --no-cache"
	@echo "  bash            Run interactive shell (GPU enabled if GPU!=none)"
	@echo "  start           Start detached container (sleep infinity, GPU enabled if GPU!=none)"
	@echo "  exec            Open a Bash shell in the running container"
	@echo "  attach          Attach to PID1 of the running container"
	@echo "  logs            Follow container logs"
	@echo "  stop            Stop running container"
	@echo "  rm              Remove container"
	@echo "  jupyter         Run JupyterLab (NO AUTH, local dev only, GPU enabled if GPU!=none)"
	@echo "  jupyter-secure  Run JupyterLab with token auth (recommended)"
	@echo "  cuda-check      Verify nvidia-smi + torch sees CUDA"
	@echo "  nvidia-smi      Run nvidia-smi inside the container"
	@echo "  prune           Remove dangling images/volumes"
	@echo ""
	@echo "Config examples:"
	@echo "  make bash GPU=all"
	@echo "  make bash GPU=0"
	@echo "  make bash GPU=none"
	@echo "  make bash SHM=1g"

build:
	docker build --build-arg UID=$(UID) --build-arg GID=$(GID) -t $(IMAGE) .

rebuild:
	docker build --no-cache --build-arg UID=$(UID) --build-arg GID=$(GID) -t $(IMAGE) .

bash:
	$(DOCKER_RUN_BASE) $(IMAGE) bash

start:
	docker run -d --name $(CONTAINER) \
		$(DOCKER_GPU_FLAGS) \
		$(DOCKER_SHM_FLAG) \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
		-p $(PORT):8888 \
		-v "$(WORKDIR_ABS)":"$(WORKDIR_ABS)" \
		-w "$(WORKDIR_ABS)" \
		-u $(UID):$(GID) \
		-e HOME="$(WORKDIR_ABS)" \
		-e TZ=UTC \
		--network=host \
		$(IMAGE) sleep infinity

exec:
	docker exec -it $(CONTAINER) bash

attach:
	docker attach $(CONTAINER)

logs:
	docker logs -f $(CONTAINER)

stop:
	-docker stop $(CONTAINER)

rm:
	-docker rm -f $(CONTAINER)

# WARNING: disables token/password; use only on trusted local machine/network
jupyter:
	$(DOCKER_RUN_BASE) $(IMAGE) sh -c "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
	  --ServerApp.token='' --ServerApp.password='' --ServerApp.root_dir='$(WORKDIR_ABS)'"

# Safer default: Jupyter will print a token URL in logs
jupyter-secure:
	$(DOCKER_RUN_BASE) $(IMAGE) sh -c "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
	  --ServerApp.root_dir='$(WORKDIR_ABS)'"

# GPU sanity checks
nvidia-smi:
	$(DOCKER_RUN_BASE) $(IMAGE) nvidia-smi

cuda-check:
	$(DOCKER_RUN_BASE) $(IMAGE) sh -c "\
	  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo 'nvidia-smi not found'; \
	  python -c \"import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('torch.version.cuda', torch.version.cuda)\" \
	"

prune:
	-docker image prune -f
	-docker volume prune -f
