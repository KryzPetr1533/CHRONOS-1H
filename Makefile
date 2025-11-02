# ------- Config -------
IMAGE ?= btcusdt-dev:latest
CONTAINER ?= btcusdt-dev
PORT ?= 8888

# Resolve host UID/GID to avoid permission issues when writing to bind mounts
UID := $(shell id -u 2>/dev/null || echo 1000)
GID := $(shell id -g 2>/dev/null || echo 1000)

# Absolute working directory on host (the dir you call `make` from)
WORKDIR_ABS := $(shell pwd)

# Common docker run flags:
# - bind mount the current working directory at the same path inside the container
# - set working dir to that path
# - run processes as the host UID:GID to avoid file permission conflicts
DOCKER_RUN_BASE = docker run --rm -it --name $(CONTAINER) \
	-p $(PORT):8888 \
	-v "$(WORKDIR_ABS)":"$(WORKDIR_ABS)" \
	-w "$(WORKDIR_ABS)" \
	-u $(UID):$(GID) \
	-e HOME="$(WORKDIR_ABS)" \
	-e TZ=UTC

# ------- Targets -------
.PHONY: help build rebuild bash start exec attach stop rm logs jupyter jupyter-secure prune

help:
	@echo "Targets:"
	@echo "  build           Build image"
	@echo "  rebuild         Rebuild with --no-cache"
	@echo "  bash            Run interactive shell with project bind-mounted at the same path"
	@echo "  start           Start detached container (sleep infinity)"
	@echo "  exec            Open a Bash shell in the running container"
	@echo "  attach          Attach to PID1 of the running container"
	@echo "  logs            Follow container logs"
	@echo "  stop            Stop running container"
	@echo "  rm              Remove container"
	@echo "  jupyter         Run JupyterLab (NO AUTH, local dev only)"
	@echo "  jupyter-secure  Run JupyterLab with token auth (recommended)"
	@echo "  prune           Remove dangling images/volumes"

build:
	docker build --build-arg UID=$(UID) --build-arg GID=$(GID) -t $(IMAGE) .

rebuild:
	docker build --no-cache --build-arg UID=$(UID) --build-arg GID=$(GID) -t $(IMAGE) .

bash:
	$(DOCKER_RUN_BASE) $(IMAGE) bash

start:
	docker run -d --name $(CONTAINER) \
		-p $(PORT):8888 \
		-v "$(WORKDIR_ABS)":"$(WORKDIR_ABS)" \
		-w "$(WORKDIR_ABS)" \
		-u $(UID):$(GID) \
		-e HOME="$(WORKDIR_ABS)" \
		-e TZ=UTC \
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

prune:
	-docker image prune -f
	-docker volume prune -f
