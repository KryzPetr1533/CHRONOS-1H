# ------- Config -------
IMAGE     ?= btcusdt-dev:latest
CONTAINER ?= btcusdt-dev
PORT      ?= 8888

# GPU controls:
#   GPU=all        expose all GPUs (Linux + NVIDIA)
#   GPU=0          expose only GPU 0
#   GPU=none       disable GPU (Colima/macOS default — no NVIDIA in VM)
ifeq ($(shell uname -s),Darwin)
GPU ?= none
else
GPU ?= all
endif

# Optional: increase shared memory for PyTorch DataLoader / multiprocessing
# SHM ?= 1g
SHM ?=

# Resolve host UID/GID to avoid permission issues when writing to bind mounts
UID := $(shell id -u 2>/dev/null || echo 1000)
GID := $(shell id -g 2>/dev/null || echo 1000)

# Absolute working directory on host (the dir you call `make` from)
WORKDIR_ABS := $(shell pwd)

# Compose CLI: Colima often lacks `docker compose` plugin — prefer docker-compose from brew
DOCKER_COMPOSE      ?= $(shell if command -v docker-compose >/dev/null 2>&1; then echo docker-compose; else echo "docker compose"; fi)

# MinIO / dataset upload (scripts/upload_datasets_s3.py)
MLFLOW_DIR          ?= infra/mlflow
MLFLOW_ENV          := $(MLFLOW_DIR)/.env
MLFLOW_NETWORK      ?= mlflow_internal
DATASET_SRC         ?= outputs/datasets
DATASET_TAG         ?= $(shell date +%Y%m%d)
S3_DATASET_PREFIX   ?= chronos/datasets
MINIO_ENDPOINT      ?= http://minio:9000

# Memory for training containers (Colima default is often too small for full datasets)
DOCKER_MEMORY       ?= 6g

# Non-interactive dev container on the MLflow compose network (MinIO host: minio)
DOCKER_RUN_MLFLOW = docker run --rm \
	--memory=$(DOCKER_MEMORY) \
	--network $(MLFLOW_NETWORK) \
	--env-file $(MLFLOW_ENV) \
	-e MLFLOW_S3_ENDPOINT_URL=$(MINIO_ENDPOINT) \
	-e PYTHONPATH="$(WORKDIR_ABS)" \
	-v "$(WORKDIR_ABS)":"$(WORKDIR_ABS)" \
	-w "$(WORKDIR_ABS)" \
	-u $(UID):$(GID) \
	-e HOME="$(WORKDIR_ABS)" \
	-e TZ=UTC

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

# Common docker run flags (batch: no -it; interactive: DOCKER_RUN_BASE adds -it + port)
DOCKER_RUN = docker run --rm \
	$(DOCKER_GPU_FLAGS) \
	$(DOCKER_SHM_FLAG) \
	-e NVIDIA_VISIBLE_DEVICES=all \
	-e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
	-e PYTHONPATH="$(WORKDIR_ABS)" \
	-v "$(WORKDIR_ABS)":"$(WORKDIR_ABS)" \
	-w "$(WORKDIR_ABS)" \
	-u $(UID):$(GID) \
	-e HOME="$(WORKDIR_ABS)" \
	-e TZ=UTC

DOCKER_RUN_BASE = $(DOCKER_RUN) -it --name $(CONTAINER) \
	-p $(PORT):8888

# ------- Targets -------
.PHONY: help build rebuild bash start exec attach stop rm logs jupyter jupyter-secure prune cuda-check nvidia-smi \
        build-clf-dataset eda-clf-dataset train-clf train-clf-sweep train-sweep report-leaderboard \
        mlflow-up mlflow-down mlflow-check-env minio-check-running docker-image-check \
        upload-datasets s3-ls-datasets train-final train-smoke register-prd predict-prd token-transformer

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
	@echo "MLflow / S3 (MinIO):"
	@echo "  mlflow-up           Start Postgres + MLflow + MinIO (needs infra/mlflow/.env)"
	@echo "  mlflow-down         Stop MLflow stack"
	@echo "  upload-datasets     Upload DATASET_SRC to MinIO via dev image (needs build + mlflow-up)"
	@echo "  s3-ls-datasets      List dataset prefixes in MinIO (dev image)"
	@echo "  train-final         Train + log to MLflow + promote PRD"
	@echo "  train-smoke         Quick MLflow train (logreg, rich data) to verify stack"
	@echo "                      Colima: colima start --memory 8  (train-final needs more RAM)"
	@echo "  predict-prd         Load models:/chronos_1h_prd@prd and predict"
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

# ------- Classification pipeline -------
# Build the feature matrix + sanity-check + label preview (runs in Docker)
build-clf-dataset:
	$(DOCKER_RUN) $(IMAGE) python scripts/build_clf_dataset.py

eda-clf-dataset:
	$(DOCKER_RUN) $(IMAGE) python scripts/eda_clf_dataset.py

report-leaderboard:
	$(DOCKER_RUN) $(IMAGE) python scripts/report_leaderboard.py

# Train single run (default: vol_regime + catboost)
train-clf:
	$(DOCKER_RUN) $(IMAGE) python scripts/train_classifier.py

# Multirun sweep over all label families and models (no MLflow)
train-clf-sweep:
	$(DOCKER_RUN) $(IMAGE) python scripts/train_classifier.py -m label=direction,large_move,vol_regime,horizon_dir,return_token model=logreg,catboost,lightgbm

# MLflow multirun (no PRD); results visible at http://localhost:5050
train-sweep: mlflow-check-env minio-check-running docker-image-check
	$(DOCKER_RUN_MLFLOW) -e MLFLOW_TRACKING_URI=http://chronos_mlflow:5000 $(IMAGE) \
	  python scripts/train_mlflow.py -m \
	  label=vol_regime,large_move,direction \
	  model=logreg,catboost \
	  mlflow.promote_to_prd=false \
	  mlflow.tracking_uri=http://chronos_mlflow:5000 \
	  mlflow.s3_endpoint_url=http://minio:9000 cv_splits=2

# ------- MLflow stack -------
mlflow-check-env:
	@test -f $(MLFLOW_ENV) || (echo "Missing $(MLFLOW_ENV) — run: cp $(MLFLOW_DIR)/.env.example $(MLFLOW_ENV)" && exit 1)
	@if [ -d mlflow ] && [ -f mlflow/docker-compose.yml ] 2>/dev/null; then \
	  echo "WARNING: ./mlflow/ shadows pip mlflow when PYTHONPATH includes the repo. Migrate: cp mlflow/.env infra/mlflow/.env && rm -rf mlflow"; \
	fi

minio-check-running:
	@docker ps --format '{{.Names}}' | grep -qx chronos_minio || \
	  (echo "MinIO is not running. Run: make mlflow-up" && exit 1)

docker-image-check:
	@docker image inspect $(IMAGE) >/dev/null 2>&1 || \
	  (echo "Docker image $(IMAGE) not found. Run: make build" && exit 1)

mlflow-up: mlflow-check-env
	cd $(MLFLOW_DIR) && $(DOCKER_COMPOSE) up -d

mlflow-down:
	cd $(MLFLOW_DIR) && $(DOCKER_COMPOSE) down

# Upload local datasets to MinIO (dev image on mlflow_internal → http://minio:9000).
# Override: make upload-datasets DATASET_TAG=v2 DATASET_SRC=outputs/datasets
upload-datasets: mlflow-check-env minio-check-running docker-image-check
	@test -d "$(DATASET_SRC)" || (echo "DATASET_SRC not found: $(DATASET_SRC)" && exit 1)
	$(DOCKER_RUN_MLFLOW) $(IMAGE) python scripts/upload_datasets_s3.py \
	  --src "$(DATASET_SRC)" --tag "$(DATASET_TAG)" --prefix "$(S3_DATASET_PREFIX)"

s3-ls-datasets: mlflow-check-env minio-check-running docker-image-check
	$(DOCKER_RUN_MLFLOW) $(IMAGE) python scripts/upload_datasets_s3.py \
	  --list --prefix "$(S3_DATASET_PREFIX)"

# ------- MLflow-integrated training (T2-P2 bonus) -------
# Quick end-to-end check (logreg, no PRD promotion)
train-smoke: mlflow-check-env minio-check-running docker-image-check
	$(DOCKER_RUN_MLFLOW) -e MLFLOW_TRACKING_URI=http://chronos_mlflow:5000 $(IMAGE) \
	  python scripts/train_mlflow.py model=logreg label=vol_regime data=btcusdt_rich \
	  mlflow.promote_to_prd=false mlflow.tracking_uri=http://chronos_mlflow:5000 \
	  mlflow.s3_endpoint_url=http://minio:9000 cv_splits=2

# Register latest local catboost_vol_regime joblib to Model Registry (if train-final logging failed)
register-prd: mlflow-check-env minio-check-running docker-image-check
	$(DOCKER_RUN_MLFLOW) -e MLFLOW_TRACKING_URI=http://chronos_mlflow:5000 $(IMAGE) \
	  python scripts/register_prd_from_artifacts.py \
	  --tracking-uri http://chronos_mlflow:5000 --s3-endpoint http://minio:9000

# Train final model and promote to PRD (requires mlflow-up)
train-final: mlflow-check-env minio-check-running docker-image-check
	$(DOCKER_RUN_MLFLOW) -e MLFLOW_TRACKING_URI=http://chronos_mlflow:5000 $(IMAGE) \
	  python scripts/train_mlflow.py experiment=final mlflow.promote_to_prd=true \
	  mlflow.tracking_uri=http://chronos_mlflow:5000 mlflow.s3_endpoint_url=http://minio:9000

# Load PRD model and predict (requires mlflow-up + trained PRD)
predict-prd: mlflow-check-env minio-check-running docker-image-check
	$(DOCKER_RUN_MLFLOW) -e MLFLOW_TRACKING_URI=http://chronos_mlflow:5000 $(IMAGE) \
	  python scripts/predict_prd.py mlflow.tracking_uri=http://chronos_mlflow:5000 \
	  mlflow.s3_endpoint_url=http://minio:9000

# Train token transformer (no MLflow required)
token-transformer:
	$(DOCKER_RUN) $(IMAGE) python scripts/train_token_transformer.py
