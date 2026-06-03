# CHRONOS-1H — BTCUSDT 1H Forecasting Project Summary

## 1. Project goal

The project aims to forecast the **next 1-hour BTCUSDT log-return** using **public Binance data only** and later deliver the forecast through a Telegram bot.

Target used for supervised learning:

- raw column in the merged dataset: `log_ret_1h(t) = log(close_t / close_{t-1})`
- forecasting target: `target_log_ret_1h(t) = log_ret_1h(t+1) = log(close_{t+1} / close_t)`

This target shift was verified during EDA and then enforced in the training scripts.

The **current production path** reframes the problem as **classification** (volatility regime, large moves, direction, return tokens) with **MLflow** tracking and a **PRD** model in the registry. See [MODEL_CARD.md](MODEL_CARD.md).

---

## Quick start tutorial (classification + MLflow)

End-to-end workflow on a fresh machine. Requires **Docker** and the dev image (`make build` once).

### 1. One-time setup

```bash
git clone <repo> && cd CHRONOS-1H
cp infra/mlflow/.env.example infra/mlflow/.env   # MinIO + Postgres credentials
make build                                      # image: btcusdt-dev:latest
```

On macOS (Colima), `GPU=none` is the default in the Makefile. On Linux with NVIDIA, use `GPU=all` if needed.

**Important:** do not keep a top-level `mlflow/` folder in the repo — it shadows the Python `mlflow` package. The Docker stack lives in **`infra/mlflow/`**. If you have an old `mlflow/` directory: `rm -rf mlflow`.

### 2. Start tracking stack

```bash
make mlflow-up
```

| Service | URL |
|---------|-----|
| MLflow UI | http://localhost:5050 |
| MinIO console | http://localhost:9001 (`admin` / `password` from `.env`) |

### 3. Build data and run EDA

```bash
# needs data/btcusdt_1h_merged.csv
make build-clf-dataset
make eda-clf-dataset          # → outputs/reports/clf_eda_summary.md
```

### 4. Train and register PRD model

```bash
make train-smoke              # fast check: logreg on rich data, MLflow run, no PRD
make train-final              # official: CatBoost vol_regime → chronos_1h_prd@prd (~15–20 min)
make predict-prd              # load models:/chronos_1h_prd@prd and print sample preds
```

Open the run in the UI (params, metrics, artifacts). Latest PRD run ID is recorded in [MODEL_CARD.md](MODEL_CARD.md).

If registry logging fails but local artifacts exist:

```bash
make register-prd
make predict-prd
```

### 5. Optional: sweeps and S3

```bash
make train-sweep              # Hydra multirun → several MLflow runs (no PRD)
make report-leaderboard       # refresh outputs/reports/phase1_leaderboard.csv
make upload-datasets          # push outputs/datasets to MinIO
make s3-ls-datasets
```

### 6. Train without MLflow (local Hydra only)

```bash
make train-clf                # default: catboost + vol_regime
make train-clf-sweep          # all label families × logreg/catboost/lightgbm
```

Configs live under `conf/`. Example:

```bash
docker run --rm -e PYTHONPATH="$(pwd)" -v "$(pwd):$(pwd)" -w "$(pwd)" btcusdt-dev:latest \
  python scripts/train_classifier.py label=large_move model=catboost
```

### 7. Remote server (SSH)

```bash
ssh -L 5050:localhost:5050 -L 9001:localhost:9001 user@host
# then open http://localhost:5050 on your laptop
```

Detailed task plans (local, gitignored): `plans/README.md`.

---

## 2. Data sources used

We worked with publicly available Binance futures / market features:

- OHLCV / trades
- quote volume
- taker buy base / quote volume
- premium index / basis proxy
- funding rate
- taker imbalance and buy/sell ratio
- open interest history

We also created derived features such as:

- `premium_chg`
- `vol_chg`
- `trades_chg`
- `taker_buy_share`
- `rv_6`, `rv_24`, `rv_72`, `rv_168`
- calendar features (`hour_sin/cos`, `dow_sin/cos`, funding-cycle encodings)

---

## 3. What EDA showed

### 3.1 Mean return is very hard to predict

The ACF/PACF analysis of hourly returns showed that the **mean of next-hour return is close to white noise**. That means simple autoregressive structure in raw returns is weak.

### 3.2 Volatility is forecastable

Absolute and squared returns had clear persistence. This led us to the conclusion that:

- volatility clustering is real
- volatility-aware features are important
- volatility models are likely more useful than plain AR-style return models

### 3.3 Feature coverage is heterogeneous

Not all features have the same historical support.

We therefore split the problem into:

- **core / long-history datasets**
- **rich / short-history datasets**

This was important because features like taker imbalance / OI-related series were available only on a shorter recent slice.

---

## 4. Dataset versions we built

### 4.1 Wide tabular datasets

We created two general-purpose tabular datasets:

- `btcusdt_core_tabular.csv`
- `btcusdt_rich_tabular.csv`

The core dataset preserved long history. The rich dataset added more recent market microstructure features.

### 4.2 Compact datasets for focused experiments

Then we created narrower datasets for specific experiment families:

- `btcusdt_core_mean_small.csv` — compact regression dataset for mean-return models
- `btcusdt_core_vol_small.csv` — compact dataset for volatility models
- `btcusdt_core_seq_small.csv` — compact sequence dataset for RNN/Transformer-style models
- `btcusdt_timexer.csv` — dataset prepared for TimeXer
- `chronos2_panel.csv` — dataset prepared for AutoGluon Chronos-2

---

## 5. Models we trained and what happened

## 5.1 Linear and tree baselines

### Ridge (core dataset)

Result: almost flat.

Key test metrics:

- RMSE ≈ `0.00366`
- R² ≈ `-0.037`
- Pearson ≈ `0.0118`
- Directional accuracy ≈ `0.504`

Interpretation: slightly better than trivial on some metrics, but not enough to call it useful.

### CatBoost (core dataset)

Result: also close to flat.

Key test metrics:

- RMSE ≈ `0.00360`
- R² ≈ `-0.00119`
- Pearson ≈ `0.0257`
- Directional accuracy ≈ `0.502`

Interpretation: bigger nonlinear tabular model did not unlock meaningful one-step return signal on the long-history core dataset.

### CatBoost (rich dataset)

This experiment looked much better on paper, but the sample was tiny:

- train / val / test = `299 / 99 / 101`

Key test metrics:

- RMSE ≈ `0.00495`
- R² ≈ `0.0024`
- Pearson ≈ `0.1007`
- Directional accuracy ≈ `0.614`

Interpretation: interesting, but **too small-sample** to trust as a production conclusion.

---

## 5.2 Classical mean-return models

### SARIMAX / AR-only

Best result was actually **AR-only** with `(2,0,1)` and no exogenous variables.

Key test metrics:

- RMSE ≈ `0.00360`
- R² ≈ `0`
- Pearson ≈ `-0.0015`
- Directional accuracy ≈ `0.497`

Interpretation:

- it improved RMSE over the naive previous-return baseline
- but mainly by collapsing toward a near-zero forecast
- it did **not** produce useful directional signal

So classical mean-return forecasting was not promising.

---

## 5.3 Volatility models

### HAR-style volatility model

This was the first genuinely useful result.

Target: next-hour absolute return.

Key test metrics vs baseline:

Baseline:

- RMSE ≈ `0.00330`
- R² ≈ `-0.546`
- Pearson ≈ `0.227`

HAR model:

- RMSE ≈ `0.00254`
- R² ≈ `0.085`
- Pearson ≈ `0.332`
- Spearman ≈ `0.343`

Interpretation: volatility forecasting worked materially better than naive persistence.

### EGARCH

Best GARCH-family model was EGARCH, but it still lost to HAR.

Key test metrics:

- RMSE ≈ `0.00315`
- R² ≈ `-0.409`
- Pearson ≈ `0.259`

Interpretation: useful as a benchmark, but HAR remained the best classical volatility model in our runs.

---

## 5.4 Neural sequence models

### GRU / LSTM sequence regression

Best sequence run:

- model: `GRU`
- lookback: `168`
- hidden size: `32`

Key test metrics:

- RMSE ≈ `0.00360`
- R² ≈ `-0.00129`
- Pearson ≈ `0.0303`
- Directional accuracy ≈ `0.493`

Interpretation: sequence modeling did not meaningfully improve one-step return regression.

---

## 5.5 TimeXer

We selected TimeXer because it is explicitly designed for **forecasting with exogenous variables**.

Our tuned best run used:

- encoder length = `168`
- batch size = `256`
- gradient accumulation = `4`
- best parameters:
  - lr = `1e-3`
  - hidden size = `128`
  - heads = `4`
  - encoder layers = `2`
  - FF size = `512`
  - dropout = `0.2`
  - patch length = `24`

Key test metrics:

- RMSE ≈ `0.00360`
- R² ≈ `-0.00133`
- Pearson ≈ `0.0182`
- Directional accuracy ≈ `0.486`

Interpretation: TimeXer did not beat the naive baseline enough to justify more scaling on this setup.

---

## 5.6 Chronos-2 Small

We moved to **AutoGluon Chronos-2 Small**, because it supports:

- cross-learning across items
- past covariates
- known future covariates
- fine-tuning with LoRA

Due to GPU limits (~4 GB VRAM), we used:

- `chronos-2-small`
- shorter context (`512`)
- larger but still safe batches
- LoRA fine-tuning for the fine-tuned variant

The first full custom evaluation loop was too slow because we accidentally requested thousands of one-step backtest windows. We then reduced evaluation to **256 sampled windows**.

### Chronos2SmallZeroShot

On the sampled 256-window backtest:

- RMSE ≈ `0.00531`
- R² ≈ `-0.088`
- Pearson ≈ `-0.161`
- Directional accuracy ≈ `0.527`
- Top-20% directional accuracy ≈ `0.577`

### Chronos2SmallFineTuned

On the same sampled 256-window backtest:

- RMSE ≈ `0.00531`
- R² ≈ `-0.090`
- Pearson ≈ `-0.160`
- Directional accuracy ≈ `0.512`
- Top-20% directional accuracy ≈ `0.577`

Interpretation:

- zero-shot and fine-tuned versions were very close
- zero-shot was slightly better than the fine-tuned model in this sampled evaluation
- results are **not directly comparable** to the earlier full-horizon core metrics because the evaluation protocol changed to a sparse sampled backtest (`256` windows)
- still, Chronos-2 is the most promising scalable foundation-model direction because it naturally supports covariates and multi-series training

---

## 6. External model references and why we tried them

### PatchTST

Why considered:
- strong practical baseline for transformer-style time series forecasting
- NeuralForecast implementation with exogenous support in the docs

Reference task:
- general long-range time series forecasting with patch-based transformer inputs

Links:
- NeuralForecast PatchTST docs: https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html

Note:
- our installed implementation rejected future exogenous variables, so we stopped there.

### TimeXer

Why considered:
- specifically designed for forecasting with exogenous variables

Reference task:
- endogenous + exogenous time-series forecasting

Links:
- PyTorch Forecasting docs: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.timexer._timexer.TimeXer.html
- Official repository: https://github.com/thuml/TimeXer

### Chronos-2

Why considered:
- universal time-series foundation model
- supports univariate, multivariate, and covariate-informed forecasting
- supports cross-learning across related series
- supports LoRA fine-tuning

Reference task:
- zero-shot and fine-tuned forecasting with covariates and related-series transfer

Links:
- AutoGluon Chronos-2 tutorial: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html
- Chronos-2 model card: https://huggingface.co/autogluon/chronos-2
- Chronos-2 Small model card: https://huggingface.co/autogluon/chronos-2-small
- Chronos forecasting releases (LoRA support): https://github.com/amazon-science/chronos-forecasting/releases

---

## 7. Main conclusions

### 7.1 What did **not** work well

For **next-hour raw return regression**, the following were all close to flat:

- Ridge
- CatBoost on the long-history core dataset
- SARIMAX / AR-only
- GRU sequence regression
- TimeXer

This strongly suggests that **one-step BTC return mean** is extremely noisy and hard to predict in this formulation.

### 7.2 What **did** work

The best clear positive result was **volatility forecasting**:

- HAR outperformed both naive persistence and EGARCH
- volatility is much more forecastable than mean return in this setup

### 7.3 What is the most promising path forward

If the project must remain a **regression** project, the best next scaling direction is:

1. Expand from **single-series BTC** to a **multi-asset panel**:
   - BTCUSDT
   - ETHUSDT
   - BNBUSDT
   - SOLUSDT
   - XRPUSDT
   - DOGEUSDT

2. Reuse the same covariate schema for all assets.

3. Train **Chronos-2 Small / Chronos-2** on that panel with:
   - past covariates
   - known future calendar covariates
   - cross-learning enabled

4. Keep the volatility model alongside the return model as a confidence signal.

---

## 8. Final status

At the end of this iteration:

- the training pipeline was fully rebuilt into scripts
- multiple dataset variants were created
- classical, tree, sequence, transformer, and foundation-model routes were tested
- the strongest result was volatility forecasting via HAR
- the best scalable next bet for raw return regression is **multi-series Chronos-2**

This means the project now has:

- a reproducible experimentation structure
- a realistic understanding of where the signal is and is not
- a clear next plan for scaling
