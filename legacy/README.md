# legacy/ — Archive of superseded files

Files are moved here (not deleted) to preserve history. Use `git log -- legacy/<file>`
to see original commit history.

## regression_mean/ — Next-hour mean return regression (superseded by Task 1 Phase 1)

| Original path | Reason moved | Replaced by |
|---|---|---|
| `configs/train_ridge_core.yaml` | one-step return regression (R²≈0, dir-acc≈0.50) | `conf/label/direction.yaml` + `scripts/train_classifier.py` |
| `configs/train_catboost_core.yaml` | same | same |
| `configs/train_catboost_rich.yaml` | same (too few samples to trust) | same |
| `configs/train_enet_b_aggr.yaml` | ElasticNet regression for the serving app | PRD classifier (Task 2) |
| `scripts/train_sarimax.py` | AR mean-return model (dir-acc≈0.497) | vol_regime / direction classifiers |
| `scripts/train_enet_b_aggr.py` | ElasticNet regression pipeline | PRD classifier (Task 2) |

**What is NOT moved (still active):**
- `chronos_ts/trainer.py`, `metrics.py`, `splits.py`, `dataset.py` — reused by both paths
- `scripts/train_har_vol.py`, `scripts/train_garch.py` — volatility signal, feeds `vol_regime` target
- `configs/build_core.yaml`, `configs/build_rich.yaml` — dataset building still needed
- `scripts/train_seq.py`, `train_timexer.py`, `train_patchtst.py`, `train_chronos2.py` — parked for Task 1 Phase 2 (event-bar + tokenized sequence models)

## Notes

The empirical basis for the move: README §7.1 — every mean-return regression model
(Ridge, CatBoost, SARIMAX, GRU, TimeXer, Chronos-2) yielded R²≈0 and directional
accuracy 0.49–0.53 on the core dataset, confirming the target is near-white-noise
at the 1h horizon. Volatility (HAR: test R²≈0.085, Pearson≈0.33) remains active.
</content>
