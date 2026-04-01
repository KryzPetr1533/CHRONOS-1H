# CHRONOS-1H — сводка по проекту прогноза BTCUSDT 1H

## 1. Цель проекта

Цель проекта — прогнозировать **следующий 1-часовой лог-доход BTCUSDT** только на основе **публичных данных Binance**, а затем отдавать прогноз через Telegram-бота.

Целевая переменная для обучения:

- сырой столбец в merged-датасете: `log_ret_1h(t) = log(close_t / close_{t-1})`
- таргет для supervised learning: `target_log_ret_1h(t) = log_ret_1h(t+1) = log(close_{t+1} / close_t)`

Этот сдвиг таргета был отдельно проверен на этапе EDA и затем зафиксирован во всех обучающих скриптах.

---

## 2. Какие данные использовались

Мы работали с публичными Binance futures / market features:

- OHLCV / trades
- quote volume
- taker buy base / quote volume
- premium index / basis proxy
- funding rate
- taker imbalance и buy/sell ratio
- open interest history

Также были построены производные признаки:

- `premium_chg`
- `vol_chg`
- `trades_chg`
- `taker_buy_share`
- `rv_6`, `rv_24`, `rv_72`, `rv_168`
- календарные признаки (`hour_sin/cos`, `dow_sin/cos`, funding-cycle encodings)

---

## 3. Что показал EDA

### 3.1 Средний следующий доход предсказывается очень плохо

ACF/PACF для почасовых доходностей показал, что **среднее следующего часового дохода близко к white noise**. Это означает, что простая AR-структура в сырых доходностях очень слабая.

### 3.2 Волатильность предсказуема

Абсолютные и квадратные доходности показали устойчивую зависимость. Отсюда вывод:

- кластеризация волатильности действительно есть
- признаки, связанные с волатильностью, важны
- модели волатильности полезнее, чем простые AR-модели для средней доходности

### 3.3 Покрытие признаков неоднородно

Не все признаки имеют одинаковую историю.

Поэтому задачу пришлось разделить на:

- **core / long-history datasets**
- **rich / short-history datasets**

Это было критично, потому что признаки вроде taker imbalance / OI доступны только на коротком недавнем участке.

---

## 4. Какие версии датасетов были построены

### 4.1 Широкие tabular-датасеты

Мы создали два основных табличных датасета:

- `btcusdt_core_tabular.csv`
- `btcusdt_rich_tabular.csv`

Core-датасет сохранял длинную историю. Rich-датасет добавлял более свежие микроструктурные признаки рынка.

### 4.2 Узкие датасеты под отдельные семейства экспериментов

Далее были созданы более компактные датасеты:

- `btcusdt_core_mean_small.csv` — компактный regression-датасет для моделей средней доходности
- `btcusdt_core_vol_small.csv` — компактный датасет для моделей волатильности
- `btcusdt_core_seq_small.csv` — компактный sequence-датасет для RNN/Transformer-моделей
- `btcusdt_timexer.csv` — датасет под TimeXer
- `chronos2_panel.csv` — датасет под AutoGluon Chronos-2

---

## 5. Какие модели мы обучили и что получилось

## 5.1 Линейные и деревья как baseline

### Ridge (core dataset)

Результат: почти плоско.

Основные test-метрики:

- RMSE ≈ `0.00366`
- R² ≈ `-0.037`
- Pearson ≈ `0.0118`
- Directional accuracy ≈ `0.504`

Интерпретация: немного лучше тривиального варианта по отдельным метрикам, но недостаточно, чтобы назвать модель полезной.

### CatBoost (core dataset)

Результат: тоже почти плоско.

Основные test-метрики:

- RMSE ≈ `0.00360`
- R² ≈ `-0.00119`
- Pearson ≈ `0.0257`
- Directional accuracy ≈ `0.502`

Интерпретация: нелинейная tabular-модель не смогла извлечь существенный one-step сигнал на длинной core-истории.

### CatBoost (rich dataset)

Здесь цифры выглядели лучше, но выборка была очень маленькая:

- train / val / test = `299 / 99 / 101`

Основные test-метрики:

- RMSE ≈ `0.00495`
- R² ≈ `0.0024`
- Pearson ≈ `0.1007`
- Directional accuracy ≈ `0.614`

Интерпретация: выглядит интересно, но **слишком маленькая выборка**, чтобы делать production-выводы.

---

## 5.2 Классические модели для средней доходности

### SARIMAX / AR-only

Лучший результат дала фактически **AR-only** модель `(2,0,1)` без exogenous variables.

Основные test-метрики:

- RMSE ≈ `0.00360`
- R² ≈ `0`
- Pearson ≈ `-0.0015`
- Directional accuracy ≈ `0.497`

Интерпретация:

- по RMSE модель лучше наивного baseline по предыдущей доходности
- но это произошло в основном за счёт схлопывания прогноза к почти нулю
- полезного directional-сигнала модель не дала

То есть классический mean-return forecasting себя не оправдал.

---

## 5.3 Модели волатильности

### HAR-модель волатильности

Это был первый реально полезный результат.

Таргет: следующая часовая абсолютная доходность.

Основные test-метрики против baseline:

Baseline:

- RMSE ≈ `0.00330`
- R² ≈ `-0.546`
- Pearson ≈ `0.227`

HAR model:

- RMSE ≈ `0.00254`
- R² ≈ `0.085`
- Pearson ≈ `0.332`
- Spearman ≈ `0.343`

Интерпретация: прогноз волатильности оказался заметно лучше, чем наивная persistence-модель.

### EGARCH

Лучшей моделью из GARCH-family оказался EGARCH, но он всё равно проиграл HAR.

Основные test-метрики:

- RMSE ≈ `0.00315`
- R² ≈ `-0.409`
- Pearson ≈ `0.259`

Интерпретация: полезный benchmark, но HAR остался лучшей классической моделью волатильности.

---

## 5.4 Нейросетевые sequence-модели

### GRU / LSTM sequence regression

Лучшая sequence-модель:

- model: `GRU`
- lookback: `168`
- hidden size: `32`

Основные test-метрики:

- RMSE ≈ `0.00360`
- R² ≈ `-0.00129`
- Pearson ≈ `0.0303`
- Directional accuracy ≈ `0.493`

Интерпретация: sequence modeling не дало заметного улучшения для one-step regression по raw return.

---

## 5.5 TimeXer

Мы выбрали TimeXer, потому что он специально предназначен для **forecasting with exogenous variables**.

Лучший tuned-run использовал:

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

Основные test-метрики:

- RMSE ≈ `0.00360`
- R² ≈ `-0.00133`
- Pearson ≈ `0.0182`
- Directional accuracy ≈ `0.486`

Интерпретация: TimeXer не обыграл наивный baseline настолько, чтобы было смысл дальше масштабировать именно этот путь.

---

## 5.6 Chronos-2 Small

Дальше мы перешли к **AutoGluon Chronos-2 Small**, потому что он поддерживает:

- cross-learning across items
- past covariates
- known future covariates
- fine-tuning через LoRA

Из-за ограничений GPU (~4 GB VRAM) использовались:

- `chronos-2-small`
- укороченный context (`512`)
- безопасные batch sizes
- LoRA fine-tuning для fine-tuned варианта

Первый кастомный evaluation loop оказался слишком тяжёлым, потому что по ошибке запрашивались тысячи one-step backtest windows. Потом мы сократили оценивание до **256 sampled windows**.

### Chronos2SmallZeroShot

На sampled 256-window backtest:

- RMSE ≈ `0.00531`
- R² ≈ `-0.088`
- Pearson ≈ `-0.161`
- Directional accuracy ≈ `0.527`
- Top-20% directional accuracy ≈ `0.577`

### Chronos2SmallFineTuned

На том же sampled 256-window backtest:

- RMSE ≈ `0.00531`
- R² ≈ `-0.090`
- Pearson ≈ `-0.160`
- Directional accuracy ≈ `0.512`
- Top-20% directional accuracy ≈ `0.577`

Интерпретация:

- zero-shot и fine-tuned версии оказались очень близки
- в этом sampled evaluation zero-shot даже чуть лучше fine-tuned
- результаты **нельзя напрямую сравнивать** с предыдущими full-horizon core-метриками, потому что protocol changed: это уже разрежённый sampled backtest (`256` windows)
- несмотря на это, Chronos-2 остаётся самым перспективным foundation-model направлением для масштабирования, потому что естественно поддерживает covariates и multi-series training

---

## 6. Внешние модели и почему мы их вообще рассматривали

### PatchTST

Почему рассматривали:
- сильный практический baseline для transformer-style forecasting
- в документации NeuralForecast есть поддержка exogenous variables

Тип задачи:
- general long-range time series forecasting с patch-based transformer input

Ссылки:
- NeuralForecast PatchTST docs: https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html

Примечание:
- наша установленная версия не приняла future exogenous variables, поэтому этот путь был остановлен.

### TimeXer

Почему рассматривали:
- специально создан для forecasting with exogenous variables

Тип задачи:
- endogenous + exogenous time-series forecasting

Ссылки:
- PyTorch Forecasting docs: https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.timexer._timexer.TimeXer.html
- Official repository: https://github.com/thuml/TimeXer

### Chronos-2

Почему рассматривали:
- универсальная time-series foundation model
- поддерживает univariate, multivariate и covariate-informed forecasting
- поддерживает cross-learning across related series
- поддерживает LoRA fine-tuning

Тип задачи:
- zero-shot и fine-tuned forecasting с covariates и transfer между связанными временными рядами

Ссылки:
- AutoGluon Chronos-2 tutorial: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html
- Chronos-2 model card: https://huggingface.co/autogluon/chronos-2
- Chronos-2 Small model card: https://huggingface.co/autogluon/chronos-2-small
- Chronos forecasting releases (LoRA support): https://github.com/amazon-science/chronos-forecasting/releases

---

## 7. Главные выводы

### 7.1 Что **не** сработало хорошо

Для **next-hour raw return regression** практически плоскими оказались:

- Ridge
- CatBoost на long-history core dataset
- SARIMAX / AR-only
- GRU sequence regression
- TimeXer

Это очень сильный сигнал о том, что **one-step BTC return mean** в такой постановке задачи крайне шумный и плохо предсказываемый.

### 7.2 Что **сработало**

Лучший явный позитивный результат — это **прогноз волатильности**:

- HAR обыграл и naive persistence baseline, и EGARCH
- волатильность в этой постановке предсказывается лучше, чем mean return

### 7.3 Какой путь выглядит самым перспективным дальше

Если проект должен остаться именно **регрессионным**, то лучший путь масштабирования такой:

1. Перейти от **одного BTC-ряда** к **панели из нескольких активов**:
   - BTCUSDT
   - ETHUSDT
   - BNBUSDT
   - SOLUSDT
   - XRPUSDT
   - DOGEUSDT

2. Использовать ту же схему covariates для всех активов.

3. Обучать **Chronos-2 Small / Chronos-2** на панели с:
   - past covariates
   - known future calendar covariates
   - включённым cross-learning

4. Параллельно оставить volatility model как confidence signal.

---

## 8. Итоговый статус

На конец этой итерации:

- пайплайн обучения полностью перестроен в виде скриптов
- созданы несколько версий датасетов
- проверены classical, tree, sequence, transformer и foundation-model направления
- самый сильный результат получен на задаче волатильности через HAR
- лучший масштабируемый следующий кандидат для raw return regression — **multi-series Chronos-2**

То есть теперь в проекте есть:

- воспроизводимая структура экспериментов
- реалистичное понимание, где сигнал есть, а где его почти нет
- чёткий следующий план по масштабированию
