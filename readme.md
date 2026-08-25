# ForwardEdge
## Data-driven equity screening: from raw market data to risk-adjusted stock rankings

An end-to-end finance analytics + ML project:

```
Raw (alphavantage / fred / yfinance)
    -> Bronze -> Silver (master dataset) -> EDA -> Feature Engineering
    -> Modeling (baseline vs ML) -> Stock Ranking Engine -> Portfolio Backtest
    -> Gold (vw_* analytics tables) -> Streamlit Dashboard
```

## Business question

> Can market, technical, fundamental, and macroeconomic indicators be
> combined to predict a stock's future return and identify stocks with
> attractive forward risk-adjusted return potential?

Concretely, two prediction tasks over a 20-trading-day forward horizon:
- **Regression** — forward return magnitude
- **Classification** — forward direction (up/down)

both evaluated on a **time-ordered train/validation/test split** (never
shuffled — that would leak the future into training), then turned into an
actual **ranking + simulated portfolio** so the result is a business
decision ("which stocks to overweight"), not just a model score.

## About the raw data

The three real folders under `data/raw/` — `alphavantage/`, `fred/`,
`yfinance/` — could not be uploaded into this environment (they live on your
machine), so `src/raw_data_stub.py` generates files that **exactly reproduce
the three messy shapes** you described, so Bronze has real parsing work to
do rather than reading tidy CSVs:

| Folder | File pattern | Real shape |
|---|---|---|
| `alphavantage/` | `rsi_{TICKER}.csv` | `data` column is a JSON string shaped like AV's actual response: `{"Technical Analysis: RSI": {"<date>": {"RSI": "<val>"}}}` |
| `fred/` | `series_{ID}.csv` | `date, <SERIES_ID>` — two tidy columns, but 5 separate files at mixed frequencies (CPI/Fed funds/10Y/unemployment monthly, VIX daily) |
| `yfinance/` | `prices_{TICKER}.csv` | **wide**: a `Field` column (Open/High/Low/Close/Adj Close/Volume) + one column per date — not `date, price` |
| `yfinance/` | `fundamentals_{TICKER}.csv` | `data` column is a JSON string of a `.info`-style metrics dict |

**Drop your real 59 files into these same three folders with these same
filenames** and delete `data/raw/**/*.csv` first (or pass `force=True` to
`build_raw_layer`) — `bronze.py` parses by *inspecting* structure (which
column holds dates, which JSON key holds the series, whether `Close` or
`Adj Close` is present) rather than assuming, so real files in the same
shape parse identically to the stand-ins.

## Pipeline stages

### 1. Bronze (`src/bronze.py`) — extract only
Parses each source's actual on-disk shape into standardized long tables.
No cleaning, no joins, no feature engineering.
- `bronze_alphavantage_rsi.csv` → `date, ticker, indicator, rsi`
- `bronze_fred.csv` → `date, series, value`
- `bronze_yfinance_prices.csv` → `date, ticker, price, volume`
- `bronze_yfinance_fundamentals.csv` → `ticker, extracted_at, metric, value`

### 2. Silver (`src/silver.py`) — clean + integrate
De-duplication, numeric coercion, missing-value handling (ticker-wise
ffill/bfill for prices/RSI, sector-median for fundamentals), and date
alignment: macro is forward-filled onto the daily trading calendar
(never back-filled — a monthly CPI print isn't known before its release
date), fundamentals are broadcast as "latest known snapshot" per ticker.
Produces one **master dataset**: `date, ticker, sector, price, volume, rsi,
<5 FRED series>, <10 fundamental metrics>`.

### 3. EDA (`src/eda.py`)
Return distributions, per-ticker annualized volatility, return/RSI/macro
correlation heatmap, sector cumulative-return chart, fundamental
correlation heatmap. Figures in `outputs/figures/`.

### 4. Feature Engineering (`src/feature_engineering.py`)
- **Technical**: returns, 10/20-day momentum, 5/20/60-day moving averages
  and volatility, drawdown, volume z-score
- **Macro**: CPI YoY growth, Fed funds / 10Y yield / unemployment 20-day
  changes, VIX 5-day change
- **Fundamental**: cross-sectional (per-date) percentile scores for
  valuation, growth, profitability, leverage, combined into one composite
- **Target**: 20-day forward return + forward direction

### 5. Modeling (`src/modeling.py`)
Time-ordered 60/20/20 train/val/test split.
- **Regression**: Linear, Ridge, Random Forest, XGBoost (falls back to
  HistGradientBoosting if `xgboost` isn't installed)
- **Classification**: Logistic Regression, Random Forest, XGBoost/fallback
- **ML metrics**: RMSE/MAE/R² and Accuracy/Precision/Recall/F1/ROC-AUC
- **Financial metrics**: for each classifier, simulates the actual
  long-only equal-weight portfolio of "predicted positive" tickers and
  reports Hit Rate, Sharpe, Sortino, Max Drawdown — translating a
  classification score into a real trading outcome

### 6. Stock Ranking Engine (`src/ranking.py`)
Combines `predicted_return`, `predicted_probability`, `technical_score`,
`fundamental_score`, and `risk_score` (all converted to 0–1 cross-sectional
percentiles first) into one weighted `final_signal`, ranked per date.

### 7. Portfolio Backtest (`src/backtest.py`)
Every 20 trading days, buys an equal-weight basket of the **top-3** ranked
tickers and holds until the next rebalance — compared against an
equal-weight full-universe benchmark, entirely inside the test period.

### 8. Gold (`src/gold.py`) — the dashboard's data source
Writes 10 tables to `data/gold/`:

| View | Grain | Contents |
|---|---|---|
| `vw_stock_performance` | ticker × date | price, returns, MAs, RSI, drawdown |
| `vw_sector_performance` | sector × date | avg/dispersion return, cum. return |
| `vw_macro_overlay` | date | market return vs macro indicators |
| `vw_fundamental_scorecard` | ticker × date | ratios + value/growth/profit/leverage scores |
| `vw_risk_metrics` | ticker | vol, beta, Sharpe, Sortino, VaR, CVaR |
| `vw_volume_anomalies` | flagged days | \|volume z-score\| ≥ 2.5 |
| `vw_forecast_input` | ticker × date | full model-ready feature panel |
| `vw_model_predictions` | ticker × test date | predicted return + probability |
| `vw_stock_rankings` | ticker × test date | component scores + final signal + rank |
| `vw_portfolio_backtest` | test date | strategy vs benchmark cumulative return |

### 9. Streamlit Dashboard (`dashboard/app.py`)
Six pages, reading only from `data/gold/`:
**Executive Summary · Stock Analytics · Fundamentals · Macro & Risk ·
ML Intelligence · Portfolio Simulator**

## Running it

```bash
pip install -r requirements.txt
python main.py                       # runs the whole pipeline, Raw -> Gold
streamlit run dashboard/app.py        # launches the dashboard
```

`xgboost`, `pyarrow` are optional — the pipeline degrades gracefully
(HistGradientBoosting / CSV) if they aren't installed.

**Automated refresh**: re-running `python main.py` regenerates every Gold
table; the dashboard has no stale caching beyond a 30-second TTL, so
refreshing the browser after a pipeline run shows the latest data — this is
the "New Data → ETL → Features → Predictions → Dashboard Refresh" loop from
the target architecture, without needing an actual scheduler wired in yet.

## Honest results on the synthetic data

Daily-return-driven price series are close to a random walk by
construction, so — as should be expected and is worth stating plainly
rather than dressing up — no model here beats the naive baseline by much
(`outputs/reports/*_comparison.csv`), and the ML-ranked portfolio doesn't
reliably beat the benchmark in this backtest either. That is the correct,
defensible finding for short-horizon return forecasting on this data. With
**your real fundamentals/macro/RSI data**, expect genuine (if still modest)
separation to emerge, especially in the classification framing and in
feature importances shifting toward fundamentals/macro.

## Keeping data current

`END_DATE` in `src/config.py` is computed fresh every run (`date.today()`),
not a fixed string — so the pipeline always targets "through today."

`build_raw_layer()` in `src/raw_data_stub.py` is freshness-aware for its own
stand-in data: it writes a `.stub_generated` marker file, and on each run
checks the newest date already on disk against today.
- **Stand-in data present, fresh (< 4 days old)** → skipped, no regeneration
- **Stand-in data present, stale** → regenerated through today automatically
- **Real files present (no marker)** → never touched. Refreshing real data
  means re-pulling from Alpha Vantage / FRED / yfinance yourself and
  re-running `python main.py` — this project doesn't call those live APIs.

So the normal daily habit is just: `python main.py` → `streamlit run
dashboard/app.py`, and (for the stand-in data) it will always extend itself
to the latest date without you touching anything.

## Project layout

```
ForwardEdge/
├── data/
│   ├── raw/{alphavantage,fred,yfinance}/   # source files (stand-ins here)
│   ├── bronze/                              # parsed long tables
│   ├── silver/master_dataset.csv            # cleaned, integrated
│   └── gold/                                # vw_* tables (dashboard reads here)
├── src/
│   ├── config.py             # paths, universe, constants
│   ├── raw_data_stub.py      # stand-in raw data generator
│   ├── bronze.py
│   ├── silver.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── ranking.py
│   └── backtest.py
├── dashboard/app.py
├── outputs/{figures,reports}/
├── main.py
└── requirements.txt
```

## Extending this

- Swap in your real 59 files (see table above) and delete the stand-ins.
- Add more tickers/sectors/FRED series in `src/config.py`.
- Point `_save()` in `src/gold.py` at a real warehouse (Postgres/Neon)
  instead of CSV/Parquet — the rest of the pipeline is warehouse-agnostic.
- Add a scheduler (cron / Airflow / GitHub Actions) around `python main.py`
  for the fully "automated pipeline" loop.
# forwarded
