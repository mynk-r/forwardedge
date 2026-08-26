"""
STAND-IN RAW DATA
===================
Generates files in data/raw/{alphavantage,fred,yfinance}/ that deliberately
reproduce the three messy real-world shapes described for this project:

  alphavantage/rsi_{TICKER}.csv
      columns: ticker, indicator, interval, extracted_at, row_count, params, data
      -> `data` is a JSON string shaped like Alpha Vantage's actual RSI
         response: {"Technical Analysis: RSI": {"<date>": {"RSI": "<val>"}}}

  fred/series_{SERIES_ID}.csv
      columns: date, <SERIES_ID>   (already close to tidy — just two columns)

  yfinance/prices_{TICKER}.csv
      WIDE format: a "Field" column (Open/High/Low/Close/Adj Close/Volume)
      followed by one column per date. NOT a simple date|price table.

  yfinance/fundamentals_{TICKER}.csv
      columns: ticker, extracted_at, data
      -> `data` is a JSON string of a yfinance-`.info`-style metrics dict.

If you already have your real 59 files, just drop them into these same
three folders with these same filenames/columns — bronze.py doesn't care
whether the source is this generator or the real APIs.
"""
import json
import re
from datetime import date

import numpy as np
import pandas as pd

from src.config import (
    RAW, RAW_AV, RAW_FRED, RAW_YF, TICKERS, SECTOR_MAP, FRED_SERIES, FRED_MONTHLY,
    START_DATE, END_DATE, RANDOM_SEED,
)

rng = np.random.default_rng(RANDOM_SEED)

SECTOR_DRIFT = {
    "Technology": 0.00050, "Financials": 0.00022, "Healthcare": 0.00028,
    "Energy": 0.00016, "Consumer Discretionary": 0.00032,
}
SECTOR_VOL = {
    "Technology": 0.022, "Financials": 0.015, "Healthcare": 0.016,
    "Energy": 0.024, "Consumer Discretionary": 0.020,
}


def _trading_dates():
    return pd.bdate_range(START_DATE, END_DATE)


def _simulate_ohlcv(ticker: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    sector = SECTOR_MAP[ticker]
    n = len(dates)
    drift = SECTOR_DRIFT[sector] + rng.normal(0, 0.00008)
    vol = SECTOR_VOL[sector] * rng.uniform(0.85, 1.2)
    ret = drift + rng.normal(0, vol, n)
    close = rng.uniform(30, 400) * np.cumprod(1 + ret)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = low + (high - low) * rng.uniform(0, 1, n)
    adj_close = close * rng.uniform(0.97, 1.0)  # pretend a couple of small dividends
    volume = rng.uniform(2e5, 6e6) * (1 + 3 * np.abs(ret)) * rng.lognormal(0, 0.3, n)
    return pd.DataFrame({
        "date": dates, "Open": open_, "High": high, "Low": low,
        "Close": close, "Adj Close": adj_close, "Volume": volume.astype(int),
    })


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def write_yfinance_prices(ticker: str, ohlcv: pd.DataFrame) -> None:
    """WIDE format: rows are OHLCV fields, columns are dates."""
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    wide = ohlcv.set_index("date")[fields].T
    wide.columns = wide.columns.strftime("%Y-%m-%d")
    wide.insert(0, "Field", fields)
    wide.to_csv(RAW_YF / f"prices_{ticker}.csv", index=False)


def write_yfinance_fundamentals(ticker: str, close_last: float) -> None:
    info = {
        "marketCap": float(close_last * rng.uniform(2e8, 3e9)),
        "trailingPE": float(np.clip(rng.normal(22, 8), 5, 80)),
        "forwardPE": float(np.clip(rng.normal(19, 7), 4, 70)),
        "priceToBook": float(np.clip(rng.normal(4, 2.5), 0.4, 25)),
        "returnOnEquity": float(np.clip(rng.normal(0.15, 0.10), -0.3, 0.7)),
        "debtToEquity": float(np.clip(rng.normal(80, 60), 0, 400)),
        "profitMargins": float(np.clip(rng.normal(0.14, 0.09), -0.2, 0.5)),
        "revenueGrowth": float(np.clip(rng.normal(0.08, 0.10), -0.3, 0.6)),
        "dividendYield": float(max(0, rng.normal(0.012, 0.011))),
        "beta": float(np.clip(rng.normal(1.1, 0.4), 0.2, 2.5)),
    }
    row = pd.DataFrame([{
        "ticker": ticker,
        "extracted_at": pd.Timestamp(END_DATE).isoformat(),
        "data": json.dumps(info),
    }])
    row.to_csv(RAW_YF / f"fundamentals_{ticker}.csv", index=False)


def write_alphavantage_rsi(ticker: str, dates: pd.DatetimeIndex, rsi: pd.Series) -> None:
    payload = {
        "Technical Analysis: RSI": {
            d.strftime("%Y-%m-%d"): {"RSI": f"{v:.4f}"}
            for d, v in zip(dates, rsi) if pd.notna(v)
        }
    }
    row = pd.DataFrame([{
        "ticker": ticker, "indicator": "RSI", "interval": "daily",
        "extracted_at": pd.Timestamp(END_DATE).isoformat(),
        "row_count": len(payload["Technical Analysis: RSI"]),
        "params": json.dumps({"time_period": 14, "series_type": "close"}),
        "data": json.dumps(payload),
    }])
    row.to_csv(RAW_AV / f"rsi_{ticker}.csv", index=False)


def write_fred_series(series_id: str) -> None:
    freq = "MS" if series_id in FRED_MONTHLY else "B"
    dates = pd.date_range(START_DATE, END_DATE, freq=freq)
    n = len(dates)
    base = {
        "CPIAUCSL": 280.0, "FEDFUNDS": 0.5, "GS10": 1.8,
        "UNRATE": 4.0, "VIXCLS": 18.0,
    }[series_id]
    step = {
        "CPIAUCSL": 0.6, "FEDFUNDS": 0.09, "GS10": 0.05,
        "UNRATE": 0.06, "VIXCLS": 1.0,
    }[series_id]
    vals = base + np.cumsum(rng.normal(step, step * 1.5, n))
    if series_id == "VIXCLS":
        vals = np.clip(vals, 10, 60)
    if series_id == "UNRATE":
        vals = np.clip(vals, 2.5, 9)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), series_id: vals.round(4)})
    # FRED occasionally has a few NA rows around release lags — keep it real
    na_idx = df.sample(n=max(1, n // 60), random_state=RANDOM_SEED).index
    df.loc[na_idx, series_id] = np.nan
    df.to_csv(RAW_FRED / f"series_{series_id}.csv", index=False)


def _last_date_in_raw() -> pd.Timestamp | None:
    """Peek at one yfinance prices file to see how fresh the raw layer is."""
    sample = next(iter(sorted(RAW_YF.glob("prices_*.csv"))), None)
    if sample is None:
        return None
    raw = pd.read_csv(sample, nrows=1)
    date_cols = [c for c in raw.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c))]
    if not date_cols:
        return None
    return pd.to_datetime(max(date_cols))


def build_raw_layer(force: bool = False) -> None:
    marker = RAW / ".stub_generated"
    existing = list(RAW_AV.glob("*.csv")) + list(RAW_FRED.glob("*.csv")) + list(RAW_YF.glob("*.csv"))

    if existing and not marker.exists() and not force:
        # Real files (no marker) -- never auto-overwrite. Refreshing these to
        # "today" means re-pulling from Alpha Vantage / FRED / yfinance
        # yourself; this generator only manages its own stand-in data.
        print(f"Raw data already present ({len(existing)} files, no stub marker found) "
              "— treating as your real data and leaving it untouched.")
        return

    if existing and marker.exists() and not force:
        last_date = _last_date_in_raw()
        today = pd.Timestamp(date.today())
        # weekend-aware: don't regenerate every Sat/Sun just because "today"
        # isn't a trading day yet
        stale = last_date is not None and (today - last_date).days > 3
        if not stale:
            print(f"Stand-in raw data already up to date (last date: "
                  f"{last_date.date() if last_date is not None else 'unknown'}) — skipping regeneration.")
            return
        print(f"Stand-in raw data is stale (last date: {last_date.date()}, today: {today.date()}) "
              "— regenerating through today.")

    print("Generating stand-in raw data (alphavantage / fred / yfinance)...")
    dates = _trading_dates()
    for ticker in TICKERS:
        ohlcv = _simulate_ohlcv(ticker, dates)
        write_yfinance_prices(ticker, ohlcv)
        write_yfinance_fundamentals(ticker, ohlcv["Close"].iloc[-1])
        rsi = _rsi(ohlcv["Close"])
        write_alphavantage_rsi(ticker, dates, rsi)

    for series_id in FRED_SERIES:
        write_fred_series(series_id)

    marker.write_text(f"generated_through={END_DATE}\n")

    n_files = len(list(RAW_AV.glob("*.csv"))) + len(list(RAW_FRED.glob("*.csv"))) + len(list(RAW_YF.glob("*.csv")))
    print(f"  wrote {n_files} files across alphavantage/ fred/ yfinance/ (through {END_DATE})")


if __name__ == "__main__":
    build_raw_layer(force=True)
