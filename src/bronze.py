"""
Bronze Layer
=============
Extract-only: parse each raw source's *actual* on-disk shape into a
standardized long format. No cleaning, no feature engineering, no
cross-source joins here — that's Silver's job. Each function inspects the
real structure before assuming anything (per the source spec), rather than
blindly reshaping.

Outputs (data/bronze/):
    bronze_alphavantage_rsi.csv   date | ticker | indicator | rsi
    bronze_fred.csv               date | series  | value
    bronze_yfinance_prices.csv    date | ticker | price   (+ price_field used)
    bronze_yfinance_fundamentals.csv  ticker | extracted_at | metric | value
"""
import json
import re

import numpy as np
import pandas as pd

from src.config import RAW_AV, RAW_FRED, RAW_YF, BRONZE


def parse_alphavantage_rsi() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_AV.glob("rsi_*.csv")):
        raw = pd.read_csv(path)
        assert set(["ticker", "indicator", "interval", "extracted_at", "row_count",
                     "params", "data"]).issubset(raw.columns), f"unexpected columns in {path.name}"

        for _, r in raw.iterrows():
            payload = json.loads(r["data"])
            # inspect the actual key rather than assuming "Technical Analysis: RSI"
            series_key = next((k for k in payload if "Technical Analysis" in k), None)
            if series_key is None:
                # fall back: single nested dict of date -> {metric: value}
                series_key = list(payload.keys())[0]
            series = payload[series_key]
            for date_str, obs in series.items():
                # obs is itself a dict, e.g. {"RSI": "45.12"}
                value_key = next(iter(obs))
                rows.append({
                    "date": date_str, "ticker": r["ticker"],
                    "indicator": r["indicator"], "rsi": float(obs[value_key]),
                })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"[bronze] alphavantage RSI: {df.shape[0]:,} rows, {df['ticker'].nunique()} tickers")
    return df


def parse_fred() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_FRED.glob("series_*.csv")):
        raw = pd.read_csv(path)
        assert "date" in raw.columns, f"{path.name} missing date column"
        value_cols = [c for c in raw.columns if c != "date"]
        assert len(value_cols) == 1, f"{path.name} has more than one value column: {value_cols}"
        series_id_from_col = value_cols[0]
        series_id_from_name = re.sub(r"^series_|\.csv$", "", path.name)
        # trust the column name; filename is just a convenience label
        series_id = series_id_from_col if series_id_from_col else series_id_from_name

        sub = raw.rename(columns={series_id_from_col: "value"})
        sub["series"] = series_id
        rows.append(sub[["date", "series", "value"]])
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["series", "date"]).reset_index(drop=True)
    print(f"[bronze] fred: {df.shape[0]:,} rows, {df['series'].nunique()} series")
    return df


def parse_yfinance_prices() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_YF.glob("prices_*.csv")):
        raw = pd.read_csv(path)
        ticker = re.sub(r"^prices_|\.csv$", "", path.name)

        # identify which columns are actual calendar dates vs. the field label
        date_cols = [c for c in raw.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c))]
        label_col = [c for c in raw.columns if c not in date_cols]
        assert len(label_col) == 1, f"{path.name}: expected exactly one non-date column, got {label_col}"
        label_col = label_col[0]

        # confirm which row represents the price series before assuming Close
        available_fields = set(raw[label_col])
        price_field = "Close" if "Close" in available_fields else (
            "Adj Close" if "Adj Close" in available_fields else None
        )
        if price_field is None:
            raise ValueError(f"{path.name}: no Close/Adj Close field found among {available_fields}")

        price_row = raw.loc[raw[label_col] == price_field, date_cols]
        long = price_row.T.reset_index()
        long.columns = ["date", "price"]
        long["ticker"] = ticker
        long["price_field"] = price_field

        if "Volume" in available_fields:
            vol_row = raw.loc[raw[label_col] == "Volume", date_cols]
            vol_long = vol_row.T.reset_index()
            vol_long.columns = ["date", "volume"]
            long = long.merge(vol_long, on="date", how="left")
        else:
            long["volume"] = np.nan

        rows.append(long)

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"[bronze] yfinance prices: {df.shape[0]:,} rows, {df['ticker'].nunique()} tickers "
          f"(price field used: {df['price_field'].unique().tolist()})")
    return df[["date", "ticker", "price", "volume", "price_field"]]


def parse_yfinance_fundamentals() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_YF.glob("fundamentals_*.csv")):
        raw = pd.read_csv(path)
        assert set(["ticker", "extracted_at", "data"]).issubset(raw.columns), \
            f"unexpected columns in {path.name}"
        for _, r in raw.iterrows():
            metrics = json.loads(r["data"])
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) or value is None:
                    rows.append({
                        "ticker": r["ticker"], "extracted_at": r["extracted_at"],
                        "metric": metric, "value": value,
                    })
                # non-numeric metrics (strings, nested dicts) are intentionally
                # dropped here -- Bronze should not silently invent numerics
    df = pd.DataFrame(rows)
    df["extracted_at"] = pd.to_datetime(df["extracted_at"])
    print(f"[bronze] yfinance fundamentals: {df.shape[0]:,} rows, "
          f"{df['ticker'].nunique()} tickers, {df['metric'].nunique()} metrics")
    return df


def run_bronze():
    print("=" * 60, "\nBRONZE LAYER\n", "=" * 60, sep="")
    av_rsi = parse_alphavantage_rsi()
    fred = parse_fred()
    yf_prices = parse_yfinance_prices()
    yf_fund = parse_yfinance_fundamentals()

    av_rsi.to_csv(BRONZE / "bronze_alphavantage_rsi.csv", index=False)
    fred.to_csv(BRONZE / "bronze_fred.csv", index=False)
    yf_prices.to_csv(BRONZE / "bronze_yfinance_prices.csv", index=False)
    yf_fund.to_csv(BRONZE / "bronze_yfinance_fundamentals.csv", index=False)
    print(f"Bronze tables written -> {BRONZE}")

    return av_rsi, fred, yf_prices, yf_fund


if __name__ == "__main__":
    run_bronze()
