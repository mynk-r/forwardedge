"""
Silver Layer
=============
Takes the four Bronze long-format tables and produces one clean, aligned
Master Dataset at Date x Ticker grain:

    date | ticker | sector | price | rsi | <fred series as columns> | <fundamental metrics as columns>

Steps: de-duplication, numeric coercion, date alignment onto the trading
calendar (macro is monthly/daily -> forward-filled onto trading days;
fundamentals are point-in-time snapshots -> broadcast as "latest known" per
ticker, which is a deliberate simplification worth calling out rather than
hiding), and missing-value handling. No technical/fundamental *feature
engineering* happens here — only cleaning + integration.
"""
import numpy as np
import pandas as pd

from src.config import SILVER, SECTOR_MAP


def _dedupe(df: pd.DataFrame, subset) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="last")
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped} duplicate rows on {subset}")
    return df


def clean_prices(yf_prices: pd.DataFrame) -> pd.DataFrame:
    df = _dedupe(yf_prices, ["date", "ticker"]).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df = df.sort_values(["ticker", "date"])
    n_missing = df["price"].isna().sum()
    df["price"] = df.groupby("ticker")["price"].transform(lambda s: s.ffill().bfill())
    df["volume"] = df.groupby("ticker")["volume"].transform(lambda s: s.fillna(s.median()))
    print(f"[silver] prices: imputed {n_missing} missing price points (ticker-wise ffill/bfill)")
    return df[["date", "ticker", "price", "volume"]]


def clean_rsi(av_rsi: pd.DataFrame) -> pd.DataFrame:
    df = _dedupe(av_rsi, ["date", "ticker"]).copy()
    df["rsi"] = pd.to_numeric(df["rsi"], errors="coerce")
    df = df.sort_values(["ticker", "date"])
    df["rsi"] = df.groupby("ticker")["rsi"].transform(lambda s: s.ffill().bfill())
    return df[["date", "ticker", "rsi"]]


def clean_macro(fred: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Pivot to wide (date x series), then forward-fill each series onto the
    daily trading calendar -- monthly releases (CPI, Fed funds, 10Y, UNRATE)
    are only known as of their release date, so ffill is the leakage-safe
    choice (never bfill macro)."""
    df = _dedupe(fred, ["date", "series"]).copy()
    wide = df.pivot(index="date", columns="series", values="value").sort_index()
    wide = wide.reindex(wide.index.union(trading_dates)).sort_index()
    wide = wide.ffill()
    wide = wide.reindex(trading_dates)
    wide = wide.ffill().bfill()  # bfill only to cover the very first rows before any release
    wide = wide.rename_axis("date").reset_index()
    return wide


def clean_fundamentals(yf_fund: pd.DataFrame) -> pd.DataFrame:
    """Pivot metric rows to columns; one row per ticker (point-in-time
    snapshot). Broadcasting this across dates happens in build_master()."""
    df = _dedupe(yf_fund, ["ticker", "metric"]).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    wide = df.pivot(index="ticker", columns="metric", values="value").reset_index()
    # a few missing metrics (late/unreported) -> sector-median impute
    wide["sector"] = wide["ticker"].map(SECTOR_MAP)
    metric_cols = [c for c in wide.columns if c not in ("ticker", "sector")]
    for c in metric_cols:
        wide[c] = wide.groupby("sector")[c].transform(lambda s: s.fillna(s.median()))
        wide[c] = wide[c].fillna(wide[c].median())
    return wide


def build_master(prices: pd.DataFrame, rsi: pd.DataFrame, macro: pd.DataFrame,
                  fundamentals: pd.DataFrame) -> pd.DataFrame:
    master = prices.merge(rsi, on=["date", "ticker"], how="left")
    master["sector"] = master["ticker"].map(SECTOR_MAP)
    master = master.merge(macro, on="date", how="left")
    master = master.merge(fundamentals.drop(columns=["sector"]), on="ticker", how="left")

    # any remaining gaps (e.g. RSI's 14-day warmup window) -> ticker-wise fill
    fill_cols = [c for c in master.columns if c not in ("date", "ticker", "sector")]
    master[fill_cols] = master.groupby("ticker")[fill_cols].transform(lambda s: s.ffill().bfill())

    master = master.sort_values(["ticker", "date"]).reset_index(drop=True)
    return master


def run_silver(av_rsi, fred, yf_prices, yf_fund):
    print("=" * 60, "\nSILVER LAYER\n", "=" * 60, sep="")
    prices = clean_prices(yf_prices)
    rsi = clean_rsi(av_rsi)
    trading_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    macro = clean_macro(fred, trading_dates)
    fundamentals = clean_fundamentals(yf_fund)

    master = build_master(prices, rsi, macro, fundamentals)

    remaining_na = master.isna().sum().sum()
    print(f"[silver] master dataset: {master.shape[0]:,} rows x {master.shape[1]} cols, "
          f"{remaining_na} remaining NaNs")

    master.to_csv(SILVER / "master_dataset.csv", index=False)
    print(f"Master dataset written -> {SILVER / 'master_dataset.csv'}")
    return master


if __name__ == "__main__":
    from src.bronze import run_bronze
    av_rsi, fred, yf_prices, yf_fund = run_bronze()
    run_silver(av_rsi, fred, yf_prices, yf_fund)
