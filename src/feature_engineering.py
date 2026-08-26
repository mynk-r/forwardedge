"""
Feature Engineering
=====================
Builds three feature blocks on top of the Silver master dataset:

  Technical  — returns, momentum, moving averages, volatility, drawdown
               (RSI itself already arrives from Bronze/Silver)
  Macro      — CPI YoY growth, rate changes, yield changes, VIX changes
  Fundamental — valuation / growth / profitability / leverage scores,
                 built as cross-sectional percentile ranks per date so
                 they're comparable across tickers of very different scale
"""
import numpy as np
import pandas as pd

from src.config import ROLLING_WINDOWS


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", group_keys=False)

    df["daily_return"] = g["price"].apply(lambda s: s.pct_change())
    df["momentum_10"] = g["price"].apply(lambda s: s.pct_change(10))
    df["momentum_20"] = g["price"].apply(lambda s: s.pct_change(20))

    for name, w in ROLLING_WINDOWS.items():
        df[f"ma_{w}"] = g["price"].apply(lambda s, w=w: s.rolling(w).mean())
        df[f"volatility_{w}"] = g["daily_return"].apply(
            lambda s, w=w: s.rolling(w).std() * np.sqrt(252)
        )

    df["cum_return"] = g["daily_return"].apply(lambda s: (1 + s.fillna(0)).cumprod() - 1)
    running_max = g["price"].apply(lambda s: s.cummax())
    df["drawdown"] = (df["price"] - running_max) / running_max

    vol_mean = g["volume"].apply(lambda s: s.rolling(20).mean())
    vol_std = g["volume"].apply(lambda s: s.rolling(20).std())
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)

    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    if "CPIAUCSL" in df.columns:
        df["cpi_yoy"] = df.groupby("ticker")["CPIAUCSL"].transform(lambda s: s.pct_change(252))
    if "FEDFUNDS" in df.columns:
        df["fedfunds_change_20d"] = df.groupby("ticker")["FEDFUNDS"].transform(lambda s: s.diff(20))
    if "GS10" in df.columns:
        df["gs10_change_20d"] = df.groupby("ticker")["GS10"].transform(lambda s: s.diff(20))
    if "UNRATE" in df.columns:
        df["unrate_change_20d"] = df.groupby("ticker")["UNRATE"].transform(lambda s: s.diff(20))
    if "VIXCLS" in df.columns:
        df["vix_change_5d"] = df.groupby("ticker")["VIXCLS"].transform(lambda s: s.diff(5))
    return df


def add_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (per-date) percentile scores across four factor
    families. Lower PE/PB/D-E is "better" for value/leverage, so those are
    rank-inverted; ROE/margins/growth are rank-as-is."""
    df = df.copy()

    def pct_rank(s: pd.Series, ascending: bool) -> pd.Series:
        return s.rank(pct=True, ascending=ascending)

    grp = df.groupby("date")
    if {"trailingPE", "priceToBook"}.issubset(df.columns):
        df["valuation_score"] = (
            grp["trailingPE"].transform(lambda s: pct_rank(s, ascending=False)) * 0.5
            + grp["priceToBook"].transform(lambda s: pct_rank(s, ascending=False)) * 0.5
        )
    if "revenueGrowth" in df.columns:
        df["growth_score"] = grp["revenueGrowth"].transform(lambda s: pct_rank(s, ascending=True))
    if {"profitMargins", "returnOnEquity"}.issubset(df.columns):
        df["profitability_score"] = (
            grp["profitMargins"].transform(lambda s: pct_rank(s, ascending=True)) * 0.5
            + grp["returnOnEquity"].transform(lambda s: pct_rank(s, ascending=True)) * 0.5
        )
    if "debtToEquity" in df.columns:
        df["leverage_score"] = grp["debtToEquity"].transform(lambda s: pct_rank(s, ascending=False))

    score_cols = [c for c in ["valuation_score", "growth_score", "profitability_score",
                               "leverage_score"] if c in df.columns]
    df["fundamental_composite_score"] = df[score_cols].mean(axis=1)
    return df


def add_forward_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    def _fwd(g: pd.DataFrame) -> pd.Series:
        return g["price"].shift(-horizon) / g["price"] - 1

    df["forward_return"] = df.groupby("ticker", group_keys=False).apply(_fwd)
    df["forward_direction"] = (df["forward_return"] > 0).astype(int)
    return df


def run_feature_engineering(master: pd.DataFrame, horizon: int) -> pd.DataFrame:
    print("=" * 60, "\nFEATURE ENGINEERING\n", "=" * 60, sep="")
    df = add_technical_features(master)
    df = add_macro_features(df)
    df = add_fundamental_features(df)
    df = add_forward_target(df, horizon)
    print(f"[features] engineered panel: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


if __name__ == "__main__":
    from src.config import SILVER, FORWARD_HORIZON_DAYS
    master = pd.read_csv(SILVER / "master_dataset.csv", parse_dates=["date"])
    run_feature_engineering(master, FORWARD_HORIZON_DAYS)
