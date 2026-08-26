"""
Gold Layer
===========
Writes the final, dashboard/BI-ready analytics tables. Two families:

  Descriptive (built straight from the engineered feature panel):
    vw_stock_performance, vw_sector_performance, vw_macro_overlay,
    vw_fundamental_scorecard, vw_risk_metrics, vw_volume_anomalies,
    vw_forecast_input

  ML-driven (built from modeling.py / ranking.py / backtest.py output):
    vw_model_predictions, vw_stock_rankings, vw_portfolio_backtest

Every table is written to data/gold/<name>.csv (parquet if pyarrow is
available) — this is the layer the Streamlit dashboard reads from.
"""
import numpy as np
import pandas as pd

from src.config import GOLD, RISK_FREE_ANNUAL

TRADING_DAYS = 252


def _save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    try:
        df.to_parquet(GOLD / f"{name}.parquet", index=False)
        path = GOLD / f"{name}.parquet"
    except (ImportError, ValueError):
        df.to_csv(GOLD / f"{name}.csv", index=False)
        path = GOLD / f"{name}.csv"
    df.to_csv(GOLD / f"{name}.csv", index=False)  # always also write csv for the dashboard
    print(f"  -> {name}: {df.shape[0]:,} rows, {df.shape[1]} cols [{path}]")
    return df


def build_vw_stock_performance(feat: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "sector", "date", "price", "volume", "daily_return", "cum_return",
            "ma_5", "ma_20", "ma_60", "volatility_20", "rsi", "drawdown", "momentum_20"]
    return _save(feat[[c for c in cols if c in feat.columns]].copy(), "vw_stock_performance")


def build_vw_sector_performance(feat: pd.DataFrame) -> pd.DataFrame:
    g = feat.groupby(["date", "sector"])
    perf = g.agg(avg_daily_return=("daily_return", "mean"),
                 return_dispersion=("daily_return", "std"),
                 num_tickers=("ticker", "nunique"),
                 total_volume=("volume", "sum")).reset_index()
    perf = perf.sort_values(["sector", "date"])
    perf["sector_cum_return"] = perf.groupby("sector")["avg_daily_return"] \
        .apply(lambda s: (1 + s.fillna(0)).cumprod() - 1).reset_index(level=0, drop=True)
    return _save(perf, "vw_sector_performance")


def build_vw_macro_overlay(feat: pd.DataFrame) -> pd.DataFrame:
    macro_cols = [c for c in ["CPIAUCSL", "FEDFUNDS", "GS10", "UNRATE", "VIXCLS",
                               "cpi_yoy", "fedfunds_change_20d", "gs10_change_20d",
                               "unrate_change_20d", "vix_change_5d"] if c in feat.columns]
    overlay = feat.groupby("date").agg(
        market_return=("daily_return", "mean"),
        **{c: (c, "mean") for c in macro_cols},
    ).reset_index()
    overlay["market_cum_return"] = (1 + overlay["market_return"].fillna(0)).cumprod() - 1
    return _save(overlay, "vw_macro_overlay")


def build_vw_fundamental_scorecard(feat: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "sector", "date", "trailingPE", "priceToBook", "returnOnEquity",
            "debtToEquity", "profitMargins", "revenueGrowth", "dividendYield", "beta",
            "valuation_score", "growth_score", "profitability_score", "leverage_score",
            "fundamental_composite_score"]
    df = feat[[c for c in cols if c in feat.columns]].copy()
    df["sector_rank"] = df.groupby(["date", "sector"])["fundamental_composite_score"] \
        .rank(ascending=False, method="min")
    return _save(df, "vw_fundamental_scorecard")


def build_vw_risk_metrics(feat: pd.DataFrame) -> pd.DataFrame:
    market_ret = feat.groupby("date")["daily_return"].mean().rename("market_return")
    f = feat.merge(market_ret, on="date", how="left")

    rows = []
    for ticker, g in f.groupby("ticker"):
        r = g["daily_return"].dropna()
        if len(r) < 30:
            continue
        rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        ann_return = (1 + r.mean()) ** TRADING_DAYS - 1
        downside = r[r < 0]
        sortino = ((ann_return - RISK_FREE_ANNUAL) / (downside.std() * np.sqrt(TRADING_DAYS))
                   if downside.std() > 0 else np.nan)
        sharpe = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan
        var_95 = np.percentile(r, 5)
        cvar_95 = r[r <= var_95].mean()
        max_dd = g["drawdown"].min()
        cov = g["daily_return"].cov(g["market_return"])
        var = g["market_return"].var()
        beta_mkt = cov / var if var else np.nan
        rows.append({"ticker": ticker, "sector": g["sector"].iloc[0],
                     "annualized_return": ann_return, "annualized_volatility": ann_vol,
                     "sharpe_ratio": sharpe, "sortino_ratio": sortino,
                     "market_beta": beta_mkt, "max_drawdown": max_dd,
                     "var_95_daily": var_95, "cvar_95_daily": cvar_95, "n_obs": len(r)})
    risk = pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False)
    risk["risk_rank"] = risk["sharpe_ratio"].rank(ascending=False, method="min")
    return _save(risk, "vw_risk_metrics")


def build_vw_volume_anomalies(feat: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    df = feat[["ticker", "sector", "date", "volume", "volume_zscore", "daily_return"]].copy()
    df["is_anomaly"] = df["volume_zscore"].abs() >= z_threshold
    anomalies = df[df["is_anomaly"]].sort_values("volume_zscore", ascending=False)
    print(f"  [vw_volume_anomalies] flagged {len(anomalies):,} of {len(df):,} "
          f"ticker-days ({len(anomalies)/len(df):.2%})")
    return _save(anomalies, "vw_volume_anomalies")


def build_vw_forecast_input(feat: pd.DataFrame) -> pd.DataFrame:
    cols = ["ticker", "sector", "date", "daily_return", "momentum_10", "momentum_20",
            "ma_5", "ma_20", "ma_60", "volatility_5", "volatility_20", "volatility_60",
            "rsi", "cum_return", "drawdown", "volume_zscore", "cpi_yoy",
            "fedfunds_change_20d", "gs10_change_20d", "unrate_change_20d", "vix_change_5d",
            "valuation_score", "growth_score", "profitability_score", "leverage_score",
            "fundamental_composite_score", "forward_return", "forward_direction"]
    df = feat[[c for c in cols if c in feat.columns]].dropna(subset=["ma_60", "volatility_60"])
    return _save(df, "vw_forecast_input")


def build_vw_model_predictions(model_results: dict) -> pd.DataFrame:
    reg = model_results["regression_test_predictions"]
    clf = model_results["classification_test_predictions"]
    merged = reg.merge(clf[["date", "ticker", "predicted_probability", "forward_return"]],
                        on=["date", "ticker"], how="outer")
    return _save(merged, "vw_model_predictions")


def build_vw_stock_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    return _save(rankings, "vw_stock_rankings")


def build_vw_portfolio_backtest(backtest_results: dict) -> pd.DataFrame:
    return _save(backtest_results["equity_curve"], "vw_portfolio_backtest")


def run_gold(feat: pd.DataFrame, model_results: dict, rankings: pd.DataFrame,
             backtest_results: dict) -> dict:
    print("=" * 60, "\nGOLD LAYER\n", "=" * 60, sep="")
    views = {
        "vw_stock_performance": build_vw_stock_performance(feat),
        "vw_sector_performance": build_vw_sector_performance(feat),
        "vw_macro_overlay": build_vw_macro_overlay(feat),
        "vw_fundamental_scorecard": build_vw_fundamental_scorecard(feat),
        "vw_risk_metrics": build_vw_risk_metrics(feat),
        "vw_volume_anomalies": build_vw_volume_anomalies(feat),
        "vw_forecast_input": build_vw_forecast_input(feat),
        "vw_model_predictions": build_vw_model_predictions(model_results),
        "vw_stock_rankings": build_vw_stock_rankings(rankings),
        "vw_portfolio_backtest": build_vw_portfolio_backtest(backtest_results),
    }
    # backtest summary + holdings log are small tables worth persisting too
    _save(backtest_results["comparison"].reset_index().rename(columns={"index": "portfolio"}),
          "vw_backtest_summary")
    if not backtest_results["holdings_log"].empty:
        _save(backtest_results["holdings_log"], "vw_backtest_holdings_log")
    return views
