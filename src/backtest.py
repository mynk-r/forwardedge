"""
Portfolio Backtest
====================
Turns the Stock Ranking Engine's output into an actual (simulated)
strategy: every REBALANCE_DAYS trading days, buy an equal-weight basket of
the TOP_N ranked tickers as of that date and hold until the next rebalance.
Compared against an equal-weight benchmark that holds the full universe the
whole time. Both run strictly inside the test period (no train/val
leakage), and rebalancing only ever looks at the ranking known *as of* the
rebalance date.
"""
import numpy as np
import pandas as pd

from src.config import TOP_N_PORTFOLIO, REBALANCE_DAYS, RISK_FREE_ANNUAL


def _perf_stats(daily_returns: pd.Series) -> dict:
    r = daily_returns.dropna()
    if r.empty or r.std() == 0:
        return {"total_return": np.nan, "annualized_return": np.nan, "annualized_vol": np.nan,
                "sharpe": np.nan, "sortino": np.nan, "max_drawdown": np.nan, "hit_rate": np.nan}
    rf_daily = RISK_FREE_ANNUAL / 252
    cum = (1 + r).cumprod()
    total_return = cum.iloc[-1] - 1
    ann_return = (1 + r.mean()) ** 252 - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() - rf_daily) / r.std() * np.sqrt(252)
    downside = r[r < 0]
    sortino = ((r.mean() - rf_daily) / downside.std() * np.sqrt(252)
               if len(downside) and downside.std() > 0 else np.nan)
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    hit_rate = (r > 0).mean()
    return {"total_return": total_return, "annualized_return": ann_return,
            "annualized_vol": ann_vol, "sharpe": sharpe, "sortino": sortino,
            "max_drawdown": max_dd, "hit_rate": hit_rate}


def run_backtest(rankings: pd.DataFrame, test_panel: pd.DataFrame) -> dict:
    print("=" * 60, "\nPORTFOLIO BACKTEST\n", "=" * 60, sep="")

    daily = test_panel[["date", "ticker", "daily_return"]].dropna()
    dates = sorted(daily["date"].unique())

    strategy_rows, benchmark_rows, holdings_log = [], [], []
    i = 0
    while i < len(dates):
        rebal_date = dates[i]
        period_dates = dates[i + 1: i + 1 + REBALANCE_DAYS]
        if not period_dates:
            break

        day_rank = rankings[rankings["date"] == rebal_date]
        top_tickers = day_rank.nsmallest(TOP_N_PORTFOLIO, "rank")["ticker"].tolist()
        if top_tickers:
            holdings_log.append({"rebalance_date": rebal_date, "tickers": ", ".join(top_tickers)})

        period = daily[daily["date"].isin(period_dates)]
        strat = (
            period[period["ticker"].isin(top_tickers)]
            .groupby("date")["daily_return"].mean()
        )
        bench = period.groupby("date")["daily_return"].mean()  # equal-weight full universe

        strategy_rows.append(strat)
        benchmark_rows.append(bench)
        i += REBALANCE_DAYS

    strategy_returns = pd.concat(strategy_rows).sort_index() if strategy_rows else pd.Series(dtype=float)
    benchmark_returns = pd.concat(benchmark_rows).sort_index() if benchmark_rows else pd.Series(dtype=float)

    strategy_stats = _perf_stats(strategy_returns)
    benchmark_stats = _perf_stats(benchmark_returns)

    comparison = pd.DataFrame({"Top-N ML Strategy": strategy_stats, "Equal-Weight Benchmark": benchmark_stats}).T
    print(comparison.round(4).to_string())

    equity_curve = pd.DataFrame({
        "date": strategy_returns.index,
        "strategy_cum_return": (1 + strategy_returns.fillna(0)).cumprod() - 1,
        "benchmark_cum_return": (1 + benchmark_returns.reindex(strategy_returns.index).fillna(0)).cumprod() - 1,
    })

    holdings = pd.DataFrame(holdings_log)

    return {"comparison": comparison, "equity_curve": equity_curve, "holdings_log": holdings,
            "strategy_returns": strategy_returns, "benchmark_returns": benchmark_returns}
