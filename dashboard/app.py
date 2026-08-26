"""
ForwardEdge — Streamlit Dashboard
=============================================
Reads exclusively from data/gold/*.csv. Re-running `python main.py` and then
refreshing the browser is the whole "automated pipeline" refresh loop —
there's no caching here that would show stale data after a re-run.

Run:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import GOLD  # noqa: E402

st.set_page_config(page_title="ForwardEdge", layout="wide", page_icon="📈")


@st.cache_data(ttl=30)
def load_gold(name: str) -> pd.DataFrame:
    path = GOLD / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in df.columns:
        if c in ("date", "quarter_date", "extracted_at", "rebalance_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


# ---------------------------------------------------------------- sidebar --
st.sidebar.title("📈 ForwardEdge")
st.sidebar.caption("Data-driven equity screening: from raw market data to risk-adjusted stock rankings")
page = st.sidebar.radio("Navigate", [
    "Executive Summary", "Stock Analytics", "Fundamentals",
    "Macro & Risk", "ML Intelligence", "Portfolio Simulator",
])
st.sidebar.caption("Data source: data/gold/  ·  refresh by re-running `python main.py`")

stock_perf = load_gold("vw_stock_performance")
sector_perf = load_gold("vw_sector_performance")
macro = load_gold("vw_macro_overlay")
fundamentals = load_gold("vw_fundamental_scorecard")
risk = load_gold("vw_risk_metrics")
vol_anomalies = load_gold("vw_volume_anomalies")
rankings = load_gold("vw_stock_rankings")
predictions = load_gold("vw_model_predictions")
backtest_curve = load_gold("vw_portfolio_backtest")
backtest_summary = load_gold("vw_backtest_summary")

if stock_perf.empty:
    st.warning("No data found in data/gold/. Run `python main.py` first.")
    st.stop()

ALL_TICKERS = sorted(stock_perf["ticker"].unique())
ALL_SECTORS = sorted(stock_perf["sector"].dropna().unique())


# =========================================================== EXEC SUMMARY ==
if page == "Executive Summary":
    st.title("Executive Summary")
    st.caption("Can market, technical, fundamental, and macro indicators predict "
               "forward risk-adjusted return potential?")

    latest_date = stock_perf["date"].max()
    latest = stock_perf[stock_perf["date"] == latest_date]
    market_cum = macro.sort_values("date")["market_cum_return"].iloc[-1] if not macro.empty else np.nan
    best_sharpe = risk.sort_values("sharpe_ratio", ascending=False).iloc[0] if not risk.empty else None
    strat_row = backtest_summary[backtest_summary["portfolio"] == "Top-N ML Strategy"] \
        if not backtest_summary.empty else pd.DataFrame()
    bench_row = backtest_summary[backtest_summary["portfolio"] == "Equal-Weight Benchmark"] \
        if not backtest_summary.empty else pd.DataFrame()

    kpi_row([
        ("Tickers Tracked", f"{len(ALL_TICKERS)}", None),
        ("Sectors", f"{len(ALL_SECTORS)}", None),
        ("Universe Cum. Return", f"{market_cum:.1%}" if pd.notna(market_cum) else "—", None),
        ("Best Sharpe Ticker", best_sharpe["ticker"] if best_sharpe is not None else "—",
         f"{best_sharpe['sharpe_ratio']:.2f}" if best_sharpe is not None else None),
    ])

    if not strat_row.empty and not bench_row.empty:
        kpi_row([
            ("ML Strategy Total Return", f"{strat_row['total_return'].iloc[0]:.1%}", None),
            ("Benchmark Total Return", f"{bench_row['total_return'].iloc[0]:.1%}", None),
            ("ML Strategy Sharpe", f"{strat_row['sharpe'].iloc[0]:.2f}", None),
            ("Benchmark Sharpe", f"{bench_row['sharpe'].iloc[0]:.2f}", None),
        ])

    st.subheader("Sector cumulative return")
    fig = px.line(sector_perf, x="date", y="sector_cum_return", color="sector")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top / bottom ranked tickers (most recent test date)")
    if not rankings.empty:
        last_rank_date = rankings["date"].max()
        snap = rankings[rankings["date"] == last_rank_date].sort_values("rank")
        c1, c2 = st.columns(2)
        c1.write("**Top 5**")
        c1.dataframe(snap.head(5)[["ticker", "sector", "final_signal", "rank"]], hide_index=True)
        c2.write("**Bottom 5**")
        c2.dataframe(snap.tail(5)[["ticker", "sector", "final_signal", "rank"]], hide_index=True)
    else:
        st.info("No ranking data available yet.")


# ============================================================ STOCK PAGE ==
elif page == "Stock Analytics":
    st.title("Stock Analytics")
    ticker = st.selectbox("Ticker", ALL_TICKERS)
    df = stock_perf[stock_perf["ticker"] == ticker].sort_values("date")

    kpi_row([
        ("Latest Price", f"${df['price'].iloc[-1]:,.2f}", None),
        ("20D Momentum", f"{df['momentum_20'].iloc[-1]:.1%}" if pd.notna(df['momentum_20'].iloc[-1]) else "—", None),
        ("RSI (14)", f"{df['rsi'].iloc[-1]:.1f}", None),
        ("Max Drawdown", f"{df['drawdown'].min():.1%}", None),
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["price"], name="Price"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_20"], name="MA 20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma_60"], name="MA 60", line=dict(dash="dot")))
    fig.update_layout(title=f"{ticker} price & moving averages", height=420)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_rsi = px.line(df, x="date", y="rsi", title="RSI (14)")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        st.plotly_chart(fig_rsi, use_container_width=True)
    with c2:
        fig_vol = px.line(df, x="date", y="volatility_20", title="20D annualized volatility")
        st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Volume anomalies")
    tick_anom = vol_anomalies[vol_anomalies["ticker"] == ticker]
    if not tick_anom.empty:
        st.dataframe(tick_anom.sort_values("date", ascending=False), hide_index=True)
    else:
        st.caption("No flagged volume anomalies for this ticker.")


# ======================================================= FUNDAMENTALS ====
elif page == "Fundamentals":
    st.title("Fundamentals")
    sector_filter = st.multiselect("Sector", ALL_SECTORS, default=ALL_SECTORS)
    latest_date = fundamentals["date"].max()
    snap = fundamentals[(fundamentals["date"] == latest_date) & (fundamentals["sector"].isin(sector_filter))]

    st.subheader("Fundamental scorecard (latest)")
    show_cols = [c for c in ["ticker", "sector", "trailingPE", "priceToBook", "returnOnEquity",
                              "debtToEquity", "profitMargins", "revenueGrowth", "dividendYield",
                              "fundamental_composite_score", "sector_rank"] if c in snap.columns]
    st.dataframe(snap[show_cols].sort_values("fundamental_composite_score", ascending=False),
                 hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(snap, x="trailingPE", y="returnOnEquity", color="sector",
                          size="fundamental_composite_score", hover_name="ticker",
                          title="Valuation (PE) vs Profitability (ROE)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(snap.sort_values("fundamental_composite_score", ascending=True),
                      x="fundamental_composite_score", y="ticker", color="sector",
                      orientation="h", title="Fundamental composite score")
        st.plotly_chart(fig2, use_container_width=True)


# ======================================================= MACRO & RISK ====
elif page == "Macro & Risk":
    st.title("Macro & Risk")

    st.subheader("Market return vs macro backdrop")
    macro_metric = st.selectbox("Macro series", [c for c in
        ["CPIAUCSL", "FEDFUNDS", "GS10", "UNRATE", "VIXCLS"] if c in macro.columns])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macro["date"], y=macro["market_cum_return"], name="Market cum. return"))
    fig.add_trace(go.Scatter(x=macro["date"], y=macro[macro_metric], name=macro_metric, yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="Market cum. return"),
        yaxis2=dict(title=macro_metric, overlaying="y", side="right"),
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk metrics by ticker")
    st.dataframe(risk.sort_values("sharpe_ratio", ascending=False), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_r = px.bar(risk.sort_values("sharpe_ratio"), x="sharpe_ratio", y="ticker",
                        orientation="h", color="sector", title="Sharpe ratio by ticker")
        st.plotly_chart(fig_r, use_container_width=True)
    with c2:
        fig_v = px.scatter(risk, x="annualized_volatility", y="annualized_return",
                            color="sector", size="market_beta", hover_name="ticker",
                            title="Risk / return map (bubble = beta)")
        st.plotly_chart(fig_v, use_container_width=True)


# ======================================================= ML INTELLIGENCE =
elif page == "ML Intelligence":
    st.title("ML Intelligence")
    st.caption("Baseline vs ML model comparison, computed on a held-out, time-ordered test period.")

    reg_path = GOLD.parent.parent / "outputs" / "reports" / "regression_comparison.csv"
    clf_path = GOLD.parent.parent / "outputs" / "reports" / "classification_comparison.csv"
    fin_path = GOLD.parent.parent / "outputs" / "reports" / "financial_metrics_comparison.csv"

    c1, c2 = st.columns(2)
    if reg_path.exists():
        reg_df = pd.read_csv(reg_path, index_col=0)
        with c1:
            st.write("**Regression — forward return**")
            st.dataframe(reg_df, use_container_width=True)
            st.plotly_chart(px.bar(reg_df.reset_index(), x="index", y="R2",
                                    title="R² by model"), use_container_width=True)
    if clf_path.exists():
        clf_df = pd.read_csv(clf_path, index_col=0)
        with c2:
            st.write("**Classification — forward direction**")
            st.dataframe(clf_df, use_container_width=True)
            st.plotly_chart(px.bar(clf_df.reset_index(), x="index", y="ROC_AUC",
                                    title="ROC-AUC by model"), use_container_width=True)

    if fin_path.exists():
        st.write("**Financial outcome of each classifier's predicted-positive portfolio**")
        st.dataframe(pd.read_csv(fin_path, index_col=0), use_container_width=True)

    st.subheader("Latest stock rankings (final ML signal)")
    if not rankings.empty:
        last_date = rankings["date"].max()
        snap = rankings[rankings["date"] == last_date].sort_values("rank")
        st.dataframe(snap, hide_index=True, use_container_width=True)
        st.plotly_chart(
            px.bar(snap.sort_values("final_signal"), x="final_signal", y="ticker",
                   orientation="h", color="sector", title="Final ML signal, latest test date"),
            use_container_width=True,
        )


# ==================================================== PORTFOLIO SIMULATOR
elif page == "Portfolio Simulator":
    st.title("Portfolio Simulator")
    st.caption("Top-N ranked stocks, equal-weight, rebalanced periodically — vs an "
               "equal-weight full-universe benchmark. Backtested strictly on the test period.")

    if not backtest_summary.empty:
        st.dataframe(backtest_summary.set_index("portfolio").round(4), use_container_width=True)

    if not backtest_curve.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=backtest_curve["date"], y=backtest_curve["strategy_cum_return"],
                                  name="Top-N ML Strategy"))
        fig.add_trace(go.Scatter(x=backtest_curve["date"], y=backtest_curve["benchmark_cum_return"],
                                  name="Equal-Weight Benchmark"))
        fig.update_layout(title="Cumulative return: strategy vs benchmark", height=450)
        st.plotly_chart(fig, use_container_width=True)

    holdings_path = GOLD / "vw_backtest_holdings_log.csv"
    if holdings_path.exists():
        st.subheader("Rebalance holdings log")
        st.dataframe(pd.read_csv(holdings_path), hide_index=True, use_container_width=True)
