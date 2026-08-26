"""
EDA
====
Explores the Silver master dataset before any feature engineering: return
distributions, volatility, correlations, missingness, macro relationships,
fundamental analysis, and stock/sector-level summaries. Saves figures +
a text summary; doesn't mutate the data.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import OUT_FIGURES, OUT_REPORTS

sns.set_theme(style="whitegrid")


def run_eda(master: pd.DataFrame) -> None:
    print("=" * 60, "\nEDA\n", "=" * 60, sep="")

    df = master.sort_values(["ticker", "date"]).copy()
    df["daily_return"] = df.groupby("ticker")["price"].pct_change()

    # 1) missingness
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        print("Missingness:\n", (miss * 100).round(2).to_string())
    else:
        print("No missing values in master dataset.")

    # 2) return distributions + volatility
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["daily_return"].dropna(), bins=80, ax=axes[0], color="steelblue")
    axes[0].set_title("Daily return distribution (all tickers)")

    vol = df.groupby("ticker")["daily_return"].std() * np.sqrt(252)
    vol.sort_values().plot(kind="barh", ax=axes[1], color="darkorange")
    axes[1].set_title("Annualized volatility by ticker")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "01_returns_volatility.png", dpi=130)
    plt.close(fig)

    # 3) correlation: technicals(rsi)/macro relationships
    macro_cols = [c for c in ["CPIAUCSL", "FEDFUNDS", "GS10", "UNRATE", "VIXCLS"] if c in df.columns]
    corr_cols = ["daily_return", "rsi"] + macro_cols
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Return / RSI / Macro correlations")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "02_macro_correlations.png", dpi=130)
    plt.close(fig)

    # 4) sector-level cumulative return
    df["cum_return"] = df.groupby("ticker")["daily_return"].transform(
        lambda s: (1 + s.fillna(0)).cumprod() - 1
    )
    sector_cum = df.groupby(["sector", "date"])["cum_return"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for sector, g in sector_cum.groupby("sector"):
        ax.plot(g["date"], g["cum_return"], label=sector, linewidth=1.3)
    ax.set_title("Cumulative return by sector")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "03_sector_cumulative_return.png", dpi=130)
    plt.close(fig)

    # 5) fundamental snapshot correlations
    fund_cols = [c for c in ["trailingPE", "priceToBook", "returnOnEquity", "debtToEquity",
                              "profitMargins", "revenueGrowth", "dividendYield", "beta"]
                 if c in df.columns]
    if fund_cols:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        snap = df.drop_duplicates("ticker")[fund_cols]
        sns.heatmap(snap.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Fundamental metric correlations (cross-sectional)")
        fig.tight_layout()
        fig.savefig(OUT_FIGURES / "04_fundamental_correlations.png", dpi=130)
        plt.close(fig)

    summary = {
        "rows": len(df), "tickers": df["ticker"].nunique(), "sectors": df["sector"].nunique(),
        "date_range": f"{df.date.min().date()} to {df.date.max().date()}",
        "avg_daily_return": round(df["daily_return"].mean(), 5),
        "avg_annualized_vol": round(vol.mean(), 4),
    }
    with open(OUT_REPORTS / "eda_summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print("EDA summary ->", summary)


if __name__ == "__main__":
    import pandas as pd
    from src.config import SILVER
    master = pd.read_csv(SILVER / "master_dataset.csv", parse_dates=["date"])
    run_eda(master)
