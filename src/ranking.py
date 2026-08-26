"""
Stock Ranking Engine
======================
Combines everything the pipeline has produced about a ticker on a given
test date into one comparable Final ML Signal:

    predicted_return        (from the best regression model)
    predicted_probability   (from the best classification model)
    technical_score         (momentum + RSI, cross-sectional percentile)
    fundamental_score       (valuation/growth/profitability/leverage composite)
    risk_score              (inverse volatility, cross-sectional percentile)

Each is converted to a 0-1 cross-sectional percentile *per date* first (so a
regression output in return-units and a classifier probability are on the
same footing), then combined with fixed weights into final_signal. Stocks
are ranked within each date by that signal — this ranking is what
backtest.py acts on.
"""
import pandas as pd

WEIGHTS = {
    "predicted_return": 0.35,
    "predicted_probability": 0.25,
    "fundamental_score": 0.20,
    "technical_score": 0.10,
    "risk_score": 0.10,
}


def _pct_rank_by_date(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True, ascending=ascending)


def build_stock_rankings(test_panel: pd.DataFrame, reg_preds: pd.DataFrame,
                          clf_preds: pd.DataFrame) -> pd.DataFrame:
    base = test_panel[["date", "ticker", "sector", "rsi", "momentum_20",
                        "volatility_20", "fundamental_composite_score", "forward_return"]].copy()

    df = base.merge(reg_preds[["date", "ticker", "predicted_return"]], on=["date", "ticker"], how="left")
    df = df.merge(clf_preds[["date", "ticker", "predicted_probability"]], on=["date", "ticker"], how="left")

    df["technical_raw"] = df["momentum_20"].fillna(0) + (df["rsi"].fillna(50) - 50) / 100
    df["technical_score"] = _pct_rank_by_date(df, "technical_raw", ascending=True)
    df["risk_score"] = _pct_rank_by_date(df, "volatility_20", ascending=False)  # lower vol -> higher score
    df["fundamental_score"] = df["fundamental_composite_score"].fillna(df["fundamental_composite_score"].median())
    df["predicted_return_score"] = _pct_rank_by_date(df, "predicted_return", ascending=True)
    df["predicted_probability"] = df["predicted_probability"].fillna(0.5)

    df["final_signal"] = (
        WEIGHTS["predicted_return"] * df["predicted_return_score"]
        + WEIGHTS["predicted_probability"] * df["predicted_probability"]
        + WEIGHTS["fundamental_score"] * df["fundamental_score"]
        + WEIGHTS["technical_score"] * df["technical_score"]
        + WEIGHTS["risk_score"] * df["risk_score"]
    )
    df["rank"] = df.groupby("date")["final_signal"].rank(ascending=False, method="min")

    cols = ["date", "ticker", "sector", "predicted_return", "predicted_probability",
            "technical_score", "fundamental_score", "risk_score", "final_signal",
            "rank", "forward_return"]
    ranked = df[cols].sort_values(["date", "rank"]).reset_index(drop=True)
    print(f"[ranking] built stock rankings: {ranked.shape[0]:,} rows across "
          f"{ranked['date'].nunique()} test dates")
    return ranked


def run_ranking(model_results: dict) -> pd.DataFrame:
    print("=" * 60, "\nSTOCK RANKING ENGINE\n", "=" * 60, sep="")
    test_panel = model_results["test"]
    reg_preds = model_results["regression_test_predictions"]
    clf_preds = model_results["classification_test_predictions"]
    return build_stock_rankings(test_panel, reg_preds, clf_preds)
