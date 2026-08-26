"""
Modeling
=========
Time-ordered TRAIN / VALIDATION / TEST split (never random — that would
leak the future into training). Two parallel tasks:

  Regression      predict forward_return  (Linear/Ridge, RandomForest, XGBoost)
  Classification   predict forward_direction (Logistic, RandomForest, XGBoost)

XGBoost is used if installed; otherwise falls back to sklearn's
HistGradientBoosting with a clear note, so the pipeline never hard-fails on
a missing optional dependency.

Evaluation has two layers:
  ML metrics        RMSE/MAE/R2 (regression), Accuracy/F1/ROC-AUC (classification)
  Financial metrics  for each classification model, forming a simple
                     long-only equal-weight portfolio of "predicted positive"
                     tickers on each test date and computing Hit Rate,
                     Sharpe, Sortino, and Max Drawdown on *that* portfolio
                     — i.e. translating ML accuracy into an actual trading
                     outcome, not just a classification score.
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from src.config import TRAIN_FRAC, VAL_FRAC, RANDOM_SEED, RISK_FREE_ANNUAL, OUT_REPORTS

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[modeling] xgboost not installed -> using HistGradientBoosting as the "
          "'XGBoost' slot instead. `pip install xgboost` to use the real thing.")

FEATURE_COLS = [
    "daily_return", "momentum_10", "momentum_20", "ma_5", "ma_20", "ma_60",
    "volatility_5", "volatility_20", "volatility_60", "rsi", "cum_return",
    "drawdown", "volume_zscore",
    "cpi_yoy", "fedfunds_change_20d", "gs10_change_20d", "unrate_change_20d", "vix_change_5d",
    "valuation_score", "growth_score", "profitability_score", "leverage_score",
    "fundamental_composite_score",
]


def time_split(df: pd.DataFrame):
    df = df.dropna(subset=["forward_return", "forward_direction"]).copy()
    for c in FEATURE_COLS:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    df = df.sort_values("date")

    dates = df["date"].sort_values().unique()
    n = len(dates)
    train_cut = dates[int(n * TRAIN_FRAC)]
    val_cut = dates[int(n * (TRAIN_FRAC + VAL_FRAC))]

    train = df[df["date"] < train_cut]
    val = df[(df["date"] >= train_cut) & (df["date"] < val_cut)]
    test = df[df["date"] >= val_cut]
    print(f"[modeling] time split -> train {len(train):,} rows (< {pd.Timestamp(train_cut).date()}) "
          f"/ val {len(val):,} rows / test {len(test):,} rows (>= {pd.Timestamp(val_cut).date()})")
    return train, val, test


def _cols_present(df):
    return [c for c in FEATURE_COLS if c in df.columns]


def _reg_metrics(y_true, y_pred):
    return {"RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def _clf_metrics(y_true, y_pred, y_proba):
    return {"Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan}


def run_regression(train, val, test):
    cols = _cols_present(train)
    X_train, y_train = train[cols], train["forward_return"]
    X_test, y_test = test[cols], test["forward_return"]

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    results, models, preds = {}, {}, {}

    lin = LinearRegression().fit(Xtr_s, y_train)
    results["Linear Regression"] = _reg_metrics(y_test, lin.predict(Xte_s))

    ridge = Ridge(alpha=5.0).fit(Xtr_s, y_train)
    results["Ridge Regression"] = _reg_metrics(y_test, ridge.predict(Xte_s))
    preds["Ridge Regression"] = ridge.predict(Xte_s)

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=15,
                                random_state=RANDOM_SEED, n_jobs=-1).fit(X_train, y_train)
    results["Random Forest"] = _reg_metrics(y_test, rf.predict(X_test))
    preds["Random Forest"] = rf.predict(X_test)

    if HAS_XGBOOST:
        xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                            random_state=RANDOM_SEED, n_jobs=-1).fit(X_train, y_train)
    else:
        xgb = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300,
                                             random_state=RANDOM_SEED).fit(X_train, y_train)
    results["XGBoost"] = _reg_metrics(y_test, xgb.predict(X_test))
    preds["XGBoost"] = xgb.predict(X_test)

    models = {"linear": lin, "ridge": ridge, "random_forest": rf, "xgboost": xgb, "scaler": scaler}
    fi = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    best_name = pd.DataFrame(results).T["R2"].idxmax()
    best_pred_col = "Random Forest" if best_name == "Random Forest" else (
        "XGBoost" if best_name == "XGBoost" else "Ridge Regression"
    )
    test_predictions = test[["date", "ticker"]].copy()
    test_predictions["predicted_return"] = preds.get(best_pred_col, rf.predict(X_test))
    test_predictions["model_used"] = best_pred_col

    return pd.DataFrame(results).T, models, fi, test_predictions


def run_classification(train, val, test):
    cols = _cols_present(train)
    X_train, y_train = train[cols], train["forward_direction"]
    X_test, y_test = test[cols], test["forward_direction"]

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

    results, preds_proba = {}, {}

    logit = LogisticRegression(max_iter=1000).fit(Xtr_s, y_train)
    results["Logistic Regression"] = _clf_metrics(
        y_test, logit.predict(Xte_s), logit.predict_proba(Xte_s)[:, 1])
    preds_proba["Logistic Regression"] = logit.predict_proba(Xte_s)[:, 1]

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=15,
                                 random_state=RANDOM_SEED, n_jobs=-1).fit(X_train, y_train)
    results["Random Forest"] = _clf_metrics(
        y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])
    preds_proba["Random Forest"] = rf.predict_proba(X_test)[:, 1]

    if HAS_XGBOOST:
        xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                             random_state=RANDOM_SEED, n_jobs=-1, eval_metric="logloss").fit(X_train, y_train)
    else:
        xgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300,
                                              random_state=RANDOM_SEED).fit(X_train, y_train)
    results["XGBoost"] = _clf_metrics(
        y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1])
    preds_proba["XGBoost"] = xgb.predict_proba(X_test)[:, 1]

    models = {"logistic": logit, "random_forest": rf, "xgboost": xgb, "scaler": scaler}
    fi = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)

    best_name = pd.DataFrame(results).T["ROC_AUC"].idxmax()
    test_predictions = test[["date", "ticker", "forward_return"]].copy()
    test_predictions["predicted_probability"] = preds_proba[best_name]
    test_predictions["model_used"] = best_name

    return pd.DataFrame(results).T, models, fi, test_predictions, preds_proba, y_test


def financial_metrics_from_predictions(test: pd.DataFrame, y_proba: np.ndarray,
                                        threshold: float = 0.5) -> dict:
    """Simulate a naive daily long-only equal-weight portfolio: on each date,
    go long every ticker the model flags as predicted-positive, hold for one
    day (proxy using daily_return), and measure the resulting portfolio's
    real financial performance -- not just classification accuracy."""
    sim = test[["date", "ticker", "daily_return"]].copy()
    sim["signal"] = (y_proba >= threshold).astype(int)
    port = (
        sim[sim["signal"] == 1]
        .groupby("date")["daily_return"].mean()
        .rename("portfolio_return")
    )
    if port.empty or port.std() == 0:
        return {"Hit_Rate": np.nan, "Sharpe": np.nan, "Sortino": np.nan, "Max_Drawdown": np.nan}

    rf_daily = RISK_FREE_ANNUAL / 252
    sharpe = (port.mean() - rf_daily) / port.std() * np.sqrt(252)
    downside = port[port < 0]
    sortino = ((port.mean() - rf_daily) / downside.std() * np.sqrt(252)
               if len(downside) and downside.std() > 0 else np.nan)
    hit_rate = (port > 0).mean()
    cum = (1 + port.fillna(0)).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    return {"Hit_Rate": hit_rate, "Sharpe": sharpe, "Sortino": sortino, "Max_Drawdown": max_dd}


def run_modeling(feature_panel: pd.DataFrame):
    print("=" * 60, "\nMODELING: BASELINE vs ML (REGRESSION + CLASSIFICATION)\n", "=" * 60, sep="")
    train, val, test = time_split(feature_panel)

    reg_results, reg_models, reg_fi, reg_test_preds = run_regression(train, val, test)
    clf_results, clf_models, clf_fi, clf_test_preds, clf_probas, y_test = run_classification(train, val, test)

    print("\n--- Regression: forward_return ---")
    print(reg_results.round(4).to_string())
    print("\n--- Classification: forward_direction (ML metrics) ---")
    print(clf_results.round(4).to_string())

    fin_rows = {}
    for name, proba in clf_probas.items():
        fin_rows[name] = financial_metrics_from_predictions(test, proba)
    fin_results = pd.DataFrame(fin_rows).T
    print("\n--- Classification: financial metrics of predicted-positive portfolio ---")
    print(fin_results.round(4).to_string())

    reg_results.round(4).to_csv(OUT_REPORTS / "regression_comparison.csv")
    clf_results.round(4).to_csv(OUT_REPORTS / "classification_comparison.csv")
    fin_results.round(4).to_csv(OUT_REPORTS / "financial_metrics_comparison.csv")
    reg_fi.round(4).to_csv(OUT_REPORTS / "regression_feature_importance.csv")
    clf_fi.round(4).to_csv(OUT_REPORTS / "classification_feature_importance.csv")

    return {
        "train": train, "val": val, "test": test,
        "regression_results": reg_results, "regression_models": reg_models,
        "regression_feature_importance": reg_fi, "regression_test_predictions": reg_test_preds,
        "classification_results": clf_results, "classification_models": clf_models,
        "classification_feature_importance": clf_fi, "classification_test_predictions": clf_test_preds,
        "financial_results": fin_results,
    }


if __name__ == "__main__":
    from src.config import SILVER, FORWARD_HORIZON_DAYS
    from src.feature_engineering import run_feature_engineering
    master = pd.read_csv(SILVER / "master_dataset.csv", parse_dates=["date"])
    feat = run_feature_engineering(master, FORWARD_HORIZON_DAYS)
    run_modeling(feat)
