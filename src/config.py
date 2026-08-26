"""
Central configuration. One place for paths, universe, and constants so
every stage (and anyone pointing this at their own data/raw) only has to
edit one file.
"""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
RAW_AV = RAW / "alphavantage"
RAW_FRED = RAW / "fred"
RAW_YF = RAW / "yfinance"

BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"

OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_REPORTS = ROOT / "outputs" / "reports"

for p in [RAW_AV, RAW_FRED, RAW_YF, BRONZE, SILVER, GOLD, OUT_FIGURES, OUT_REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- universe ----
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "META": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "JPM": "Financials", "XOM": "Energy", "JNJ": "Healthcare",
}
TICKERS = sorted(SECTOR_MAP.keys())

FRED_SERIES = ["CPIAUCSL", "FEDFUNDS", "GS10", "UNRATE", "VIXCLS"]
FRED_MONTHLY = {"CPIAUCSL", "FEDFUNDS", "GS10", "UNRATE"}  # VIXCLS is daily

START_DATE = "2022-01-03"
# Always run up through today's date -- recomputed fresh every time this
# module is imported, so every pipeline run picks up the latest available
# trading day rather than a date frozen at whenever this file was written.
END_DATE = date.today().strftime("%Y-%m-%d")

RANDOM_SEED = 42

# --------------------------------------------------------- feature/model ---
FORWARD_HORIZON_DAYS = 20
ROLLING_WINDOWS = {"short": 5, "medium": 20, "long": 60}

# time-ordered 3-way split (fractions of the date range, in order)
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20   # remainder

TOP_N_PORTFOLIO = 3          # stocks held each rebalance in the backtest
REBALANCE_DAYS = FORWARD_HORIZON_DAYS
RISK_FREE_ANNUAL = 0.02
