"""
ForwardEdge — end-to-end pipeline
============================================
Raw (alphavantage/fred/yfinance) -> Bronze -> Silver (master dataset) -> EDA
-> Feature Engineering -> Modeling (baseline vs ML, regression + classification)
-> Stock Ranking Engine -> Portfolio Backtest -> Gold (vw_* tables)
-> [Streamlit dashboard reads Gold]

Run:  python main.py
Then: streamlit run dashboard/app.py
"""
from src.raw_data_stub import build_raw_layer
from src.bronze import run_bronze
from src.silver import run_silver
from src.eda import run_eda
from src.feature_engineering import run_feature_engineering
from src.modeling import run_modeling
from src.ranking import run_ranking
from src.backtest import run_backtest
from src.gold import run_gold
from src.config import FORWARD_HORIZON_DAYS


def main():
    # 1) Raw — no-op if data/raw already has real files in it
    build_raw_layer()

    # 2) Bronze — parse each source's real (messy) shape into long tables
    av_rsi, fred, yf_prices, yf_fund = run_bronze()

    # 3) Silver — clean, align, integrate into one Date x Ticker master dataset
    master = run_silver(av_rsi, fred, yf_prices, yf_fund)

    # 4) EDA — understand the master dataset before engineering anything
    run_eda(master)

    # 5) Feature engineering — technical / macro / fundamental feature blocks + target
    feature_panel = run_feature_engineering(master, FORWARD_HORIZON_DAYS)

    # 6) Modeling — time-ordered train/val/test; baseline vs ML; ML + financial eval
    model_results = run_modeling(feature_panel)

    # 7) Stock Ranking Engine — combine model outputs + scores into one signal
    rankings = run_ranking(model_results)

    # 8) Portfolio Backtest — top-N ranked stocks vs equal-weight benchmark
    backtest_results = run_backtest(rankings, model_results["test"])

    # 9) Gold — write every vw_* analytics table the dashboard reads from
    views = run_gold(feature_panel, model_results, rankings, backtest_results)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("Bronze   -> data/bronze/*.csv")
    print("Silver   -> data/silver/master_dataset.csv")
    print("Gold     -> data/gold/*.csv (dashboard reads from here)")
    print("Figures  -> outputs/figures/*.png")
    print("Reports  -> outputs/reports/*.csv")
    print("\nNext: streamlit run dashboard/app.py")
    return views, model_results, rankings, backtest_results


if __name__ == "__main__":
    main()
