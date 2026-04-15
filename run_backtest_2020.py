from backtest_engine import BacktestEngine

if __name__ == "__main__":
    print("Starting MARS Backtest (2020-2026)...")
    bt = BacktestEngine(start_date="2020-01-01")
    bt.prepare_data()
    bt.run_strategy("Baseline")
    bt.run_strategy("Inertia_Latch")
    bt.run_strategy("Probabilistic")
    bt.run_strategy("EFF_Filter")
    bt.compare()
