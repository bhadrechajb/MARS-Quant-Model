import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inference_engine import InferenceEngine

class TrendExperiment:
    def __init__(self, start_date="2024-01-01", fetch_start="2023-01-01", initial_capital=100000.0):
        self.start_date = start_date
        self.fetch_start = fetch_start
        self.initial_capital = initial_capital
        self.inference = InferenceEngine()
        self.df = None
        self.results = {}

    def prepare_data(self):
        print(f"--- Trend Experiment: Fetching Data from {self.fetch_start} for EMA context ---")
        # Fetch data starting earlier to warm up EMAs
        self.df = self.inference.run_inference(start_date=self.fetch_start)
        
        # Check if we have the new raw columns
        if 'Nifty50' not in self.df.columns:
            raise ValueError("Raw 'Nifty50' price not found in dataframe. Check data_engine.py")

        # 1. Compute Indicators (EMA 200)
        print("Computing Technical Indicators...")
        self.df['Nifty50_EMA200'] = self.df['Nifty50'].rolling(window=200).mean()
        self.df['Nifty_Trend_Bullish'] = self.df['Nifty50'] > self.df['Nifty50_EMA200']
        
        # 2. Compute Returns
        self.df['Nifty50_Ret'] = np.exp(self.df['Nifty50_LogRet']) - 1
        self.df['Smallcap_Ret'] = np.exp(self.df['Smallcap_LogRet']) - 1
        self.df['Midcap_Ret'] = np.exp(self.df['Midcap_LogRet']) - 1
        
        # Cash Proxy (Liquid Fund ~6.75%)
        daily_liquid_rate = (1 + 0.0675)**(1/252) - 1
        self.df['Cash_Ret'] = daily_liquid_rate

        # 3. Smallcap Check Data
        if "Smallcap_Index_Check" in self.df.columns:
            self.df['Smallcap_Index_Ret'] = np.exp(self.df['Smallcap_Index_Check_LogRet']) - 1
        
        # Slice to Backtest Period
        print(f"Slicing data to start from {self.start_date}...")
        self.df = self.df.loc[self.start_date:]
        
        if self.df.empty:
            raise ValueError("Dataframe empty after slicing!")

        # Map Regimes to Ideal Allocations
        def get_ideal_allocation(row):
            regime = row['Regime_ID']
            if regime == 0: # Bear
                return {'Cash': 1.0, 'Nifty': 0.0, 'Smallcap': 0.0}
            elif regime == 2: # Bull
                return {'Cash': 0.0, 'Nifty': 0.0, 'Smallcap': 1.0}
            else: # Sideways (1, 3)
                return {'Cash': 0.0, 'Nifty': 1.0, 'Smallcap': 0.0}
                
        allocs = self.df.apply(get_ideal_allocation, axis=1)
        self.ideal_weights = pd.DataFrame(allocs.tolist(), index=self.df.index)
        
        # Shift weights by 1 day (Trade Implementation Lag)
        self.ideal_weights = self.ideal_weights.shift(1).fillna(0)
        
        # Drop first row due to shift
        self.df = self.df.iloc[1:]
        self.ideal_weights = self.ideal_weights.iloc[1:]

    def analyze_smallcap_drag(self):
        print("\n" + "="*60)
        print("SMALLCAP LINDY CHECK: ETF (HDFCSML250) vs INDEX (^NSESMLCP250)")
        print("="*60)
        
        if 'Smallcap_Index_Ret' not in self.df.columns:
            print("Index data not available for comparison.")
            return

        etf_cum = (1 + self.df['Smallcap_Ret']).cumprod()
        idx_cum = (1 + self.df['Smallcap_Index_Ret']).cumprod()
        
        etf_total = (etf_cum.iloc[-1] - 1) * 100
        idx_total = (idx_cum.iloc[-1] - 1) * 100
        
        print(f"Period: {self.df.index[0].date()} to {self.df.index[-1].date()}")
        print(f"ETF Total Return:   {etf_total:.2f}%")
        print(f"Index Total Return: {idx_total:.2f}%")
        print(f"Tracking Diff:      {etf_total - idx_total:.2f}%")
        
        # Tracking Error (Std Dev of Diff)
        diff = self.df['Smallcap_Ret'] - self.df['Smallcap_Index_Ret']
        te = diff.std() * np.sqrt(252) * 100
        print(f"Tracking Error (Ann): {te:.2f}%")
        print("="*60)

    def run_strategy(self, strategy_name="Baseline"):
        print(f"--- Running Strategy: {strategy_name} ---")
        
        weights = pd.DataFrame(0.0, index=self.ideal_weights.index, columns=['Cash', 'Nifty', 'Smallcap'])
        
        if strategy_name == "Baseline":
            final_weights = self.ideal_weights.copy()
            
        elif strategy_name == "Inertia_Latch":
            # 80/20 Retention Rule
            weight_list = []
            current_target = np.array([1.0, 0.0, 0.0])
            for idx, row in self.ideal_weights.iterrows():
                ideal_w = row.values
                if idx == self.ideal_weights.index[0]:
                    current_target = ideal_w
                else:
                    prev_ideal = self.ideal_weights.loc[:idx].iloc[-2].values
                    if not np.array_equal(ideal_w, prev_ideal):
                        current_target = 0.8 * ideal_w + 0.2 * current_target
                        current_target /= current_target.sum()
                weight_list.append(current_target)
            final_weights = pd.DataFrame(weight_list, index=self.ideal_weights.index, columns=['Cash', 'Nifty', 'Smallcap'])
            
        elif strategy_name == "Trend_Override":
            # Baseline + Rule: If HMM=Bear (Cash) but Nifty > 200EMA, Stay Nifty
            # Note: We apply this logic to the *Signal* before Latching (if we combine them)
            # Or just apply to Baseline for purity first.
            # User request: "I'm keeping the Inertia Latch as the core logic. I want a 'Trend Override' script."
            # So: Trend Override + Inertia Latch? Or Trend Override standalone?
            # "If the HMM says 'Sell' but the 200-EMA says 'Bull Trend,' we do nothing."
            # "Do nothing" implies staying in previous asset? Or forcing Nifty?
            # "Override the cash exit and stay in Nifty 50." -> Force Nifty.
            
            # Let's Implement: Inertia Latch + Trend Override
            
            weight_list = []
            current_target = np.array([1.0, 0.0, 0.0])
            
            # We need to access the EMA state from self.df (aligned by index) 
            
            for idx, row in self.ideal_weights.iterrows():
                ideal_w = row.values # This is HMM signal
                
                # Check Trend Override
                # We need yesterday's EMA check to trade today? 
                # self.ideal_weights is already shifted. 
                # So row corresponds to Trade Date. Signal was generated T-1.
                # We should check EMA at T-1.
                # self.df index matches self.ideal_weights index (Trade Date).
                # So we look at self.df.loc[idx] which has columns from T (Price T).
                # Wait. prepare_data shifted weights.
                # self.df is trade date. 
                # We need EMA from T-1.
                # Actually, data_engine lags features, but Prices are contemporaneous.
                # If we trade at Close T, we know Price T. 
                # If we trade Open T, we know Price T-1.
                # Let's assume we check EMA from T-1 (Signal Generation Time).
                
                # Use shift(1) for EMA check in vector op, or iloc in loop. 
                # Simpler: The Signal (ideal_w) comes from Regime T-1.
                # We check Trend T-1.
                # The dataframe has 'Nifty_Trend_Bullish'. We should shift it to align with weights?
                # Weights are shifted. So `ideal_weights` at `idx` comes from `df` at `idx-1`.
                # So we check `df['Nifty_Trend_Bullish']` at `idx-1`.
                
                # Let's pre-calculate the override signal aligned with ideal_weights
                # It's getting complex in the loop. 
                
                pass # Logic implemented below loop
            
            # Vectorized approach for Override Signal
            # 1. Align Trend Signal: Shifted by 1 to match Trade Date
            trend_bullish = self.df['Nifty_Trend_Bullish'].shift(1).fillna(False)
            
            # 2. Modify Ideal Weights (The Signal)
            modified_ideal = self.ideal_weights.copy()
            
            # Mask: HMM=Cash (Bear) AND Trend=Bullish
            # HMM Bear is where Cash=1.0
            bear_signal = (self.ideal_weights['Cash'] == 1.0)
            override_mask = bear_signal & trend_bullish
            
            print(f"DEBUG: Trend Override triggered on {override_mask.sum()} days.")
            if override_mask.sum() > 0:
                print("DEBUG: Dates:", self.ideal_weights.index[override_mask].date)

            # Apply Override: Force Nifty (State 1 equivalent)
            # Why Nifty? User said "Stay in Nifty 50".
            # Bear signal usually comes after Nifty/Smallcap. 
            # If we are in Smallcap, and Bear hits, and Trend is Bull -> Switch to Nifty? or Stay Smallcap?
            # User: "Override the cash exit and stay in Nifty 50."
            # This implies the alternative to Cash is Nifty.
            
            modified_ideal.loc[override_mask, 'Cash'] = 0.0
            modified_ideal.loc[override_mask, 'Nifty'] = 1.0
            modified_ideal.loc[override_mask, 'Smallcap'] = 0.0
            
            # 3. Apply Inertia Latch to Modified Signal
            weight_list = []
            current_target = np.array([1.0, 0.0, 0.0]) # Start Cash
            
            for idx, row in modified_ideal.iterrows():
                ideal_w = row.values
                if idx == modified_ideal.index[0]:
                    current_target = ideal_w
                else:
                    prev_ideal = modified_ideal.loc[:idx].iloc[-2].values
                    if not np.array_equal(ideal_w, prev_ideal):
                        current_target = 0.8 * ideal_w + 0.2 * current_target
                        current_target /= current_target.sum()
                weight_list.append(current_target)
            
            final_weights = pd.DataFrame(weight_list, index=self.ideal_weights.index, columns=['Cash', 'Nifty', 'Smallcap'])

        # Calculate Returns
        asset_rets = self.df[['Cash_Ret', 'Nifty50_Ret', 'Smallcap_Ret']]
        asset_rets.columns = ['Cash', 'Nifty', 'Smallcap']
        
        strat_ret = (final_weights * asset_rets).sum(axis=1)
        cum_ret = (1 + strat_ret).cumprod()
        equity_curve = self.initial_capital * cum_ret
        
        self.results[strategy_name] = {
            'Daily_Ret': strat_ret,
            'Equity': equity_curve,
            'Weights': final_weights
        }
        return equity_curve

    def compare(self):
        nifty_ret = self.df['Nifty50_Ret']
        nifty_equity = self.initial_capital * (1 + nifty_ret).cumprod()
        self.results['Benchmark (Nifty)'] = {'Equity': nifty_equity}
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON (Jan 2024 - Jan 2026)")
        print("="*60)
        metrics = []
        for name, data in self.results.items():
            eq = data['Equity']
            total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
            daily_std = eq.pct_change().std()
            sharpe = (eq.pct_change().mean() / daily_std) * np.sqrt(252) if daily_std > 0 else 0
            roll_max = eq.cummax()
            dd = (eq - roll_max) / roll_max
            max_dd = dd.min()
            metrics.append({
                'Strategy': name,
                'Total Return %': total_ret * 100,
                'Sharpe Ratio': sharpe,
                'Max Drawdown %': max_dd * 100,
                'Trades': 'N/A' # To implement trade counter if needed
            })
        print(pd.DataFrame(metrics).set_index('Strategy').round(2))
        print("="*60)

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("RUN 1: RECENT BULL MARKET (Jan 2024 - Jan 2026)")
    print("#"*60)
    exp = TrendExperiment(start_date="2024-01-01", fetch_start="2023-01-01")
    try:
        exp.prepare_data()
        exp.analyze_smallcap_drag()
        exp.run_strategy("Baseline")
        exp.run_strategy("Inertia_Latch")
        exp.run_strategy("Trend_Override")
        exp.compare()
    except Exception as e:
        print(f"Error running Run 1: {e}")

    print("\n" + "#"*60)
    print("RUN 2: FULL HISTORY STRESS TEST (Jan 2016 - Jan 2026)")
    print("#"*60)
    # Fetch from 2015 for EMA warmup
    exp_full = TrendExperiment(start_date="2016-01-01", fetch_start="2015-01-01")
    try:
        exp_full.prepare_data()
        exp_full.run_strategy("Baseline")
        exp_full.run_strategy("Inertia_Latch")
        exp_full.run_strategy("Trend_Override")
        exp_full.compare()
    except Exception as e:
        print(f"Error running Run 2: {e}")
