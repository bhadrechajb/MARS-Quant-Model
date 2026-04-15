import pandas as pd
import numpy as np
from inference_engine import InferenceEngine

class BacktestEngine:
    def __init__(self, start_date="2024-01-01", initial_capital=100000.0):
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.inference = InferenceEngine()
        self.df = None
        self.results = {}

    def prepare_data(self):
        print("--- Backtest: Preparing Data & Regimes ---")
        self.df = self.inference.run_inference(start_date=self.start_date)
        
        self.df['Nifty50_Ret'] = np.exp(self.df['Nifty50_LogRet']) - 1
        self.df['Smallcap_Ret'] = np.exp(self.df['Smallcap_LogRet']) - 1
        self.df['Midcap_Ret'] = np.exp(self.df['Midcap_LogRet']) - 1
        
        # Step 1: Yield-Enhanced Cash Proxy
        # Replacing 6% fixed with Liquid Fund / Overnight Fund proxy ~6.5% - 7.0%
        # Let's use 6.75% as a midpoint for Liquid Fund
        daily_liquid_rate = (1 + 0.0675)**(1/252) - 1
        self.df['Cash_Ret'] = daily_liquid_rate
        
        # Map Regimes to Ideal Assets (0/1 Allocations)
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
        
        # Lag weights by 1 day (Regime T -> Trade T+1)
        self.ideal_weights = self.ideal_weights.shift(1).fillna(0)
        
        # Clean NaNs
        self.df = self.df.iloc[1:]
        self.ideal_weights = self.ideal_weights.iloc[1:]

    def run_strategy(self, strategy_name="Baseline"):
        print(f"--- Running Strategy: {strategy_name} ---")
        
        weights = pd.DataFrame(0.0, index=self.ideal_weights.index, columns=['Cash', 'Nifty', 'Smallcap'])
        
        # --- Strategy Implementations ---
        
        if strategy_name == "Baseline":
            final_weights = self.ideal_weights.copy()
            
        elif strategy_name == "Inertia_Latch":
            # Previous experiment logic (80/20 Latch)
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

        elif strategy_name == "Probabilistic":
            # Step 2: Probabilistic Weighting (Soft Switching)
            # Allocation = P(Bull)*Smallcap + P(Sideways)*Nifty + P(Bear)*Cash
            
            # Need probabilities, shifted by 1 day (prediction T -> trade T+1)
            # Prob columns: Prob_State_0 (Bear), Prob_State_1 (Sideways), Prob_State_2 (Bull), Prob_State_3 (Sideways)
            
            probs = self.df[[c for c in self.df.columns if 'Prob_' in c]].shift(1).fillna(0)
            
            # Vectorized calculation
            w_cash = probs['Prob_State_0']
            w_smallcap = probs['Prob_State_2']
            w_nifty = probs['Prob_State_1'] + probs['Prob_State_3']
            
            weights['Cash'] = w_cash
            weights['Smallcap'] = w_smallcap
            weights['Nifty'] = w_nifty
            
            # Normalize
            final_weights = weights.div(weights.sum(axis=1), axis=0)

        elif strategy_name == "EFF_Filter":
            # Step 3: Economic Friction Filter
            # Switch only if E[R_new] - E[R_curr] > 2 * Slippage (0.1% * 2 = 0.2%)
            # We need Expected Returns for states. 
            # Using mapped values from GEMINI.md: Bear: -0.48, Bull: +0.06, Sideways: ~0.0 (Neutral)
            # Let's approximate daily expected returns (log-space / 252 or similar scaling? No, these are stationary log-returns means)
            # Wait, -0.48% is huge for daily. It must be annualized? 
            # Or maybe they are standardized Z-scores?
            # Re-checking GEMINI.md: "State 0 (Bear): Expected Mean ~ -0.48"
            # If that's daily %, it's -0.48%. If it's log-return, it's ~ -0.48%.
            # Let's assume these are daily % terms based on typical HMM outputs for returns.
            
            # Let's define E[R] vector for states [0, 1, 2, 3]
            # State 0 (Bear): -0.48% -> -0.0048
            # State 2 (Bull): +0.06% -> +0.0006
            # State 1,3 (Sideways): 0.0 -> 0.0
            
            # Threshold = 0.2% = 0.002
            
            state_means = {0: -0.0048, 1: 0.0, 2: 0.0006, 3: 0.0}
            threshold = 0.002
            
            weight_list = []
            current_w = np.array([1.0, 0.0, 0.0]) # Start Cash
            current_state_er = state_means[0] # Assume start in Bear
            
            # We iterate because decision depends on current state
            # Using inferred Regime_ID from ideal_weights logic
            
            # Recover Regime ID sequence from inference (shifted)
            regime_seq = self.df['Regime_ID'].shift(1).fillna(0).astype(int)
            
            # But wait, self.df is already cleaned and shifted in prepare_data? 
            # No, self.df is aligned. self.ideal_weights is shifted.
            # Let's use self.ideal_weights to infer "New Ideal"
            
            curr_w = np.array([1.0, 0.0, 0.0]) # Actual holding
            curr_regime_idx = 0 # Assume start State 0
            
            for date, row in self.ideal_weights.iterrows():
                # What is the "Signal" regime?
                # We can infer from the row:
                # [1,0,0] -> Bear(0), [0,0,1] -> Bull(2), [0,1,0] -> Sideways(1/3)
                
                if row['Cash'] == 1.0:
                    signal_regime = 0
                elif row['Smallcap'] == 1.0:
                    signal_regime = 2
                else:
                    signal_regime = 1 # or 3, treating same return profile
                
                signal_er = state_means[signal_regime]
                current_er = state_means[curr_regime_idx]
                
                # Check EFF Constraint
                # "Switch if E[R_New] - E[R_Curr] > Threshold"
                # Note: This implies we only switch to "Better" states?
                # What if we need to switch to "Safe" state (Cash) to avoid loss?
                # E[R_Bear] is -0.48. E[R_Bull] is +0.06.
                # If moving Bull -> Bear: New(-0.48) - Curr(+0.06) = -0.54. 
                # This is < Threshold. So we wouldn't switch? 
                # That would be disastrous (staying in Bull during Bear).
                
                # Interpretation of "Alpha Gain":
                # It likely means: Is the new state *statistically distinct enough*?
                # Or specifically for "Opportunity Entry"?
                # "Switch if E[R_New] - E[R_Curr] > ..." usually applies to "Entering Alpha".
                # For "Exiting Risk", the cost is secondary to safety.
                
                # Let's refine the logic:
                # If New State is "Defensive" (Cash), ALWAYS Switch (Safety First).
                # If New State is "Aggressive" (Smallcap/Nifty), Only Switch if Gain > Cost.
                
                if signal_regime == 0: # Bear Signal
                    # Always respect Risk signal
                    curr_w = row.values
                    curr_regime_idx = signal_regime
                else:
                    # Bull or Sideways Signal
                    # Check if improvement over current
                    if signal_er - current_er > threshold:
                        curr_w = row.values
                        curr_regime_idx = signal_regime
                    else:
                        # Suppress switch, hold previous weights
                        # BUT update current_regime_idx? No, we are technically still in old regime mentally?
                        # Or do we acknowledge the regime changed but we chose not to trade?
                        pass 
                
                weight_list.append(curr_w)
                
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
        import matplotlib.pyplot as plt
        nifty_ret = self.df['Nifty50_Ret']
        nifty_equity = self.initial_capital * (1 + nifty_ret).cumprod()
        self.results['Benchmark'] = {'Equity': nifty_equity}
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON (With Enhancements)")
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
                'Final Capital': eq.iloc[-1]
            })
        print(pd.DataFrame(metrics).set_index('Strategy').round(2))
        print("="*60)

if __name__ == "__main__":
    bt = BacktestEngine(start_date="2024-01-01")
    bt.prepare_data()
    bt.run_strategy("Baseline")
    bt.run_strategy("Inertia_Latch")
    bt.run_strategy("Probabilistic")
    bt.run_strategy("EFF_Filter")
    bt.compare()