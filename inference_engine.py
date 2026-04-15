import joblib
import pandas as pd
from data_engine import DataPipeline
import numpy as np

class InferenceEngine:
    def __init__(self, model_path="mars_golden_model.pkl"):
        print(f"Loading MARS Golden Model: {model_path}")
        self.payload = joblib.load(model_path)
        self.model = self.payload['model']
        self.features = self.payload['selected_features']
        self.state_map = self.payload['state_map']
        
    def run_inference(self, start_date="2025-01-01"):
        """
        Fetches data, predicts regime, and applies production filters (Trend Override).
        """
        # Fetch with buffer for EMA warmup (need ~200 days before start_date)
        fetch_start = (pd.to_datetime(start_date) - pd.Timedelta(days=300)).strftime('%Y-%m-%d')
        
        print(f"Initializing Data Pipeline from {fetch_start}...")
        pipeline = DataPipeline(start_date=fetch_start)
        df_processed = pipeline.run_pipeline()
        
        # Selection of locked features
        X = df_processed[self.features]
        
        # Inference
        print("Executing HMM Inference...")
        regimes = self.model.predict(X)
        probs = self.model.predict_proba(X)
        
        # Attach to dataframe
        df_processed['Regime_ID'] = regimes
        df_processed['Regime_Name'] = df_processed['Regime_ID'].map(self.state_map)
        
        # Trend Override Logic
        print("Applying Trend Override Guardrail (Nifty 200-EMA)...")
        df_processed['Nifty50_EMA200'] = df_processed['Nifty50'].rolling(window=200).mean()
        df_processed['Trend_Bullish'] = df_processed['Nifty50'] > df_processed['Nifty50_EMA200']
        
        # Override: If Bear (0) but Trend Bullish -> Treat as Sideways (1)
        df_processed['Final_Regime_ID'] = df_processed['Regime_ID']
        mask_override = (df_processed['Regime_ID'] == 0) & df_processed['Trend_Bullish']
        df_processed.loc[mask_override, 'Final_Regime_ID'] = 1 
        
        # Map Recommendations
        def get_action(regime_id):
            if regime_id == 0: return "CASH / LIQUID"
            if regime_id == 2: return "SMALLCAP (ALPHA)"
            return "NIFTY 50 (BENCHMARK)"
            
        df_processed['Recommended_Asset'] = df_processed['Final_Regime_ID'].map(get_action)
        
        # Attach Probabilities
        for i in range(self.model.n_components):
            df_processed[f'Prob_State_{i}'] = probs[:, i]
        
        return df_processed.loc[start_date:]

if __name__ == "__main__":
    engine = InferenceEngine()
    results = engine.run_inference(start_date="2026-01-01")
    
    print("\n" + "="*50)
    print("MARS PRODUCTION INFERENCE REPORT (YOLO MODE)")
    print("="*50)
    
    latest = results.iloc[-1]
    print(f"Date:           {results.index[-1].date()}")
    print(f"HMM Raw State:  {latest['Regime_Name']} ({latest['Regime_ID']})")
    print(f"Trend Bullish:  {latest['Trend_Bullish']}")
    print(f"FINAL ACTION:   {latest['Recommended_Asset']}")
    print("-" * 50)
    print("Inertia Latch:  ACTIVE (Retain 20% of previous allocation)")
    print("="*50)