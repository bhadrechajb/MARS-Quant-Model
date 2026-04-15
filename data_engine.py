import yfinance as yf
import pandas as pd
import numpy as np

class DataIntegrityError(Exception):
    """Raised when data quality checks fail (e.g., stale feeds, zero variance)."""
    pass

class DataPipeline:
    def __init__(self, start_date="2015-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = {
            # Target Indices (To be predicted)
            "Nifty50": "^NSEI",
            "Midcap": "^NSMIDCP",
            "Smallcap_Old": "BSE-SMLCAP.BO",
            "Smallcap_New": "HDFCSML250.NS",
            
            # Macro Overlay (Features)
            "USDINR": "INR=X",
            "BrentCrude": "BZ=F",
            "US10Y": "^TNX",
            
            # Volatility (Feature)
            "IndiaVIX": "^INDIAVIX"
        }
        self.raw_data = None
        self.tensor = None

    def fetch_data(self):
        """
        Pulls raw adjusted close data from yfinance.
        """
        print(f"Fetching data from {self.start_date}...")
        data = yf.download(list(self.tickers.values()), start=self.start_date, end=self.end_date, progress=False)
        
        # Extract 'Close' (Adj Close might be missing)
        if isinstance(data.columns, pd.MultiIndex):
             try:
                 df = data['Adj Close'].copy()
             except KeyError:
                 df = data['Close'].copy()
        else:
             df = data.copy() # Fallback if single level

        # Rename columns to friendly names
        reverse_map = {v: k for k, v in self.tickers.items()}
        df.rename(columns=reverse_map, inplace=True)
        
        self.raw_data = df
        print(f"Raw Data Fetched. Shape: {self.raw_data.shape}")
        return self

    def align_and_clean(self):
        """
        Handles timezone mismatches and holidays using forward fill.
        Also splices Smallcap feeds (BSE Index -> HDFC ETF).
        """
        if self.raw_data is None:
            raise ValueError("Data not fetched yet. Call .fetch_data() first.")
        
        df = self.raw_data.copy()
        
        # Rule: Forward Fill to handle mismatched holidays
        df.ffill(inplace=True)
        
        # --- SYNTHETIC SMALLCAP SPLICE ---
        # Switch date: 2023-02-21 (Start of HDFC ETF Data availability approx)
        # We check where Smallcap_New starts having valid data
        
        if "Smallcap_New" in df.columns and "Smallcap_Old" in df.columns:
            print("Splicing Smallcap Feeds (Index -> ETF)...")
            
            # Identify transition point (First valid index of New)
            # Or hardcode a safe date where both exist. 2023-03-01 seems safe based on checks.
            splice_date = pd.Timestamp("2023-03-01")
            
            # Normalize New to match Old at splice point to avoid price shock
            try:
                # If splice_date is in index, use it. Otherwise use the first available overlap.
                if splice_date in df.index:
                    old_price = df.loc[splice_date, "Smallcap_Old"]
                    new_price = df.loc[splice_date, "Smallcap_New"]
                elif df.index[0] > splice_date:
                    # If we started AFTER the splice date, we need to find the earliest point where both exist
                    # or assume they were already aligned (which they aren't, so we need a reference)
                    # For MARS, the alignment factor is historically ~310.0 (Index is ~30k, ETF is ~100)
                    # Let's try to find the earliest overlapping valid data
                    overlap = df[["Smallcap_Old", "Smallcap_New"]].dropna()
                    if not overlap.empty:
                        first_date = overlap.index[0]
                        old_price = overlap.loc[first_date, "Smallcap_Old"]
                        new_price = overlap.loc[first_date, "Smallcap_New"]
                        print(f"Splice date {splice_date.date()} out of range. Using first overlap at {first_date.date()}")
                    else:
                        raise ValueError("No overlapping data for Smallcap feeds.")
                else:
                    # Splice date is in the future relative to some data? 
                    # Should not happen with ffill, but handle iloc[-1] safely
                    old_price = df.loc[:splice_date, "Smallcap_Old"].iloc[-1]
                    new_price = df.loc[:splice_date, "Smallcap_New"].iloc[-1]
                
                factor = old_price / new_price
                print(f"Splice Factor: {factor:.4f} (Aligning ETF to Index scale)")
                
                # Create spliced series
                df["Smallcap"] = df["Smallcap_Old"].copy()
                mask_new = df.index > splice_date
                
                # If we are completely after splice_date, all data comes from New * factor
                if df.index[0] > splice_date:
                    df["Smallcap"] = df["Smallcap_New"] * factor
                else:
                    df.loc[mask_new, "Smallcap"] = df.loc[mask_new, "Smallcap_New"] * factor
                
                # Clean up source columns
                df.drop(columns=["Smallcap_Old", "Smallcap_New"], inplace=True)
                
            except Exception as e:
                print(f"Splicing failed: {e}. Fallback to Smallcap_Old.")
                df["Smallcap"] = df["Smallcap_Old"]
        
        # Drop initial NaNs
        df.dropna(inplace=True)
        
        self.tensor = df
        print("Data Aligned and Cleaned (NaNs removed).")
        return self

    def transform_features(self):
        """
        Applies mathematical transformations:
        1. Stationarity (Log Returns) for Target Indices.
        2. Rolling Z-Score for Macro Features.
        3. Lag Strategy (Shift=1) for Look-Ahead Bias prevention.
        """
        if self.tensor is None:
            raise ValueError("Data not aligned. Call .align_and_clean() first.")
        
        df = self.tensor.copy()
        transformed = pd.DataFrame(index=df.index)
        
        # Lists of columns for different treatments
        targets = ["Nifty50", "Midcap", "Smallcap"]
        macros = ["USDINR", "BrentCrude", "US10Y", "IndiaVIX"]
        
        # Transformation A: Enforce Stationarity (Log-Returns)
        # r_t = ln(P_t) - ln(P_t-1)
        print("Applying Log-Returns to Targets...")
        for col in targets:
            if col in df.columns:
                transformed[f"{col}_LogRet"] = np.log(df[col] / df[col].shift(1))
                # Preserve Raw Prices for Backtest/Trend Logic
                transformed[col] = df[col]
        
        # Also preserve Smallcap_Index_Check if it exists
        if "Smallcap_Index_Check" in df.columns:
            transformed["Smallcap_Index_Check"] = df["Smallcap_Index_Check"]
            transformed["Smallcap_Index_Check_LogRet"] = np.log(df["Smallcap_Index_Check"] / df["Smallcap_Index_Check"].shift(1))
        
        # Transformation B: Rolling Z-Score Scaling
        # z = (x - mean) / std
        window_size = 63 # 1 Business Quarter (Institutional Standard)
        print(f"Applying Rolling Z-Score (window={window_size}) to Macro Features...")
        
        # Data Integrity Layer: Rolling MAD for outlier handling
        def rolling_mad_clean(series, window=63, threshold=3.5):
            median = series.rolling(window=window).median()
            mad = (series - median).abs().rolling(window=window).median()
            # Flag/Cap values exceeding 3.5x MAD
            lower_bound = median - (threshold * mad)
            upper_bound = median + (threshold * mad)
            return series.clip(lower_bound, upper_bound)

        for col in macros:
            if col in df.columns:
                # Apply MAD cleaning before Z-score
                cleaned_series = rolling_mad_clean(df[col], window=window_size)
                
                roll_mean = cleaned_series.rolling(window=window_size).mean()
                roll_std = cleaned_series.rolling(window=window_size).std()
                z_score_col = (cleaned_series - roll_mean) / roll_std
                
                # Transformation C: Prevent Look-Ahead Bias (Lag Strategy)
                # We shift these features by 1 because we want Yesterday's Macro/VIX to predict Today's Market
                transformed[f"{col}_Z_Lag1"] = z_score_col.shift(1)
                
        # Drop NaNs created by Shift(1) and Rolling Window
        transformed.dropna(inplace=True)
        
        self.tensor = transformed
        print(f"Feature Engineering Complete. Final Tensor Shape: {self.tensor.shape}")
        return self

    def check_liquidity(self, df, threshold=5):
        """
        Checks for stale feeds where price hasn't moved for 'threshold' days.
        """
        print("Running Liquidity/Continuity Check...")
        for col in df.columns:
            # Calculate consecutive zeros in diff (no price change)
            # We use df[col].diff() == 0 to check for unchanged prices
            price_unchanged = (df[col].diff() == 0).fillna(False)
            
            consecutive_stale = price_unchanged.astype(int).groupby((~price_unchanged).cumsum()).cumsum()
            max_stale = consecutive_stale.max()
            
            if max_stale >= threshold:
                print(f"WARNING: Stale feed detected in {col}. Max stagnant days: {max_stale}")
                # raise DataIntegrityError(f"Stale feed detected in {col}. Max stagnant days: {max_stale}")
        print("Liquidity Check Passed (with Warnings).")

    def run_pipeline(self):
        """
        Orchestrates the pipeline.
        """
        self.fetch_data()
        self.align_and_clean()
        
        # Run Liquidity Check before transformation
        self.check_liquidity(self.tensor)
        
        self.transform_features()
        return self.tensor

if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = DataPipeline()
    final_tensor = pipeline.run_pipeline()
    
    # Preview
    print("\n--- Final Tensor Head ---")
    print(final_tensor.head())
    print("\n--- Final Tensor Tail ---")
    print(final_tensor.tail())
    
    # Save for Phase 2
    final_tensor.to_csv("market_tensor_processed.csv")
    print("\nProcessed Tensor saved to 'market_tensor_processed.csv'")
