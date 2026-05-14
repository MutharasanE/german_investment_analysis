import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
import os

def run_basic_eda(file_path):
    print(f"--- Exploratory Data Analysis (EDA) ---")
    print(f"Analyzing File: {os.path.basename(file_path)}")
    print("-" * 40)
    
    # 1. Load Data
    df = pd.read_csv(file_path)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
    print(f"Data Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Missing Values Check
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    if len(missing_cols) > 0:
        print("\n1. Missing Values (Data Quality):")
        print("Action Needed: Imputation (e.g., Forward-fill for trading holidays)")
        print(missing_cols)
    else:
        print("\n1. Missing Values (Data Quality):")
        print("Status: PERFECT. No missing values detected.")

    # Select a target column to test (e.g., Close price or Return)
    target_col = 'Close' if 'Close' in df.columns else 'return_1y'
    if target_col not in df.columns:
         target_col = df.select_dtypes(include=[np.number]).columns[0]
            
    print(f"\nAnalyzing column: '{target_col}'")
    
    # Drop NaNs just for the statistical tests
    test_data = df[target_col].dropna()

    # 3. Stationarity Test (Augmented Dickey-Fuller)
    # Required for Time Series forecasting reliability
    print("\n2. Stationarity Test (Augmented Dickey-Fuller Test):")
    print("Goal: Check if the data has a constant mean and variance over time.")
    adf_result = adfuller(test_data)
    p_value_adf = adf_result[1]
    
    print(f"   p-value: {p_value_adf:.4f}")
    if p_value_adf < 0.05:
        print("   Result: Stationary (Safe for time-series modeling directly).")
    else:
        print("   Result: Non-Stationary (Needs differencing or percentage returns BEFORE modeling).")

    # 4. Non-Gaussianity Test (Shapiro-Wilk)
    # Critical for Phase 6 (DirectLiNGAM Causal Discovery)
    print("\n3. Normality Test (Shapiro-Wilk Test):")
    print("Goal: DirectLiNGAM REQUIRES Non-Gaussian (non-normal) data.")
    
    # Shapiro-Wilk test works best on sample sizes < 5000. 
    stat, p_value_sw = stats.shapiro(test_data.sample(min(len(test_data), 4999), random_state=42))
    
    print(f"   p-value: {p_value_sw:.4e}")
    if p_value_sw < 0.05:
        print("   Result: Non-Gaussian (SUCCESS! The data is suitable for LiNGAM).")
        print("   Reason: Financial data typically has 'fat tails' and skewness, rejecting perfect normality.")
    else:
        print("   Result: Gaussian/Normal (WARNING: LiNGAM might struggle).")

if __name__ == "__main__":
    # Test on one of the Raw Market Data files
    sample_file = "data/BMW_DE.csv"
    if os.path.exists(sample_file):
        run_basic_eda(sample_file)
    else:
        print(f"File {sample_file} not found.")
    
    # Also test on the final engineered dataset
    final_file = "data/investment_dataset.csv"
    if os.path.exists(final_file):
        print("\n========================================")
        run_basic_eda(final_file)
