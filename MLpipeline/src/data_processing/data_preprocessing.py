"""
data_preprocessing.py

Handles:
- Data cleaning
- Feature engineering (lags + rolling stats)
- Target creation
- Train-test split (time-series safe)
- Saving processed data (ticker-based)
"""

import os
import logging
import pandas as pd
from typing import Tuple


# ---------------------- #
# LOGGING
# ---------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------- #
# LOAD DATA
# ---------------------- #

def load_raw_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


# ---------------------- #
# CLEANING
# ---------------------- #

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")

    df = df.drop_duplicates()
    df = df.dropna()

    logging.info("Data cleaned successfully")
    return df


# ---------------------- #
# FEATURE ENGINEERING
# ---------------------- #

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["return"] = df["close"].pct_change()

    # Moving averages
    df["ma_7"] = df["close"].rolling(window=7).mean()
    df["ma_14"] = df["close"].rolling(window=14).mean()
    df["ma_21"] = df["close"].rolling(window=21).mean()

    # Volatility
    df["volatility_7"] = df["return"].rolling(window=7).std()

    # ---------------------- #
    # LAG FEATURES (IMPORTANT)
    # ---------------------- #
    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_3"] = df["close"].shift(3)

    df["return_lag_1"] = df["return"].shift(1)
    df["momentum"] = df["close"] - df["lag_3"]

    # Drop NA from rolling + lag
    df = df.dropna()

    logging.info("Features added successfully")
    return df


# ---------------------- #
# TARGET CREATION
# ---------------------- #

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["target"] = df["close"].pct_change().shift(-1)

    df = df.dropna()

    logging.info("Target column created")
    return df


# ---------------------- #
# TRAIN-TEST SPLIT
# ---------------------- #

def split_data(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * split_ratio)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df


# ---------------------- #
# SAVE DATA (DYNAMIC)
# ---------------------- #

def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, ticker: str):
    base_path = os.path.join("MLpipeline", "data", "processed", ticker)
    os.makedirs(base_path, exist_ok=True)

    train_path = os.path.join(base_path, "train.csv")
    test_path = os.path.join(base_path, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logging.info(f"Processed data saved for {ticker}")


# ---------------------- #
# FULL PIPELINE
# ---------------------- #

def run_preprocessing(file_path: str, ticker: str):
    df = load_raw_data(file_path)

    df = clean_data(df)
    df = add_features(df)
    df = create_target(df)

    train_df, test_df = split_data(df)

    save_processed_data(train_df, test_df, ticker)

    logging.info(f" Preprocessing pipeline completed for {ticker}")