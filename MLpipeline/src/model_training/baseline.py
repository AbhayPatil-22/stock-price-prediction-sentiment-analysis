"""
baseline.py

Handles:
- Baseline prediction (naive)
- Evaluation of baseline model
"""

import os
import logging
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

def load_test_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded test data from {path}")
        return df
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        raise


# ---------------------- #
# BASELINE PREDICTION
# ---------------------- #

def baseline_predict(df: pd.DataFrame):
    """
    Naive baseline:
    Predict next day's price = today's close
    """
    return df["close"]


# ---------------------- #
# EVALUATION
# ---------------------- #

def evaluate_baseline(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    logging.info("📊 Baseline Results:")
    logging.info(f"MAE  : {mae:.4f}")
    logging.info(f"MSE  : {mse:.4f}")
    logging.info(f"RMSE : {rmse:.4f}")
    logging.info(f"R2   : {r2:.4f}")

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }


# ---------------------- #
# FULL PIPELINE
# ---------------------- #

def run_baseline_pipeline(ticker: str):
    test_path = os.path.join(
        "MLpipeline", "data", "processed", ticker, "test.csv"
    )

    df = load_test_data(test_path)

    y_true = df["target"]
    y_pred = baseline_predict(df)

    results = evaluate_baseline(y_true, y_pred)

    logging.info(f" Baseline evaluation completed for {ticker}")

    return results