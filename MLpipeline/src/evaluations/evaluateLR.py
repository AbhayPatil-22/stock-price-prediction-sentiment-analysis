"""
evaluate_rf.py

Handles:
- Loading test data (ticker-based)
- Loading trained model
- Model evaluation
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------- #
# LOGGING
# ---------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ---------------------- #
# LOAD TEST DATA
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
# LOAD MODEL
# ---------------------- #

def load_model(path: str):
    try:
        model = joblib.load(path)
        logging.info(f"Loaded model from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


# ---------------------- #
# FEATURES & TARGET
# ---------------------- #

def get_features_target(df: pd.DataFrame):
    X = df.drop(columns=["target", "date"])
    y = df["target"]
    return X, y


# ---------------------- #
# EVALUATE MODEL
# ---------------------- #

def evaluate_model(model, X, y):
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)

    logging.info("📊 Evaluation Results:")
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

def run_evaluation_pipeline(ticker: str):
    test_path = os.path.join(
        "MLpipeline", "data", "processed", ticker, "test.csv"
    )

    model_path = os.path.join(
        "MLpipeline", "models", ticker, "lr.pkl"
    )

    # Load data
    df = load_test_data(test_path)

    # Features
    X, y = get_features_target(df)

    # Load model
    model = load_model(model_path)

    # Evaluate
    results = evaluate_model(model, X, y)

    logging.info(f"Evaluation pipeline completed for {ticker}")

    return results