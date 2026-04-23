"""
train_xgb.py

Handles:
- Loading training data (ticker-based)
- Feature/target split
- Model training (XGBoost)
- Saving trained model
"""

import os
import logging
import pandas as pd
import joblib

from xgboost import XGBRegressor


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

def load_train_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded training data from {path}")
        return df
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        raise


# ---------------------- #
# FEATURES & TARGET
# ---------------------- #

def get_features_target(df: pd.DataFrame):
    X = df.drop(columns=["target", "date"])
    y = df["target"]
    return X, y


# ---------------------- #
# TRAIN MODEL
# ---------------------- #

def train_model(X, y):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    logging.info("XGBoost training completed")

    return model


# ---------------------- #
# SAVE MODEL
# ---------------------- #

def save_model(model, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logging.info(f"Model saved at {path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise


# ---------------------- #
# FULL PIPELINE
# ---------------------- #

def run_training_pipeline(ticker: str):
    train_path = os.path.join(
        "MLpipeline", "data", "processed", ticker, "train.csv"
    )

    model_path = os.path.join(
        "MLpipeline", "models", ticker, "xgb.pkl"
    )

    # Load
    df = load_train_data(train_path)

    # Features
    X, y = get_features_target(df)

    # Train
    model = train_model(X, y)

    # Save
    save_model(model, model_path)

    logging.info(f" XGBoost training completed for {ticker}")