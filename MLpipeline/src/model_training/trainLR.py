"""
train_lr.py

Handles:
- Loading training data (ticker-based)
- Feature/target split
- Model training (Linear Regression)
- Saving trained model
"""

import os
import logging
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression


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
    model = LinearRegression()

    model.fit(X, y)
    logging.info("Linear Regression training completed")

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
        "MLpipeline", "models", ticker, "lr.pkl"
    )

    # Load
    df = load_train_data(train_path)

    # Features
    X, y = get_features_target(df)

    # Train
    model = train_model(X, y)

    # Save
    save_model(model, model_path)

    logging.info(f" Linear Regression training completed for {ticker}")