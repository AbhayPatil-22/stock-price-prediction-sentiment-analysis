"""
prediction_pipeline.py

Handles:
- Fetching latest stock data
- Preprocessing
- Auto-training if model missing
- Prediction
"""

import os
import logging
import joblib
import pandas as pd
from datetime import datetime, timedelta
from MLpipeline.src.data_collection.stock_data_collection import fetch_stock_data
from MLpipeline.src.data_processing.data_preprocessing import clean_data, add_features

#  Import training function
from MLpipeline.src.pipeline.train_pipeline import train_single_ticker


# ---------------------- #
# CONFIG
# ---------------------- #

MODEL_DIR = os.path.join("MLpipeline", "models")


# ---------------------- #
# LOAD MODEL (AUTO-TRAIN)
# ---------------------- #

def load_model(ticker: str, model_type: str = "lr"):
    model_path = os.path.join(MODEL_DIR, ticker, f"{model_type}.pkl")

    #  AUTO-TRAIN TRIGGER
    if not os.path.exists(model_path):
        logging.warning(f"Model not found for {ticker}. Starting auto-training...")
        train_single_ticker(ticker)

    # Check again
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found even after training for {ticker}")

    model = joblib.load(model_path)
    logging.info(f"Loaded model for {ticker}")

    return model


# ---------------------- #
# PREPARE INPUT
# ---------------------- #

def prepare_input_data(df: pd.DataFrame) -> pd.DataFrame:
    latest_row = df.iloc[-1:].copy()

    # Must match training
    latest_row = latest_row.drop(columns=["date"], errors="ignore")

    return latest_row


# ---------------------- #
# MAIN PIPELINE
# ---------------------- #

def run_prediction_pipeline(
    ticker: str,
    model_type: str = "lr"
):
    try:
        logging.info(f"Running prediction pipeline for {ticker}")

        # 🔥 Dynamic date range
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Step 1: Fetch data
        df = fetch_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        if df.empty:
            return {"error": "No data fetched"}

        # Step 2: Clean
        df = clean_data(df)

        # Step 3: Features
        df = add_features(df)

        latest_date = df.iloc[-1]["date"]
        print(f"[DEBUG] Latest data date used: {latest_date}")

        # Step 4: Load model (auto-train if needed)
        model = load_model(ticker, model_type)

        # Step 5: Prepare input
        latest_row = prepare_input_data(df)

        # Step 6: Predict
        predicted_return = model.predict(latest_row)[0]

        # Get current price
        current_price = df.iloc[-1]["close"]

        # Convert return → price
        predicted_price = current_price * (1 + predicted_return)

        return {
        "ticker": ticker,
        "model_used": model_type,
        "current_price": round(float(current_price), 2),
        "predicted_return": round(float(predicted_return), 4),
        "predicted_price": round(float(predicted_price), 2)
        }

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return {"error": str(e)}