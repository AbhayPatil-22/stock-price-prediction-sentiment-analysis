"""
train_pipeline.py

Handles:
- Training for single ticker
- Training for multiple tickers
"""

import logging
from datetime import datetime
from MLpipeline.src.data_collection.stock_data_collection import collect_stock_data
from MLpipeline.src.data_processing.data_preprocessing import run_preprocessing
from MLpipeline.src.model_training.trainLR import run_training_pipeline


# ---------------------- #
# CONFIG
# ---------------------- #

START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


# ---------------------- #
# TRAIN SINGLE TICKER
# ---------------------- #

def train_single_ticker(ticker: str):
    try:
        logging.info(f"Training started for {ticker}")

        # Step 1: Fetch data
        file_path = collect_stock_data(
            ticker=ticker,
            start_date=START_DATE,
            end_date=END_DATE
        )

        # Step 2: Preprocess
        run_preprocessing(file_path, ticker)

        # Step 3: Train model
        run_training_pipeline(ticker)

        logging.info(f"Training completed for {ticker}")

    except Exception as e:
        logging.error(f"Training failed for {ticker}: {e}")


# ---------------------- #
# TRAIN MULTIPLE TICKERS
# ---------------------- #

def train_multiple_tickers(tickers: list):
    for ticker in tickers:
        train_single_ticker(ticker)

