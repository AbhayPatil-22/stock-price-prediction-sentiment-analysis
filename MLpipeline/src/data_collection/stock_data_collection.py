"""
stock_data_collection.py

Handles:
- Fetching stock data from Yahoo Finance
- Saving raw stock data

"""

import os
import logging
from typing import Optional
import pandas as pd
import yfinance as yf

# ---------------------- #
#  CONFIGURATION
# ---------------------- #

BASE_PATH = os.path.join("MLpipeline", "data", "raw")
os.makedirs(BASE_PATH, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------- #
#  FETCH STOCK DATA
# ---------------------- #

def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.

    Args:
        ticker (str): Stock symbol (e.g., 'AAPL', 'TSLA')
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        interval (str): Data frequency (default: '1d')

    Returns:
        pd.DataFrame
    """
    try:
        logging.info(f"Fetching data for {ticker}")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")

        df.reset_index(inplace=True)

        # Standardize column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]
        logging.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logging.error(f"Error fetching stock data: {e}")
        raise


# ---------------------- #
#  SAVE DATA
# ---------------------- #

def save_stock_data(
    df: pd.DataFrame,
    ticker: str,
    filename: Optional[str] = None
) -> str:
    """
    Save stock data to CSV.
    """
    try:
        if filename is None:
            filename = f"{ticker}_stock_data.csv"

        file_path = os.path.join(BASE_PATH, filename)

        df.to_csv(file_path, index=False)

        logging.info(f"Saved data to {file_path}")
        return file_path

    except Exception as e:
        logging.error(f"Error saving stock data: {e}")
        raise


# ---------------------- #
#  PIPELINE FUNCTION
# ---------------------- #

def collect_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Full pipeline:
    Fetch + Save stock data

    Returns:
        str: Saved file path
    """
    df = fetch_stock_data(ticker, start_date, end_date)
    return save_stock_data(df, ticker)


