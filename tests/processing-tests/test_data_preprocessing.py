from MLpipeline.src.data_processing.data_preprocessing import run_preprocessing

run_preprocessing(
    file_path="MLpipeline/data/raw/MSFT_stock_data.csv",
    ticker="MSFT"
)