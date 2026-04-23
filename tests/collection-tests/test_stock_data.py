from MLpipeline.src.data_collection.stock_data_collection import collect_stock_data

if __name__ == "__main__":
    file_path = collect_stock_data(
        ticker="MSFT",
        start_date="2020-01-01",
        end_date="2024-01-01"
    )

    print(f"Stock data saved at: {file_path}")
    