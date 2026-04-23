import os
from sentiment.src.data_collection.finviz_scraper import scrape_finviz_news
from sentiment.src.processing.sentiment_analyzer import analyze_sentiment
from sentiment.src.processing.sentiment_aggregator import aggregate_sentiment


def run_news_pipeline(ticker: str):
    # Step 1: Scrape
    df = scrape_finviz_news(ticker)

    if df.empty:
        print("[WARNING] No data fetched.")
        return None

    # Step 2: Sentiment Analysis
    df = analyze_sentiment(df)

    # Step 3: Aggregate Sentiment
    summary = aggregate_sentiment(df)

    # Step 4: Save raw + sentiment data
    save_path = f"sentiment/data/news/finviz_{ticker}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"[INFO] Pipeline completed for {ticker}")

    return {
        "ticker": ticker,
        "summary": summary,
    
    }   


 ###"data": df  # optional (for debugging / future use)

