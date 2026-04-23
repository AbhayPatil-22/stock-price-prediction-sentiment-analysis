from sentiment.src.data_collection.finviz_scraper import scrape_finviz_news
from sentiment.src.processing.sentiment_analyzer import analyze_sentiment
from sentiment.src.processing.sentiment_aggregator import aggregate_sentiment


df = scrape_finviz_news("AAPL")
df = analyze_sentiment(df)

summary = aggregate_sentiment(df)

print(summary)