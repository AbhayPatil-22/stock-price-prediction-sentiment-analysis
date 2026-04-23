from sentiment.src.data_collection.finviz_scraper import scrape_finviz_news
from sentiment.src.processing.sentiment_analyzer import analyze_sentiment

df = scrape_finviz_news("AAPL")
df = analyze_sentiment(df)

print(df.head())