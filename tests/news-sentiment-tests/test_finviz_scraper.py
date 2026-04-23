from sentiment.src.data_collection.finviz_scraper import scrape_finviz_news

df = scrape_finviz_news("AAPL")

print(df.head())
print("\nShape:", df.shape)