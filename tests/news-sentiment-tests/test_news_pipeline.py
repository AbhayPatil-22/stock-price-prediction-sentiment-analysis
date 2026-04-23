from sentiment.src.pipeline.news_pipeline import run_news_pipeline

result = run_news_pipeline("AAPL")
print(result["summary"])