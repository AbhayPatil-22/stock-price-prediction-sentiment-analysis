import pandas as pd


def aggregate_sentiment(df: pd.DataFrame) -> dict:
    """
    Aggregates sentiment for a ticker.
    Returns a summary dictionary.
    """

    if df.empty:
        return {}

    total = len(df)

    positive = (df["sentiment_label"] == "positive").sum()
    negative = (df["sentiment_label"] == "negative").sum()
    neutral = (df["sentiment_label"] == "neutral").sum()

    avg_score = float(df["sentiment_score"].mean())

    # Overall sentiment
    if avg_score >= 0.05:
        overall = "positive"
    elif avg_score <= -0.05:
        overall = "negative"
    else:
        overall = "neutral"

    summary = {
        "total_news": total,
        "positive": int(positive),
        "negative": int(negative),
        "neutral": int(neutral),
        "avg_score": round(avg_score, 4),
        "overall_sentiment": overall
    }

    return summary