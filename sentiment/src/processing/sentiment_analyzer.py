from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


# Initialize once (important for performance)
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sentiment score and label to news DataFrame.
    """

    if df.empty:
        print("[WARNING] Empty DataFrame received")
        return df

    scores = []
    labels = []

    for title in df["title"]:
        sentiment = analyzer.polarity_scores(title)
        compound = sentiment["compound"]

        scores.append(compound)

        # Labeling logic
        if compound >= 0.05:
            labels.append("positive")
        elif compound <= -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")

    df["sentiment_score"] = scores
    df["sentiment_label"] = labels

    print(f"[INFO] Sentiment analysis completed for {len(df)} rows")

    return df