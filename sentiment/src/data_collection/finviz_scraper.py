import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_finviz_news(ticker: str) -> pd.DataFrame:
    """
    Scrapes latest news headlines from Finviz for a given ticker.
    Returns a cleaned DataFrame.
    """

    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")

    if news_table is None:
        print("[WARNING] No news table found")
        return pd.DataFrame()

    rows = news_table.find_all("tr")

    news = []
    current_date = None

    for row in rows:
        title = row.a.text.strip()  #  cleaned title
        timestamp = row.td.text.strip()

        if " " in timestamp:
            current_date, time = timestamp.split(" ")
        else:
            time = timestamp

        news.append({
            "ticker": ticker,        #  added here
            "date": current_date,    #  fixed date handling
            "time": time,
            "title": title
        })

    df = pd.DataFrame(news)

    print(f"[INFO] Scraped {len(df)} news articles for {ticker}")

    return df