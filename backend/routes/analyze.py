# backend/routes/analyze.py

from fastapi import APIRouter
from pydantic import BaseModel

# Import your pipelines
from MLpipeline.src.pipeline.prediction_pipeline import run_prediction_pipeline
from sentiment.src.pipeline.news_pipeline import run_news_pipeline

router = APIRouter()

# --------------------------
# Request Schema
# --------------------------
class AnalyzeRequest(BaseModel):
    ticker: str
    model_type: str = "lr"


# --------------------------
# Route
# --------------------------
@router.post("/analyze")
def analyze_stock(request: AnalyzeRequest):
    
    ticker = request.ticker
    model_type = request.model_type

    # 1. Prediction
    prediction = run_prediction_pipeline(
        ticker=ticker,
        model_type=model_type
    )

    # 2. Sentiment
    sentiment = run_news_pipeline(ticker)

    # 3. Response
    return {
        "ticker": ticker,
        "prediction": prediction["predicted_price"],
        "current_price": prediction["current_price"],
        "sentiment": sentiment["summary"]
    }