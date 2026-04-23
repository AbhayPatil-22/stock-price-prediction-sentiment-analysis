from MLpipeline.src.pipeline.prediction_pipeline import run_prediction_pipeline

# Use a ticker you have NOT trained before
result = run_prediction_pipeline("AAPL")

print("\nAuto-Training Prediction Result:")
print(result)