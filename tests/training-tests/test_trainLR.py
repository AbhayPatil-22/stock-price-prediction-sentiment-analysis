from MLpipeline.src.model_training.trainLR import run_training_pipeline


def test_training_pipeline():
    run_training_pipeline("AAPL")
    print(" Linear Regression training successful")


if __name__ == "__main__":
    test_training_pipeline()