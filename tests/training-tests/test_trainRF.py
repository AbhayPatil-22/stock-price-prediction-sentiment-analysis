
from MLpipeline.src.model_training.trainRF import run_training_pipeline


def test_training_pipeline():
    run_training_pipeline("MSFT")
    print( "Training successful")


if __name__ == "__main__":
    test_training_pipeline()