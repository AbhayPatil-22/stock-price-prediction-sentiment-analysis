
from MLpipeline.src.evaluations.evaluateRF import run_evaluation_pipeline


def test_evaluation_pipeline():
    results = run_evaluation_pipeline("MSFT")

    print("\n Evaluation Successful")
    print(results)


if __name__ == "__main__":
    test_evaluation_pipeline()