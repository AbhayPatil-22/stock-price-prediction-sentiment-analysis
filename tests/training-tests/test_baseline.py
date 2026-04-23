from MLpipeline.src.model_training.baseline import run_baseline_pipeline


def test_baseline():
    results = run_baseline_pipeline("MSFT")

    print("\n Baseline Test Successful")
    print(results)


if __name__ == "__main__":
    test_baseline()