import pandas as pd
import joblib
import numpy as np

def test_scoring_pipeline():
    # Load the trained model pipeline
    model_path = "models/linear_regression_pipeline.joblib"
    pipeline = joblib.load(model_path)

    # Create sample input data
    df = pd.DataFrame({"eruptions": [1.5, 2.0, 3.0]})

    # Generate predictions
    preds = pipeline.predict(df[["eruptions"]])

    # a) Assert number of predictions matches number of inputs
    assert len(preds) == len(df), "Number of predictions does not match number of inputs"

    # b) Assert all predictions are finite
    assert np.all(np.isfinite(preds)), "Some predictions are NaN or infinite"

    # c) Assert all predictions are positive
    assert np.all(preds > 0), "Some predictions are not positive"

