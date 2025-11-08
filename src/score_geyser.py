from pathlib import Path
import pandas as pd
import joblib


def main() -> None:
    # Define project paths
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data/raw" / "geyser.tsv"
    model_path = project_root / "models" / "linear_regression_pipeline.joblib"
    
    # Load the dataset
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep="\t")
    print(f"Loaded {len(df)} records")
    
    # Load the trained model pipeline
    print(f"Loading model from: {model_path}")
    pipeline = joblib.load(model_path)
    
    # Define feature column (same as training)
    feature_column = "eruptions"
    
    # Prepare features for prediction
    X = df[[feature_column]]
    
    # Generate predictions
    print("Generating predictions...")
    predictions = pipeline.predict(X)
    
    # Add predictions to dataframe
    df_scored = df.copy()
    df_scored["predicted_waiting"] = predictions
    
    # Create output directory
    scored_dir = project_root / "data" / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    
    # Export scored dataset
    output_path = scored_dir / "geyser_scored.tsv"
    df_scored.to_csv(output_path, sep="\t", index=False)
    
    print(f"Scored dataset exported to: {output_path}")
    print(f"Total records scored: {len(df_scored)}")
    
    # Display sample predictions
    print("\nSample predictions (first 5 rows):")
    print(df_scored[["eruptions", "waiting", "predicted_waiting"]].head())


if __name__ == "__main__":
    main()
