import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error


class DamagePredictorInference:
    """A class for making predictions using the trained Damage Predictor model."""

    def __init__(self, model_folder, config):
        self.model_folder = model_folder
        self.CONFIG = config
        self.scaler = joblib.load(os.path.join(model_folder, "best_scaler.pkl"))
        self.pca = joblib.load(os.path.join(model_folder, "best_pca.pkl"))
        self.model = joblib.load(os.path.join(model_folder, "best_model.pkl"))

    def predict_from_test_csv(self, test_csv_path):
        """Loads a precomputed test dataset from CSV and returns predictions.

        Args:
            test_csv_path (str): Path to the CSV file containing the test set.

        Returns:
            pd.DataFrame: A DataFrame containing bam_name, predicted_age, and optionally true_age.
        """
        df = pd.read_csv(test_csv_path)

        labels = df['label'] if 'label' in df.columns else None
        bam_names = df['bam_name'].astype(str)

        # Drop metadata columns
        X = df.drop(columns=['label', 'bam_name', 'batch_name'], errors='ignore')
        X = X.fillna(X.median())

        # Transform and predict
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        predictions = self.model.predict(X_pca)

        result_df = pd.DataFrame({
            'bam_name': bam_names,
            'predicted_age': predictions
        })

        if labels is not None:
            result_df['true_age'] = labels

        return result_df


def inference_from_test_set():
    """Main function to run inference and evaluate performance."""
    # Load config
    with open("config/config_harvard_60_cv5.json", 'r') as f:
        CONFIG = json.load(f)

    # Define paths
    exp_folder = CONFIG["exp_folder"]
    model_folder = os.path.join(exp_folder, "best_model")
    test_csv_path = os.path.join(exp_folder, "SPLIT_DATA/test.csv")

    # Output folder
    output_folder = os.path.join(exp_folder, "test")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "test_predictions.csv")

    # Run inference
    predictor = DamagePredictorInference(model_folder, CONFIG)
    prediction_df = predictor.predict_from_test_csv(test_csv_path)

    # Save predictions
    prediction_df.to_csv(output_path, index=False)

    # Evaluation
    if 'true_age' in prediction_df.columns:
        y_true = prediction_df['true_age'].values
        y_pred = prediction_df['predicted_age'].values

        avg_predicted_age = np.mean(y_pred)
        avg_true_age = np.mean(y_true)

        # Model errors
        mae_model = mean_absolute_error(y_true, y_pred)
        rmse_model = mean_squared_error(y_true, y_pred)

        # Baseline (predicting average age)
        y_baseline = np.full_like(y_true, avg_true_age)
        mae_baseline = mean_absolute_error(y_true, y_baseline)
        rmse_baseline = mean_squared_error(y_true, y_baseline)

        # Print results
        print("\n📊 Evaluation Metrics:")
        print(f"  ▸ Average predicted age:         {avg_predicted_age:.2f}")
        print(f"  ▸ Average true age:              {avg_true_age:.2f}")
        print(f"  ▸ Model MAE (mean abs error):    {mae_model:.2f}")
        print(f"  ▸ Model RMSE:                    {rmse_model:.2f}")
        print(f"  ▸ Baseline MAE (mean predictor): {mae_baseline:.2f}")
        print(f"  ▸ Baseline RMSE:                 {rmse_baseline:.2f}")
    else:
        print("\nℹ No true labels found in test set, evaluation skipped.")

    print(f"\n✅ Predictions saved to: {output_path}")
    print("\n🔍 First 5 predictions:\n", prediction_df.head())


if __name__ == '__main__':
    inference_from_test_set()
