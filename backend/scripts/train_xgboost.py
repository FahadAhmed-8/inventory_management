import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib # For saving and loading the model
import os

# Define the path to your processed CSV file
PROCESSED_CSV_FILE_PATH = os.path.join("data", "processed", "retail_inventory_forecast_processed.csv")
# Define the path where the trained XGBoost model will be saved
XGBOOST_MODEL_SAVE_PATH = os.path.join("models", "xgboost_model.joblib")

def train_xgboost_model(input_csv_path, model_save_path):
    """
    Loads processed data, trains an XGBoost model for demand forecasting,
    evaluates it, and saves the trained model.
    """
    print(f"Starting XGBoost model training using data from {input_csv_path}...")

    try:
        # Load the processed dataset
        df = pd.read_csv(input_csv_path)
        print(f"Loaded dataset shape: {df.shape}")

        # Ensure 'Date' column is datetime for proper splitting
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort data by Date to ensure correct time-series split
        df.sort_values(by='Date', inplace=True)

        # Define features (X) and target (y)
        features = [
            'Price', 'Discount', 'Holiday/Promotion', 'Competitor Pricing',
            'day_of_week', 'month', 'year',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_30',
            'units_sold_rolling_avg_7', 'units_sold_rolling_avg_30'
        ]
        target = 'Units Sold'

        # Check if all required features and target exist in the DataFrame
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Error: Missing required features in the dataset: {missing_features}")
            print("Please ensure your data_preprocessing.py generated these columns correctly.")
            print("Available columns in the DataFrame:")
            print(df.columns.tolist())
            return

        if target not in df.columns:
            print(f"Error: Missing target column '{target}' in the dataset.")
            return

        X = df[features]
        y = df[target]

        # Split data into training and validation sets
        split_point = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]

        print(f"Training set size: {len(X_train)} samples")
        print(f"Validation set size: {len(X_val)} samples")

        # Initialize and train the XGBoost Regressor model
        print("Training XGBoost Regressor model...")
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        print("XGBoost model training complete.")
        print("Note: Early stopping not available in this XGBoost version. Using all 1000 trees.")

        # Evaluate the model on the validation set
        print("Evaluating model on validation set...")
        val_predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_predictions)
        print(f"XGBoost Validation Mean Absolute Error (MAE): {mae:.2f}")

        # Save the trained model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure directory exists
        joblib.dump(model, model_save_path)
        print(f"XGBoost model saved to {model_save_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found. Please ensure it's in the 'backend' directory and data_preprocessing.py was run.")
    except Exception as e:
        print(f"An unexpected error occurred during XGBoost training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_xgboost_model(PROCESSED_CSV_FILE_PATH, XGBOOST_MODEL_SAVE_PATH)

