import pandas as pd
import numpy as np
import joblib # For loading models and saving weights
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from prophet import Prophet # Ensure Prophet is imported for loading
import json # For saving weights
import os

# Define paths
PROCESSED_CSV_FILE_PATH = os.path.join("data", "processed", "retail_inventory_forecast_processed.csv")
XGBOOST_MODEL_SAVE_PATH = os.path.join("models", "xgboost_model.joblib")
LSTM_MODEL_SAVE_PATH = os.path.join("models", "lstm_model.h5")
SCALER_X_SAVE_PATH = os.path.join("models", "scaler_X.joblib")
SCALER_Y_SAVE_PATH = os.path.join("models", "scaler_y.joblib")
PROPHET_MODEL_SAVE_PATH = os.path.join("models", "prophet_model.joblib")
ENSEMBLE_WEIGHTS_SAVE_PATH = os.path.join("models", "ensemble_weights.json")

# Define sequence length (must match what was used in train_lstm.py)
LOOK_BACK = 7

# MODIFIED: create_sequences_for_ensemble to directly process group data
def create_sequences_for_ensemble(data_df, look_back, scaler_X, scaler_y, sequence_features, target_feature):
    """
    Creates sequences of features (X) and corresponding actual target values (y) for LSTM prediction.
    Applies scaling and groups data by Store ID and Product ID.
    Returns both the input sequences and the true target values for those sequences.
    """
    X_sequences, y_actuals_aligned = [], []

    grouped = data_df.groupby(['Store ID', 'Product ID'])

    for name, group in grouped:
        # Extract features and target for the current group
        group_features_data = group[sequence_features].values
        group_target_data = group[target_feature].values

        # Scale features for this group
        group_scaled_X = scaler_X.transform(group_features_data)
        
        if len(group_scaled_X) > look_back: # Ensure enough data to create at least one sequence
            for i in range(len(group_scaled_X) - look_back):
                # Features for the sequence
                X_sequences.append(group_scaled_X[i:(i + look_back), :])
                # The actual target value for this sequence (raw, not scaled)
                y_actuals_aligned.append(group_target_data[i + look_back])

    return np.array(X_sequences), np.array(y_actuals_aligned)


def create_ensemble(input_csv_path, xgboost_model_path, lstm_model_path,
                    scaler_x_path, scaler_y_path, prophet_model_path,
                    weights_save_path):
    """
    Loads trained models, generates predictions on a validation set,
    calculates weights based on MAE, creates an ensemble forecast,
    and saves the weights.
    """
    print("Starting ensemble creation...")

    try:
        # --- 1. Load Data and Split Validation Set ---
        df = pd.read_csv(input_csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)

        # Define features and target (must match what was used in training)
        xgboost_features = [
            'Price', 'Discount', 'Holiday/Promotion', 'Competitor Pricing',
            'day_of_week', 'month', 'year',
            'units_sold_lag_1', 'units_sold_lag_7', 'units_sold_lag_30',
            'units_sold_rolling_avg_7', 'units_sold_rolling_avg_30'
        ]
        lstm_sequence_features = [
            'Units Sold',
            'day_of_week',
            'month'
        ]
        target = 'Units Sold'

        # Split point for validation set (must match training scripts)
        split_point = int(len(df) * 0.8)
        val_df = df.iloc[split_point:].copy()

        # Get actual target values for validation (for XGBoost)
        y_val_actual_xgboost = val_df[target].values
        print(f"Validation set size for ensemble: {len(val_df)} samples")

        # --- 2. Load Trained Models ---
        print("Loading trained models...")
        xgboost_model = joblib.load(xgboost_model_path)
        lstm_model = tf.keras.models.load_model(lstm_model_path)
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        prophet_model = joblib.load(prophet_model_path)
        print("Models loaded successfully.")

        # --- 3. Generate Predictions on Validation Set ---

        # XGBoost Predictions
        print("Generating XGBoost predictions...")
        X_val_xgboost = val_df[xgboost_features]
        xgboost_predictions = xgboost_model.predict(X_val_xgboost)
        print(f"XGBoost predictions shape: {xgboost_predictions.shape}")

        # LSTM Predictions
        print("Generating LSTM predictions...")
        X_val_lstm_sequences, y_val_actual_lstm_aligned = create_sequences_for_ensemble(
            val_df, LOOK_BACK, scaler_X, scaler_y, lstm_sequence_features, target
        )

        if X_val_lstm_sequences.size > 0:
            lstm_predictions_scaled = lstm_model.predict(X_val_lstm_sequences)
            lstm_predictions = scaler_y.inverse_transform(lstm_predictions_scaled).flatten()
            print(f"LSTM predictions shape: {lstm_predictions.shape}")
            print(f"LSTM actuals (aligned) shape: {y_val_actual_lstm_aligned.shape}")
        else:
            print("Warning: Not enough data in validation set to create LSTM sequences. Skipping LSTM predictions and MAE.")
            lstm_predictions = np.array([])
            y_val_actual_lstm_aligned = np.array([])


        # Prophet Predictions
        print("Generating Prophet predictions...")
        val_prophet_df = val_df.groupby('Date')[target].sum().reset_index()
        val_prophet_df.rename(columns={'Date': 'ds', target: 'y'}, inplace=True)

        prophet_regressors_val = val_df.groupby('Date')['Competitor Pricing'].mean().reset_index()
        prophet_regressors_val.rename(columns={'Date': 'ds'}, inplace=True)
        val_prophet_df = pd.merge(val_prophet_df, prophet_regressors_val, on='ds', how='left')
        val_prophet_df['Competitor Pricing'] = val_prophet_df['Competitor Pricing'].fillna(val_prophet_df['Competitor Pricing'].mean())

        future_prophet_val = prophet_model.make_future_dataframe(
            periods=len(val_prophet_df),
            include_history=False,
            freq='D'
        )
        if 'Competitor Pricing' in val_prophet_df.columns:
            future_prophet_val = pd.merge(future_prophet_val, val_prophet_df[['ds', 'Competitor Pricing']], on='ds', how='left')
            future_prophet_val['Competitor Pricing'] = future_prophet_val['Competitor Pricing'].fillna(val_prophet_df['Competitor Pricing'].mean())


        prophet_forecast = prophet_model.predict(future_prophet_val)

        prophet_predictions_aligned = prophet_forecast.set_index('ds')['yhat']
        prophet_actual_aligned = val_prophet_df.set_index('ds')['y']

        common_dates_prophet = prophet_actual_aligned.index.intersection(prophet_predictions_aligned.index)
        prophet_predictions = prophet_predictions_aligned[common_dates_prophet].values
        prophet_actuals = prophet_actual_aligned[common_dates_prophet].values
        print(f"Prophet predictions shape: {prophet_predictions.shape}")
        print(f"Prophet actuals shape: {prophet_actuals.shape}")


        # --- 4. Calculate MAE for Each Model ---
        print("\nCalculating MAE for individual models on validation set...")

        mae_xgboost = mean_absolute_error(y_val_actual_xgboost, xgboost_predictions)
        print(f"XGBoost Validation MAE: {mae_xgboost:.2f}")

        mae_lstm = float('inf')
        if lstm_predictions.size > 0:
            mae_lstm = mean_absolute_error(y_val_actual_lstm_aligned, lstm_predictions)
            print(f"LSTM Validation MAE: {mae_lstm:.2f}")
        else:
            print("LSTM MAE not calculated due to insufficient validation data for sequences.")

        mae_prophet = mean_absolute_error(prophet_actuals, prophet_predictions)
        print(f"Prophet Validation MAE: {mae_prophet:.2f}")


        # --- 5. Determine Weights (Inverse Proportional to MAE) ---
        inv_mae_xgboost = 1 / mae_xgboost if mae_xgboost > 0 else 0
        inv_mae_lstm = 1 / mae_lstm if mae_lstm > 0 else 0
        inv_mae_prophet = 1 / mae_prophet if mae_prophet > 0 else 0

        total_inv_mae = inv_mae_xgboost + inv_mae_lstm + inv_mae_prophet

        if total_inv_mae == 0:
            w_xgboost, w_lstm, w_prophet = 1/3, 1/3, 1/3
            print("Warning: All inverse MAEs are zero. Using equal weights.")
        else:
            w_xgboost = inv_mae_xgboost / total_inv_mae
            w_lstm = inv_mae_lstm / total_inv_mae
            w_prophet = inv_mae_prophet / total_inv_mae


        ensemble_weights = {
            "xgboost": w_xgboost,
            "lstm": w_lstm,
            "prophet": w_prophet
        }
        print("\nCalculated Ensemble Weights:")
        print(f"XGBoost Weight: {w_xgboost:.4f}")
        print(f"LSTM Weight: {w_lstm:.4f}")
        print(f"Prophet Weight: {w_prophet:.4f}")
        print(f"Sum of Weights: {w_xgboost + w_lstm + w_prophet:.4f}")

        os.makedirs(os.path.dirname(weights_save_path), exist_ok=True) # Ensure directory exists
        with open(weights_save_path, 'w') as f:
            json.dump(ensemble_weights, f, indent=4)
        print(f"Ensemble weights saved to {weights_save_path}")

        # --- 6. Create Ensemble Forecast (for the validation set) ---
        print("\nCreating Ensemble Forecast (for daily total sales on validation set)...")

        val_df['xgboost_pred'] = xgboost_predictions
        xgboost_daily_predictions = val_df.groupby('Date')['xgboost_pred'].sum().reindex(common_dates_prophet)
        xgboost_daily_predictions = xgboost_daily_predictions.fillna(0).values

        print("Note: LSTM predictions could not be reliably aggregated to daily totals for direct ensemble sum due to different granularity.")
        print("Ensemble will only directly combine XGBoost and Prophet predictions for the final forecast, but LSTM's weight still influences the overall balance.")

        sum_inv_mae_xg_prophet = inv_mae_xgboost + inv_mae_prophet
        if sum_inv_mae_xg_prophet > 0:
            final_w_xgboost = inv_mae_xgboost / sum_inv_mae_xg_prophet
            final_w_prophet = inv_mae_prophet / sum_inv_mae_xg_prophet
        else:
            final_w_xgboost, final_w_prophet = 0.5, 0.5

        final_ensemble_predictions = (final_w_xgboost * xgboost_daily_predictions +
                                      final_w_prophet * prophet_predictions)

        ensemble_mae = mean_absolute_error(prophet_actuals, final_ensemble_predictions)
        print(f"Ensemble (XGBoost + Prophet) Validation MAE: {ensemble_mae:.2f}")


    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e}. Please ensure all models and processed data exist.")
    except Exception as e:
        print(f"An unexpected error occurred during ensemble creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_ensemble(
        PROCESSED_CSV_FILE_PATH,
        XGBOOST_MODEL_SAVE_PATH,
        LSTM_MODEL_SAVE_PATH,
        SCALER_X_SAVE_PATH,
        SCALER_Y_SAVE_PATH,
        PROPHET_MODEL_SAVE_PATH,
        ENSEMBLE_WEIGHTS_SAVE_PATH
    )

