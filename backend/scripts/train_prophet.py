import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import joblib # For saving the Prophet model
import os

# Define paths
PROCESSED_CSV_FILE_PATH = os.path.join("data", "processed", "retail_inventory_forecast_processed.csv")
PROPHET_MODEL_SAVE_PATH = os.path.join("models", "prophet_model.joblib")

def train_prophet_model(input_csv_path, model_save_path):
    """
    Loads processed data, trains a Prophet model for demand forecasting,
    evaluates it, and saves the trained model.
    """
    print(f"Starting Prophet model training using data from {input_csv_path}...")

    try:
        # Load the processed dataset
        df = pd.read_csv(input_csv_path)
        print(f"Loaded dataset shape: {df.shape}")

        # Ensure 'Date' column is datetime and sort for time-series processing
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by='Date', inplace=True)

        # Aggregate Units Sold by Date for a global forecast (simplification for demonstration)
        df_prophet = df.groupby('Date')['Units Sold'].sum().reset_index()
        df_prophet.rename(columns={'Date': 'ds', 'Units Sold': 'y'}, inplace=True)

        print(f"Prophet aggregated dataset shape: {df_prophet.shape}")

        # Create a dataframe for holidays for Prophet
        holidays_df = df[df['Holiday/Promotion'] == 1][['Date']].drop_duplicates()
        holidays_df.rename(columns={'Date': 'ds'}, inplace=True)
        holidays_df['holiday'] = 'promotion'
        holidays_df['lower_window'] = 0
        holidays_df['upper_window'] = 0
        print(f"Created {len(holidays_df)} holiday entries for Prophet.")


        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            holidays=holidays_df
        )

        # Add 'Competitor Pricing' as an extra regressor
        df_prophet_regressors = df.groupby('Date')['Competitor Pricing'].mean().reset_index()
        df_prophet_regressors.rename(columns={'Date': 'ds'}, inplace=True)
        df_prophet = pd.merge(df_prophet, df_prophet_regressors, on='ds', how='left')

        if 'Competitor Pricing' in df_prophet.columns:
            df_prophet['Competitor Pricing'] = df_prophet['Competitor Pricing'].fillna(df_prophet['Competitor Pricing'].mean())
            model.add_regressor('Competitor Pricing')
            print("Added 'Competitor Pricing' as an extra regressor.")
        else:
            print("Warning: 'Competitor Pricing' not found in aggregated data, skipping as regressor.")


        # Split data chronologically for Prophet
        split_date = df_prophet['ds'].iloc[int(len(df_prophet) * 0.8)]
        train_prophet_df = df_prophet[df_prophet['ds'] <= split_date]
        val_prophet_df = df_prophet[df_prophet['ds'] > split_date]

        print(f"Prophet Training set size: {len(train_prophet_df)} samples")
        print(f"Prophet Validation set size: {len(val_prophet_df)} samples")

        # Train the Prophet model
        print("Training Prophet model...")
        model.fit(train_prophet_df)
        print("Prophet model training complete.")

        # Make future dataframe for validation period
        future = model.make_future_dataframe(periods=len(val_prophet_df), include_history=False)

        if 'Competitor Pricing' in df_prophet.columns:
            future = pd.merge(future, df_prophet[['ds', 'Competitor Pricing']], on='ds', how='left')
            future['Competitor Pricing'] = future['Competitor Pricing'].fillna(df_prophet['Competitor Pricing'].mean())


        # Generate predictions
        forecast = model.predict(future)

        # Evaluate the model on the validation set
        print("Evaluating Prophet model on validation set...")
        prophet_val_actual = val_prophet_df.set_index('ds')['y']
        prophet_val_predictions = forecast.set_index('ds')['yhat']

        common_dates = prophet_val_actual.index.intersection(prophet_val_predictions.index)
        aligned_actual = prophet_val_actual[common_dates]
        aligned_predictions = prophet_val_predictions[common_dates]

        prophet_mae = mean_absolute_error(aligned_actual, aligned_predictions)
        print(f"Prophet Validation Mean Absolute Error (MAE): {prophet_mae:.2f}")

        # Save the trained model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure directory exists
        joblib.dump(model, model_save_path)
        print(f"Prophet model saved to {model_save_path}")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found. Please ensure it's in the 'backend' directory and data_preprocessing.py was run.")
    except Exception as e:
        print(f"An unexpected error occurred during Prophet training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_prophet_model(PROCESSED_CSV_FILE_PATH, PROPHET_MODEL_SAVE_PATH)

