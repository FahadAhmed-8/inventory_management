import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib # For saving scalers
import os

# Define paths
PROCESSED_CSV_FILE_PATH = os.path.join("data", "processed", "retail_inventory_forecast_processed.csv")
LSTM_MODEL_SAVE_PATH = os.path.join("models", "lstm_model.h5")
SCALER_X_SAVE_PATH = os.path.join("models", "scaler_X.joblib")
SCALER_Y_SAVE_PATH = os.path.join("models", "scaler_y.joblib")

# Define sequence length (how many past time steps to look back)
LOOK_BACK = 7 # Using 7 days (1 week) as a sequence length for now

def create_sequences(data, look_back):
    """
    Creates sequences of features and corresponding target values for LSTM.
    For each (Store ID, Product ID) group, creates sequences.
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        # Features are the 'look_back' preceding values
        X.append(data[i:(i + look_back), :])
        # Target is the 'Units Sold' at the next time step
        y.append(data[i + look_back, 0]) # Assuming scaled 'Units Sold' is the first column

    return np.array(X), np.array(y)

def train_lstm_model(input_csv_path, model_save_path, scaler_x_path, scaler_y_path):
    """
    Loads processed data, prepares it for LSTM, trains an LSTM model,
    evaluates it, and saves the trained model and scalers.
    """
    print(f"Starting LSTM model training using data from {input_csv_path}...")

    try:
        # Load the processed dataset
        df = pd.read_csv(input_csv_path)
        print(f"Loaded dataset shape: {df.shape}")

        # Ensure 'Date' column is datetime and sort for time-series processing
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values(by=['Store ID', 'Product ID', 'Date'], inplace=True)

        # Define features and target for LSTM
        sequence_features = [
            'Units Sold',
            'day_of_week',
            'month'
        ]

        # Check if all required features and target exist
        missing_features = [f for f in sequence_features if f not in df.columns]
        if missing_features:
            print(f"Error: Missing required features in the dataset for LSTM: {missing_features}")
            print("Please ensure your data_preprocessing.py generated these columns correctly.")
            print("Available columns in the DataFrame:")
            print(df.columns.tolist())
            return

        # Prepare data for scaling
        data_to_scale = df[sequence_features].values

        # Scale features
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler_X.fit_transform(data_to_scale)

        # We will also need a scaler for the 'Units Sold' specifically to inverse transform predictions
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_y.fit(data_to_scale[:, 0].reshape(-1, 1))

        # Save scalers for later use (e.g., during prediction)
        os.makedirs(os.path.dirname(scaler_x_path), exist_ok=True) # Ensure directory exists
        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        print(f"MinMax Scalers saved to {scaler_x_path} and {scaler_y_path}")

        # Create sequences for LSTM
        X_sequences, y_sequences = [], []
        grouped = df.groupby(['Store ID', 'Product ID'])

        for name, group in grouped:
            group_scaled_data = scaled_data[group.index] # Use original index to slice scaled_data
            if len(group_scaled_data) > LOOK_BACK:
                X_g, y_g = create_sequences(group_scaled_data, LOOK_BACK)
                X_sequences.append(X_g)
                y_sequences.append(y_g)

        if not X_sequences:
            print("Error: No sequences could be created. Check your data and LOOK_BACK value.")
            return

        X_lstm = np.vstack(X_sequences)
        y_lstm = np.hstack(y_sequences)

        print(f"Created {len(X_lstm)} LSTM sequences.")
        print(f"Input sequence shape (X): {X_lstm.shape}")
        print(f"Output target shape (y): {y_lstm.shape}")


        # Split data chronologically for time series
        split_point = int(len(X_lstm) * 0.8)
        X_train, X_val = X_lstm[:split_point], X_lstm[split_point:]
        y_train, y_val = y_lstm[:split_point], y_lstm[split_point:]

        print(f"LSTM Training set size: {len(X_train)} samples")
        print(f"LSTM Validation set size: {len(X_val)} samples")

        # --- Build LSTM Model ---
        print("Building LSTM model architecture...")
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.summary()

        # Callbacks for training
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure directory exists
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
            ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        # Train the LSTM model
        print("Training LSTM model...")
        history = model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=1)

        print("LSTM model training complete.")

        best_model = tf.keras.models.load_model(model_save_path)

        # Evaluate the best model on the validation set
        print("Evaluating best LSTM model on validation set...")
        val_predictions_scaled = best_model.predict(X_val)

        val_predictions = scaler_y.inverse_transform(val_predictions_scaled)
        y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1, 1))

        lstm_mae = mean_absolute_error(y_val_actual, val_predictions)
        print(f"LSTM Validation Mean Absolute Error (MAE): {lstm_mae:.2f}")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found. Please ensure it's in the 'backend' directory and data_preprocessing.py was run.")
    except Exception as e:
        print(f"An unexpected error occurred during LSTM training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_lstm_model(PROCESSED_CSV_FILE_PATH, LSTM_MODEL_SAVE_PATH, SCALER_X_SAVE_PATH, SCALER_Y_SAVE_PATH)

