import pandas as pd
import numpy as np
import os

# Define the path to your raw CSV file
CSV_FILE_PATH = os.path.join("data", "raw", "retail_inventory_forecast.csv")
# Define the path where the processed CSV will be saved
PROCESSED_CSV_FILE_PATH = os.path.join("data", "processed", "retail_inventory_forecast_processed.csv")

def clean_and_feature_engineer_data(input_csv_path, output_csv_path):
    """
    Loads the raw inventory data, cleans it, creates new features,
    and saves the processed data to a new CSV file.
    """
    print(f"Starting data preprocessing for {input_csv_path}...")

    try:
        # Load the dataset
        df = pd.read_csv(input_csv_path)
        print(f"Original dataset shape: {df.shape}")

        # --- Debugging Step: Print columns to verify ---
        print("\nColumns in the loaded DataFrame (before standardization):")
        print(df.columns.tolist())
        print("-" * 30)

        # --- Column Name Standardization ---
        column_mapping = {
            'C Holiday/Pr': 'Holiday/Promotion', # Standardize this name
            'C Holiday/Pr ': 'Holiday/Promotion',
            'C Holiday/Promo': 'Holiday/Promotion',
            'Competitio': 'Competitor Pricing',   # Standardize this name
            'Competition': 'Competitor Pricing',
            'Competitor': 'Competitor Pricing',
            'Competitio ': 'Competitor Pricing'
        }

        new_columns = []
        for col in df.columns:
            found_mapping = False
            for old_name, new_name in column_mapping.items():
                if col.strip() == old_name.strip():
                    new_columns.append(new_name)
                    found_mapping = True
                    break
            if not found_mapping:
                new_columns.append(col)

        df.columns = new_columns
        print("\nColumns in the DataFrame (after standardization):")
        print(df.columns.tolist())
        print("-" * 30)


        # --- 1. Data Cleaning ---
        print("Cleaning data: Handling missing values and outliers...")

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        initial_rows = df.shape[0]
        df.dropna(subset=['Date'], inplace=True)
        if df.shape[0] < initial_rows:
            print(f"Dropped {initial_rows - df.shape[0]} rows due to invalid 'Date' format.")

        df['Discount'] = df['Discount'].fillna(0)
        print("Missing 'Discount' values filled with 0.")

        if 'Competitor Pricing' in df.columns: # Use the standardized name
            if df['Competitor Pricing'].isnull().any():
                mean_competitio = df['Competitor Pricing'].mean()
                df['Competitor Pricing'] = df['Competitor Pricing'].fillna(mean_competitio)
                print(f"Missing 'Competitor Pricing' values filled with mean: {mean_competitio:.2f}.")
            else:
                print("No missing 'Competitor Pricing' values found.")
        else:
            print("Warning: 'Competitor Pricing' column not found after standardization. Skipping fillna for this column.")

        numerical_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price'] # Updated to match new column names
        for col in numerical_cols:
            if col in df.columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(0)
                    print(f"Missing '{col}' values filled with 0.")
            else:
                print(f"Warning: '{col}' column not found. Skipping fillna for this column.")


        # --- 2. Feature Creation ---
        print("Creating new features...")

        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
            df['month'] = df['Date'].dt.month
            df['year'] = df['Date'].dt.year
            print("Extracted day_of_week, week_of_year, month, year from 'Date'.")
        else:
            print("Error: 'Date' column is not in datetime format. Cannot extract time-based features.")


        if all(col in df.columns for col in ['Store ID', 'Product ID', 'Date']):
            df.sort_values(by=['Store ID', 'Product ID', 'Date'], inplace=True)
            print("Data sorted by 'Store ID', 'Product ID', 'Date'.")
        else:
            print("Warning: Missing 'Store ID', 'Product ID', or 'Date' columns. Skipping sorting for lag/rolling features.")


        if 'Units Sold' in df.columns and 'Store ID' in df.columns and 'Product ID' in df.columns:
            print("Creating lag features for 'Units Sold'...")
            df['units_sold_lag_1'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(1)
            df['units_sold_lag_7'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(7)
            df['units_sold_lag_30'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(30)
            df['units_sold_lag_1'] = df['units_sold_lag_1'].fillna(0)
            df['units_sold_lag_7'] = df['units_sold_lag_7'].fillna(0)
            df['units_sold_lag_30'] = df['units_sold_lag_30'].fillna(0)
            print("Created units_sold_lag_1, units_sold_lag_7, units_sold_lag_30.")
        else:
            print("Warning: Missing 'Units Sold', 'Store ID', or 'Product ID' columns. Skipping lag feature creation.")


        if 'Units Sold' in df.columns and 'Store ID' in df.columns and 'Product ID' in df.columns:
            print("Creating rolling average features for 'Units Sold'...")
            df['units_sold_rolling_avg_7'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            df['units_sold_rolling_avg_30'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].transform(
                lambda x: x.rolling(window=30, min_periods=1).mean()
            )
            print("Created units_sold_rolling_avg_7, units_sold_rolling_avg_30.")
        else:
            print("Warning: Missing 'Units Sold', 'Store ID', or 'Product ID' columns. Skipping rolling average creation.")


        # --- Save Processed Data ---
        print(f"Saving processed data to {output_csv_path}...")
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Processed dataset shape: {df.shape}")
        print("Data preprocessing complete!")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found. Please ensure it's in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred during data preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    clean_and_feature_engineer_data(CSV_FILE_PATH, PROCESSED_CSV_FILE_PATH)

