import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB Atlas connection details from .env
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("MONGO_DB_NAME")
COLLECTION_NAME = "forecast_data_processed"
# Define the path to your PROCESSED CSV file
CSV_FILE_PATH = os.path.join("data", "processed", "retail_inventory_forecast_processed.csv")

def load_csv_to_mongodb(csv_file, mongo_uri, db_name, collection_name):
    """
    Loads data from a CSV file into a MongoDB collection.
    """
    if not mongo_uri or not db_name:
        print("Error: MONGO_URI or MONGO_DB_NAME not found in .env file.")
        print("Please ensure your .env file is correctly configured in the backend directory.")
        return

    try:
        # Connect to MongoDB Atlas
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        print(f"Connecting to MongoDB Atlas...")
        print(f"Using database: {db_name}, collection: {collection_name}")
        print(f"Loading data from processed CSV: {csv_file}...")

        # Read the CSV file using pandas
        df = pd.read_csv(csv_file)

        # Convert 'Date' column to datetime objects for proper storage in MongoDB
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True) # Drop rows where Date conversion failed

        # Convert DataFrame to a list of dictionaries (JSON format)
        data = df.to_dict(orient='records')

        # Clear existing data in the collection before inserting new data
        print(f"Clearing existing data in the collection '{collection_name}' (if any)...")
        collection.delete_many({})

        # Insert data into MongoDB
        print(f"Inserting {len(data)} documents into MongoDB...")
        if data: # Only insert if there's data to insert
            collection.insert_many(data)
            print("Processed data loaded successfully!")
        else:
            print("No data to insert after processing CSV.")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found. Please ensure it's in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    finally:
        if 'client' in locals() and client:
            client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    load_csv_to_mongodb(CSV_FILE_PATH, MONGO_URI, DATABASE_NAME, COLLECTION_NAME)

