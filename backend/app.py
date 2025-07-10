import os
from flask import Flask, jsonify, g # Import g for application context
from pymongo import MongoClient
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB Atlas connection details from .env
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

def get_mongo_client():
    """
    Returns the MongoDB client instance.
    Initializes it if it doesn't exist in the current application context (g object).
    This ensures the client is created once per request/process.
    """
    # Check if the client is already stored in the application context's 'g' object
    if 'mongo_client' not in g:
        if not MONGO_URI or not MONGO_DB_NAME:
            # Raise an error if environment variables are missing
            raise RuntimeError("MONGO_URI or MONGO_DB_NAME not found in .env file.")
        try:
            # Create a new MongoClient and store it in g
            g.mongo_client = MongoClient(MONGO_URI)
            # Attempt to ping the database to check connection immediately
            g.mongo_client.admin.command('ping')
            print("MongoDB connection established successfully within application context.")
        except Exception as e:
            # If connection fails, ensure any partial client is closed and raise an error
            print(f"Failed to connect to MongoDB: {e}")
            if hasattr(g, 'mongo_client') and g.mongo_client:
                g.mongo_client.close()
            # Remove the client from g to indicate a failed connection
            if 'mongo_client' in g:
                del g.mongo_client
            raise RuntimeError(f"Database connection failed: {e}")
    return g.mongo_client

# Teardown function to close the MongoDB client when the application context ends
# This is crucial for releasing resources, especially with the Flask reloader.
@app.teardown_appcontext
def close_mongo_client(exception):
    """
    Closes the MongoDB client if it was opened during the request/context.
    This function is registered to run when the application context is torn down.
    """
    # Pop the 'mongo_client' from g; if it doesn't exist, default to None
    mongo_client = g.pop('mongo_client', None)
    if mongo_client:
        mongo_client.close()
        print("MongoDB connection closed for this context.")

# Basic root route for confirmation
@app.route('/', methods=['GET'])
def home():
    db_status = "disconnected"
    try:
        # Attempt to get the client to check connection status
        get_mongo_client()
        db_status = "connected"
    except RuntimeError as e:
        print(f"Home route DB status check failed: {e}")
        db_status = "disconnected" # Explicitly set status on failure
    return jsonify({
        "message": "Walmart Inventory Backend is running. No specific API routes defined yet.",
        "db_status": db_status
    }), 200

# Basic health check route (simplified)
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # Get the MongoDB client; this will attempt to connect if not already connected
        client = get_mongo_client()
        # Ping the database to ensure it's responsive
        client.admin.command('ping')
        return jsonify({"status": "ok", "database": "connected"}), 200
    except RuntimeError as e:
        # Catch the RuntimeError raised by get_mongo_client if connection fails
        return jsonify({"status": "error", "database": "disconnected", "details": str(e)}), 500
    except Exception as e:
        # Catch any other unexpected exceptions
        return jsonify({"status": "error", "database": "disconnected", "details": "Unexpected error: " + str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    # If you continue to experience "OSError: [WinError 10038]",
    # you can try disabling the reloader for development:
    # app.run(debug=True, port=5000, use_reloader=False)
    # However, the current changes should help manage the client more gracefully.
    app.run(debug=True, port=5000)

