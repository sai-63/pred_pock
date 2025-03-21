import firebase_admin
from firebase_admin import credentials, auth, firestore
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

firebase_cred = {
    "type": os.getenv('FIREBASE_TYPE'),
    "project_id": os.getenv('FIREBASE_PROJECT_ID'),
    "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
    "private_key": os.getenv('FIREBASE_PRIVATE_KEY'),
    "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
    "client_id": os.getenv('FIREBASE_CLIENT_ID'),
    "auth_uri": os.getenv('FIREBASE_AUTH_URI'),
    "token_uri": os.getenv('FIREBASE_TOKEN_URI'),
    "auth_provider_x509_cert_url": os.getenv('FIREBASE_AUTH_PROVIDER_X509_CERT_URL'),
    "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_X509_CERT_URL'),
    "universe_domain": os.getenv('FIREBASE_UNIVERSE_DOMAIN')
}

# Initialize Firebase Admin SDK
cred = credentials.Certificate(firebase_cred)
firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client()

# Initialize a dictionary to store models for each category
category_models = {}

# Signup endpoint
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    email = data['email']
    password = data['password']
    
    try:
        # Create a new Firebase user with email and password
        user = auth.create_user(
            email=email,
            password=password
        )
        return jsonify({"message": f"User {email} signed up successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Login endpoint
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data['email']
    password = data['password']
    
    try:
        # Sign in user using Firebase Authentication
        user = auth.get_user_by_email(email)
        # For simplicity, we do not handle password verification here. In a real app, you would verify the password.
        return jsonify({"message": f"User {email} logged in successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Endpoint to upload CSV data to Firestore
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    data = request.json
    email = data['email']
    csv_data = data['csv_data']
    
    try:
        # Upload data to the transactions collection under the user's email
        transactions_ref = db.collection('transactions').document(email)
        
        # Also save individual records in the user_data collection
        for record in csv_data:
            transactions_ref.collection('user_data').add(record)
        
        return jsonify({"message": "CSV data uploaded successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Endpoint to handle logout
@app.route('/logout', methods=['POST'])
def logout():
    try:
        # Assuming Firebase doesn't need to handle session on backend, logout is client-side logic.
        # Here we can handle session invalidation if using server-side sessions.
        return jsonify({"message": "User logged out successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/get_transactions', methods=['POST'])
def get_transactions():
    data = request.json
    email = data.get('email')

    try:
        # Fetch the transactions from Firestore
        transactions_ref = db.collection('transactions').document(email).collection('user_data')
        transactions = transactions_ref.stream()

        # Convert the Firestore documents to a list of dictionaries
        transactions_list = []
        for transaction in transactions:
            transaction_dict = transaction.to_dict()
            # Normalize keys to lowercase
            transaction_dict = {key.lower(): value for key, value in transaction_dict.items()}
            transactions_list.append(transaction_dict)

        if not transactions_list:
            return jsonify({"error": "No transactions found for this user"}), 404

        return jsonify({"transactions": transactions_list}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

def train_model(transactions):
    global category_models

    if not transactions:
        return None, jsonify({"error": "No transactions data provided."}), 400

    df = pd.DataFrame(transactions)

    # Ensure columns are correct and strip spaces
    df.columns = df.columns.str.strip().str.lower()

    # Check if the required columns exist
    required_columns = ['date', 'category', 'expense']
    if not all(col in df.columns for col in required_columns):
        return None, jsonify({"error": f"Missing one or more required columns: {required_columns}"}), 400

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows where date conversion failed
    if df['date'].isnull().any():
        df = df.dropna(subset=['date'])

    # Sort the dataframe by category and date
    df = df.sort_values(by=['category', 'date'])

    # Create lag features for the last 3 months of expenses for each category
    df['expense_last_month'] = df.groupby('category')['expense'].shift(1)
    df['expense_two_months_ago'] = df.groupby('category')['expense'].shift(2)
    df['expense_three_months_ago'] = df.groupby('category')['expense'].shift(3)

    # Drop rows where there are missing values for the lag features
    df = df.dropna(subset=['expense_last_month', 'expense_two_months_ago', 'expense_three_months_ago'])

    # Debug: Print the processed dataframe
    print("Processed DataFrame:")
    print(df)

    # Loop through each unique category to train a model
    for category in df['category'].unique():
        category_data = df[df['category'] == category]

        # Skip categories with insufficient data (less than 4 data points)
        if len(category_data) < 4:
            print(f"Skipping category '{category}' due to insufficient data.")
            continue

        # Define the feature set (past 3 months' expenses) and the target variable (current month's expense)
        X = category_data[['expense_last_month', 'expense_two_months_ago', 'expense_three_months_ago']]
        y = category_data['expense']

        # Split the data into training and test sets (80% for training, 20% for testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Initialize and train the Random Forest Regressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Store the trained model in the dictionary
        category_models[category] = model
    print("Category Models in train data:", category_models)  # Debug: Log category models

    # Return the dataframe with lag features for prediction
    return df

@app.route('/predict_expenses', methods=['POST'])
def predict_expenses():
    global category_models
    data = request.get_json()

    # Check if transaction data is provided
    if 'transactions' not in data:
        return jsonify({"error": "Missing 'transactions' in request data."}), 400

    transactions = data['transactions']

    # Train the model and get the dataframe with lag features
    df = train_model(transactions)

    # Predict the next month's expense for each category
    predictions = {}
    print("Category Models:", category_models)  # Debug: Log category models

    for category in category_models:
        # Get the last 3 months' expenses for the category
        category_data = df[df['category'] == category]
        if len(category_data) == 0:
            print(f"No data available for category: {category}")
            continue  # Skip categories with no data

        last_month_expense = category_data['expense_last_month'].iloc[-1]
        two_months_ago_expense = category_data['expense_two_months_ago'].iloc[-1]
        three_months_ago_expense = category_data['expense_three_months_ago'].iloc[-1]

        # Get the model for the category
        model = category_models[category]

        # Prepare the feature set for prediction
        X_new = np.array([[last_month_expense, two_months_ago_expense, three_months_ago_expense]])

        # Predict the next month's expense
        predicted_expense = model.predict(X_new)[0]

        predictions[category] = predicted_expense

    print("Predictions:", predictions)  # Debug: Log predictions
    return jsonify({"predictions": predictions})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5000)
