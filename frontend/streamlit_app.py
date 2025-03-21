import streamlit as st
import requests
import pandas as pd
import io


# Backend API URL
FLASK_API_URL = "https://pred-pock-backend.onrender.com"

# Function to handle login
def login():
    st.title("Login")

    with st.form(key="login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Log In")

        if login_button:
            # Send the login data to Flask backend for Firebase Authentication
            login_data = {"email": email, "password": password}
            response = requests.post(f"{FLASK_API_URL}/login", json=login_data)

            if response.status_code == 200:
                st.success(f"Login successful! Welcome, {email}")
                st.session_state.logged_in = True
                st.session_state.email = email
                st.experimental_rerun()  # Re-render to go to the home page
            else:
                st.error(f"Error: {response.json().get('error')}")

    if st.button("Don't have an account? Sign Up"):
        st.session_state.page = "signup"
        st.experimental_rerun()

# Function to handle signup
def signup():
    st.title("Sign Up")

    with st.form(key="signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        signup_button = st.form_submit_button("Sign Up")

        if signup_button:
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                signup_data = {"email": email, "password": password}
                response = requests.post(f"{FLASK_API_URL}/signup", json=signup_data)

                if response.status_code == 200:
                    st.success(f"Signup successful! You can now log in with your email.")
                    st.session_state.page = "login"
                    st.experimental_rerun()
                else:
                    st.error(f"Error: {response.json().get('error')}")

    if st.button("Already have an account? Log In"):
        st.session_state.page = "login"
        st.experimental_rerun()

def get_transactions():
    st.write("Fetching your transactions...")
    response = requests.post(f"{FLASK_API_URL}/get_transactions", json={"email": st.session_state.email})

    if response.status_code == 200:
        transactions = response.json().get("transactions")
        return transactions
    else:
        st.error(f"Error: {response.json().get('error')}")
        return []

def predict_expenses(transactions):
    st.write("Predicting your next month's expenses...")
    response = requests.post(f"{FLASK_API_URL}/predict_expenses", json={"transactions": transactions})
    
    # DEBUG: Print raw response before parsing JSON
    st.write(f"Response Status Code: {response.status_code}")
    st.write(f"Raw Response Text: {response.text}")

    if response.status_code == 200:
        try:
            predictions = response.json().get("predictions")
            if predictions:
                for category, prediction in predictions.items():
                    st.write(f"Predicted expense for {category} next month: ${float(prediction):.2f}")
            else:
                st.warning("No predictions available. Ensure you have sufficient transaction data.")
        except requests.exceptions.JSONDecodeError:
            st.error("Error decoding JSON response. Check the backend response format.")
    else:
        st.error(f"Error: {response.text}")

# Home page after login
def home_page():
    st.title(f"Welcome, {st.session_state.email}!")
    st.write("You are now logged in!")
    st.write("You can now perform any transactions or changes.")

    # File upload functionality
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the CSV data into a DataFrame
        try:
            df = pd.read_csv(uploaded_file)
            st.write("CSV Preview:")
            st.write(df)

            # Upload data to Firestore when the user confirms
            if st.button("Upload to Firestore"):
                # Send the CSV data to Flask backend for processing
                csv_data = df.to_dict(orient="records")  # Convert DataFrame to a list of dicts
                upload_data = {
                    "email": st.session_state.email,
                    "csv_data": csv_data
                }
                response = requests.post(f"{FLASK_API_URL}/upload_csv", json=upload_data)

                if response.status_code == 200:
                    st.success("CSV data uploaded to Firestore successfully!")
                else:
                    st.error(f"Error: {response.json().get('error')}")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

    # Predict expenses functionality
    if st.button("Predict Next Month's Expenses"):
        transactions = get_transactions()
        if transactions:
            predict_expenses(transactions)

    # Logout functionality
    if st.button("Logout"):
        # Clear session state
        st.session_state.logged_in = False
        st.session_state.email = ""
        st.session_state.page = "login"
        st.experimental_rerun()

# Main entry point
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "login"
    if "email" not in st.session_state:
        st.session_state.email = ""

    if st.session_state.logged_in:
        home_page()
    else:
        if st.session_state.page == "login":
            login()
        elif st.session_state.page == "signup":
            signup()

if __name__ == '__main__':
    main()
