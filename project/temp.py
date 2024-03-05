import streamlit as st
import pandas as pd
import joblib

loaded_model = joblib.load("C:/Users/chait/Downloads/project/train_model (1).joblib")

# Function to preprocess user input and make prediction
def predict_loan_approval(applicant_data):
    # Preprocess user input similar to the training data
    applicant_data.replace({'education': {'Not Graduate': 0, 'Graduate': 1},
                            'self_employed': {'No': 0, 'Yes': 1},
                            'loan_status': {'Not Approved': 0, 'Approved': 1}}, inplace=True)

    # Select relevant features and order them
    selected_features = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
                         'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                         'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value',
                         'Pyschometric_Analysis']

    # Ensure the order of features matches the order used during training
    applicant_data = applicant_data[selected_features]

    # Make prediction using the loaded model
    prediction = loaded_model.predict(applicant_data.values)

    return prediction[0]


# Streamlit UI
st.title("Loan Approval Prediction App")

# Get user input
no_of_dependents = st.slider("Number of Dependents", 0, 4, 0)
education = st.selectbox("Education", ["Not Graduate", "Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
income_annum = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Amount Term")
cibil_score = st.number_input("CIBIL Score")
residential_assets_value = st.number_input("Residential Assets Value")
commercial_assets_value = st.number_input("Commercial Assets Value")
luxury_assets_value = st.number_input("Luxury Assets Value")
bank_asset_value = st.number_input("Bank Asset Value")
psychometric_analysis = st.number_input("Psychometric Analysis")

# Create a dictionary with user input
user_input = {'no_of_dependents': no_of_dependents, 'education': education,
              'self_employed': self_employed, 'income_annum': income_annum,
              'loan_amount': loan_amount, 'loan_term': loan_term,
              'cibil_score': cibil_score, 'residential_assets_value': residential_assets_value,
              'commercial_assets_value': commercial_assets_value,
              'luxury_assets_value': luxury_assets_value, 'bank_asset_value': bank_asset_value,
              'Pyschometric_Analysis': psychometric_analysis}

# Convert user input into a DataFrame
user_data = pd.DataFrame([user_input])

# Button to trigger prediction
# Button to trigger prediction
if st.button("Submit"):
    # Make prediction
    prediction = predict_loan_approval(user_data)
    
    # Display result and corresponding image or GIF
    st.write(f"The loan is {'Approved' if prediction == 1 else 'Not Approved'}")
    
    if prediction == 1:
        st.image('http://surl.li/rfzdl', caption='Loan Approved', use_column_width=True)
    else:
        st.image('http://surl.li/rfzdl', caption='Loan Not Approved', use_column_width=True)

