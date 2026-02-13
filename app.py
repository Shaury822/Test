
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the trained model and feature columns ---
rf_regressor = joblib.load('random_forest_regressor_model_reduced.pkl') # Load the reduced model
model_features = joblib.load('model_features.pkl')

# --- 2. Define unique values for dropdowns (hardcoded as they were derived from the original df) ---
unique_job_titles = ['AI Developer', 'AI Engineer', 'AI Research Engineer', 'Analytics Engineer', 'Applied Scientist', 'Applied Machine Learning Scientist', 'Associate Data Scientist', 'Big Data Engineer', 'Big Data Architect', 'Business Analyst', 'Business Intelligence Developer', 'Cloud Data Engineer', 'Computer Vision Engineer', 'Computer Vision Software Engineer', 'Data Analyst', 'Data Architect', 'Data Engineer', 'Data Integration Engineer', 'Data Quality Analyst', 'Data Scientist', 'Data Strategist', 'Deep Learning Engineer', 'Deep Learning Researcher', 'ETL Developer', 'Financial Data Analyst', 'Lead Data Analyst', 'Lead Data Engineer', 'Lead Data Scientist', 'Machine Learning Engineer', 'Machine Learning Infrastructure Engineer', 'Machine Learning Operations Engineer', 'Machine Learning Scientist', 'ML Engineer', 'NLP Engineer', 'Power BI Developer', 'Principal Data Analyst', 'Principal Data Engineer', 'Principal Data Scientist', 'Research Scientist', 'Robotics Engineer', 'Software Data Engineer']
unique_locations = ['Australia', 'Canada', 'Germany', 'India', 'USA']

# --- Streamlit App Layout ---
st.set_page_config(page_title='Data Science Salary Predictor', layout='centered')
st.title('📊 Data Science Salary Predictor')
st.write("Enter details below to predict a data science job salary.")

# --- Input Widgets ---
with st.sidebar:
    st.header("Job Details")

    selected_job_title = st.selectbox(
        'Job Title',
        unique_job_titles
    )

    selected_experience_level = st.selectbox(
        'Experience Level',
        ['Entry', 'Mid', 'Senior', 'Lead']
    )

    selected_location = st.selectbox(
        'Location',
        unique_locations
    )

    selected_company_size = st.selectbox(
        'Company Size',
        ['Small', 'Medium', 'Large']
    )

    selected_employment_type = st.selectbox(
        'Employment Type',
        ['Full-time', 'Part-time', 'Contract']
    )

    selected_remote_ratio = st.selectbox(
        'Remote Ratio (%)',
        [0, 50, 100]
    )

    selected_years_experience = st.slider(
        'Years of Experience',
        min_value=0,
        max_value=30,
        value=5
    )

# --- Prediction Button ---
if st.button('Predict Salary'):
    # --- Preprocessing User Input ---
    user_data = {
        'remote_ratio': selected_remote_ratio,
        'years_experience': selected_years_experience,
        'job_title': selected_job_title,
        'experience_level': selected_experience_level,
        'location': selected_location,
        'company_size': selected_company_size,
        'employment_type': selected_employment_type
    }

    # Create a DataFrame from user input
    user_df = pd.DataFrame([user_data])

    # Initialize a DataFrame with all model_features columns and zeros
    processed_input = pd.DataFrame(0, index=[0], columns=model_features)

    # Populate the `processed_input` DataFrame
    processed_input['remote_ratio'] = user_df['remote_ratio'][0]
    processed_input['years_experience'] = user_df['years_experience'][0]

    col_name = f"job_title_{user_df['job_title'][0]}"
    if col_name in processed_input.columns:
        processed_input[col_name] = 1

    col_name = f"experience_level_{user_df['experience_level'][0]}"
    if col_name in processed_input.columns:
        processed_input[col_name] = 1

    col_name = f"location_{user_df['location'][0]}"
    if col_name in processed_input.columns:
        processed_input[col_name] = 1

    col_name = f"company_size_{user_df['company_size'][0]}"
    if col_name in processed_input.columns:
        processed_input[col_name] = 1

    col_name = f"employment_type_{user_df['employment_type'][0]}"
    if col_name in processed_input.columns:
        processed_input[col_name] = 1

    # --- Make Prediction ---
    predicted_salary = rf_regressor.predict(processed_input)[0]

    # --- Display Prediction ---
    st.success(f"Predicted Salary: ${predicted_salary:,.2f}")
