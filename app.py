import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

def load_transformer(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def apply_categorical_mappings(df):
    # Define mappings
    travel_mapping = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    department_mapping = {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2}
    education_field_mapping = {'Other': 0, 'Life Sciences': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5}
    gender_mapping = {'Female': 0, 'Male': 1}
    job_role_mapping = {
        'Laboratory Technician': 0, 'Research Scientist': 1, 'Sales Executive': 2, 'Sales Representative': 3,
        'Human Resources': 4, 'Healthcare Representative': 5, 'Manufacturing Director': 6, 'Manager': 7, 'Research Director': 8
    }
    marital_status_mapping = {'Single': 0, 'Married': 1, 'Divorced': 2}
    overtime_mapping = {'No': 0, 'Yes': 1}

    # Apply mappings
    mappings = {
        'BusinessTravel': travel_mapping,
        'Department': department_mapping,
        'EducationField': education_field_mapping,
        'Gender': gender_mapping,
        'JobRole': job_role_mapping,
        'MaritalStatus': marital_status_mapping,
        'OverTime': overtime_mapping
    }

    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)

    return df

def preprocess_data(df, continuous_columns, min_max_scaler, standard_scaler, normalizer):
    df_continuous = df[continuous_columns]

    df_rescaled = pd.DataFrame(min_max_scaler.transform(df_continuous), columns=continuous_columns)
    df_standardized = pd.DataFrame(standard_scaler.transform(df_rescaled), columns=continuous_columns)
    df_normalized = pd.DataFrame(normalizer.transform(df_standardized), columns=continuous_columns)

    df[continuous_columns] = df_normalized

    return df

def main():
    # Load the model and transformers
    model = joblib.load("/content/random_forest_model_for_attrition.joblib")
    min_max_scaler = load_transformer("/content/transformers/min_max_scaler.pkl")
    standard_scaler = load_transformer("/content/transformers/standard_scaler.pkl")
    normalizer = load_transformer("/content/transformers/normalizer.pkl")

    # Title of the application
    st.title('Employee Attrition Prediction')

    # Create sidebar for input features
    st.sidebar.header('Input Features')

    # Collect user inputs
    input_data = {
        'Age': st.sidebar.slider('Age', 18, 60, 36),
        'BusinessTravel': st.sidebar.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel']),
        'DailyRate': st.sidebar.slider('Daily Rate', 102, 1499, 802),
        'Department': st.sidebar.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources']),
        'DistanceFromHome': st.sidebar.slider('Distance from Home', 1, 29, 9),
        'Education': st.sidebar.slider('Education', 1, 5, 3),
        'EducationField': st.sidebar.selectbox('Education Field', ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources']),
        'EmployeeCount': st.sidebar.slider('Employee Count', 1, 10, 1),  # Fixed value
        'EmployeeNumber': st.sidebar.slider('Employee Number', 1, 2068, 1024),
        'EnvironmentSatisfaction': st.sidebar.slider('Environment Satisfaction', 1, 4, 3),
        'Gender': st.sidebar.selectbox('Gender', ['Female', 'Male']),
        'HourlyRate': st.sidebar.slider('Hourly Rate', 30, 100, 66),
        'JobInvolvement': st.sidebar.slider('Job Involvement', 1, 4, 3),
        'JobLevel': st.sidebar.slider('Job Level', 1, 5, 2),
        'JobRole': st.sidebar.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']),
        'JobSatisfaction': st.sidebar.slider('Job Satisfaction', 1, 4, 3),
        'MaritalStatus': st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced']),
        'MonthlyIncome': st.sidebar.slider('Monthly Income', 1000, 20000, 6500),
        'MonthlyRate': st.sidebar.slider('Monthly Rate', 2000, 27000, 14300),
        'NumCompaniesWorked': st.sidebar.slider('Num Companies Worked', 0, 9, 3),
        'OverTime': st.sidebar.selectbox('Over Time', ['Yes', 'No']),
        'PercentSalaryHike': st.sidebar.slider('Percent Salary Hike', 11, 25, 15),
        'PerformanceRating': st.sidebar.slider('Performance Rating', 3, 4, 3),
        'RelationshipSatisfaction': st.sidebar.slider('Relationship Satisfaction', 1, 4, 3),
        'StandardHours': st.sidebar.slider('Standard Hours', 80, 90, 80),  # Fixed value
        'StockOptionLevel': st.sidebar.slider('Stock Option Level', 0, 3, 1),
        'TotalWorkingYears': st.sidebar.slider('Total Working Years', 0, 40, 11),
        'TrainingTimesLastYear': st.sidebar.slider('Training Times Last Year', 0, 6, 3),
        'WorkLifeBalance': st.sidebar.slider('Work Life Balance', 1, 4, 3),
        'YearsAtCompany': st.sidebar.slider('Years At Company', 0, 40, 7),
        'YearsInCurrentRole': st.sidebar.slider('Years In Current Role', 0, 18, 4),
        'YearsSinceLastPromotion': st.sidebar.slider('Years Since Last Promotion', 0, 15, 2),
        'YearsWithCurrManager': st.sidebar.slider('Years With Current Manager', 0, 17, 4)
    }

    input_df = pd.DataFrame([input_data])

    # Apply categorical mappings
    input_df = apply_categorical_mappings(input_df)

    # Define the continuous columns
    continuous_columns = [
        'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
        'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
        'YearsWithCurrManager'
    ]

    # Preprocess the data
    input_df = preprocess_data(input_df, continuous_columns, min_max_scaler, standard_scaler, normalizer)

    # Prediction button
    if st.button('Predict Attrition'):
        # Predict the class and the confidence
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Map the prediction to the attrition status
        attrition_status = 'Yes' if prediction[0] == 1 else 'No'

        st.write(f'Predicted Attrition: {attrition_status}')
        st.write(f'Confidence Score: {prediction_proba[0][prediction[0]]:.2f}')

if __name__ == '__main__':
    main()
