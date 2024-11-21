import pandas as pd
from joblib import load

model_age_above_25 = load('artifacts/model_age_above_25.joblib')
model_age_under_25 = load('artifacts/model_age_under_25.joblib')

scaler_age_above_25 = load('artifacts/scaler_age_above_25.joblib')
scaler_age_under_25 = load('artifacts/scaler_age_under_25.joblib')

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }

    diseases = medical_history.lower().split(" & ")

    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  

    max_score = 14 
    min_score = 0  

    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score


def handle_scaling(age, df):

    if age <= 25:
        scaler_object = scaler_age_under_25
    else:
        scaler_object = scaler_age_above_25

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None 
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    mappings = {
        'Gender': {'Male': 'gender_Male'},
        'Region': {'Northwest': 'region_Northwest', 'Southeast': 'region_Southeast', 'Southwest': 'region_Southwest'},
        'Marital Status': {'Unmarried': 'marital_status_Unmarried'},
        'BMI Category': {
            'Obesity': 'bmi_category_Obesity',
            'Overweight': 'bmi_category_Overweight',
            'Underweight': 'bmi_category_Underweight'
        },
        'Smoking Status': {
            'Occasional': 'smoking_status_Occasional',
            'Regular': 'smoking_status_Regular'
        },
        'Employment Status': {
            'Salaried': 'employment_status_Salaried',
            'Self-Employed': 'employment_status_Self-Employed'
        },
    }
    
    for key, value in input_dict.items():
        if key in mappings:  
            if value in mappings[key]:
                df[mappings[key][value]] = 1
        elif key == 'Insurance Plan': 
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key in ['Age', 'Number of Dependants', 'Income in Lakhs', 'Genetical Risk']:  
            df[key.lower().replace(' ', '_')] = value


    
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)

    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)
    if input_dict['Age'] <= 25:
        prediction = model_age_under_25.predict(input_df)
    else:
        prediction = model_age_above_25.predict(input_df)

    return int(prediction[0])