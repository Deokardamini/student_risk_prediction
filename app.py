import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(
    page_title="Student Risk Assessment",
    page_icon="🎓",
    layout="wide"
)

# Embed Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load the logistic regression model
@st.cache_resource
def load_model():
    with open("logistic.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Header
st.title("🎓 Student Academic Performance Predictor")
st.write("Enter the student details below to assess risk level.")
st.markdown("---")

# Input Form
with st.form(key="prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Academic Metrics")
        attendance = st.slider("Attendance Rate (%)", 0.0, 100.0, 85.0)
        study_hours = st.number_input("Weekly Study Hours", min_value=0.0, max_value=100.0, value=15.0)
        past_failures = st.number_input("Past Failures", min_value=0, max_value=10, value=0)
        assignments_completed_pct = st.slider("Assignments Completed (%)", 0.0, 100.0, 90.0)
        previous_grade = st.number_input("Previous Grade (%)", min_value=0.0, max_value=100.0, value=75.0)
        final_score = st.number_input("Final Score (%)", min_value=0.0, max_value=100.0, value=80.0)

    with col2:
        st.subheader("🏠 Personal & Socioeconomic Factors")
        parental_education = st.selectbox(
            "Parental Education Level",
            options=["None", "High School", "Bachelor's", "Master's", "Doctorate"]
        )
        family_income = st.selectbox(
            "Family Income Level",
            options=["Low", "Medium", "High"]
        )
        extracurricular = st.selectbox(
            "Extracurricular Activities",
            options=["No", "Yes"]
        )
        internet_access = st.selectbox(
            "Internet Access at Home",
            options=["No", "Yes"]
        )

    submit_button = st.form_submit_button(label="Predict Student Risk Level")

# Prediction Execution
if submit_button:
    # Map categorical text options to categorical codes/types if expected numerically or as object categories
    input_data = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "past_failures": past_failures,
        "assignments_completed_pct": assignments_completed_pct,
        "parental_education": pd.Categorical([parental_education], categories=["None", "High School", "Bachelor's", "Master's", "Doctorate"])[0],
        "family_income": pd.Categorical([family_income], categories=["Low", "Medium", "High"])[0],
        "extracurricular": pd.Categorical([extracurricular], categories=["No", "Yes"])[0],
        "internet_access": pd.Categorical([internet_access], categories=["No", "Yes"])[0],
        "previous_grade": previous_grade,
        "final_score": final_score
    }])

    st.markdown("---")
    
    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        
        if prediction == "At-Risk":
            st.error(f"⚠️ Assessment Status: **{prediction}**")
        elif prediction == "High-Risk":
            st.warning(f"🚨 Assessment Status: **{prediction}**")
        else:
            st.success(f"✅ Assessment Status: **{prediction}**")
            
        st.markdown("---")
        st.write("### Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": [f"{p * 100:.2f}%" for p in probabilities]
        })
        st.table(prob_df)
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("If your pickled model contains a custom preprocessor/pipeline encoding steps, ensure categorical features match the exact numerical or encoded types expected by your fitted model.")
