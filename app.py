import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Inject custom CSS for custom colors and shadow effects
st.markdown("""
    <style>
    /* Main container styling */
    .reportview-container {
        background: #f5f7fa;
    }
    
    /* Card / Container styling with elegant shadow */
    .prediction-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #4A90E2;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Input widget customization */
    .stNumberInput, .stSlider {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 5px;
    }
    
    /* Title styling */
    h1 {
        color: #2C3E50;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
    }
    
    /* Custom Button styling */
    div.stButton > button:first-child {
        background-color: #4A90E2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(74, 144, 226, 0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #357ABD;
        box-shadow: 0 6px 12px rgba(74, 144, 226, 0.3);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🎓 Student Performance Predictor")
st.markdown("Enter the student metrics below to estimate their final performance index using the pre-trained KNN Regression model.")

# Load the model securely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please ensure the model file is in the same directory as this script.")
    st.stop()

# Form / Inputs Layout
st.subheader("📝 Input Metrics")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input(
        "Hours Studied", 
        min_value=0.0, 
        max_value=24.0, 
        value=5.0, 
        step=0.5,
        help="Number of hours spent studying per day."
    )
    
    attendance_percent = st.slider(
        "Attendance (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=85.0, 
        step=1.0,
        help="Percentage of classes attended."
    )

with col2:
    sleep_hours = st.number_input(
        "Sleep Hours", 
        min_value=0.0, 
        max_value=24.0, 
        value=7.0, 
        step=0.5,
        help="Average hours of sleep per night."
    )
    
    previous_scores = st.slider(
        "Previous Scores", 
        min_value=0.0, 
        max_value=100.0, 
        value=70.0, 
        step=1.0,
        help="Scores achieved in previous evaluations."
    )

st.markdown("---")

# Predict button
if st.button("Predict Performance"):
    # Features structured exactly as expected by your model
    features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
    
    # Run prediction
    prediction = model.predict(features)[0]
    
    # Display Result within the shadow card layout
    st.markdown(f"""
        <div class="prediction-card">
            <h3 style='margin-top:0; color: #2C3E50;'>Predicted Performance Index</h3>
            <p style='font-size: 32px; font-weight: bold; color: #4A90E2; margin: 0;'>
                {prediction:.2f}
            </p>
            <p style='color: #7F8C8D; font-size: 14px; margin-top: 5px;'>
                Based on the provided study habits, sleep schedules, and attendance metrics.
            </p>
        </div>
    """, unsafe_allow_html=True)
