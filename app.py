import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config for beautiful layout
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF2B2B 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.6);
        background: linear-gradient(135deg, #FF2B2B 0%, #FF0000 100%);
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 12px;
        background: #1f2937;
        border: 1px solid #374151;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        margin-top: 1.5rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .status-churn {
        color: #ef4444;
    }
    .status-stay {
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Load the model assets
@st.cache_resource
def load_assets():
    assets_path = os.path.join(os.path.dirname(__file__), "Scripts", "churn_model_assets.pkl")
    with open(assets_path, "rb") as f:
        return pickle.load(f)

try:
    assets = load_assets()
    model = assets["model"]
    scaler = assets["scaler"]
    label_encoders = assets["label_encoders"]
    scale_cols = assets["scale_cols"]
    feature_order = assets["feature_order"]
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

# Header / Title Section
st.title("🔮 Telecom Customer Churn Predictor")
st.markdown("""
This interactive web app predicts the likelihood of a customer churning (switching to another service provider) 
using a state-of-the-art Voting Classifier (combining **Gradient Boosting**, **AdaBoost**, and **Logistic Regression**).
""")

# Sidebar info
st.sidebar.image("https://raw.githubusercontent.com/Vansh-Varshney-07/Telecom-Customer-Churn-prediction/main/output/customer%20churn.jpeg", use_column_width=True)
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.info("""
- **Validation Accuracy**: ~84.6%
- **ROC AUC Score**: ~0.84
- **Final Classifier**: Voting Ensemble
""")

# Main form layout
st.markdown("### 👤 Enter Customer details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12)

with col2:
    st.subheader("Services Subscribed")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Account & Payment")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=65.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=10.0, max_value=10000.0, value=780.0, step=1.0)

# Prediction execution
if st.button("🔮 Predict Churn Probability"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": float(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges)
    }])

    # Align columns to matching order of features
    input_data = input_data[feature_order]

    # Encode categorical columns
    for col, mapping_dict in label_encoders.items():
        val = input_data.loc[0, col]
        # Map string value to its integer code. Fallback to 0 if not found
        input_data.loc[0, col] = mapping_dict["mapping"].get(val, 0)

    # Scale numerical columns
    input_data[scale_cols] = scaler.transform(input_data[scale_cols])

    # Predict probability
    prob = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    # Display results
    st.markdown("---")
    st.markdown("### 📊 Prediction Output")
    
    if prediction == 1 or prob >= 0.5:
        st.markdown(f"""
        <div class="prediction-card">
            <h3>Prediction Results</h3>
            <div class="metric-value status-churn">⚠️ HIGH RISK OF CHURN</div>
            <p>Probability: <strong>{prob:.1%}</strong></p>
            <p style="color: #9ca3af;">This customer matches the behavior patterns of users who switched providers. Retention strategies should be offered immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-card">
            <h3>Prediction Results</h3>
            <div class="metric-value status-stay">✅ LOW RISK (LOYAL)</div>
            <p>Probability: <strong>{prob:.1%}</strong></p>
            <p style="color: #9ca3af;">This customer is likely to stay with the current service provider.</p>
        </div>
        """, unsafe_allow_html=True)
