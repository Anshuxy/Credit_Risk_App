import streamlit as st
import pandas as pd
import joblib

# 1. Load the Brains (Logistic and Random Forest)
@st.cache_resource
def load_all_models():
    # Ensure these filenames match what you saved in train_model.py
    lr = joblib.load('logistic_model.pkl')
    rf = joblib.load('rf_model.pkl')
    return lr, rf

try:
    lr_model, rf_model = load_all_models()
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first to generate .pkl files.")
    st.stop()

# 2. Luxury Dark Theme CSS
st.set_page_config(page_title="Credit Risk AI", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #050505;
        color: #FDFD96;
        font-family: 'Inter', sans-serif;
    }

    /* Modern Header */
    .main-header {
        font-size: 40px;
        font-weight: 800;
        color: #FDFD96;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }

    /* Glass Cards for Inputs */
    .input-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(253, 253, 150, 0.1);
        margin-bottom: 20px;
    }

    /* Clean Underline Inputs */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(253, 253, 150, 0.3) !important;
        border-radius: 0px !important;
        color: #FDFD96 !important;
    }

    /* FIX: Premium Button - Black text on Pale Yellow background */
    .stButton > button {
        width: 100%;
        background-color: #FDFD96 !important;
        color: #000000 !important; /* Forces text to be visible */
        border: none !important;
        padding: 12px !important;
        font-weight: 800 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 8px !important;
        transition: 0.4s ease;
    }

    .stButton > button:hover {
        background-color: #ffffff !important;
        box-shadow: 0px 0px 20px rgba(253, 253, 150, 0.5);
        color: #000000 !important;
    }

    label {
        color: #FDFD96 !important;
        opacity: 0.8;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Hide Streamlit components for a cleaner look */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Header Section
st.markdown('<div class="main-header">CREDIT RISK INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; opacity:0.6;">Advanced Machine Learning Comparison Protocol</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 4. Input Layout
with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("👤 Applicant Profile")
        age = st.number_input("Age", min_value=18, max_value=100, value=25)
        income = st.number_input("Annual Income ($)", min_value=1, value=65000)
        emp_exp = st.number_input("Employment (Years)", min_value=0, value=4)
        home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("💰 Loan Financials")
        loan_amt = st.number_input("Loan Amount Requested ($)", min_value=1, value=15000)
        rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.5)
        intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        st.markdown('</div>', unsafe_allow_html=True)

# 5. History and Default Selection
st.markdown('<div class="input-card">', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    cred_hist = st.number_input("Credit History Length (Years)", min_value=0, value=5)
with c4:
    default = st.selectbox("Historical Default on Record?", ["N", "Y"])
st.markdown('</div>', unsafe_allow_html=True)

# 6. Feature Engineering & Prediction Logic
if 'results' not in st.session_state:
    st.session_state.results = {}

# Prepare data exactly as the model expects
percent_income = loan_amt / income
input_df = pd.DataFrame([[
    age, income, emp_exp, loan_amt, rate, percent_income, cred_hist, 
    home, intent, grade, default
]], columns=[
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
])

st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("PREDICT WITH LOGISTIC"):
        prob = lr_model.predict_proba(input_df)[0][1]
        st.session_state.results['Logistic Regression'] = prob

with btn_col2:
    if st.button("PREDICT WITH RANDOM FOREST"):
        prob = rf_model.predict_proba(input_df)[0][1]
        st.session_state.results['Random Forest'] = prob

# 7. Comparison Summary Table
if st.session_state.results:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Comparison Summary")
    
    comparison_data = []
    for model_name, prob in st.session_state.results.items():
        status = "❌ REJECT" if prob > 0.5 else "✅ APPROVE"
        comparison_data.append({
            "Model Engine": model_name,
            "Risk Probability": f"{prob:.2%}",
            "Decision": status
        })
    
    st.table(pd.DataFrame(comparison_data))
    
    # Show variance if both are calculated
    if len(st.session_state.results) > 1:
        diff = abs(st.session_state.results['Logistic Regression'] - st.session_state.results['Random Forest'])
        st.markdown(f"<p style='text-align:center;'>Model Variance (Difference): <b>{diff:.2%}</b></p>", unsafe_allow_html=True)