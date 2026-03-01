import streamlit as st
import pandas as pd
import joblib
import os

# 1. Load the Models with relative paths for Streamlit Cloud
@st.cache_resource
def load_all_models():
    # Make sure these filenames match your GitHub repository exactly
    lr_path = 'logistic_model.pkl'
    rf_path = 'rf_model.pkl'
    
    if not os.path.exists(lr_path) or not os.path.exists(rf_path):
        return None, None
        
    lr = joblib.load(lr_path)
    rf = joblib.load(rf_path)
    return lr, rf

lr_model, rf_model = load_all_models()

# 2. Luxury Dark Theme CSS
st.set_page_config(page_title="Credit Risk AI", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #FDFD96; font-family: 'Inter', sans-serif; }
    .main-header { font-size: 40px; font-weight: 800; color: #FDFD96; text-align: center; margin-bottom: 10px; }
    .input-card { background: rgba(255, 255, 255, 0.03); padding: 25px; border-radius: 15px; border: 1px solid rgba(253, 253, 150, 0.1); margin-bottom: 20px; }
    .stButton > button { width: 100%; background-color: #FDFD96 !important; color: #000!important; font-weight: 800!important; border-radius: 8px!important; }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

if lr_model is None:
    st.error("⚠️ Model files (.pkl) not found. Please upload 'logistic_model.pkl' and 'rf_model.pkl' to your GitHub repo.")
    st.stop()

# 3. Header
st.markdown('<div class="main-header">CREDIT RISK INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; opacity:0.6;">Traditional ML Comparison Protocol (No GenAI)</p>', unsafe_allow_html=True)

# 4. Input Layout
with st.container():
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("👤 Applicant Profile")
        age = st.number_input("Age", 18, 100, 25)
        income = st.number_input("Annual Income ($)", 1, 1000000, 65000)
        emp_exp = st.number_input("Employment (Years)", 0, 50, 4)
        home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("💰 Loan Financials")
        loan_amt = st.number_input("Loan Amount Requested ($)", 1, 500000, 15000)
        rate = st.number_input("Interest Rate (%)", 0.0, 30.0, 10.5)
        intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-card">', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    cred_hist = st.number_input("Credit History Length (Years)", 0, 40, 5)
with c4:
    default = st.selectbox("Historical Default on Record?", ["N", "Y"])
st.markdown('</div>', unsafe_allow_html=True)

# 5. Data Preparation (Crucial for Scikit-Learn Pipeline)
# This calculates the feature used in the technical project plan
percent_income = loan_amt / income if income > 0 else 0

input_df = pd.DataFrame([[
    age, income, emp_exp, loan_amt, rate, percent_income, cred_hist, 
    home, intent, grade, default
]], columns=[
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 
    'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
])

# 6. Prediction
if 'results' not in st.session_state:
    st.session_state.results = {}

st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("RUN LOGISTIC REGRESSION"):
        # We use [0][1] because predict_proba returns [prob_0, prob_1]
        prob = lr_model.predict_proba(input_df)[0][1]
        st.session_state.results['Logistic Regression'] = prob

with btn_col2:
    if st.button("RUN RANDOM FOREST"):
        prob = rf_model.predict_proba(input_df)[0][1]
        st.session_state.results['Random Forest'] = prob

# 7. Comparison Table
if st.session_state.results:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Model Comparison Report")
    
    res_list = []
    for model_name, prob in st.session_state.results.items():
        # Using 0.5 as a standard threshold for Risk classification
        status = "❌ HIGH RISK (REJECT)" if prob > 0.5 else "✅ LOW RISK (APPROVE)"
        res_list.append({
            "Algorithm": model_name,
            "Default Probability": f"{prob:.2%}",
            "Recommended Action": status
        })
    
    st.table(pd.DataFrame(res_list))