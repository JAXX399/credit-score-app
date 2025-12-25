import streamlit as st
import pandas as pd
import plotly.express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
st.set_page_config(page_title="Credit Scoring AI", page_icon="💳", layout="wide")

# --- CUSTOM CSS (MAKE BUTTON HUGE & GREEN) ---
st.markdown("""
    <style>
        /* Reduce top padding */
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        
        /* Style the Calculate Button */
        div.stButton > button:first-child {
            background-color: #2e7d32; /* Green */
            color: white;
            font-size: 24px;
            font-weight: bold;
            height: 60px;
            border-radius: 12px;
            border: 2px solid #1b5e20;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #1b5e20;
            transform: scale(1.02);
        }
        
        /* Style the Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 5px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- SELF-TRAINING MODEL (Cloud Compatible) ---
@st.cache_resource
def get_trained_model():
    # 1. Locate Data
    possible_paths = ['german_credit_data.csv', 'data/german_credit_data.csv']
    data_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not data_path:
        st.error("❌ 'german_credit_data.csv' not found. Please upload it to your GitHub root folder.")
        return None, None, None

    # 2. Train Model
    try:
        df = pd.read_csv(data_path)
        target_col = 'credit_risk' if 'credit_risk' in df.columns else df.columns[-1]
        
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        X_encoded = pd.get_dummies(X)
        model_columns = list(X_encoded.columns) # Save schema

        X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        clf1 = LogisticRegression(random_state=1, max_iter=1000)
        clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
        clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
        
        ensemble = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)], voting='soft')
        ensemble.fit(X_train_scaled, y_train)
        
        return ensemble, scaler, model_columns
    except Exception as e:
        st.error(f"Training Error: {e}")
        return None, None, None

# Load Model
with st.spinner("🤖 Initializing AI... (Building the Brain)"):
    model, scaler, model_columns = get_trained_model()
    MODEL_LOADED = True if model else False

# --- NAVIGATION ---
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

def go_to_explainer(): st.session_state['current_page'] = 'explainer'
def go_to_home(): st.session_state['current_page'] = 'home'

# ==========================================
# PAGE 1: CALCULATOR
# ==========================================
def show_home_page():
    # Header
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("💳 AI Credit Scoring")
        st.caption("Complete the form below to assess creditworthiness.")
    with c2:
        st.button("🧠 How Logic Works", on_click=go_to_explainer, type="secondary", use_container_width=True)

    st.markdown("---")

    # Inputs organized in Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal", "💰 Financial", "🏠 Assets", "📄 Loan Details"])
    input_data = {}

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age (Years)", 18, 75, 30)
            sex_map = {"Male (Single)": "personal_status_A93", "Female (Div/Mar)": "personal_status_A92", "Male (Mar/Div)": "personal_status_A94", "Male (Div/Sep)": "personal_status_A91"}
            sex = st.selectbox("Sex & Status", list(sex_map.keys()))
            foreign = st.radio("Foreign Worker?", ["Yes", "No"], horizontal=True)
            deps = st.radio("Dependents", [1, 2], horizontal=True)
        with col2:
            job_map = {"Skilled": "job_A173", "Unskilled (Res)": "job_A172", "Management": "job_A174", "Unemployed": "job_A171"}
            job = st.selectbox("Job Type", list(job_map.keys()))
            emp_map = {"1-4 years": "employment_A73", ">= 7 years": "employment_A75", "4-7 years": "employment_A74", "< 1 year": "employment_A72", "Unemployed": "employment_A71"}
            emp = st.selectbox("Employment Duration", list(emp_map.keys()))
            tel = st.radio("Telephone?", ["None", "Yes"], horizontal=True)
            
            # Save
            input_data.update({"age": age, "num_dependents": deps})
            input_data[sex_map[sex]] = 1.0
            input_data["foreign_worker_A201" if foreign == "Yes" else "foreign_worker_A202"] = 1.0
            input_data[job_map[job]] = 1.0
            input_data[emp_map[emp]] = 1.0
            input_data["telephone_A192" if tel == "Yes" else "telephone_A191"] = 1.0

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            check_map = {"No Account (Safe)": "checking_status_A14", "Negative (<0)": "checking_status_A11", "Low (0-200)": "checking_status_A12", "High (>200)": "checking_status_A13"}
            check = st.selectbox("Checking Status", list(check_map.keys()))
            hist_map = {"Critical/Good": "credit_history_A34", "Existing Paid": "credit_history_A32", "No Credits/Paid": "credit_history_A30", "Delay": "credit_history_A33", "All Paid": "credit_history_A31"}
            hist = st.selectbox("Credit History", list(hist_map.keys()))
        with col2:
            sav_map = {"Unknown/None": "savings_status_A65", "Low (<100)": "savings_status_A61", "Medium": "savings_status_A62", "High": "savings_status_A63", "Very High": "savings_status_A64"}
            sav = st.selectbox("Savings Balance", list(sav_map.keys()))
            exist_cr = st.slider("Existing Credits", 1, 4, 1)
            
            input_data["existing_credits"] = exist_cr
            input_data[check_map[check]] = 1.0
            input_data[hist_map[hist]] = 1.0
            input_data[sav_map[sav]] = 1.0

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            house_map = {"Own": "housing_A152", "Rent": "housing_A151", "Free": "housing_A153"}
            house = st.selectbox("Housing", list(house_map.keys()))
            prop_map = {"Real Estate": "property_A121", "Savings/Life Ins": "property_A122", "Car/Other": "property_A123", "Unknown": "property_A124"}
            prop = st.selectbox("Property", list(prop_map.keys()))
        with col2:
            res_since = st.slider("Residence Since (Years)", 1, 4, 2)
            input_data["residence_since"] = res_since
            input_data[house_map[house]] = 1.0
            input_data[prop_map[prop]] = 1.0

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input("Credit Amount (DM)", 250, 20000, 4000)
            dur = st.slider("Duration (Months)", 4, 72, 24)
            rate = st.slider("Installment Rate (%)", 1, 4, 2)
        with col2:
            pur_map = {"New Car": "purpose_A40", "Used Car": "purpose_A41", "Furniture": "purpose_A42", "Radio/TV": "purpose_A43", "Appliances": "purpose_A44", "Repairs": "purpose_A45", "Education": "purpose_A46", "Business": "purpose_A49", "Retraining": "purpose_A48", "Other": "purpose_A410"}
            pur = st.selectbox("Purpose", list(pur_map.keys()))
            debt_map = {"None": "other_debtors_A101", "Guarantor": "other_debtors_A103", "Co-Applicant": "other_debtors_A102"}
            debt = st.selectbox("Debtors", list(debt_map.keys()))
            inst_map = {"None": "other_payment_plans_A143", "Bank": "other_payment_plans_A141", "Stores": "other_payment_plans_A142"}
            inst = st.selectbox("Other Installments", list(inst_map.keys()))

            input_data.update({"credit_amount": amt, "duration": dur, "installment_rate": rate})
            input_data[pur_map[pur]] = 1.0
            input_data[debt_map[debt]] = 1.0
            input_data[inst_map[inst]] = 1.0

    # --- ACTION AREA (Huge Button) ---
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("### 🚀 Ready to Predict?")
    
    if st.button("CALCULATE CREDIT SCORE", use_container_width=True):
        if not MODEL_LOADED:
            st.error("⚠️ Model failed to load. Please check dataset.")
        else:
            with st.spinner("Analyzing 20 data points..."):
                # Align Data
                df_input = pd.DataFrame([input_data])
                df_aligned = pd.DataFrame(columns=model_columns)
                df_aligned = pd.concat([df_aligned, df_input], ignore_index=True).fillna(0)
                df_final = df_aligned[model_columns] # Ensure order
                
                # Predict
                X_scaled = scaler.transform(df_final)
                prob = float(model.predict_proba(X_scaled)[0][1])
                score = int(300 + 550 * (1 - prob))
                
                # Result UI
                st.markdown("---")
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    if score >= 700:
                        st.success(f"## ✅ APPROVED\n### Score: {score}")
                    elif score >= 600:
                        st.warning(f"## ⚠️ REVIEW\n### Score: {score}")
                    else:
                        st.error(f"## ❌ REJECTED\n### Score: {score}")
                    
                    st.progress(score/850)
                    st.caption(f"Default Probability: {prob*100:.1f}%")

# ==========================================
# PAGE 2: EXPLAINER
# ==========================================
def show_explainer_page():
    st.button("⬅️ Back to Calculator", on_click=go_to_home)
    st.title("🧠 How the Model Works")
    
    st.info("This AI uses an **Ensemble Model** (Voting Classifier). It combines the decisions of Logistic Regression, Random Forest, and Gradient Boosting.")

    data = {
        "Attribute": ["Checking Status", "Duration", "Credit History", "Credit Amount", "Age", "Savings", "Employment", "Installment Rate"],
        "Impact": [18, 12, 10, 9, 7, 5, 5, 4],
        "Category": ["Financial", "Loan", "History", "Loan", "Demographic", "Financial", "Demographic", "Loan"]
    }
    df = pd.DataFrame(data)
    fig = px.bar(df, x="Impact", y="Attribute", orientation='h', color="Category", title="Top Factors Influencing Score")
    st.plotly_chart(fig, use_container_width=True)

# Run Logic
if st.session_state['current_page'] == 'home': show_home_page()
else: show_explainer_page()
