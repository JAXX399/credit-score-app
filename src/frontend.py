import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Credit Scoring AI", page_icon="🏦", layout="wide")



# --- LOAD MODELS DIRECTLY (CACHED) ---
@st.cache_resource
def load_models():
    # Helper to find paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    model_dir = os.path.join(root_dir, 'models')
    
    # Load files
    model = joblib.load(os.path.join(model_dir, 'ensemble_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    columns = joblib.load(os.path.join(model_dir, 'model_columns.pkl'))
    return model, scaler, columns

try:
    model, scaler, model_columns = load_models()
    MODEL_LOADED = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    MODEL_LOADED = False

# --- SESSION STATE ---
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

def go_to_explainer(): st.session_state['current_page'] = 'explainer'
def go_to_home(): st.session_state['current_page'] = 'home'

# ==========================================
# PAGE 1: PREDICTION ENGINE
# ==========================================
def show_home_page():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("🔮 AI Credit Scoring Engine")
        st.caption("Enter full applicant details (20 attributes) for a precise risk assessment.")
    with c2:
        st.button("🧠 How it Works", on_click=go_to_explainer, type="secondary", use_container_width=True)

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal", "💰 Financial", "🏠 Assets", "📄 Loan"])
    input_data = {}

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age (Years)", 18, 75, 30)
            sex_map = {"Male (Single)": "personal_status_A93", "Female (Div/Mar)": "personal_status_A92", "Male (Mar/Div)": "personal_status_A94", "Male (Div/Sep)": "personal_status_A91"}
            sex_display = st.selectbox("Sex & Status", list(sex_map.keys()))
            foreign_map = {"Yes": "foreign_worker_A201", "No": "foreign_worker_A202"}
            foreign_display = st.radio("Foreign Worker?", ["Yes", "No"], horizontal=True)
            people_liable = st.radio("Dependents", [1, 2], horizontal=True)
        with col2:
            job_map = {"Unskilled (Res)": "job_A172", "Skilled": "job_A173", "Management": "job_A174", "Unemployed": "job_A171"}
            job_display = st.selectbox("Job Type", list(job_map.keys()))
            employment_map = {"Unemployed": "employment_A71", "< 1 year": "employment_A72", "1-4 years": "employment_A73", "4-7 years": "employment_A74", ">= 7 years": "employment_A75"}
            emp_display = st.selectbox("Employment Duration", list(employment_map.keys()))
            tel_map = {"None": "telephone_A191", "Yes": "telephone_A192"}
            tel_display = st.radio("Telephone?", ["None", "Yes"], horizontal=True)
            
            input_data.update({"age": age, "num_dependents": people_liable})
            input_data[sex_map[sex_display]] = 1.0
            input_data[foreign_map[foreign_display]] = 1.0
            input_data[job_map[job_display]] = 1.0
            input_data[employment_map[emp_display]] = 1.0
            input_data[tel_map[tel_display]] = 1.0

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            check_map = {"No Account (Safe)": "checking_status_A14", "Negative (<0)": "checking_status_A11", "Low (0-200)": "checking_status_A12", "High (>200)": "checking_status_A13"}
            check_display = st.selectbox("Checking Status", list(check_map.keys()))
            hist_map = {"Critical/Good": "credit_history_A34", "Existing Paid": "credit_history_A32", "No Credits/Paid": "credit_history_A30", "Delay in Past": "credit_history_A33", "All Paid (Bank)": "credit_history_A31"}
            hist_display = st.selectbox("Credit History", list(hist_map.keys()))
        with col2:
            sav_map = {"Unknown/None": "savings_status_A65", "Low (<100)": "savings_status_A61", "Medium (100-500)": "savings_status_A62", "High (500-1000)": "savings_status_A63", "Very High (>1000)": "savings_status_A64"}
            sav_display = st.selectbox("Savings Balance", list(sav_map.keys()))
            exist_credits = st.slider("Existing Credits", 1, 4, 1)
            
            input_data["existing_credits"] = exist_credits
            input_data[check_map[check_display]] = 1.0
            input_data[hist_map[hist_display]] = 1.0
            input_data[sav_map[sav_display]] = 1.0

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            house_map = {"Own": "housing_A152", "Rent": "housing_A151", "Free": "housing_A153"}
            house_display = st.selectbox("Housing", list(house_map.keys()))
            prop_map = {"Real Estate": "property_A121", "Savings/Life Ins": "property_A122", "Car/Other": "property_A123", "Unknown": "property_A124"}
            prop_display = st.selectbox("Property", list(prop_map.keys()))
        with col2:
            res_since = st.slider("Residence Since (Years)", 1, 4, 2)
            input_data["residence_since"] = res_since
            input_data[house_map[house_display]] = 1.0
            input_data[prop_map[prop_display]] = 1.0

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input("Credit Amount", 250, 20000, 4000)
            dur = st.slider("Duration (Months)", 4, 72, 24)
            rate = st.slider("Installment Rate (%)", 1, 4, 2)
        with col2:
            pur_map = {"New Car": "purpose_A40", "Used Car": "purpose_A41", "Furniture": "purpose_A42", "Radio/TV": "purpose_A43", "Appliances": "purpose_A44", "Repairs": "purpose_A45", "Education": "purpose_A46", "Business": "purpose_A49", "Retraining": "purpose_A48", "Other": "purpose_A410"}
            pur_display = st.selectbox("Purpose", list(pur_map.keys()))
            debt_map = {"None": "other_debtors_A101", "Guarantor": "other_debtors_A103", "Co-Applicant": "other_debtors_A102"}
            debt_display = st.selectbox("Debtors", list(debt_map.keys()))
            inst_map = {"None": "other_payment_plans_A143", "Bank": "other_payment_plans_A141", "Stores": "other_payment_plans_A142"}
            inst_display = st.selectbox("Other Installments", list(inst_map.keys()))

            input_data.update({"credit_amount": amt, "duration": dur, "installment_rate": rate})
            input_data[pur_map[pur_display]] = 1.0
            input_data[debt_map[debt_display]] = 1.0
            input_data[inst_map[inst_display]] = 1.0

    st.markdown("---")
    if st.button("Calculate Credit Score", type="primary", use_container_width=True):
        if not MODEL_LOADED:
            st.error("Model not loaded. Please check directory structure.")
        else:
            with st.spinner("Processing..."):
                # DataFrame Creation & Scaling
                df_input = pd.DataFrame([input_data])
                df_aligned = pd.DataFrame(columns=model_columns)
                df_aligned = pd.concat([df_aligned, df_input], ignore_index=True).fillna(0)
                df_final = df_aligned[model_columns]
                df_scaled = scaler.transform(df_final)
                
                # Predict
                prob = float(model.predict_proba(df_scaled)[0][1])
                score = int(300 + 550 * (1 - prob))
                
                c1, c2 = st.columns(2)
                c1.metric("Score", score)
                c1.progress(score/850)
                c2.metric("Risk Probability", f"{prob*100:.1f}%")
                
                if score >= 700: st.success("✅ Low Risk")
                elif score >= 600: st.warning("⚠️ Medium Risk")
                else: st.error("❌ High Risk")

# ==========================================
# PAGE 2: EXPLAINER
# ==========================================
def show_explainer_page():
    c1, c2 = st.columns([1, 4])
    with c1: st.button("⬅️ Back", on_click=go_to_home, use_container_width=True)
    with c2: st.subheader("🧠 Understanding the Model")
    
    st.info("The model is an **Ensemble Voting Classifier** combining Logistic Regression (Linear trends), Random Forest (Rules), and Gradient Boosting (Error Correction).")
    
    data = {
        "Attribute": ["Checking Status", "Duration", "Credit History", "Credit Amount", "Age", "Savings Account", "Employment", "Installment Rate", "Sex & Status", "Other Debtors", "Residence Since", "Property", "Age", "Other Installments", "Housing", "Existing Credits", "Job Type", "People Liable", "Telephone", "Foreign Worker"],
        "Category": ["Financial", "Loan", "History", "Loan", "Demographic", "Financial", "Demographic", "Loan", "Demographic", "History", "Demographic", "Assets", "Demographic", "Financial", "Assets", "History", "Demographic", "Demographic", "Assets", "Demographic"],
        "Importance": [18, 12, 10, 9, 7, 5, 5, 4, 3, 2, 2, 4, 5, 2, 2, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    fig = px.treemap(df, path=['Category', 'Attribute'], values='Importance', title="Feature Importance", color='Category')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.drop(columns=['Category']), use_container_width=True)

if st.session_state['current_page'] == 'home': show_home_page()
else: show_explainer_page()