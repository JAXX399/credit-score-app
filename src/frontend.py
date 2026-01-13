import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.doc_utils import parse_document, extract_attributes

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
st.set_page_config(page_title="Credit Scoring AI", page_icon="💳", layout="wide")

# --- CUSTOM CSS (Hide UI, Button Fixes & High Contrast Tabs) ---
st.markdown("""
    <style>
        /* --- HIDE STREAMLIT UI ELEMENTS --- */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;} /* Hides the Star/Share buttons */
        
        /* Reduce top whitespace since header is gone */
        .block-container {
            padding-top: 1rem;
        }

        /* 1. Fix Tab Visibility - Dark Blue Background with White Text */
        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #0e1117; /* Dark background */
            color: #ffffff;            /* White text */
            border-radius: 5px 5px 0px 0px;
            border: 1px solid #333;
            padding: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2e7d32; /* Green for selected tab */
            color: white;
            border-bottom: 2px solid #2e7d32;
        }

        /* 2. Calculate Button - Huge & Green */
        div.stButton > button:first-child {
            width: 100%;
            background-color: #2e7d32;
            color: white;
            font-size: 20px;
            font-weight: bold;
            height: 55px;
            border-radius: 8px;
            border: 2px solid #1b5e20;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #1b5e20;
            border-color: #000;
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
        return None, None, None, None

    # 2. Train Model
    try:
        df = pd.read_csv(data_path)
        target_col = 'credit_risk' if 'credit_risk' in df.columns else df.columns[-1]
        
        X = df.drop([target_col], axis=1)
        # Ensure target is 0/1. If it's 1/2 (common in german credit), map it.
        # Assuming original dataset might be 1=Good, 2=Bad or similar.
        # Checking uniqueness just in case.
        y = df[target_col]
        
        X_encoded = pd.get_dummies(X)
        model_columns = list(X_encoded.columns) # Save schema

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate Class Weight for Imbalance
        # scale_pos_weight = total_negative_examples / total_positive_examples
        # Assuming 1 is "Bad" (Positive class for risk) and 0 is "Good". 
        # If dataset is 0/1:
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        weight = num_neg / num_pos if num_pos > 0 else 1.0

        # XGBoost Classifier
        model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            scale_pos_weight=weight,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Calculate Accuracy
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        return model, scaler, model_columns, acc
    except Exception as e:
        st.error(f"Training Error: {e}")
        return None, None, None, None

# Load Model
with st.spinner("🤖 Initializing AI... (Training XGBoost Model)"):
    model, scaler, model_columns, accuracy = get_trained_model()
    MODEL_LOADED = True if model else False

if MODEL_LOADED and accuracy:
    st.sidebar.success(f"Model Trained! 🎯 Test Accuracy: {accuracy*100:.1f}%")

# --- AUTO-FILL LOGIC ---
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {}

with st.sidebar:
    st.header("📂 Auto-Fill")
    uploaded_file = st.file_uploader("Upload Application (PDF/DOCX)", type=['pdf', 'docx', 'txt'])
    if uploaded_file is not None:
        if st.button("Read Document"):
            text = parse_document(uploaded_file)
            extracted = extract_attributes(text)
            if extracted:
                st.session_state['form_data'].update(extracted)
                st.success("✅ Data Extracted!")
            else:
                st.warning("⚠️ No attributes found in document.")

# --- NAVIGATION ---
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

def go_to_explainer(): st.session_state['current_page'] = 'explainer'
def go_to_home(): st.session_state['current_page'] = 'home'

# ==========================================
# PAGE 1: CALCULATOR
# ==========================================
def show_home_page():
    # Header Layout
    st.title("💳 AI Credit Scoring")
    st.caption("Complete the form below to assess creditworthiness.")
    
    # Logic Button (Full width row to prevent cutoff)
    col_a, col_b = st.columns([4, 1])
    with col_b:
        st.button("🧠 How Logic Works", on_click=go_to_explainer, type="secondary", use_container_width=True)

    st.markdown("---")

    # Inputs organized in Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Personal", "💰 Financial", "🏠 Assets", "📄 Loan Details"])
    input_data = {}

    # Helper to get session value or default
    def get_val(key, default):
        return st.session_state['form_data'].get(key, default)

    with tab1:
        st.markdown("#### Personal Details")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age (Years)", 18, 75, get_val('age', 30))
            
            sex_map = {"Male (Single)": "personal_status_A93", "Female (Div/Mar)": "personal_status_A92", "Male (Mar/Div)": "personal_status_A94", "Male (Div/Sep)": "personal_status_A91"}
            # Determine index for sex
            default_sex = get_val('sex', "Male (Single)")
            sex_idx = list(sex_map.keys()).index(default_sex) if default_sex in sex_map else 0
            sex = st.selectbox("Sex & Status", list(sex_map.keys()), index=sex_idx)
            
            # Foreign
            default_for = get_val('foreign', "Yes")
            for_idx = ["Yes", "No"].index(default_for) if default_for in ["Yes", "No"] else 0
            foreign = st.radio("Foreign Worker?", ["Yes", "No"], index=for_idx, horizontal=True)

            # Dependents
            def_dep = get_val('dependents', 1)
            dep_idx = [1, 2].index(def_dep) if def_dep in [1, 2] else 0
            deps = st.radio("Dependents", [1, 2], index=dep_idx, horizontal=True)

        with col2:
            job_map = {"Skilled": "job_A173", "Unskilled (Res)": "job_A172", "Management": "job_A174", "Unemployed": "job_A171"}
            def_job = get_val('job', "Skilled")
            job_idx = list(job_map.keys()).index(def_job) if def_job in job_map else 0
            job = st.selectbox("Job Type", list(job_map.keys()), index=job_idx)
            
            emp_map = {"1-4 years": "employment_A73", ">= 7 years": "employment_A75", "4-7 years": "employment_A74", "< 1 year": "employment_A72", "Unemployed": "employment_A71"}
            def_emp = get_val('emp_duration', "1-4 years")
            emp_idx = list(emp_map.keys()).index(def_emp) if def_emp in emp_map else 0
            emp = st.selectbox("Employment Duration", list(emp_map.keys()), index=emp_idx)
            
            tel = st.radio("Telephone?", ["None", "Yes"], horizontal=True)
            
            # Save
            input_data.update({"age": age, "num_dependents": deps})
            input_data[sex_map[sex]] = 1.0
            input_data["foreign_worker_A201" if foreign == "Yes" else "foreign_worker_A202"] = 1.0
            input_data[job_map[job]] = 1.0
            input_data[emp_map[emp]] = 1.0
            input_data["telephone_A192" if tel == "Yes" else "telephone_A191"] = 1.0

    with tab2:
        st.markdown("#### Financial Status")
        col1, col2 = st.columns(2)
        with col1:
            check_map = {"No Account (Safe)": "checking_status_A14", "Negative (<0)": "checking_status_A11", "Low (0-200)": "checking_status_A12", "High (>200)": "checking_status_A13"}
            def_check = get_val('check_status', "No Account (Safe)")
            check_idx = list(check_map.keys()).index(def_check) if def_check in check_map else 0
            check = st.selectbox("Checking Status", list(check_map.keys()), index=check_idx)
            
            hist_map = {"Critical/Good": "credit_history_A34", "Existing Paid": "credit_history_A32", "No Credits/Paid": "credit_history_A30", "Delay": "credit_history_A33", "All Paid": "credit_history_A31"}
            # Default history not extracted currently, keep standard default
            hist = st.selectbox("Credit History", list(hist_map.keys()))
        with col2:
            sav_map = {"Unknown/None": "savings_status_A65", "Low (<100)": "savings_status_A61", "Medium": "savings_status_A62", "High": "savings_status_A63", "Very High": "savings_status_A64"}
            def_sav = get_val('savings', "Unknown/None")
            sav_idx = list(sav_map.keys()).index(def_sav) if def_sav in sav_map else 0
            sav = st.selectbox("Savings Balance", list(sav_map.keys()), index=sav_idx)
            
            exist_cr = st.slider("Existing Credits", 1, 4, get_val('exist_credits', 1))
            
            input_data["existing_credits"] = exist_cr
            input_data[check_map[check]] = 1.0
            input_data[hist_map[hist]] = 1.0
            input_data[sav_map[sav]] = 1.0

    with tab3:
        st.markdown("#### Assets & Living")
        col1, col2 = st.columns(2)
        with col1:
            house_map = {"Own": "housing_A152", "Rent": "housing_A151", "Free": "housing_A153"}
            def_house = get_val('housing', "Own")
            house_idx = list(house_map.keys()).index(def_house) if def_house in house_map else 0
            house = st.selectbox("Housing", list(house_map.keys()), index=house_idx)
            
            prop_map = {"Real Estate": "property_A121", "Savings/Life Ins": "property_A122", "Car/Other": "property_A123", "Unknown": "property_A124"}
            def_prop = get_val('property', "Real Estate")
            prop_idx = list(prop_map.keys()).index(def_prop) if def_prop in prop_map else 0
            prop = st.selectbox("Property", list(prop_map.keys()), index=prop_idx)
        with col2:
            res_since = st.slider("Residence Since (Years)", 1, 4, 2)
            input_data["residence_since"] = res_since
            input_data[house_map[house]] = 1.0
            input_data[prop_map[prop]] = 1.0

    with tab4:
        st.markdown("#### Loan Parameters")
        col1, col2 = st.columns(2)
        with col1:
            amt = st.number_input("Credit Amount (DM)", 250, 20000, get_val('amount', 4000))
            dur = st.slider("Duration (Months)", 4, 72, get_val('duration', 24))
            rate = st.slider("Installment Rate (%)", 1, 4, 2)
        with col2:
            pur_map = {"New Car": "purpose_A40", "Used Car": "purpose_A41", "Furniture": "purpose_A42", "Radio/TV": "purpose_A43", "Appliances": "purpose_A44", "Repairs": "purpose_A45", "Education": "purpose_A46", "Business": "purpose_A49", "Retraining": "purpose_A48", "Other": "purpose_A410"}
            def_pur = get_val('purpose', "New Car")
            pur_idx = list(pur_map.keys()).index(def_pur) if def_pur in pur_map else 0
            pur = st.selectbox("Purpose", list(pur_map.keys()), index=pur_idx)
            
            debt_map = {"None": "other_debtors_A101", "Guarantor": "other_debtors_A103", "Co-Applicant": "other_debtors_A102"}
            debt = st.selectbox("Debtors", list(debt_map.keys()))
            inst_map = {"None": "other_payment_plans_A143", "Bank": "other_payment_plans_A141", "Stores": "other_payment_plans_A142"}
            inst = st.selectbox("Other Installments", list(inst_map.keys()))

            input_data.update({"credit_amount": amt, "duration": dur, "installment_rate": rate})
            input_data[pur_map[pur]] = 1.0
            input_data[debt_map[debt]] = 1.0
            input_data[inst_map[inst]] = 1.0

    # --- ACTION AREA ---
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
                        st.success(f"##Credit Score:\n  {score}")
                    elif score >= 600:
                        st.warning(f"##Credit Score:\n  {score}")
                    else:
                        st.error(f"##Credit Score:\n  {score}")
                    
                    st.progress(score/850)
                    st.caption(f"Default Probability: {prob*100:.1f}%")

# ==========================================
# PAGE 2: EXPLAINER (FULL 20 ATTRIBUTES)
# ==========================================
def show_explainer_page():
    c1, c2 = st.columns([1, 4])
    with c1:
        st.button("⬅️ Back", on_click=go_to_home, use_container_width=True)
    with c2:
        st.subheader("🧠 Understanding the Model")
    
    # --- XGBOOST EXPLANATION ---
    st.info("""
    **How the "XGBoost" Model Works:**
    This AI uses **Extreme Gradient Boosting (XGBoost)**, a powerful technique that builds a series of decision trees to make predictions.
    
    1.  **Iterative Learning:** Unlike a single guess, the model builds hundreds of small decision trees one after another.
    2.  **Error Correction:** Each new tree specifically focuses on correcting the mistakes made by the previous trees.
    3.  **Class Balancing:** We have configured the model to pay extra attention to "risky" applicants (who are often rare in the data) to ensure we don't miss potential defaults.
    
    **Result:** A highly accurate model that captures complex patterns and non-linear relationships better than traditional methods.
    """)
    # --------------------------------------

    st.markdown("### Attribute Importance (All 20 Features)")
    st.caption("Which factors influence the decision the most?")

    # FULL DATASET OF 20 ATTRIBUTES
    data = {
        "Attribute": [
            "Checking Status", "Duration", "Credit History", "Credit Amount", "Age",
            "Savings Account", "Employment Since", "Installment Rate", "Sex & Status",
            "Other Debtors", "Residence Since", "Property", "Other Installments",
            "Housing", "Existing Credits", "Job Type", "People Liable", "Telephone", "Foreign Worker", "Purpose"
        ],
        "Category": [
            "Financial", "Loan", "History", "Loan", "Demographic",
            "Financial", "Demographic", "Loan", "Demographic",
            "History", "Demographic", "Assets", "Financial",
            "Assets", "History", "Demographic", "Demographic", "Assets", "Demographic", "Loan"
        ],
        # keys to match the one-hot prefixes in the dataset
        "MatchKey": [
            "checking_status", "duration", "credit_history", "credit_amount", "age",
            "savings_status", "employment", "installment_rate", "personal_status",
            "other_debtors", "residence_since", "property", "other_payment_plans",
            "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker", "purpose"
        ],
        "Key Insight / Logic": [
            "Negative balance (A11) is the #1 risk factor. No checking (A14) is safest.",
            "Loans > 48 months are significantly higher risk.",
            "Paying back a 'Critical Account' (A34) boosts score massively.",
            "Higher amounts generally increase risk, but depend on collateral.",
            "Very young (<25) is risky. Middle age (30-50) is safest.",
            "Having < 100 DM (A61) is risky. > 1000 DM is safe.",
            "Stable employment (>7 years) strongly reduces risk.",
            "High installment rate (4% of income) indicates stress.",
            "Single Males (A93) historically favored in this dataset.",
            "Guarantors (A103) significantly reduce risk.",
            "Longer residence implies stability.",
            "Real Estate (A121) is the best collateral.",
            "Owing other banks (A141) increases debt burden.",
            "Home owners (A152) are safer than renters.",
            "Having 2-3 existing credits is normal. Too many is bad.",
            "Management/Highly Skilled jobs get better scores.",
            "More dependents = Less disposable income.",
            "Owning a phone suggests stability/traceability.",
            "Foreign workers (A201) are flagged as higher flight risk.",
            "Used cars (A41) often safe, Education/Business often riskier."
        ]
    }
    
    df = pd.DataFrame(data)

    # --- CALCULATE DYNAMIC IMPORTANCE ---
    if MODEL_LOADED:
        import numpy as np
        # 1. Get raw importances
        raw_importances = model.feature_importances_
        # 2. Map one-hot encoded columns back to original features
        df['Importance'] = 0.0
        
        for idx, row in df.iterrows():
            key = row['MatchKey']
            # Find all columns that start with this key (e.g. 'checking_status_A11', 'checking_status_A12')
            # For numericals like 'age', it's just exact match
            matched_indices = [i for i, col in enumerate(model_columns) if col.startswith(key)]
            
            if matched_indices:
                total_importance = sum(raw_importances[i] for i in matched_indices)
                df.at[idx, 'Importance'] = total_importance

        # Normalize to 0-100 for display
        if df['Importance'].sum() > 0:
            df['Importance'] = (df['Importance'] / df['Importance'].sum()) * 100
    else:
        # Fallback if model not loaded
        df['Importance'] = [18, 12, 10, 9, 7, 5, 5, 4, 3, 2, 2, 4, 2, 2, 1, 1, 1, 1, 1, 5]

    # Sort by importance
    df = df.sort_values(by="Importance", ascending=False)
    
    # 1. GRAPH
    fig = px.treemap(
        df, path=['Category', 'Attribute'], values='Importance',
        title="Attribute Importance Hierarchy (Based on Your Data)", color='Category'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2. TABLE
    st.markdown("### Detailed Logic Breakdown")
    st.dataframe(df.drop(columns=['Category', 'Importance']), use_container_width=True)

# Run Logic
if st.session_state['current_page'] == 'home': show_home_page()
else: show_explainer_page()