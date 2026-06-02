import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set page configuration with medical clinical themes
st.set_page_config(
    page_title="CardioShield AI | Clinical Decision Support Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# STYLING & CUSTOM CSS
# ----------------------------------------------------
# We will inject premium medical styled CSS to wow the user at first glance
st.markdown("""
<style>
    /* Main Layout Aesthetics */
    .main {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, #8b0000 0%, #d32f2f 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 30px rgba(211, 47, 47, 0.15);
    }
    .header-card h1 {
        font-weight: 800;
        font-size: 2.5rem;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-card p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Visual Cards for inputs grouping */
    .input-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }
    .input-card h3 {
        color: #d32f2f;
        font-size: 1.25rem;
        margin-top: 0;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #f8d7da;
        padding-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Result Card Styling */
    .result-box {
        border-radius: 12px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        border-left: 8px solid;
    }
    .result-box.danger {
        background-color: #fff5f5;
        border-color: #e53e3e;
        color: #742a2a;
    }
    .result-box.success {
        background-color: #f0fff4;
        border-color: #38a169;
        color: #22543d;
    }

    /* Info Badges */
    .clinical-badge {
        background-color: #e9ecef;
        color: #495057;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# MODEL & DATA LOADING
# ----------------------------------------------------
@st.cache_resource
def load_assets():
    """Loads and caches the model and scaler."""
    try:
        model = joblib.load("heart_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure train_model.py has been run.")
        return None, None

@st.cache_data
def load_dataset():
    """Loads and caches the training dataset."""
    if os.path.exists("heart diesease dataset.xlsx"):
        return pd.read_excel("heart diesease dataset.xlsx")
    return None

model, scaler = load_assets()
df = load_dataset()

# ----------------------------------------------------
# CATEGORICAL MAPPINGS DEFINITION
# ----------------------------------------------------
# Define clean translations between UI descriptive text and raw model encoded integers.
# We explicitly write the raw integer in the string (e.g. "Male (1)") so that the user
# knows exactly what each input maps to under the hood!

SEX_MAP = {
    "Female (0)": 0,
    "Male (1)": 1
}

CP_MAP = {
    "Asymptomatic (0)": 0,
    "Atypical Angina (1)": 1,
    "Non-anginal Pain (2)": 2,
    "Typical Angina (3)": 3
}

FBS_MAP = {
    "Greater than 120 mg/ml (0)": 0,
    "Lower than 120 mg/ml (1)": 1
}

ECG_MAP = {
    "Left Ventricular Hypertrophy (0)": 0,
    "Normal (1)": 1,
    "ST-T Wave Abnormality (2)": 2
}

EXANG_MAP = {
    "No (0)": 0,
    "Yes (1)": 1
}

SLOPE_MAP = {
    "Downsloping (0)": 0,
    "Flat (1)": 1,
    "Upsloping (2)": 2
}

CA_MAP = {
    "Four (0)": 0,
    "One (1)": 1,
    "Three (2)": 2,
    "Two (3)": 3,
    "Zero (4)": 4
}

THAL_MAP = {
    "Fixed Defect (0)": 0,
    "No (1)": 1,
    "Normal (2)": 2,
    "Reversable Defect (3)": 3
}

# ----------------------------------------------------
# CLINICAL PRESET PATIENTS FOR QUICK FILL
# ----------------------------------------------------
PRESETS = {
    "Custom Input (Reset)": None,
    "Patient A: Low Risk Profile (Healthy)": {
        "age": 35, "sex": "Female (0)", "cp": "Asymptomatic (0)", "bp": 110,
        "chol": 180, "fbs": "Lower than 120 mg/ml (1)", "ecg": "Normal (1)",
        "thalach": 175, "exang": "No (0)", "oldpeak": 0.0, "slope": "Upsloping (2)",
        "ca": "Zero (4)", "thal": "Normal (2)"
    },
    "Patient B: High Risk Profile (Critical Vitals)": {
        "age": 67, "sex": "Male (1)", "cp": "Typical Angina (3)", "bp": 145,
        "chol": 294, "fbs": "Greater than 120 mg/ml (0)", "ecg": "Normal (1)",
        "thalach": 115, "exang": "Yes (1)", "oldpeak": 2.6, "slope": "Flat (1)",
        "ca": "Zero (4)", "thal": "Reversable Defect (3)"
    },
    "Patient C: Moderate Risk Profile": {
        "age": 53, "sex": "Male (1)", "cp": "Atypical Angina (1)", "bp": 130,
        "chol": 240, "fbs": "Lower than 120 mg/ml (1)", "ecg": "ST-T Wave Abnormality (2)",
        "thalach": 142, "exang": "No (0)", "oldpeak": 1.2, "slope": "Flat (1)",
        "ca": "One (1)", "thal": "Normal (2)"
    }
}

# ----------------------------------------------------
# MAIN HEADER PAGE
# ----------------------------------------------------
st.markdown("""
<div class="header-card">
    <h1>❤️ CardioShield AI Dashboard</h1>
    <p>Clinical Decision Support System for Heart Disease Detection & Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# ----------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/822/822143.png", width=90)
st.sidebar.title("Navigation & Tools")

tab_choice = st.sidebar.radio(
    "Choose Dashboard View",
    ["📋 Patient Diagnosis Panel", "📊 Dataset Analytics", "🔬 Model Health & Weights"]
)

st.sidebar.divider()
st.sidebar.subheader("Clinical Patient Presets")
st.sidebar.caption("Use these presets to instantly populate valid patient variables:")
selected_preset = st.sidebar.selectbox("Load Patient Profile", list(PRESETS.keys()))

# Update values in Session State if a preset is selected
preset_data = PRESETS[selected_preset]
if preset_data is not None:
    for key, val in preset_data.items():
        st.session_state[f"inp_{key}"] = val

# ----------------------------------------------------
# TAB 1: PATIENT DIAGNOSIS PANEL
# ----------------------------------------------------
if tab_choice == "📋 Patient Diagnosis Panel":
    
    st.subheader("📋 Patient Clinical Variables Entry")
    st.write("Fill in the fields below. Categorical labels display their corresponding mathematical model code in brackets `(x)` to ensure diagnostic mapping transparency.")

    # 1. Expandable Medical Reference Table
    with st.expander("📖 View Medical Variable Glossary & Encodings"):
        st.markdown("""
        ### Medical Glossary & Mapping Scheme
        The machine learning model uses standardized medical terms representing specific diagnostic markers. Below is the mapping from descriptive clinical terms to the raw integer representations required by the scaled Logistic Regression classifier:

        | Variable | Type | Clinical Description | Raw Encodings Meaning |
        | :--- | :--- | :--- | :--- |
        | **Age** | Numerical | Patient age in years | Valid range: `1` to `120` |
        | **Sex** | Categorical | Biological sex of patient | `0` = Female, `1` = Male |
        | **Chest Pain Type** | Categorical | Type of subjective pain described by patient | `0` = Asymptomatic (No symptoms)<br>`1` = Atypical Angina (Non-cardiac chest pain)<br>`2` = Non-anginal Pain (Chest pain not related to heart)<br>`3` = Typical Angina (Classic heart chest pain) |
        | **Blood Pressure** | Numerical | Resting systolic blood pressure (mmHg) | Normal range: `90` to `200` |
        | **Cholesterol** | Numerical | Serum cholesterol levels (mg/dl) | Normal range: `100` to `600` |
        | **Fasting Blood Sugar** | Categorical | Fasting glucose level | `0` = Greater than 120 mg/ml (High)<br>`1` = Lower than 120 mg/ml (Normal) |
        | **Rest ECG** | Categorical | Resting electrocardiographic results | `0` = Left Ventricular Hypertrophy<br>`1` = Normal<br>`2` = ST-T Wave Abnormality |
        | **Max Heart Rate** | Numerical | Maximum heart rate achieved during stress test | Range: `60` to `220` bpm |
        | **Exercise Angina** | Categorical | Exercise-induced angina (chest pain on exertion) | `0` = No, `1` = Yes |
        | **Old Peak** | Numerical | ST depression induced by exercise relative to rest | Range: `0.0` to `10.0` |
        | **Slope** | Categorical | Slope of peak exercise ST segment | `0` = Downsloping, `1` = Flat, `2` = Upsloping |
        | **Vessels** | Categorical | Number of major colored vessels (`0-3`) by fluoroscopy | `0` = Four, `1` = One, `2` = Three, `3` = Two, `4` = Zero |
        | **Thalassemia** | Categorical | Blood genetic thalassemia type | `0` = Fixed Defect, `1` = No, `2` = Normal, `3` = Reversable Defect |
        """)

    # 2. Form Input Grid (3 Cards in Columns)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("👤 Demographics & Vitals")
        
        age = st.number_input(
            "Age (Years)", 
            min_value=1, max_value=120, value=50, step=1,
            key="inp_age", help="Patient's age in years."
        )
        
        sex_str = st.selectbox(
            "Biological Sex", 
            list(SEX_MAP.keys()), 
            index=1,
            key="inp_sex", help="Select biological sex. Female is encoded as 0, Male as 1."
        )
        
        bp = st.number_input(
            "Resting Blood Pressure (mmHg)", 
            min_value=50, max_value=250, value=120, step=1,
            key="inp_bp", help="Resting blood pressure measured in mmHg upon admission."
        )
        
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)", 
            min_value=80, max_value=600, value=200, step=1,
            key="inp_chol", help="Serum cholesterol level measured in mg/dl."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🩺 Symptoms & History")
        
        cp_str = st.selectbox(
            "Chest Pain Type", 
            list(CP_MAP.keys()), 
            index=3,
            key="inp_cp", help="Select the patient's chest pain classification."
        )
        
        fbs_str = st.selectbox(
            "Fasting Blood Sugar (> 120 mg/ml)", 
            list(FBS_MAP.keys()), 
            index=1,
            key="inp_fbs", help="Indicates if fasting blood sugar is greater than 120 mg/ml."
        )
        
        exang_str = st.selectbox(
            "Exercise Induced Angina", 
            list(EXANG_MAP.keys()), 
            index=0,
            key="inp_exang", help="Does the patient experience chest pain during physical exertion?"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("⚡ Diagnostic Tests")
        
        ecg_str = st.selectbox(
            "Resting ECG Results", 
            list(ECG_MAP.keys()), 
            index=1,
            key="inp_ecg", help="Electrocardiographic results at rest."
        )
        
        thalach = st.number_input(
            "Max Heart Rate Achieved", 
            min_value=50, max_value=250, value=150, step=1,
            key="inp_thalach", help="Maximum heart rate achieved during cardiac stress test."
        )
        
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)", 
            min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f",
            key="inp_oldpeak", help="ST depression induced by exercise relative to rest."
        )
        
        slope_str = st.selectbox(
            "ST Slope Type", 
            list(SLOPE_MAP.keys()), 
            index=2,
            key="inp_slope", help="Slope of peak exercise ST segment."
        )
        
        ca_str = st.selectbox(
            "Fluoroscopy Colored Vessels", 
            list(CA_MAP.keys()), 
            index=4,
            key="inp_ca", help="Number of major vessels colored by fluoroscopy."
        )
        
        thal_str = st.selectbox(
            "Thalassemia Status", 
            list(THAL_MAP.keys()), 
            index=2,
            key="inp_thal", help="Patient's thalassemia evaluation."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Model Prediction Execution
    st.divider()
    predict_btn = st.button("🚀 Run Diagnostic Assessment", type="primary", use_container_width=True)

    if predict_btn:
        if model is None or scaler is None:
            st.error("Model assets not loaded. Please run the training script.")
        else:
            # MAP dropdown text variables to their raw numeric equivalents
            sex = SEX_MAP[sex_str]
            cp = CP_MAP[cp_str]
            fbs = FBS_MAP[fbs_str]
            ecg = ECG_MAP[ecg_str]
            exang = EXANG_MAP[exang_str]
            slope = SLOPE_MAP[slope_str]
            ca = CA_MAP[ca_str]
            thal = THAL_MAP[thal_str]

            # Build feature array in EXACT dataset order
            sample = np.array([[
                age,
                sex,
                cp,
                bp,
                chol,
                fbs,
                ecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]])

            # Scale inputs
            sample_scaled = scaler.transform(sample)

            # Perform prediction
            result = model.predict(sample_scaled)
            probs = model.predict_proba(sample_scaled)[0] # [Prob_No_Disease, Prob_Disease]
            risk_percentage = probs[1] * 100

            # Display Results Cards
            st.subheader("🎯 Diagnostic Evaluation Result")

            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                # Risk Meter Visual Display
                st.metric(label="Calculated Risk Probability", value=f"{risk_percentage:.1f}%")
                if risk_percentage < 30:
                    st.progress(risk_percentage / 100, text="Low Risk Classification")
                elif risk_percentage < 70:
                    st.progress(risk_percentage / 100, text="Moderate Risk Classification")
                else:
                    st.progress(risk_percentage / 100, text="⚠️ High Risk Classification")

            with res_col2:
                # Clinical Recommendations & Diagnosis
                if result[0] == 1:
                    st.markdown(f"""
                    <div class="result-box danger">
                        <h3>⚠️ Heart Disease Indicated</h3>
                        <p><strong>Clinical Diagnostic Summary:</strong> The Logistic Regression classifier evaluated the patient's cardiovascular parameters and detected markers strongly associated with coronary heart disease (Probability: <strong>{risk_percentage:.1f}%</strong>).</p>
                        <hr style="border-top: 1px solid #feb2b2;">
                        <p><strong>Recommended Actionable Advisories:</strong></p>
                        <ul>
                            <li>Schedule an immediate follow-up consultation with a Board-Certified Cardiologist.</li>
                            <li>Order a comprehensive 12-lead Electrocardiogram (ECG) and cardiac enzyme panel.</li>
                            <li>Consider scheduling a nuclear stress test or cardiac catheterization (angiography) to examine blood vessel flow.</li>
                            <li>Instruct patient to avoid strenuous physical exertion pending further cardiovascular clinical clearances.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box success">
                        <h3>✅ No Heart Disease Indicated</h3>
                        <p><strong>Clinical Diagnostic Summary:</strong> The patient's parameters present a standard healthy configuration. The prediction classifier did not find significant clinical features indicative of active coronary disease (Probability: <strong>{risk_percentage:.1f}%</strong>).</p>
                        <hr style="border-top: 1px solid #c6f6d5;">
                        <p><strong>Recommended Actionable Advisories:</strong></p>
                        <ul>
                            <li>Support maintenance of regular heart-healthy physical activities (150 mins aerobic activity/week).</li>
                            <li>Recommend maintaining dietary controls (low saturated fats, high fiber).</li>
                            <li>Schedule a routine annual check-up to track lipid profiles and blood pressure levels.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            # Debugging box for clinician transparency
            with st.expander("🛠️ View Raw Scaled Feature Vector Submitted to Model"):
                raw_df = pd.DataFrame(sample, columns=[
                    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 
                    'cholestoral', 'fasting_blood_sugar', 'rest_ecg', 
                    'max_heart_rate_achieved', 'exercise_induced_angina', 
                    'oldpeak', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia'
                ])
                st.caption("Raw Features (with alphabetical mappings correctly applied):")
                st.dataframe(raw_df)
                
                scaled_df = pd.DataFrame(sample_scaled, columns=raw_df.columns)
                st.caption("Standard-Scaled Features passed into Logistic Regression:")
                st.dataframe(scaled_df)

# ----------------------------------------------------
# TAB 2: DATASET ANALYTICS
# ----------------------------------------------------
elif tab_choice == "📊 Dataset Analytics":
    st.subheader("📊 Dataset Exploratory Analysis")
    st.write("This tab provides descriptive statistics and interactive distributions of the clinical dataset used to train the classifier.")
    
    if df is not None:
        # Show statistical cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Patients", f"{df.shape[0]}")
        m_col2.metric("Male Patients", f"{df[df['sex'] == 'Male'].shape[0]}")
        m_col3.metric("Female Patients", f"{df[df['sex'] == 'Female'].shape[0]}")
        m_col4.metric("Diagnosed Heart Disease", f"{df[df['target'] == 1].shape[0]}")
        
        st.divider()
        
        # Grid of Charts
        g_col1, g_col2 = st.columns(2)
        
        with g_col1:
            st.subheader("💓 Heart Disease Rate by Chest Pain Type")
            # Calculate cross tab
            cp_data = df.groupby(['chest_pain_type', 'target']).size().unstack(fill_value=0)
            cp_data.columns = ['Healthy', 'Heart Disease']
            st.bar_chart(cp_data)
            st.caption("Notice how patients reporting 'Typical Angina' or 'Asymptomatic' pain levels correlate strongly to negative or positive classes.")
            
        with g_col2:
            st.subheader("📈 Age vs Serum Cholesterol Distribution")
            # Create a simple colored scatter chart using streamlit native charts or high quality plot
            # Let's map target to text for plotting
            plot_df = df.copy()
            plot_df['Diagnosis'] = plot_df['target'].map({0: 'Healthy', 1: 'Heart Disease'})
            
            # Using altair for high quality scatter plots
            import altair as alt
            chart = alt.Chart(plot_df).mark_circle(size=60).encode(
                x=alt.X('age:Q', title='Age (Years)'),
                y=alt.Y('cholestoral:Q', title='Cholesterol (mg/dl)'),
                color=alt.Color('Diagnosis:N', scale=alt.Scale(domain=['Healthy', 'Heart Disease'], range=['#38a169', '#e53e3e'])),
                tooltip=['age', 'cholestoral', 'sex', 'resting_blood_pressure', 'Diagnosis']
            ).properties(height=350).interactive()
            st.altair_chart(chart, use_container_width=True)
            st.caption("Interactive chart. Hover on points to see clinical specifics. Zoom/Pan is enabled.")
            
        st.subheader("📋 Dataset Preview (First 50 Patient Rows)")
        st.dataframe(df.head(50), use_container_width=True)
        
    else:
        st.warning("Training dataset 'heart diesease dataset.xlsx' was not found in the application directory. Cannot display analytics.")

# ----------------------------------------------------
# TAB 3: MODEL HEALTH & WEIGHTS
# ----------------------------------------------------
elif tab_choice == "🔬 Model Health & Weights":
    st.subheader("🔬 Model Health & Feature Weights")
    st.write("This section details the parameters and coefficients of the Logistic Regression classifier.")

    if model is not None and df is not None:
        st.markdown("""
        ### Underlying Predictive Classifier
        * **Model Type**: Logistic Regression (L2 Regularization, Solver: lbfgs)
        * **Iterations Limit**: 1000
        * **Scaler Profile**: Standard Scaler (Zero Mean, Unit Variance)
        """)

        st.divider()

        st.subheader("📊 Predictive Feature Weights (Coefficients)")
        st.write("Coefficients represent the impact of each standardized feature on the log-odds of a Heart Disease diagnosis. Positive coefficients increase the probability of heart disease, whereas negative coefficients act as protective indicators according to the model.")

        # Extract model coefficients
        coefs = model.coef_[0]
        # Feature order from training
        features = [
            'Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholestoral', 
            'Fasting Blood Sugar', 'Rest ECG', 'Max Heart Rate', 
            'Exercise Induced Angina', 'Old Peak', 'Slope', 'Vessels Colored', 'Thalassemia'
        ]
        
        coef_df = pd.DataFrame({
            'Clinical Variable': features,
            'Coefficient Weight': coefs
        }).sort_values(by='Coefficient Weight', ascending=True)

        # Plot coefficients beautifully using altair
        import altair as alt
        
        bar_chart = alt.Chart(coef_df).mark_bar().encode(
            x=alt.X('Coefficient Weight:Q', title='Model Weight (Coefficient Value)'),
            y=alt.Y('Clinical Variable:N', sort='-x', title='Variable'),
            color=alt.condition(
                alt.datum['Coefficient Weight'] > 0,
                alt.value('#e53e3e'), # Red for positive correlation
                alt.value('#38a169')  # Green for negative correlation
            )
        ).properties(height=400)
        
        st.altair_chart(bar_chart, use_container_width=True)
        st.caption("Interpretation: Features with red bars contribute to higher risk probabilities. Features with green bars are negatively correlated (protective markers) under the model's current weights.")

    else:
        st.warning("Model objects or dataset not found. Please ensure train_model.py is executed.")
