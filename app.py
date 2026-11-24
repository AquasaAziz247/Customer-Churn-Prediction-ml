
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }

    .stButton > button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        background-color: #FF6B6B;
        border: none;
    }

    h1 {
        color: #1f77b4;
    }

    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD MODEL AND PREPROCESSOR
# ============================================================

@st.cache_resource
def load_models():

    """Load trained model and preprocessor."""

    try:

        model_files = list(
            Path("models").glob("best_model_*.pkl")
        )

        if not model_files:
            st.error("No trained model found!")
            return None, None

        model_path = model_files[0]

        preprocessor_path = Path(
            "models/preprocessor.pkl"
        )

        if not preprocessor_path.exists():
            st.error("Preprocessor file not found!")
            return None, None

        model = joblib.load(model_path)

        preprocessor = joblib.load(
            preprocessor_path
        )

        return model, preprocessor

    except Exception as e:

        st.error(f"Error loading models: {str(e)}")

        return None, None


# ============================================================
# 4. PREPROCESS INPUT DATA
# ============================================================

def preprocess_input(input_df, preprocessor):

    """Apply the same preprocessing used during model training."""

    df = input_df.copy()

    # --------------------------------------------------------
    # STEP 1: HANDLE MISSING VALUES
    # --------------------------------------------------------

    numeric_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

        df[col] = df[col].fillna(
            preprocessor.medians.get(col, 0)
        )

    # --------------------------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # --------------------------------------------------------

    # Tenure groups

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-np.inf, 12, 24, 48, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Average monthly charge per tenure

    df['avg_monthly_per_tenure'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )

    # Number of services

    service_cols = [
        'PhoneService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport'
    ]

    df['num_services'] = 0

    for col in service_cols:

        df['num_services'] += (
            df[col].astype(str).str.strip() == 'Yes'
        ).astype(int)

    df['num_services'] += (
        df['InternetService'].astype(str).str.strip() != 'No'
    ).astype(int)

    # --------------------------------------------------------
    # STEP 3: DROP CUSTOMER ID
    # --------------------------------------------------------

    df = df.drop(
        columns=['customerID'],
        errors='ignore'
    )

    # --------------------------------------------------------
    # STEP 4: BINARY ENCODING
    # --------------------------------------------------------

    binary_mappings = {

        'Partner': {'No': 0, 'Yes': 1},

        'Dependents': {'No': 0, 'Yes': 1},

        'PhoneService': {'No': 0, 'Yes': 1},

        'PaperlessBilling': {'No': 0, 'Yes': 1},

        'gender': {'Female': 0, 'Male': 1}
    }

    for col, mapping in binary_mappings.items():

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

    # --------------------------------------------------------
    # STEP 5: CATEGORICAL ENCODING
    # --------------------------------------------------------

    categorical_cols = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    for col in categorical_cols:

        mapping = preprocessor.label_encoders.get(
            col,
            {}
        )

        values = (
            df[col]
            .fillna('Unknown')
            .astype(str)
            .str.strip()
        )

        df[col] = (
            values
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

    # --------------------------------------------------------
    # STEP 6: FEATURE ORDERING
    # --------------------------------------------------------

    expected_cols = [

        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'tenure',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod',
        'MonthlyCharges',
        'TotalCharges',
        'tenure_group',
        'avg_monthly_per_tenure',
        'num_services'
    ]

    df = df.reindex(
        columns=expected_cols
    )

    # --------------------------------------------------------
    # STEP 7: SCALE NUMERICAL FEATURES
    # --------------------------------------------------------

    numerical_cols = preprocessor.numerical_cols

    df[numerical_cols] = (
        preprocessor.scaler.transform(
            df[numerical_cols]
        )
    )

    return df


# ============================================================
# 5. MAIN APPLICATION
# ============================================================

def main():

    st.title("🔮 Customer Churn Prediction System")

    st.markdown(
        "### Predict if a customer will churn based on their profile and usage patterns"
    )

    st.markdown("---")

    # LOAD MODEL

    model, preprocessor = load_models()

    if model is None or preprocessor is None:
        st.stop()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.header("Model Information")

        st.markdown("""
        **Best Model:** Gradient Boosting

        **ROC-AUC Score:** 0.8450

        **Accuracy:** 80.55%

        **F1-Score:** 58.48%
        """)

        st.markdown("---")

        st.markdown("### 📖 How to Use")

        st.markdown("""
        1. Fill in customer details.
        2. Click Predict Churn.
        3. View prediction and probability.
        4. Check key risk factors.
        """)

        st.markdown("---")

        st.markdown("### 💡 Quick Test Profiles")

        if st.button("Load High-Risk Profile"):
            st.session_state.test_profile = "high_risk"

        if st.button("Load Low-Risk Profile"):
            st.session_state.test_profile = "low_risk"

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if 'test_profile' not in st.session_state:
        st.session_state.test_profile = None

    # --------------------------------------------------------
    # DEFAULT CUSTOMER PROFILES
    # --------------------------------------------------------

    if st.session_state.test_profile == "high_risk":

        defaults = {
            'gender': 'Female',
            'senior_citizen': 'No',
            'partner': 'No',
            'dependents': 'No',
            'tenure': 3,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'Fiber optic',
            'online_security': 'No',
            'online_backup': 'No',
            'device_protection': 'No',
            'tech_support': 'No',
            'streaming_tv': 'Yes',
            'streaming_movies': 'Yes',
            'contract': 'Month-to-month',
            'paperless_billing': 'Yes',
            'payment_method': 'Electronic check',
            'monthly_charges': 85.0,
            'total_charges': 255.0
        }

    elif st.session_state.test_profile == "low_risk":

        defaults = {
            'gender': 'Male',
            'senior_citizen': 'No',
            'partner': 'Yes',
            'dependents': 'Yes',
            'tenure': 48,
            'phone_service': 'Yes',
            'multiple_lines': 'Yes',
            'internet_service': 'Fiber optic',
            'online_security': 'Yes',
            'online_backup': 'Yes',
            'device_protection': 'Yes',
            'tech_support': 'Yes',
            'streaming_tv': 'Yes',
            'streaming_movies': 'Yes',
            'contract': 'Two year',
            'paperless_billing': 'No',
            'payment_method': 'Credit card (automatic)',
            'monthly_charges': 105.0,
            'total_charges': 5040.0
        }

    else:

        defaults = {
            'gender': 'Male',
            'senior_citizen': 'No',
            'partner': 'No',
            'dependents': 'No',
            'tenure': 12,
            'phone_service': 'Yes',
            'multiple_lines': 'No',
            'internet_service': 'DSL',
            'online_security': 'No',
            'online_backup': 'No',
            'device_protection': 'No',
            'tech_support': 'No',
            'streaming_tv': 'No',
            'streaming_movies': 'No',
            'contract': 'Month-to-month',
            'paperless_billing': 'Yes',
            'payment_method': 'Electronic check',
            'monthly_charges': 70.0,
            'total_charges': 840.0
        }

    # --------------------------------------------------------
    # CUSTOMER INPUT AND PREDICTION LAYOUT
    # --------------------------------------------------------

    col1, col2 = st.columns([2, 1])

    with col1:

        st.header("📝 Customer Information")

        with st.form("customer_form"):

            # DEMOGRAPHICS

            st.subheader("👤 Demographics")

            demo_col1, demo_col2, demo_col3 = st.columns(3)

            with demo_col1:

                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"],
                    index=0 if defaults['gender'] == "Male" else 1
                )

                senior_citizen = st.selectbox(
                    "Senior Citizen",
                    ["No", "Yes"],
                    index=0 if defaults['senior_citizen'] == "No" else 1
                )

            with demo_col2:

                partner = st.selectbox(
                    "Has Partner",
                    ["No", "Yes"],
                    index=0 if defaults['partner'] == "No" else 1
                )

                dependents = st.selectbox(
                    "Has Dependents",
                    ["No", "Yes"],
                    index=0 if defaults['dependents'] == "No" else 1
                )

            with demo_col3:

                tenure = st.slider(
                    "Tenure (months)",
                    0,
                    72,
                    defaults['tenure']
                )

            st.markdown("---")

            # ACCOUNT INFORMATION

            st.subheader("💳 Account Information")

            acc_col1, acc_col2, acc_col3 = st.columns(3)

            contract_options = [
                "Month-to-month",
                "One year",
                "Two year"
            ]

            with acc_col1:

                contract = st.selectbox(
                    "Contract Type",
                    contract_options,
                    index=contract_options.index(
                        defaults['contract']
                    )
                )

                paperless_billing = st.selectbox(
                    "Paperless Billing",
                    ["No", "Yes"],
                    index=0 if defaults['paperless_billing'] == "No" else 1
                )

            payment_options = [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]

            with acc_col2:

                payment_method = st.selectbox(
                    "Payment Method",
                    payment_options,
                    index=payment_options.index(
                        defaults['payment_method']
                    )
                )

            with acc_col3:

                monthly_charges = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    max_value=200.0,
                    value=defaults['monthly_charges'],
                    step=5.0
                )

                total_charges = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=defaults['total_charges'],
                    step=50.0
                )

            st.markdown("---")

            # SERVICES

            st.subheader("📞 Services")

            serv_col1, serv_col2 = st.columns(2)

            with serv_col1:

                phone_service = st.selectbox(
                    "Phone Service",
                    ["No", "Yes"],
                    index=0 if defaults['phone_service'] == "No" else 1
                )

                multiple_lines = st.selectbox(
                    "Multiple Lines",
                    ["No", "Yes", "No phone service"],
                    index=[
                        "No",
                        "Yes",
                        "No phone service"
                    ].index(defaults['multiple_lines'])
                )

                internet_options = [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]

                internet_service = st.selectbox(
                    "Internet Service",
                    internet_options,
                    index=internet_options.index(
                        defaults['internet_service']
                    )
                )

            with serv_col2:

                online_security = st.selectbox(
                    "Online Security",
                    ["No", "Yes", "No internet service"],
                    index=[
                        "No",
                        "Yes",
                        "No internet service"
                    ].index(defaults['online_security'])
                )

                online_backup = st.selectbox(
                    "Online Backup",
                    ["No", "Yes", "No internet service"],
                    index=[
                        "No",
                        "Yes",
                        "No internet service"
                    ].index(defaults['online_backup'])
                )

                device_protection = st.selectbox(
                    "Device Protection",
                    ["No", "Yes", "No internet service"],
                    index=[
                        "No",
                        "Yes",
                        "No internet service"
                    ].index(defaults['device_protection'])
                )

            serv_col3, serv_col4 = st.columns(2)

            with serv_col3:

                tech_support = st.selectbox(
                    "Tech Support",
                    ["No", "Yes", "No internet service"],
                    index=[
                        "No",
                        "Yes",
                        "No internet service"
                    ].index(defaults['tech_support'])
                )

            with serv_col4:

                streaming_tv = st.selectbox(
                    "Streaming TV",
                    ["No", "Yes", "No internet service"],
                    index=[
                        "No",
                        "Yes",
                        "No internet service"
                    ].index(defaults['streaming_tv'])
                )

                streaming_movies = st.selectbox(
                    "Streaming Movies",
                    ["No", "Yes", "No internet service"],
                    index=[
                        "No",
                        "Yes",
                        "No internet service"
                    ].index(defaults['streaming_movies'])
                )

            # SUBMIT BUTTON

            submitted = st.form_submit_button(
                "🔮 Predict Churn",
                use_container_width=True
            )

    # --------------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------------

    with col2:

        st.header("🎯 Prediction Results")

        if submitted:

            st.session_state.test_profile = None

            # CREATE INPUT DATAFRAME

            input_data = pd.DataFrame({

                'customerID': ['PRED-001'],

                'gender': [gender],

                'SeniorCitizen': [
                    1 if senior_citizen == "Yes" else 0
                ],

                'Partner': [partner],
                'Dependents': [dependents],

                'tenure': [tenure],

                'PhoneService': [phone_service],
                'MultipleLines': [multiple_lines],

                'InternetService': [internet_service],

                'OnlineSecurity': [online_security],
                'OnlineBackup': [online_backup],
                'DeviceProtection': [device_protection],
                'TechSupport': [tech_support],

                'StreamingTV': [streaming_tv],
                'StreamingMovies': [streaming_movies],

                'Contract': [contract],

                'PaperlessBilling': [paperless_billing],
                'PaymentMethod': [payment_method],

                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges]
            })

            try:

                # PREPROCESS INPUT

                processed_data = preprocess_input(
                    input_data,
                    preprocessor
                )

                # DEBUG INFORMATION

                with st.expander("🔍 Debug Info (Click to expand)"):

                    st.write(
                        "**Processed data shape:**",
                        processed_data.shape
                    )

                    st.write(
                        "**Processed columns:**",
                        processed_data.columns.tolist()
                    )

                    st.write("**Sample values:**")

                    st.dataframe(processed_data)

                # VERIFY MODEL FEATURES

                if processed_data.shape[1] != model.n_features_in_:

                    raise ValueError(
                        f"Expected {model.n_features_in_} features, "
                        f"but received {processed_data.shape[1]}."
                    )

                if list(processed_data.columns) != list(model.feature_names_in_):

                    raise ValueError(
                        "Input feature order does not match model training order."
                    )

                # MAKE PREDICTION

                prediction = model.predict(
                    processed_data
                )[0]

                prediction_proba = model.predict_proba(
                    processed_data
                )[0]

                churn_probability = prediction_proba[1] * 100

                no_churn_probability = prediction_proba[0] * 100

                # ------------------------------------------------
                # DISPLAY PREDICTION USING STREAMLIT COMPONENTS
                # ------------------------------------------------

                if prediction == 1:

                    with st.container(border=True):

                        st.error("⚠️ HIGH RISK")

                        st.subheader(
                            "Customer Likely to Churn"
                        )

                        st.metric(
                            "Churn Probability",
                            f"{churn_probability:.1f}%"
                        )

                    st.warning(
                        "Recommendation: Implement retention strategies immediately!"
                    )

                    st.markdown("""
                    **Suggested Actions:**

                    - Offer special discount or promotion
                    - Reach out to customer support team
                    - Propose contract upgrade with benefits
                    - Survey customer satisfaction
                    """)

                else:

                    with st.container(border=True):

                        st.success("✅ LOW RISK")

                        st.subheader(
                            "Customer Likely to Stay"
                        )

                        st.metric(
                            "Retention Probability",
                            f"{no_churn_probability:.1f}%"
                        )

                    st.success(
                        "Status: Customer retention looks good!"
                    )

                    st.markdown("""
                    **Continue to:**

                    - Maintain service quality
                    - Engage with loyalty programs
                    - Regular satisfaction checks
                    """)

                # ------------------------------------------------
                # PROBABILITY CHART
                # ------------------------------------------------

                st.markdown("### 📊 Confidence Breakdown")

                fig = go.Figure(

                    go.Bar(
                        x=[
                            no_churn_probability,
                            churn_probability
                        ],

                        y=[
                            'Will Stay',
                            'Will Churn'
                        ],

                        orientation='h',

                        marker=dict(
                            color=[
                                '#4caf50',
                                '#f44336'
                            ]
                        ),

                        text=[
                            f'{no_churn_probability:.1f}%',
                            f'{churn_probability:.1f}%'
                        ],

                        textposition='auto'
                    )
                )

                fig.update_layout(

                    height=200,

                    margin=dict(
                        l=0,
                        r=0,
                        t=0,
                        b=0
                    ),

                    xaxis=dict(
                        range=[0, 100],
                        showticklabels=False
                    ),

                    yaxis=dict(
                        showticklabels=True
                    ),

                    showlegend=False
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # ------------------------------------------------
                # RISK FACTORS
                # ------------------------------------------------

                st.markdown("### 🎯 Key Risk Factors")

                risk_factors = []

                if contract == "Month-to-month":
                    risk_factors.append(
                        "⚠️ Month-to-month contract"
                    )

                if tenure < 12:
                    risk_factors.append(
                        "⚠️ Short tenure (< 1 year)"
                    )

                if monthly_charges > 80:
                    risk_factors.append(
                        "⚠️ High monthly charges"
                    )

                if online_security == "No":
                    risk_factors.append(
                        "⚠️ No online security"
                    )

                if tech_support == "No":
                    risk_factors.append(
                        "⚠️ No tech support"
                    )

                if payment_method == "Electronic check":
                    risk_factors.append(
                        "⚠️ Electronic check payment"
                    )

                if partner == "No":
                    risk_factors.append(
                        "⚠️ No partner"
                    )

                if risk_factors:

                    for factor in risk_factors:
                        st.markdown(f"- {factor}")

                else:

                    st.markdown(
                        "✅ No major risk factors identified"
                    )

            except Exception as e:

                st.error(
                    f"❌ Prediction Error: {str(e)}"
                )

                import traceback

                st.code(
                    traceback.format_exc()
                )

        else:

            st.info(
                "👈 Fill in the customer details and click "
                "**Predict Churn** to see results"
            )

            st.markdown("### 💡 Quick Start")

            st.markdown("""
            Use the sidebar buttons to load test profiles:

            - **High-Risk Profile:** New customer, month-to-month, no services
            - **Low-Risk Profile:** Long tenure, 2-year contract, all services
            """)


# ============================================================
# 6. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
