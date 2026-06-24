import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# PAGE CONFIG
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# LOAD MODEL
model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")


# CSS 
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

            
.block-container{
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

header[data-testid="stHeader"]{
    background: transparent !important;
}

.main-header{
    background: linear-gradient(90deg,#2563eb,#06b6d4,#7c3aed);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
    color: white;
}

.kpi-card{
    background: linear-gradient(145deg,#111827,#1e293b);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 2px solid transparent;
    background:
        linear-gradient(#111827,#111827) padding-box,
        linear-gradient(135deg,#2563eb,#06b6d4,#7c3aed) border-box;
    box-shadow: 0px 4px 15px rgba(37,99,235,0.25);
}

.kpi-title{
    color:#94a3b8;
    font-size:14px;
}

.kpi-value{
    color:white;
    font-size:28px;
    font-weight:bold;
    margin-top:5px;
}

[data-testid="stSidebar"]{
    background-color:#0b1220;
}

.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
    border:none;
    padding:10px;
    border-radius:10px;
    font-weight:bold;
}

.stButton>button:hover{
    transform:scale(1.02);
}

[data-testid="metric-container"]{
    background:#1e293b;
    border-radius:10px;
    padding:10px;
    border:1px solid #334155;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

[data-testid="stIconMaterial"]{
    color: #06b6d4 !important;
}

[data-testid="stExpandSidebarButton"]:hover
[data-testid="stIconMaterial"]{
    color: #7c3aed !important;
}

[data-testid="stExpandSidebarButton"]{
    background: transparent !important;
}
            

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Intelligence Dashboard</h1>
    <p>AI Powered Customer Retention System</p>
</div>
""", unsafe_allow_html=True)


# KPI CARDS
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">👥 Customers</div>
        <div class="kpi-value">7,043</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">📉 Churn Rate</div>
        <div class="kpi-value">26.5%</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">📅 Avg Tenure</div>
        <div class="kpi-value">32</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">💰 Revenue</div>
        <div class="kpi-value">$16.1M</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# INPUTS 
st.sidebar.title("⚙️ Customer Inputs")

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

contract = st.sidebar.selectbox(
    "Contract",
    encoders["Contract"].classes_
)

internet = st.sidebar.selectbox(
    "Internet Service",
    encoders["InternetService"].classes_
)

payment = st.sidebar.selectbox(
    "Payment Method",
    encoders["PaymentMethod"].classes_
)

# MAIN LAYOUT
col1, col2 = st.columns([1, 1.5])


# PREDICTION
with col1:

    st.subheader("🔮 Prediction Panel")

    if st.button("🚀 Predict Churn"):

        input_data = pd.DataFrame({

            "tenure": [tenure],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total],

            "Contract": [encoders["Contract"].transform([contract])[0]],
            "InternetService": [encoders["InternetService"].transform([internet])[0]],
            "PaymentMethod": [encoders["PaymentMethod"].transform([payment])[0]]
        })

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        confidence = max(prob) * 100

        if prediction == 1:
            st.error("⚠ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")

        st.metric("Confidence Score", f"{confidence:.2f}%")


# CHART
with col2:

    st.subheader("📈 Feature Importance")

    feature_df = pd.DataFrame({
        "Feature": [
            "Tenure",
            "Contract",
            "Monthly Charges",
            "Internet",
            "Payment",
            "Total Charges"
        ],
        "Importance": [0.40, 0.25, 0.15, 0.10, 0.05, 0.05]
    })

    fig = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)


# DATA TABLE
st.markdown("---")

st.subheader("📋 Input Summary")

st.dataframe(pd.DataFrame({
    "Feature":[
        "Tenure","Monthly Charges","Total Charges",
        "Contract","Internet Service","Payment Method"
    ],
    "Value":[
        tenure, monthly, total,
        contract, internet, payment
    ]
}), use_container_width=True)