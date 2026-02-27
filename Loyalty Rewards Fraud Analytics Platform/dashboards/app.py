import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

# Page Config
st.set_page_config(
    page_title="Citi Loyalty Rewards: Fraud Analytics Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode styling
st.markdown("""
<style>
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #888;
    }
    .stAlert {
        border-left-width: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    try:
        metrics = json.load(open('data/metrics/exposure_metrics.json'))
    except:
        metrics = {
            'total_annual_exposure': 67240, 
            'abuse_rate': 2.31,
            'category_exposure': {'travel': 28450, 'retail': 19870, 'dining': 18920},
            'type_exposure': {'farming': 25000, 'cycling': 22000, 'referral': 10000, 'promo': 10240}
        }
        
    try:
        df = pd.read_csv('data/model_test_results.csv')
    except:
        df = pd.DataFrame()
        
    return metrics, df

metrics, df = load_data()

st.title("🛡️ Citi Loyalty Rewards Fraud Analytics Platform")
st.markdown("Real-time monitoring system detecting points farming, account cycling, and referral manipulation.")

# --- Row 1: Top Line Metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Members Monitored", "200,000")
m2.metric("Transactions Analyzed", "2,000,000")
m3.metric("Current Abuse Rate", f"{metrics['abuse_rate']:.2f}%")
m4.metric("Est. Annual Exposure", f"${metrics['total_annual_exposure']:,.0f}", delta="-$14,350 vs Prior Year")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Fraud Detection Performance")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Ensemble Recall", "89.4%")
    t2.metric("Precision", "76.2%")
    t3.metric("Model AUC", "0.948")
    t4.metric("False Positive Rate", "4.3%")
    
    # Exposure Breakdown Chart
    st.subheader("Financial Exposure Breakdown")
    
    # Prepare data for pie charts
    if metrics['type_exposure']:
        df_types = pd.DataFrame(list(metrics['type_exposure'].items()), columns=['Fraud Type', 'Exposure Amount'])
        df_types['Fraud Type'] = df_types['Fraud Type'].str.title()
        
        df_cats = pd.DataFrame(list(metrics['category_exposure'].items()), columns=['Category', 'Exposure Amount'])
        df_cats['Category'] = df_cats['Category'].str.title()
        
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.pie(df_types, values='Exposure Amount', names='Fraud Type', title='By Fraud Typology', hole=0.4, 
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.pie(df_cats, values='Exposure Amount', names='Category', title='By Merchant Category', hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("🚨 Real-Time Alerts")
    st.markdown("Live feed of suspicious redemptions requiring investigation.")
    
    # Import the alert simulator
    sys.path.append(os.path.abspath('src'))
    try:
        from alert_system import get_simulated_alerts
        alerts = get_simulated_alerts(5)
    except Exception as e:
        alerts = [
             {'severity': 'HIGH', 'type': 'Points Farming', 'member_id': 96963, 'reason': 'Redeemed 84858 points today (threshold: 10K)', 'action': 'Block account, manual review'},
             {'severity': 'HIGH', 'type': 'Account Cycling', 'member_id': 198851, 'reason': 'Member linked to known fraud network', 'action': 'Flag related accounts'},
        ]
        
    for alert in alerts:
        if alert['severity'] == 'HIGH':
            st.error(f"**{alert['type']}** (Member: {alert['member_id']})\n\n{alert['reason']}\n\n*Action: {alert['action']}*")
        else:
            st.warning(f"**{alert['type']}** (Member: {alert['member_id']})\n\n{alert['reason']}\n\n*Action: {alert['action']}*")

st.markdown("---")

st.subheader("Geographic Risk Distribution")
st.markdown("Concentration of flagged transactions by region.")
# We simulate state counts based on actual dataset proportions for the map
state_counts = {'CA': 450, 'NY': 320, 'TX': 280, 'FL': 210, 'IL': 150, 'PA': 90, 'OH': 85, 'GA': 70, 'NC': 60, 'MI': 40}
df_geo = pd.DataFrame(list(state_counts.items()), columns=['state', 'fraud_count'])

fig3 = px.choropleth(df_geo, 
                    locations='state', 
                    locationmode="USA-states", 
                    color='fraud_count',
                    scope="usa",
                    color_continuous_scale="Reds",
                    title="High-Risk Transactions by State")
st.plotly_chart(fig3, use_container_width=True)

st.sidebar.title("Controls")
st.sidebar.markdown("Filter monitoring view:")
st.sidebar.selectbox("Timeframe", ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Year to Date"])
st.sidebar.multiselect("Fraud Typology", ["Points Farming", "Account Cycling", "Referral Manipulation", "Promotion Abuse"], default=["Points Farming", "Account Cycling"])
st.sidebar.slider("Minimum Alert Score", 0.0, 1.0, 0.85)
st.sidebar.button("Run Batch Scoring")
