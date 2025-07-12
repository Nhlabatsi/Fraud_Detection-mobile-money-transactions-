import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from collections import deque
import json
import io
import random 

# Constants
REFRESH_INTERVAL = 2  # seconds
MAX_DISPLAY_TRANSACTIONS = 200
MODEL_METRICS_HISTORY = 30
API_BASE_URL = "http://localhost:8000"  # Update with your API URL

# Configure Streamlit
st.set_page_config(
    page_title="Real Time Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .dashboard-header {
        background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        animation: pulse 2s infinite;
    }
    .warning-transaction {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .real-time-indicator {
        background-color: #f44336;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        font-size: 0.8rem;
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(244, 67, 54, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0); }
    }
    .transaction-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }
    .model-metrics {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .tab-content {
        padding-top: 1rem;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# API Client
class FraudDetectionClient:
    @staticmethod
    def get_health():
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    @staticmethod
    def get_agents():
        try:
            response = requests.get(f"{API_BASE_URL}/agents/list", timeout=5)
            return response.json().get("agents", {}) if response.status_code == 200 else {}
        except:
            return {}
    
    @staticmethod
    def create_agent(name, agent_type, risk_tolerance):
        try:
            response = requests.post(
                f"{API_BASE_URL}/agents/create",
                json={"name": name, "type": agent_type, "risk_tolerance": risk_tolerance},
                timeout=5
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    @staticmethod
    def process_transaction(agent_id, transaction_type="normal"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/agents/{agent_id}/process_transaction",
                json={"transaction_type": transaction_type},
                timeout=5
            )
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    @staticmethod
    def get_statistics():
        try:
            response = requests.get(f"{API_BASE_URL}/statistics", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    @staticmethod
    def get_agent_transactions(agent_id, limit=100):
        try:
            response = requests.get(
                f"{API_BASE_URL}/agents/{agent_id}/transactions?limit={limit}",
                timeout=5
            )
            return response.json().get("transactions", []) if response.status_code == 200 else []
        except:
            return []
    
    @staticmethod
    def get_agent_alerts(agent_id, limit=100):
        try:
            response = requests.get(
                f"{API_BASE_URL}/agents/{agent_id}/alerts?limit={limit}",
                timeout=5
            )
            return response.json().get("alerts", []) if response.status_code == 200 else []
        except:
            return []

# Utility Functions
def calculate_risk_level(score):
    if score >= 0.8: return "CRITICAL"
    elif score >= 0.6: return "HIGH"
    elif score >= 0.4: return "MEDIUM"
    elif score >= 0.2: return "LOW"
    return "MINIMAL"

def risk_level_color(level):
    colors = {
        "CRITICAL": "#d32f2f",
        "HIGH": "#f57c00",
        "MEDIUM": "#ffb300",
        "LOW": "#7cb342",
        "MINIMAL": "#388e3c"
    }
    return colors.get(level, "#757575")

def process_new_transactions():
    """Fetch and process new transactions from API"""
    current_time = time.time()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = current_time
    
    if current_time - st.session_state.last_refresh >= REFRESH_INTERVAL:
        if 'selected_agent' in st.session_state and st.session_state.selected_agent:
            # Get new transaction from API
            response = FraudDetectionClient.process_transaction(
                st.session_state.selected_agent,
                np.random.choice(["normal", "suspicious", "fraudulent"], p=[0.8, 0.15, 0.05])
            )
            
            if response and response.get("status") == "success":
                transaction = response["transaction"]
                prediction = response["prediction"]
                
                # Convert timestamp if needed
                if isinstance(transaction["Timestamp"], str):
                    transaction["Timestamp"] = datetime.fromisoformat(transaction["Timestamp"])
                
                # Update transaction history
                if 'transactions' not in st.session_state:
                    st.session_state.transactions = deque(maxlen=MAX_DISPLAY_TRANSACTIONS)
                st.session_state.transactions.appendleft(transaction)
                
                # Update fraud alerts if needed
                if prediction["prediction"]:
                    if 'fraud_alerts' not in st.session_state:
                        st.session_state.fraud_alerts = deque(maxlen=50)
                    st.session_state.fraud_alerts.appendleft(transaction)
                
                # Update system stats
                if 'system_stats' not in st.session_state:
                    st.session_state.system_stats = {
                        "total_processed": 0,
                        "total_fraud": 0,
                        "avg_processing_time": 0
                    }
                
                st.session_state.system_stats["total_processed"] += 1
                if prediction["prediction"]:
                    st.session_state.system_stats["total_fraud"] += 1
                
                st.session_state.last_refresh = current_time
                st.rerun()

# Initialize session state
if 'agents' not in st.session_state:
    st.session_state.agents = FraudDetectionClient.get_agents()
if 'selected_agent' not in st.session_state and st.session_state.agents:
    st.session_state.selected_agent = next(iter(st.session_state.agents.keys()))
if 'transactions' not in st.session_state:
    st.session_state.transactions = deque(maxlen=MAX_DISPLAY_TRANSACTIONS)
if 'fraud_alerts' not in st.session_state:
    st.session_state.fraud_alerts = deque(maxlen=50)
if 'system_stats' not in st.session_state:
    st.session_state.system_stats = {
        "total_processed": 0,
        "total_fraud": 0,
        "avg_processing_time": 0
    }
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {
        "accuracy": deque(maxlen=MODEL_METRICS_HISTORY),
        "precision": deque(maxlen=MODEL_METRICS_HISTORY),
        "recall": deque(maxlen=MODEL_METRICS_HISTORY),
        "f1": deque(maxlen=MODEL_METRICS_HISTORY),
        "latency": deque(maxlen=MODEL_METRICS_HISTORY),
        "timestamps": deque(maxlen=MODEL_METRICS_HISTORY)
    }
if 'api_status' not in st.session_state:
    st.session_state.api_status = FraudDetectionClient.get_health()
if 'real_time_mode' not in st.session_state:
    st.session_state.real_time_mode = True

# Dashboard Layout
def main():
    st.markdown("""
    <div class="dashboard-header">
        <h1 style="margin:0;padding:0;">Enterprise Fraud Detection Dashboard</h1>
        <p style="margin:0;padding:0;font-size:1.1rem;">Real-time transaction monitoring & AI-powered fraud detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API status indicator
    api_status = FraudDetectionClient.get_health()
    status_color = "#4CAF50" if api_status and api_status.get("model_loaded") else "#F44336"
    status_text = "CONNECTED" if api_status and api_status.get("model_loaded") else "DISCONNECTED"
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        st.markdown(f'<div style="color:{status_color}; font-weight:bold;">API STATUS: {status_text}</div>', unsafe_allow_html=True)
    with col2:
        if st.session_state.real_time_mode:
            st.markdown('<div class="real-time-indicator">LIVE PROCESSING</div>', unsafe_allow_html=True)
    with col3:
        st.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    # System metrics
    stats = FraudDetectionClient.get_statistics() or {}
    fraud_rate = (stats.get("fraud_count", 0) / max(1, stats.get("total_transactions", 1))) * 100
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Processed", f"{stats.get('total_transactions', 0):,}")
    m2.metric("Fraud Detected", f"{stats.get('fraud_count', 0):,}", f"{fraud_rate:.2f}%")
    m3.metric("Avg Risk Score", f"{stats.get('average_fraud_score', 0):.3f}")
    m4.metric("Last Hour", f"{stats.get('last_hour', {}).get('transactions', 0)} tx, {stats.get('last_hour', {}).get('fraud_count', 0)} fraud")
    m5.metric("Model Version", api_status.get('version', 'N/A') if api_status else 'N/A')
    
    # Sidebar
    st.sidebar.header("Agent Management")
    
    # Agent selection
    if st.session_state.agents:
        agent_options = {f"{info['name']} ({id})": id for id, info in st.session_state.agents.items()}
        selected_display = st.sidebar.selectbox(
            "Select Agent",
            options=list(agent_options.keys()),
            index=0 if 'selected_agent' not in st.session_state else 
                  list(agent_options.values()).index(st.session_state.selected_agent) 
                  if st.session_state.selected_agent in agent_options.values() else 0
        )
        st.session_state.selected_agent = agent_options[selected_display]
        
        # Agent info card
        agent_info = st.session_state.agents[st.session_state.selected_agent]
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <h3>{agent_info['name']}</h3>
            <p><strong>Type:</strong> {agent_info['type']}</p>
            <p><strong>Risk Tolerance:</strong> {agent_info['risk_tolerance']:.2f}</p>
            <p><strong>Status:</strong> {agent_info['status']}</p>
            <p><strong>Transactions:</strong> {agent_info['total_transactions']}</p>
            <p><strong>Fraud Detected:</strong> {agent_info['fraud_detected']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.info("No agents available. Create one to get started.")
    
    # Agent creation form
    with st.sidebar.expander("Create New Agent"):
        with st.form("agent_form"):
            name = st.text_input("Agent Name", placeholder="e.g., Regional Fraud Monitor")
            agent_type = st.selectbox("Agent Type", ["ATM", "Mobile Money Agent", "Bank Branch", "Online Payment Gateway"])
            risk_tolerance = st.slider("Risk Tolerance", 0.0, 1.0, 0.5, 0.01)
            
            if st.form_submit_button("Create Agent"):
                response = FraudDetectionClient.create_agent(name, agent_type, risk_tolerance)
                if response and response.get("status") == "success":
                    st.session_state.agents = FraudDetectionClient.get_agents()
                    st.session_state.selected_agent = response["agent_id"]
                    st.rerun()
                else:
                    st.error("Failed to create agent")
    
    # System controls
    st.sidebar.header("System Controls")
    st.session_state.real_time_mode = st.sidebar.checkbox("Enable Real-time Processing", value=st.session_state.real_time_mode)
    
    if st.sidebar.button("Refresh Data"):
        st.session_state.agents = FraudDetectionClient.get_agents()
        st.session_state.api_status = FraudDetectionClient.get_health()
        st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Transactions", 
        "🚨 Fraud Alerts", 
        "📈 Analytics", 
        "🤖 Model Performance"
    ])
    
    with tab1:
        st.subheader("Real-time Transaction Stream")
        
        if not st.session_state.transactions:
            st.info("Waiting for transactions to process...")
        else:
            # Transaction grid
            cols = st.columns(4)
            for i, txn in enumerate(list(st.session_state.transactions)[:8]):
                with cols[i % 4]:
                    risk_level = calculate_risk_level(txn.get("Risk_Score", 0))
                    risk_color = risk_level_color(risk_level)
                    with st.container():
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 4px solid {risk_color};">
                            <div style="display: flex; justify-content: space-between;">
                                <span style="font-weight: bold;">{txn.get('Transaction_ID', 'N/A')}</span>
                                <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;">
                                ${txn.get('Amount', 0):,.2f}
                            </div>
                            <div style="font-size: 0.8rem; color: #666;">
                                {txn.get('Subscriber_Name', 'N/A')}<br>
                                {txn.get('Timestamp', '').strftime('%H:%M:%S') if isinstance(txn.get('Timestamp'), datetime) else 'N/A'} • {txn.get('Distance', 0):.1f}km<br>
                                Score: {txn.get('Risk_Score', 0):.3f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Transaction timeline
            st.subheader("Transaction Timeline")
            df = pd.DataFrame(st.session_state.transactions)
            
            if not df.empty:
                # Convert timestamp if needed
                if isinstance(df["Timestamp"].iloc[0], str):
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                
                fig_timeline = go.Figure()
                
                # Add risk level bands
                fig_timeline.add_hrect(y0=0, y1=0.2, fillcolor="rgba(56, 142, 60, 0.1)", line_width=0, annotation_text="Minimal Risk")
                fig_timeline.add_hrect(y0=0.2, y1=0.4, fillcolor="rgba(123, 179, 66, 0.1)", line_width=0, annotation_text="Low Risk")
                fig_timeline.add_hrect(y0=0.4, y1=0.6, fillcolor="rgba(255, 179, 0, 0.1)", line_width=0, annotation_text="Medium Risk")
                fig_timeline.add_hrect(y0=0.6, y1=0.8, fillcolor="rgba(245, 124, 0, 0.1)", line_width=0, annotation_text="High Risk")
                fig_timeline.add_hrect(y0=0.8, y1=1.0, fillcolor="rgba(211, 47, 47, 0.1)", line_width=0, annotation_text="Critical Risk")
                
                # Add transactions
                fig_timeline.add_trace(go.Scatter(
                    x=df["Timestamp"],
                    y=df["Risk_Score"],
                    mode='markers+lines',
                    marker=dict(
                        size=10,
                        color=df["Risk_Score"],
                        colorscale='RdYlGn_r',
                        showscale=True,
                        cmin=0,
                        cmax=1,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name='Risk Score'
                ))
                
                # Highlight fraud
                fraud_df = df[df["Fraud_Label"] == 1]
                if not fraud_df.empty:
                    fig_timeline.add_trace(go.Scatter(
                        x=fraud_df["Timestamp"],
                        y=fraud_df["Risk_Score"],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color='#d32f2f',
                            symbol='x-thin',
                            line=dict(width=2, color='DarkSlateGrey')
                        ),
                        name='Fraud Detected'
                    ))
                
                fig_timeline.update_layout(
                    height=400,
                    xaxis_title="Time",
                    yaxis_title="Risk Score",
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        st.subheader("Recent Fraud Alerts")
        
        if not st.session_state.fraud_alerts:
            st.info("No fraud alerts detected yet")
        else:
            for alert in list(st.session_state.fraud_alerts)[:5]:
                risk_level = calculate_risk_level(alert.get("Risk_Score", 0))
                st.markdown(f"""
                <div class="fraud-alert">
                    <div style="display: flex; justify-content: space-between;">
                        <h4 style="margin: 0;">🚨 {risk_level} RISK TRANSACTION</h4>
                        <span style="font-weight: bold;">${alert.get('Amount', 0):,.2f}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.5rem;">
                        <div>
                            <strong>Transaction ID:</strong> {alert.get('Transaction_ID', 'N/A')}<br>
                            <strong>Customer:</strong> {alert.get('Subscriber_Name', 'N/A')}<br>
                            <strong>Agent:</strong> {alert.get('Agent_ID', 'N/A')}
                        </div>
                        <div>
                            <strong>Time:</strong> {alert.get('Timestamp', '').strftime('%H:%M:%S') if isinstance(alert.get('Timestamp'), datetime) else 'N/A'}<br>
                            <strong>Location:</strong> {alert.get('Transaction_Lat', 0):.2f}, {alert.get('Transaction_Lon', 0):.2f}<br>
                            <strong>Score:</strong> {alert.get('Risk_Score', 0):.3f}
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <strong>Indicators:</strong><br>
                        • Amount ${alert.get('Amount', 0):,.2f}<br>
                        • Distance {alert.get('Distance', 0):.1f}km<br>
                        • {'Weekend' if alert.get('IsWeekend') else 'Weekday'} • {'Business' if alert.get('IsBusinessHours') else 'Non-business'} hours
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Transaction Analytics")
        
        # Initialize df with empty DataFrame if no transactions exist
        if not st.session_state.transactions:
            df = pd.DataFrame(columns=[
                'Transaction_ID', 'Subscriber_ID', 'Agent_ID', 'Timestamp', 
                'Amount', 'Distance', 'Multi_Subscriber_Fraud', 'Subscriber_Name',
                'Gender', 'SIM_Card_ID', 'Agent_Longitude', 'Transaction_Lat',
                'Transaction_Lon', 'Money_Fraud_Label_agent', 'Date', 'Hour',
                'DayOfWeek', 'IsWeekend', 'IsBusinessHours', 'Risk_Score',
                'National_ID', 'Fraud_Label'
            ])
            st.info("No transaction data available yet")
        else:
            df = pd.DataFrame(st.session_state.transactions)
            # Convert timestamp if needed
            if isinstance(df["Timestamp"].iloc[0], str):
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Risk Score Distribution")
                fig_dist = px.histogram(
                    df,
                    x="Risk_Score",
                    nbins=20,
                    color="Fraud_Label",
                    color_discrete_map={1: "#d32f2f", 0: "#388e3c"},
                    labels={"Risk_Score": "Risk Score", "Fraud_Label": "Fraud"},
                    height=350
                )
                fig_dist.update_layout(bargap=0.1)
                st.plotly_chart(fig_dist, use_container_width=True)
                
                st.markdown("#### Fraud by Agent")
                fraud_counts = df[df["Fraud_Label"] == 1].groupby("Agent_ID").size().reset_index(name="counts")
                if not fraud_counts.empty:
                    fig_agent = px.bar(
                        fraud_counts,
                        x="Agent_ID",
                        y="counts",
                        color="Agent_ID",
                        height=350,
                        labels={"Agent_ID": "Agent ID", "counts": "Fraud Count"}
                    )
                    st.plotly_chart(fig_agent, use_container_width=True)
                else:
                    st.info("No fraud detected by agents yet")
            
            with col2:
                st.markdown("#### Amount vs Risk Score")
                fig_scatter = px.scatter(
                    df,
                    x="Amount",
                    y="Risk_Score",
                    color="Fraud_Label",
                    color_discrete_map={1: "#d32f2f", 0: "#388e3c"},
                    hover_data=["Subscriber_Name", "Agent_ID"],
                    height=350,
                    labels={"Amount": "Amount ($)", "Risk_Score": "Risk Score"}
                )
                fig_scatter.update_traces(marker=dict(size=8, opacity=0.7))
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                st.markdown("#### Fraud by Hour of Day")
                df["hour"] = df["Timestamp"].dt.hour
                fraud_hour = df[df["Fraud_Label"] == 1].groupby("hour").size().reset_index(name="counts")
                if not fraud_hour.empty:
                    fig_hour = px.bar(
                        fraud_hour,
                        x="hour",
                        y="counts",
                        height=350,
                        labels={"hour": "Hour of Day", "counts": "Fraud Count"}
                    )
                    st.plotly_chart(fig_hour, use_container_width=True)
                else:
                    st.info("No hourly fraud pattern detected yet")
        
    with tab4:
        st.subheader("Model Performance")
        
        if not api_status:
            st.error("Unable to fetch model performance data")
        else:
            # Simulate model metrics updates
            if 'last_metrics_update' not in st.session_state or time.time() - st.session_state.last_metrics_update > 10:
                st.session_state.model_metrics["accuracy"].append(random.uniform(0.92, 0.96))
                st.session_state.model_metrics["precision"].append(random.uniform(0.88, 0.94))
                st.session_state.model_metrics["recall"].append(random.uniform(0.89, 0.95))
                st.session_state.model_metrics["f1"].append(random.uniform(0.90, 0.94))
                st.session_state.model_metrics["latency"].append(random.uniform(25, 45))
                st.session_state.model_metrics["timestamps"].append(datetime.now())
                st.session_state.last_metrics_update = time.time()
            
            metrics_df = pd.DataFrame({
                "timestamp": st.session_state.model_metrics["timestamps"],
                "Accuracy": st.session_state.model_metrics["accuracy"],
                "Precision": st.session_state.model_metrics["precision"],
                "Recall": st.session_state.model_metrics["recall"],
                "F1 Score": st.session_state.model_metrics["f1"],
                "Latency (ms)": st.session_state.model_metrics["latency"]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Metrics Over Time")
                fig_acc = px.line(
                    metrics_df,
                    x="timestamp",
                    y=["Accuracy", "Precision", "Recall", "F1 Score"],
                    height=400,
                    labels={"value": "Score", "timestamp": "Time"},
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                st.markdown("#### Processing Latency")
                fig_latency = px.line(
                    metrics_df,
                    x="timestamp",
                    y="Latency (ms)",
                    height=400,
                    labels={"timestamp": "Time"},
                    color_discrete_sequence=["#3282b8"]
                )
                st.plotly_chart(fig_latency, use_container_width=True)
            
            st.markdown("#### Current Metrics")
            latest_metrics = {
                "Accuracy": st.session_state.model_metrics["accuracy"][-1] if st.session_state.model_metrics["accuracy"] else 0,
                "Precision": st.session_state.model_metrics["precision"][-1] if st.session_state.model_metrics["precision"] else 0,
                "Recall": st.session_state.model_metrics["recall"][-1] if st.session_state.model_metrics["recall"] else 0,
                "F1 Score": st.session_state.model_metrics["f1"][-1] if st.session_state.model_metrics["f1"] else 0,
                "Latency (ms)": st.session_state.model_metrics["latency"][-1] if st.session_state.model_metrics["latency"] else 0
            }
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{latest_metrics['Accuracy']:.2%}")
            m2.metric("Precision", f"{latest_metrics['Precision']:.2%}")
            m3.metric("Recall", f"{latest_metrics['Recall']:.2%}")
            m4.metric("F1 Score", f"{latest_metrics['F1 Score']:.2%}")
            m5.metric("Latency", f"{latest_metrics['Latency (ms)']:.1f}ms")
    
    # Process new transactions if in real-time mode
    if st.session_state.real_time_mode and 'selected_agent' in st.session_state and st.session_state.selected_agent:
        process_new_transactions()

if __name__ == "__main__":
    main()