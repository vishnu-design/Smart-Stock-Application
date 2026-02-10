import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os

# --- AI & ML Libraries ---
from prophet import Prophet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor # <--- ADDED
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SmartStock AI | Assignment Prototype",
    page_icon="📦",
    layout="wide"
)

# Professional CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; background-color: #0068c9; color: white; }
    .stMetric { background-color: white; padding: 15px; border-radius: 5px; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

st.title("📦 SmartStock: Intelligent Inventory Forecasting")
st.markdown("**System Status:** Online | **Warehouse:** WH-SG-JURONG-01 | **User:** Vishnu P.")

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    if not os.path.exists("inventory_data.csv"):
        return None
    
    df = pd.read_csv("inventory_data.csv")
    # Rename for consistency
    df = df.rename(columns={'Date': 'ds', 'Quantity_Sold': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

df = load_data()

if df is None:
    st.error("⚠️ DATA ERROR: 'inventory_data.csv' not found. Please run the data generation script.")
    st.stop()

# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.subheader("1. Select Algorithm")

# NOW WITH 3 OPTIONS
model_choice = st.sidebar.radio(
    "Choose AI Technique:",
    ("Facebook Prophet (Time Series)", 
     "Bayesian Ridge (Probabilistic)", 
     "Random Forest (Ensemble ML)") # <--- NEW OPTION
)

st.sidebar.subheader("2. Forecast Parameters")
days_to_predict = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.info("ℹ️ **Model Comparison:**\n\n1. **Prophet:** Best for seasonality.\n2. **Bayesian:** Best for uncertainty.\n3. **Random Forest:** Best for complex non-linear patterns.")

# ==========================================
# 4. DASHBOARD - HISTORICAL DATA
# ==========================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total SKU Revenue", f"${df['Total_Revenue'].sum():,.0f}")
with col2:
    st.metric("Current Unit Price", f"${df['Unit_Price_SGD'].iloc[-1]:.2f}")
with col3:
    st.metric("Last Recorded Sale", f"{df['y'].iloc[-1]} Units")

st.subheader("📊 Historical Sales Trends")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual Sales', line=dict(color='#3366cc', width=1)))
fig_hist.update_layout(height=400, xaxis_title="Date", yaxis_title="Quantity Sold")
st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# 5. AI ENGINE
# ==========================================
st.markdown("### 🔮 Generate AI Prediction")

if st.button("🚀 Run Forecast Model"):
    
    with st.spinner(f"Training {model_choice} on {len(df)} records... Please wait."):
        
        fig_forecast = go.Figure()
        # Add History (Greyed out)
        fig_forecast.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='History', line=dict(color='lightgrey')))
        
        predictions = []
        future_dates = []

        # --- MODEL A: FACEBOOK PROPHET ---
        if "Prophet" in model_choice:
            m = Prophet(daily_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=days_to_predict)
            forecast = m.predict(future)
            
            future_data = forecast.tail(days_to_predict)
            predictions = future_data['yhat'].values
            future_dates = future_data['ds']
            color = 'green'

        # --- MODEL B & C: SKLEARN MODELS (Bayesian & Random Forest) ---
        else:
            # 1. Feature Engineering (Lags)
            # Both these models need "Lag Features" (Past values predict future values)
            df_ml = df[['y']].copy()
            look_back = 30 # Use past 30 days to predict today
            
            for i in range(1, look_back + 1):
                df_ml[f'lag_{i}'] = df_ml['y'].shift(i)
            
            df_ml = df_ml.dropna()
            
            X = df_ml.drop('y', axis=1)
            y = df_ml['y']
            
            # 2. Select Model
            if "Bayesian" in model_choice:
                model = BayesianRidge()
                color = 'orange'
            elif "Random Forest" in model_choice:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                color = 'red'
                
            model.fit(X, y)
            
            # 3. Recursive Forecasting (Walking forward one day at a time)
            future_preds = []
            # Start with the very last known window of data
            curr_batch = X.iloc[-1].values.reshape(1, -1)
            
            for i in range(days_to_predict):
                # Predict next day
                pred = model.predict(curr_batch)[0]
                future_preds.append(pred)
                
                # Update batch: Drop oldest day, shift left, add new prediction at the end
                new_batch = np.roll(curr_batch, 1)
                new_batch[0, 0] = pred # Update the most recent lag (lag_1)
                curr_batch = new_batch
                
            # 4. Prepare Dates
            last_date = df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
            predictions = future_preds

        # --- PLOT RESULT ---
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, 
            y=predictions, 
            name=f'{model_choice} Prediction', 
            line=dict(color=color, width=3)
        ))

        # --- METRICS ---
        total_predicted_units = np.sum(predictions)
        
        st.success(f"✅ Prediction Complete using {model_choice}!")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Predicted Demand:**\n\n{int(total_predicted_units)} Units")
        with c2:
            unit_price = df['Unit_Price_SGD'].iloc[0]
            st.info(f"**Est. Revenue Impact:**\n\n${int(total_predicted_units * unit_price):,.2f}")
        with c3:
            st.warning(f"**Action Required:**\n\nSubmit Purchase Order to Supplier {df['Supplier_ID'].iloc[0]}")