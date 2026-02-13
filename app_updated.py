import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
from datetime import datetime, timedelta

# --- AI & ML Libraries ---
from prophet import Prophet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# --- Deep Learning Libraries ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# ==========================================
# EASY EXPERT SYSTEM CLASS
# ==========================================
class SimpleExpertSystem:
    """
    A simplified, easy-to-understand expert system for inventory management.
    Focuses on basic stock days and AI spike detection.
    """
    
    def __init__(self, df, predictions=None, forecast_days=30):
        self.df = df
        self.predictions = predictions
        self.forecast_days = forecast_days
        
        # Simple Business Rules
        self.MIN_STOCK_DAYS = 21  # 3 weeks minimum
        self.MAX_STOCK_DAYS = 45  # 1.5 months maximum
        
        # State variables
        self.rules_fired = []
        self.recommendations = []
        self.alerts = []
        self.risk_level = "LOW"
        
    def apply_rules(self):
        """Apply simple IF-THEN rules"""
        # 1. Calculate basic metrics
        recent_data = self.df.tail(30)
        
        # Safely handle column names depending on if it's been renamed for Prophet yet
        if 'Quantity_Sold' in recent_data.columns:
            self.avg_daily_sales = recent_data['Quantity_Sold'].mean()
        elif 'y' in recent_data.columns:
            self.avg_daily_sales = recent_data['y'].mean()
        else:
            self.avg_daily_sales = 0
            
        self.current_stock = recent_data['Opening_Stock'].iloc[-1] if 'Opening_Stock' in recent_data.columns else 0
        self.days_of_stock = self.current_stock / self.avg_daily_sales if self.avg_daily_sales > 0 else 0

        # 2. RULE: Low Stock Check
        if self.days_of_stock < self.MIN_STOCK_DAYS:
            self.rules_fired.append("Rule 1: Low Stock Detected")
            self.alerts.append("⚠️ LOW STOCK WARNING")
            self.risk_level = "HIGH"
            self.recommendations.append({
                "priority": "HIGH",
                "category": "Replenishment",
                "action": f"Order {int(self.avg_daily_sales * 30)} units ASAP.",
                "reason": f"Inventory covers only {self.days_of_stock:.1f} days (Below {self.MIN_STOCK_DAYS} day minimum)."
            })
            
        # 3. RULE: Excess Stock Check
        elif self.days_of_stock > self.MAX_STOCK_DAYS:
            self.rules_fired.append("Rule 2: Excess Stock Detected")
            self.risk_level = "MEDIUM"
            self.recommendations.append({
                "priority": "LOW",
                "category": "Overstock",
                "action": "Pause purchasing.",
                "reason": f"Inventory covers {self.days_of_stock:.1f} days (Exceeds {self.MAX_STOCK_DAYS} day maximum)."
            })
            
        # 4. RULE: Healthy Stock
        else:
            self.rules_fired.append("Rule 3: Healthy Inventory Level")
            self.recommendations.append({
                "priority": "LOW",
                "category": "Optimization",
                "action": "Maintain current operations.",
                "reason": f"Stock is at an optimal {self.days_of_stock:.1f} days."
            })

        # 5. RULE: AI Spike Detection Check
        if self.predictions is not None:
            max_predicted = np.max(self.predictions)
            if max_predicted > (self.avg_daily_sales * 1.5):  # 50% higher than average
                self.rules_fired.append("Rule 4: AI Demand Spike Predicted")
                self.alerts.append("🔥 AI DEMAND SPIKE DETECTED")
                self.risk_level = "CRITICAL"
                self.recommendations.append({
                    "priority": "CRITICAL",
                    "category": "AI Forecast",
                    "action": "Prepare warehouse for incoming surge.",
                    "reason": f"AI predicts a peak of {int(max_predicted)} units/day (50%+ above normal)."
                })

    def generate_report(self):
        """Generate a simple dictionary report for the UI"""
        return {
            "risk_level": self.risk_level,
            "alerts": self.alerts,
            "rules_fired": self.rules_fired,
            "recommendations": self.recommendations,
            "metrics": {
                "avg_daily_sales": self.avg_daily_sales,
                "current_stock": self.current_stock,
                "days_of_stock": self.days_of_stock
            }
        }

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SmartStock AI | Simple Expert System",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { 
        width: 100%; 
        background-color: #0068c9; 
        color: white;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
    }
    .stMetric { 
        background-color: white; 
        padding: 15px; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    }
    .expert-box { 
        background-color: #fff; 
        padding: 16px; 
        border-radius: 8px; 
        border-left: 5px solid #6c757d; 
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .critical { border-left-color: #dc3545; background-color: #fff5f5; }
    .high { border-left-color: #fd7e14; background-color: #fff8f0; }
    .medium { border-left-color: #0dcaf0; background-color: #f0fcff; }
    .low { border-left-color: #198754; background-color: #f0fff4; }
    .alert-banner {
        padding: 12px 20px;
        border-radius: 6px;
        margin: 10px 0;
        font-weight: 600;
        text-align: center;
    }
    .banner-critical { background-color: #dc3545; color: white; }
    .banner-warning { background-color: #ffc107; color: #000; }
    .product-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="product-header">
    <h1>📦 SmartStock AI - Simple Inventory Expert</h1>
    <p style="margin: 5px 0;"><strong>Product:</strong> Industrial Filter v2 | <strong>Warehouse:</strong> WH-SG-01</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    filepath = "/mnt/user-data/uploads/inventory_data.csv"
    if not os.path.exists(filepath):
        filepath = "inventory_data.csv"
        if not os.path.exists(filepath):
            # Fallback mock data generation if CSV is entirely missing
            dates = pd.date_range(end=datetime.today(), periods=365)
            y = np.random.normal(75, 20, 365)
            y = np.maximum(y, 10)  # Ensure positive
            stock = np.random.normal(500, 100, 365)
            return pd.DataFrame({'ds': dates, 'y': y, 'Opening_Stock': stock, 'Total_Revenue': y*145.5, 'Unit_Price_SGD': 145.5})
            
    df = pd.read_csv(filepath)
    df = df.rename(columns={'Date': 'ds', 'Quantity_Sold': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

df = load_data()

# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ AI Configuration")

st.sidebar.subheader("1. Forecasting Engine")
model_choice = st.sidebar.radio(
    "Select Algorithm:",
    ("1D CNN (Deep Learning)", 
     "Facebook Prophet (Time Series)", 
     "Bayesian Ridge (Probabilistic)", 
     "Random Forest (Ensemble ML)")
)

st.sidebar.subheader("2. Forecast Horizon")
days_to_predict = st.sidebar.slider("Days to Forecast", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Expert System")
enable_expert_system = st.sidebar.checkbox("Enable Expert System", value=True)

if enable_expert_system:
    st.sidebar.success("🧠 **Simple Expert System: ACTIVE**")
else:
    st.sidebar.warning("Expert System: Disabled")

# ==========================================
# 4. DASHBOARD - KEY METRICS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = df['Total_Revenue'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.0f}")
with col2:
    st.metric("Unit Price", f"${df['Unit_Price_SGD'].iloc[-1]:.2f}")
with col3:
    recent_avg = df.tail(30)['y'].mean()
    st.metric("30-Day Avg Sales", f"{recent_avg:.1f} units")
with col4:
    last_stock = df['Opening_Stock'].iloc[-1]
    st.metric("Current Stock Level", f"{int(last_stock)} units")

# ==========================================
# 5. HISTORICAL TRENDS VISUALIZATION
# ==========================================
st.markdown("---")
st.subheader("📊 Historical Sales & Stock Trends")

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Daily Sales', line=dict(color='#3366cc', width=2), yaxis='y'))
fig_hist.add_trace(go.Scatter(x=df['ds'], y=df['Opening_Stock'], name='Stock Level', line=dict(color='#dc3912', width=1, dash='dot'), yaxis='y2'))

fig_hist.update_layout(
    height=400, xaxis_title="Date",
    yaxis=dict(title="Units Sold", side='left'),
    yaxis2=dict(title="Stock Level", side='right', overlaying='y'),
    hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# 6. AI FORECASTING ENGINE
# ==========================================
st.markdown("---")
st.markdown("### 🔮 AI-Powered Demand Forecasting")

if st.button("🚀 Run Forecast Model", type="primary"):
    
    with st.spinner(f"Training {model_choice}..."):
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Historical Sales', line=dict(color='lightgrey', width=1)))
        
        predictions, future_dates = [], []

        # --- 1D CNN MODEL ---
        if "CNN" in model_choice:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['y']].values)
            
            look_back = 30
            X_cnn, y_cnn = [], []
            for i in range(look_back, len(scaled_data)):
                X_cnn.append(scaled_data[i-look_back:i, 0])
                y_cnn.append(scaled_data[i, 0])
            X_cnn, y_cnn = np.array(X_cnn), np.array(y_cnn)
            X_cnn = np.reshape(X_cnn, (X_cnn.shape[0], X_cnn.shape[1], 1))
            
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
            model.fit(X_cnn, y_cnn, epochs=50, batch_size=16, verbose=0)
            
            future_preds_scaled = []
            curr_batch = scaled_data[-look_back:].reshape((1, look_back, 1))
            
            for _ in range(days_to_predict):
                pred = model.predict(curr_batch, verbose=0)[0]
                future_preds_scaled.append(pred)
                curr_batch = np.append(curr_batch[:, 1:, :], [[pred]], axis=1)
                
            predictions = scaler.inverse_transform(future_preds_scaled).flatten()
            predictions = np.maximum(predictions, 0)
            
            last_date = df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
            color = '#8a2be2'

        # --- PROPHET MODEL ---
        elif "Prophet" in model_choice:
            m = Prophet(daily_seasonality=True, weekly_seasonality=True)
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=days_to_predict)
            forecast = m.predict(future)
            
            future_data = forecast.tail(days_to_predict)
            predictions = future_data['yhat'].values
            future_dates = future_data['ds']
            color = '#00a86b'

        # --- SKLEARN MODELS ---
        else:
            df_ml = df[['y']].copy()
            look_back = 30
            for i in range(1, look_back + 1):
                df_ml[f'lag_{i}'] = df_ml['y'].shift(i)
            df_ml = df_ml.dropna()
            X = df_ml.drop('y', axis=1)
            y = df_ml['y']
            
            if "Bayesian" in model_choice:
                model = BayesianRidge()
                color = '#ff8c00'
            elif "Random Forest" in model_choice:
                model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                color = '#dc143c'
                
            model.fit(X, y)
            
            future_preds = []
            curr_batch = X.iloc[-1].values.reshape(1, -1)
            
            for i in range(days_to_predict):
                pred = max(0, model.predict(curr_batch)[0]) 
                future_preds.append(pred)
                new_batch = np.roll(curr_batch, 1)
                new_batch[0, 0] = pred
                curr_batch = new_batch
                
            last_date = df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
            predictions = future_preds

        # Plot forecast
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=predictions, name=f'Forecast', line=dict(color=color, width=3)))
        fig_forecast.update_layout(height=450, xaxis_title="Date", yaxis_title="Units", hovermode='x unified')

        st.success(f"✅ Forecast complete using {model_choice}!")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # ==========================================
        # 7. EXPERT SYSTEM ANALYSIS (SIMPLIFIED)
        # ==========================================
        if enable_expert_system:
            st.markdown("---")
            st.markdown("## 🧠 Simple Expert System Recommendations")
            
            with st.spinner("Analyzing stock levels..."):
                expert = SimpleExpertSystem(df, predictions, days_to_predict)
                expert.apply_rules()
                report = expert.generate_report()
            
            # Print Warnings
            if report['alerts']:
                for alert in report['alerts']:
                    st.markdown(f'<div class="alert-banner banner-critical">{alert}</div>', unsafe_allow_html=True)
            
            # Key Metrics
            es_col1, es_col2, es_col3 = st.columns(3)
            with es_col1:
                risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
                st.metric("Risk Assessment", f"{risk_colors.get(report['risk_level'], '⚪')} {report['risk_level']}")
            with es_col2:
                st.metric("Days of Stock Remaining", f"{report['metrics']['days_of_stock']:.1f} days")
            with es_col3:
                st.metric("Rules Triggered", f"{len(report['rules_fired'])}")
            
            st.markdown("#### 💡 Actionable Recommendations")
            
            # Display recommendations cleanly
            for rec in report['recommendations']:
                priority_class = rec['priority'].lower()
                st.markdown(f"""
                <div class="expert-box {priority_class}">
                    <strong style="font-size: 1.1em;">[{rec['priority']}] {rec['category']}</strong><br/>
                    <strong>Action:</strong> {rec['action']}<br/>
                    <em>Reasoning:</em> {rec['reason']}
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("🔍 View Simple Rule Engine"):
                st.markdown("""
                **Active Rules:**
                1. **Low Stock:** Triggers if inventory drops below 21 days of supply.
                2. **Excess Stock:** Triggers if inventory exceeds 45 days of supply.
                3. **Optimal Check:** Triggers if inventory is comfortably between 21 and 45 days.
                4. **AI Spike Detection:** Triggers if the forecasting model predicts a sudden +50% surge in demand.
                """)
