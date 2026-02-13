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

# ==========================================
# EXPERT SYSTEM CLASS
# ==========================================
class InventoryExpertSystem:
    """
    Rule-based expert system for inventory management decisions.
    Uses forward chaining inference with weighted rules.
    """
    
    def __init__(self, df, predictions=None, forecast_days=30):
        self.df = df
        self.predictions = predictions
        self.forecast_days = forecast_days
        self.rules_fired = []
        self.recommendations = []
        self.risk_level = "LOW"
        self.confidence = 0.0
        
    def calculate_metrics(self):
        """Calculate key inventory metrics"""
        recent_data = self.df.tail(30)
        
        # Sales velocity (units per day)
        self.avg_daily_sales = recent_data['y'].mean()
        self.sales_volatility = recent_data['y'].std()
        self.cv = (self.sales_volatility / self.avg_daily_sales) if self.avg_daily_sales > 0 else 0
        
        # Trend analysis
        first_half = recent_data['y'].iloc[:15].mean()
        second_half = recent_data['y'].iloc[15:].mean()
        self.trend = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
        
        # Seasonality detection (simple)
        self.has_weekend_pattern = self._detect_weekend_pattern()
        
        # Stock metrics
        if self.predictions is not None:
            self.predicted_demand = np.sum(self.predictions)
            self.predicted_avg_daily = np.mean(self.predictions)
        else:
            self.predicted_demand = self.avg_daily_sales * self.forecast_days
            self.predicted_avg_daily = self.avg_daily_sales
    
    def _detect_weekend_pattern(self):
        """Simple weekend pattern detection"""
        df_copy = self.df.copy()
        df_copy['dayofweek'] = df_copy['ds'].dt.dayofweek
        weekend_sales = df_copy[df_copy['dayofweek'] >= 5]['y'].mean()
        weekday_sales = df_copy[df_copy['dayofweek'] < 5]['y'].mean()
        
        if weekday_sales > 0:
            diff_pct = abs(weekend_sales - weekday_sales) / weekday_sales * 100
            return diff_pct > 20
        return False
    
    def apply_rules(self):
        """Apply expert system rules using forward chaining"""
        self.calculate_metrics()
        
        # RULE 1: High Volatility Rule
        if self.cv > 0.5:
            self.rules_fired.append("R1: High Demand Volatility Detected")
            self.recommendations.append({
                "priority": "HIGH",
                "category": "Safety Stock",
                "action": f"Increase safety stock by {int(self.cv * 100)}%",
                "reason": f"Coefficient of variation ({self.cv:.2f}) exceeds 0.5 threshold"
            })
            self.risk_level = "HIGH"
        
        # RULE 2: Growth Trend Rule
        if self.trend > 15:
            self.rules_fired.append("R2: Strong Growth Trend Detected")
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "Demand Planning",
                "action": f"Prepare for {self.trend:.1f}% demand increase",
                "reason": "Recent 15-day trend shows significant growth"
            })
        elif self.trend < -15:
            self.rules_fired.append("R2b: Declining Trend Detected")
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "Inventory Optimization",
                "action": "Consider reducing order quantities",
                "reason": f"Sales declining by {abs(self.trend):.1f}% over past 15 days"
            })
        
        # RULE 3: Seasonality Rule
        if self.has_weekend_pattern:
            self.rules_fired.append("R3: Weekend Seasonality Pattern")
            self.recommendations.append({
                "priority": "LOW",
                "category": "Operational Planning",
                "action": "Adjust staffing for weekend demand patterns",
                "reason": "Significant weekend vs. weekday sales variation detected"
            })
        
        # RULE 4: Reorder Point Rule
        reorder_point = self.avg_daily_sales * 7  # 7-day lead time assumption
        safety_stock = self.sales_volatility * 1.65  # 95% service level
        optimal_reorder = reorder_point + safety_stock
        
        self.rules_fired.append("R4: Reorder Point Calculation")
        self.recommendations.append({
            "priority": "HIGH",
            "category": "Replenishment",
            "action": f"Set reorder point at {int(optimal_reorder)} units",
            "reason": f"Based on {int(reorder_point)} lead time demand + {int(safety_stock)} safety stock"
        })
        
        # RULE 5: Economic Order Quantity (Simplified)
        if len(self.recommendations) > 0:
            unit_price = self.df['Unit_Price_SGD'].iloc[0]
            annual_demand = self.avg_daily_sales * 365
            holding_cost_pct = 0.25  # 25% of unit cost per year
            order_cost = 50  # Fixed cost per order (SGD)
            
            eoq = np.sqrt((2 * annual_demand * order_cost) / (unit_price * holding_cost_pct))
            
            self.rules_fired.append("R5: Economic Order Quantity")
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "Order Optimization",
                "action": f"Optimal order quantity: {int(eoq)} units",
                "reason": f"Minimizes total cost (holding + ordering) at current demand levels"
            })
        
        # RULE 6: Stockout Risk Assessment
        if self.predictions is not None:
            max_predicted = np.max(self.predictions)
            if max_predicted > (self.avg_daily_sales * 1.5):
                self.rules_fired.append("R6: Stockout Risk Alert")
                self.recommendations.append({
                    "priority": "CRITICAL",
                    "category": "Risk Management",
                    "action": f"Peak demand may reach {int(max_predicted)} units/day",
                    "reason": "Predicted spike exceeds 150% of average daily sales"
                })
                self.risk_level = "CRITICAL"
        
        # Calculate overall confidence
        self.confidence = min(100, len(self.rules_fired) * 15 + 25)
        
        return self.recommendations
    
    def generate_report(self):
        """Generate expert system report"""
        report = {
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "rules_fired": self.rules_fired,
            "recommendations": self.recommendations,
            "metrics": {
                "avg_daily_sales": self.avg_daily_sales,
                "sales_volatility": self.sales_volatility,
                "trend_pct": self.trend,
                "coefficient_variation": self.cv
            }
        }
        return report


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
    .expert-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 10px 0; }
    .critical { border-left-color: #dc3545; background-color: #f8d7da; }
    .high { border-left-color: #fd7e14; background-color: #fff3cd; }
    .medium { border-left-color: #0dcaf0; background-color: #cff4fc; }
    .low { border-left-color: #198754; background-color: #d1e7dd; }
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
st.sidebar.subheader("1. Select AI Engine")

# Model selection tabs
model_choice = st.sidebar.radio(
    "Choose Forecasting Algorithm:",
    ("Facebook Prophet (Time Series)", 
     "Bayesian Ridge (Probabilistic)", 
     "Random Forest (Ensemble ML)")
)

st.sidebar.subheader("2. Forecast Parameters")
days_to_predict = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Expert System Settings")
enable_expert_system = st.sidebar.checkbox("Enable Expert System Analysis", value=True)

if enable_expert_system:
    st.sidebar.info("🧠 **Expert System Active**\n\nRule-based AI will analyze patterns and provide intelligent recommendations.")

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
            df_ml = df[['y']].copy()
            look_back = 30
            
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
            
            # 3. Recursive Forecasting
            future_preds = []
            curr_batch = X.iloc[-1].values.reshape(1, -1)
            
            for i in range(days_to_predict):
                pred = model.predict(curr_batch)[0]
                future_preds.append(pred)
                
                new_batch = np.roll(curr_batch, 1)
                new_batch[0, 0] = pred
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
        
        # ==========================================
        # 6. EXPERT SYSTEM ANALYSIS
        # ==========================================
        if enable_expert_system:
            st.markdown("---")
            st.markdown("### 🧠 Expert System Analysis")
            st.markdown("*Rule-based AI analyzing inventory patterns and generating recommendations...*")
            
            # Initialize and run expert system
            expert = InventoryExpertSystem(df, predictions, days_to_predict)
            report = expert.generate_report()
            
            # Display risk level and confidence
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                risk_colors = {
                    "LOW": "🟢",
                    "MEDIUM": "🟡",
                    "HIGH": "🟠",
                    "CRITICAL": "🔴"
                }
                st.metric("Risk Level", f"{risk_colors.get(report['risk_level'], '⚪')} {report['risk_level']}")
            
            with col_b:
                st.metric("System Confidence", f"{report['confidence']:.0f}%")
            
            with col_c:
                st.metric("Rules Fired", len(report['rules_fired']))
            
            # Display key metrics
            st.markdown("#### 📊 Calculated Metrics")
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            with met_col1:
                st.metric("Avg Daily Sales", f"{report['metrics']['avg_daily_sales']:.1f} units")
            with met_col2:
                st.metric("Sales Volatility", f"±{report['metrics']['sales_volatility']:.1f} units")
            with met_col3:
                st.metric("Trend", f"{report['metrics']['trend_pct']:+.1f}%")
            with met_col4:
                st.metric("Demand Variability", f"{report['metrics']['coefficient_variation']:.2f}")
            
            # Display recommendations
            st.markdown("#### 💡 Expert Recommendations")
            
            # Group by priority
            priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            for priority in priority_order:
                recs = [r for r in report['recommendations'] if r['priority'] == priority]
                if recs:
                    for rec in recs:
                        priority_class = priority.lower()
                        st.markdown(f"""
                        <div class="expert-box {priority_class}">
                            <strong>🎯 [{rec['priority']}] {rec['category']}</strong><br/>
                            <strong>Action:</strong> {rec['action']}<br/>
                            <em>Reason:</em> {rec['reason']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Display fired rules
            with st.expander("🔍 View Inference Chain (Rules Fired)"):
                st.markdown("**Forward Chaining Inference:**")
                for i, rule in enumerate(report['rules_fired'], 1):
                    st.markdown(f"{i}. {rule}")
                
                st.markdown("""
                ---
                **Expert System Logic:**
                - R1: Volatility analysis (coefficient of variation)
                - R2: Trend detection (15-day moving comparison)
                - R3: Seasonality pattern recognition
                - R4: Reorder point optimization (Lead time + Safety stock)
                - R5: Economic Order Quantity calculation
                - R6: Stockout risk assessment
                """)
