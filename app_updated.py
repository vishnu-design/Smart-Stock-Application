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
# EXPERT SYSTEM CLASS - CUSTOMIZED FOR HYDRAULIC FILTERS
# ==========================================
class HydraulicFilterExpertSystem:
    """
    Domain-specific expert system for Industrial Hydraulic Filter v2 inventory management.
    Based on real product characteristics:
    - Product: Industrial Hydraulic Filter v2 (SKU-7782-X)
    - Supplier: Bosch Rexroth AG (Germany)
    - Lead Time: 14 days
    - Historical avg: 75 units/day, volatility: ±20 units
    - Growth trend: +28.8% over 2 years
    - Profit margin: 71.2% (SGD $60.50 per unit)
    """
    
    def __init__(self, df, predictions=None, forecast_days=30):
        self.df = df
        self.predictions = predictions
        self.forecast_days = forecast_days
        
        # Product-specific constants
        self.LEAD_TIME = 14  # Days from Bosch Germany to Singapore
        self.UNIT_COST = 85.0  # SGD
        self.UNIT_PRICE = 145.5  # SGD
        self.PROFIT_MARGIN = 60.5  # SGD per unit
        self.SUPPLIER = "Bosch Rexroth AG"
        self.PRODUCT = "Industrial Hydraulic Filter v2"
        
        # Business thresholds (calibrated to actual data)
        self.HIGH_VOLATILITY_THRESHOLD = 0.3  # CV > 0.3 is concerning
        self.CRITICAL_VOLATILITY_THRESHOLD = 0.45  # CV > 0.45 is critical
        self.MIN_SAFETY_STOCK_DAYS = 7  # Minimum 1 week buffer
        self.TARGET_SERVICE_LEVEL = 0.95  # 95% service level
        self.REORDER_BUFFER_DAYS = 3  # Extra buffer on lead time
        
        # State variables
        self.rules_fired = []
        self.recommendations = []
        self.risk_level = "LOW"
        self.confidence = 0.0
        self.alerts = []
        
    def calculate_metrics(self):
        """Calculate key inventory metrics from historical data"""
        # Get recent 60-day window for analysis
        recent_data = self.df.tail(60)
        
        # Basic sales statistics
        self.avg_daily_sales = recent_data['Quantity_Sold'].mean()
        self.sales_volatility = recent_data['Quantity_Sold'].std()
        self.cv = (self.sales_volatility / self.avg_daily_sales) if self.avg_daily_sales > 0 else 0
        
        self.max_daily_sales = recent_data['Quantity_Sold'].max()
        self.min_daily_sales = recent_data['Quantity_Sold'].min()
        
        # Trend analysis (30-day rolling comparison)
        first_30 = recent_data['Quantity_Sold'].iloc[:30].mean()
        second_30 = recent_data['Quantity_Sold'].iloc[30:].mean()
        self.trend_30day = ((second_30 - first_30) / first_30 * 100) if first_30> 0 else 0
        
        # Long-term trend (compare recent 90 days vs previous 90 days)
        if len(self.df) >= 180:
            recent_90 = self.df.tail(90)['Quantity_Sold'].mean()
            previous_90 = self.df.tail(180).head(90)['Quantity_Sold'].mean()
            self.trend_longterm = ((recent_90 - previous_90) / previous_90 * 100) if previous_90 > 0 else 0
        else:
            self.trend_longterm = 0
        
        # Stock analysis
        self.current_stock = recent_data['Opening_Stock'].iloc[-1]
        self.avg_stock_level = recent_data['Opening_Stock'].mean()
        self.days_of_stock = self.current_stock / self.avg_daily_sales if self.avg_daily_sales > 0 else 0
        
        # Demand forecast (if predictions available)
        if self.predictions is not None:
            self.predicted_total_demand = np.sum(self.predictions)
            self.predicted_avg_daily = np.mean(self.predictions)
            self.predicted_max_daily = np.max(self.predictions)
            self.demand_spike_factor = self.predicted_max_daily / self.avg_daily_sales if self.avg_daily_sales > 0 else 1
        else:
            self.predicted_total_demand = self.avg_daily_sales * self.forecast_days
            self.predicted_avg_daily = self.avg_daily_sales
            self.predicted_max_daily = self.max_daily_sales
            self.demand_spike_factor = 1
    
    def apply_rules(self):
        """Apply expert system rules using forward chaining inference"""
        self.calculate_metrics()
        
        # ================================================================
        # RULE 1: STOCKOUT RISK ASSESSMENT (CRITICAL)
        # ================================================================
        lead_time_demand = self.avg_daily_sales * self.LEAD_TIME
        safety_stock_95 = self.sales_volatility * 1.65 * np.sqrt(self.LEAD_TIME)  # Z=1.65 for 95%
        reorder_point = lead_time_demand + safety_stock_95
        
        stockout_risk_days = self.days_of_stock - self.LEAD_TIME
        
        if stockout_risk_days < 3:
            self.rules_fired.append("R1-CRITICAL: Imminent Stockout Risk")
            self.recommendations.append({
                "priority": "CRITICAL",
                "category": "⚠️ Stockout Prevention",
                "action": f"URGENT: Place emergency order for {int(reorder_point * 1.5)} units",
                "reason": f"Current stock ({int(self.current_stock)} units) only covers {stockout_risk_days:.1f} days beyond lead time. Stockout risk within 72 hours!",
                "impact": f"Potential revenue loss: ${int(self.avg_daily_sales * 3 * self.UNIT_PRICE):,}"
            })
            self.risk_level = "CRITICAL"
            self.alerts.append("🚨 STOCKOUT IMMINENT")
            
        elif stockout_risk_days < 7:
            self.rules_fired.append("R1-HIGH: Low Stock Warning")
            self.recommendations.append({
                "priority": "HIGH",
                "category": "⚠️ Stock Replenishment",
                "action": f"Place order for {int(reorder_point)} units within 48 hours",
                "reason": f"Stock buffer is {stockout_risk_days:.1f} days. Below minimum safety threshold of 7 days.",
                "impact": f"Current inventory: {int(self.current_stock)} units = {self.days_of_stock:.1f} days of supply"
            })
            if self.risk_level == "LOW":
                self.risk_level = "HIGH"
        
        # ================================================================
        # RULE 2: DEMAND VOLATILITY ANALYSIS
        # ================================================================
        if self.cv > self.CRITICAL_VOLATILITY_THRESHOLD:
            self.rules_fired.append("R2-CRITICAL: Extreme Demand Volatility")
            buffer_increase = int((self.cv - 0.3) * 100)
            self.recommendations.append({
                "priority": "CRITICAL",
                "category": "📊 Safety Stock Adjustment",
                "action": f"Increase safety stock by {buffer_increase}% ({int(safety_stock_95 * buffer_increase/100)} units)",
                "reason": f"Coefficient of Variation is {self.cv:.3f} (>{self.CRITICAL_VOLATILITY_THRESHOLD}). Highly unpredictable demand pattern.",
                "impact": f"Sales range: {int(self.min_daily_sales)}-{int(self.max_daily_sales)} units/day (swing of {int(self.max_daily_sales - self.min_daily_sales)} units)"
            })
            if self.risk_level != "CRITICAL":
                self.risk_level = "CRITICAL"
                
        elif self.cv > self.HIGH_VOLATILITY_THRESHOLD:
            self.rules_fired.append("R2-MEDIUM: Moderate Demand Volatility")
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "📊 Safety Stock Adjustment",
                "action": f"Maintain elevated safety stock of {int(safety_stock_95)} units",
                "reason": f"CV = {self.cv:.3f}. Demand varies by ±{int(self.sales_volatility)} units daily.",
                "impact": f"95% service level requires {int(safety_stock_95)} units buffer"
            })
        
        # ================================================================
        # RULE 3: GROWTH TREND DETECTION
        # ================================================================
        if self.trend_longterm > 20:
            self.rules_fired.append("R3-HIGH: Strong Growth Trajectory")
            increased_demand = self.avg_daily_sales * (1 + self.trend_longterm/100)
            self.recommendations.append({
                "priority": "HIGH",
                "category": "📈 Demand Planning",
                "action": f"Revise demand forecast upward to {int(increased_demand)} units/day",
                "reason": f"Long-term growth trend of +{self.trend_longterm:.1f}%. Market expansion detected.",
                "impact": f"Additional {int((increased_demand - self.avg_daily_sales) * 30)} units needed per month"
            })
            self.alerts.append("📈 GROWTH TREND")
            
        elif self.trend_30day > 15:
            self.rules_fired.append("R3-MEDIUM: Recent Demand Surge")
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "📈 Trend Analysis",
                "action": "Monitor for sustained growth pattern. Consider increasing order frequency.",
                "reason": f"30-day trend shows +{self.trend_30day:.1f}% increase",
                "impact": "May indicate seasonal uptick or market shift"
            })
            
        elif self.trend_30day < -15:
            self.rules_fired.append("R3-MEDIUM: Declining Demand Pattern")
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "📉 Inventory Optimization",
                "action": "Reduce next order quantity by 15-20%",
                "reason": f"30-day decline of {self.trend_30day:.1f}%. Avoid overstock.",
                "impact": f"Potential savings: ${int(self.avg_daily_sales * 0.15 * self.UNIT_COST):,} in carrying costs"
            })
        
        # ================================================================
        # RULE 4: OPTIMAL REORDER POINT (ABC Classification)
        # ================================================================
        self.rules_fired.append("R4: Reorder Point Calculation (ABC-Class A)")
        
        # This is a high-value, fast-moving item (Class A)
        optimal_reorder = lead_time_demand + safety_stock_95
        
        self.recommendations.append({
            "priority": "HIGH",
            "category": "🎯 Reorder Point",
            "action": f"Set reorder trigger at {int(optimal_reorder)} units",
            "reason": f"Formula: ({int(lead_time_demand)} lead time demand) + ({int(safety_stock_95)} safety stock at 95% SL)",
            "impact": f"When stock hits {int(optimal_reorder)}, place order to maintain service level"
        })
        
        # ================================================================
        # RULE 5: ECONOMIC ORDER QUANTITY WITH IMPORT LOGISTICS
        # ================================================================
        annual_demand = self.avg_daily_sales * 365
        
        # Import-specific costs
        order_cost_base = 500  # Administrative cost (SGD)
        shipping_cost = 2500  # Container from Germany (SGD)
        customs_clearance = 300  # SGD
        total_order_cost = order_cost_base + shipping_cost + customs_clearance
        
        holding_cost_pct = 0.25  # 25% annual holding cost
        holding_cost_per_unit = self.UNIT_COST * holding_cost_pct
        
        # EOQ formula: √[(2 × D × S) / H]
        eoq = np.sqrt((2 * annual_demand * total_order_cost) / holding_cost_per_unit)
        
        # Adjust for container efficiency (typically order in multiples of pallet loads)
        pallet_size = 144  # Units per pallet
        eoq_pallets = np.ceil(eoq / pallet_size)
        eoq_adjusted = eoq_pallets * pallet_size
        
        self.rules_fired.append("R5: Economic Order Quantity (International)")
        self.recommendations.append({
            "priority": "MEDIUM",
            "category": "💰 Order Optimization",
            "action": f"Optimal order size: {int(eoq_adjusted)} units ({int(eoq_pallets)} pallets)",
            "reason": f"EOQ minimizes total cost. Base EOQ={int(eoq)}, adjusted to pallet multiples.",
            "impact": f"Annual orders: {int(annual_demand/eoq_adjusted)} shipments. Total logistics cost: ${int(total_order_cost * (annual_demand/eoq_adjusted)):,}"
        })
        
        # ================================================================
        # RULE 6: PREDICTED SPIKE DETECTION
        # ================================================================
        if self.predictions is not None and self.demand_spike_factor > 1.4:
            self.rules_fired.append("R6-CRITICAL: Demand Spike Forecast")
            spike_date_idx = np.argmax(self.predictions)
            
            self.recommendations.append({
                "priority": "CRITICAL",
                "category": "🔥 Peak Demand Alert",
                "action": f"Prepare for peak demand of {int(self.predicted_max_daily)} units/day",
                "reason": f"AI forecasts {int((self.demand_spike_factor - 1) * 100)}% spike above normal (day {spike_date_idx + 1} of forecast)",
                "impact": f"Ensure {int(self.predicted_max_daily * 3)} units available to cover 3-day spike"
            })
            if self.risk_level not in ["CRITICAL"]:
                self.risk_level = "CRITICAL"
            self.alerts.append("🔥 SPIKE FORECAST")
        
        # ================================================================
        # RULE 7: SUPPLIER LEAD TIME RISK (INTERNATIONAL)
        # ================================================================
        # Bosch in Germany - 14 day lead time includes international shipping
        self.rules_fired.append("R7: International Supply Chain Risk")
        
        # Calculate buffer for disruptions
        disruption_buffer = self.avg_daily_sales * 5  # Extra 5 days for customs/delays
        
        if self.trend_longterm > 15:  # Growing demand
            self.recommendations.append({
                "priority": "MEDIUM",
                "category": "🌍 Supply Chain Resilience",
                "action": f"Maintain strategic buffer of {int(disruption_buffer)} units",
                "reason": "14-day international lead time from Germany. Brexit/customs delays possible.",
                "impact": f"Buffer covers {5} days of supply disruption"
            })
        
        # ================================================================
        # RULE 8: PROFITABILITY & CASH FLOW ANALYSIS
        # ================================================================
        self.rules_fired.append("R8: Financial Performance Analysis")
        
        current_inventory_value = self.current_stock * self.UNIT_COST
        monthly_profit_potential = self.avg_daily_sales * 30 * self.PROFIT_MARGIN
        
        # Check if overstocked
        if self.days_of_stock > 45:
            excess_stock = self.current_stock - (self.avg_daily_sales * 30)
            tied_up_capital = excess_stock * self.UNIT_COST
            
            self.recommendations.append({
                "priority": "LOW",
                "category": "💵 Cash Flow Optimization",
                "action": f"Reduce stock by {int(excess_stock)} units to improve cash flow",
                "reason": f"Current stock ({self.days_of_stock:.0f} days) exceeds optimal level (30-40 days)",
                "impact": f"Free up ${int(tied_up_capital):,} in working capital"
            })
        
        # ================================================================
        # RULE 9: CATEGORY-SPECIFIC INSIGHTS (Heavy Machinery Parts)
        # ================================================================
        self.rules_fired.append("R9: Heavy Machinery Parts - Seasonal Pattern Check")
        
        # Check if there's a day-of-week pattern
        df_copy = self.df.tail(90).copy()
        df_copy['dayofweek'] = df_copy['ds'].dt.dayofweek
        
        # Heavy machinery typically has lower weekend demand
        weekend_avg = df_copy[df_copy['dayofweek'] >= 5]['y'].mean()
        weekday_avg = df_copy[df_copy['dayofweek'] < 5]['y'].mean()
        
        if weekday_avg > 0:
            weekend_diff_pct = ((weekday_avg - weekend_avg) / weekday_avg * 100)
            
            if weekend_diff_pct > 20:
                self.recommendations.append({
                    "priority": "LOW",
                    "category": "📅 Operational Planning",
                    "action": f"Weekend demand is {weekend_diff_pct:.0f}% lower than weekdays",
                    "reason": "B2B industrial parts show typical weekday concentration pattern",
                    "impact": "Optimize warehouse staffing and receiving schedules accordingly"
                })
        
        # Calculate overall confidence (based on data quality and rules fired)
        base_confidence = 75  # Start with 75% for good quality data
        rules_bonus = min(20, len(self.rules_fired) * 2)  # +2% per rule, max 20%
        data_quality_bonus = 5 if len(self.df) > 365 else 0  # +5% if >1 year data
        
        self.confidence = min(100, base_confidence + rules_bonus + data_quality_bonus)
        
        return self.recommendations
    
    def generate_report(self):
        """Generate comprehensive expert system report"""
        report = {
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "alerts": self.alerts,
            "rules_fired": self.rules_fired,
            "recommendations": self.recommendations,
            "product_info": {
                "name": self.PRODUCT,
                "supplier": self.SUPPLIER,
                "lead_time": self.LEAD_TIME,
                "unit_cost": self.UNIT_COST,
                "unit_price": self.UNIT_PRICE,
                "profit_margin": self.PROFIT_MARGIN
            },
            "metrics": {
                "avg_daily_sales": self.avg_daily_sales,
                "sales_volatility": self.sales_volatility,
                "cv": self.cv,
                "trend_30day": self.trend_30day,
                "trend_longterm": self.trend_longterm,
                "current_stock": self.current_stock,
                "days_of_stock": self.days_of_stock,
                "max_daily_sales": self.max_daily_sales,
                "min_daily_sales": self.min_daily_sales
            }
        }
        
        if hasattr(self, 'predicted_total_demand'):
            report["forecast_metrics"] = {
                "total_demand": self.predicted_total_demand,
                "avg_daily": self.predicted_avg_daily,
                "max_daily": self.predicted_max_daily,
                "spike_factor": self.demand_spike_factor
            }
        
        return report


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SmartStock AI | Hydraulic Filter Expert System",
    page_icon="📦",
    layout="wide"
)

# Professional CSS with priority colors
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

# Product-specific header
st.markdown("""
<div class="product-header">
    <h1>📦 SmartStock AI - Hydraulic Filter Management System</h1>
    <p style="margin: 5px 0;"><strong>Product:</strong> Industrial Hydraulic Filter v2 (SKU-7782-X) | <strong>Supplier:</strong> Bosch Rexroth AG, Germany</p>
    <p style="margin: 5px 0;"><strong>Warehouse:</strong> WH-SG-JURONG-01 | <strong>Lead Time:</strong> 14 days | <strong>User:</strong> Vishnu P.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    filepath = "/mnt/user-data/uploads/inventory_data.csv"
    if not os.path.exists(filepath):
        # Fallback to local path
        filepath = "inventory_data.csv"
        if not os.path.exists(filepath):
            return None
    
    df = pd.read_csv(filepath)
    df = df.rename(columns={'Date': 'ds', 'Quantity_Sold': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    return df

df = load_data()

if df is None:
    st.error("⚠️ DATA ERROR: 'inventory_data.csv' not found. Please ensure the file is available.")
    st.stop()

# ==========================================
# 3. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ AI Configuration")

st.sidebar.subheader("1. Forecasting Engine")
model_choice = st.sidebar.radio(
    "Select Algorithm:",
    ("Facebook Prophet (Time Series)", 
     "Bayesian Ridge (Probabilistic)", 
     "Random Forest (Ensemble ML)")
)

st.sidebar.subheader("2. Forecast Horizon")
days_to_predict = st.sidebar.slider("Days to Forecast", 7, 90, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Expert System")
enable_expert_system = st.sidebar.checkbox("Enable Expert System", value=True)

if enable_expert_system:
    st.sidebar.success("🧠 **Expert System: ACTIVE**")
    st.sidebar.info("9 domain-specific rules tailored for hydraulic filter inventory management")
else:
    st.sidebar.warning("Expert System: Disabled")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**ℹ️ About the Models:**
- **Prophet:** Handles seasonality & holidays
- **Bayesian Ridge:** Probabilistic predictions
- **Random Forest:** Captures non-linear patterns
""")

# ==========================================
# 4. DASHBOARD - KEY METRICS
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_revenue = df['Total_Revenue'].sum()
    st.metric("Total Revenue (All Time)", f"${total_revenue:,.0f}")

with col2:
    st.metric("Unit Price", f"${df['Unit_Price_SGD'].iloc[-1]:.2f}")

with col3:
    recent_avg = df.tail(30)['y'].mean()
    st.metric("30-Day Avg Sales", f"{recent_avg:.1f} units/day")

with col4:
    last_stock = df['Opening_Stock'].iloc[-1]
    st.metric("Current Stock Level", f"{last_stock} units")

# ==========================================
# 5. HISTORICAL TRENDS VISUALIZATION
# ==========================================
st.markdown("---")
st.subheader("📊 Historical Sales & Stock Trends")

fig_hist = go.Figure()

# Sales trend
fig_hist.add_trace(go.Scatter(
    x=df['ds'], 
    y=df['y'], 
    name='Daily Sales',
    line=dict(color='#3366cc', width=2),
    yaxis='y'
))

# Stock levels
fig_hist.add_trace(go.Scatter(
    x=df['ds'], 
    y=df['Opening_Stock'], 
    name='Stock Level',
    line=dict(color='#dc3912', width=1, dash='dot'),
    yaxis='y2'
))

fig_hist.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis=dict(title="Units Sold", side='left'),
    yaxis2=dict(title="Stock Level", side='right', overlaying='y'),
    hovermode='x unified',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# 6. AI FORECASTING ENGINE
# ==========================================
st.markdown("---")
st.markdown("### 🔮 AI-Powered Demand Forecasting")

if st.button("🚀 Run Forecast Model", type="primary"):
    
    with st.spinner(f"Training {model_choice} on {len(df)} historical records..."):
        
        fig_forecast = go.Figure()
        
        # Historical data (greyed out)
        fig_forecast.add_trace(go.Scatter(
            x=df['ds'], 
            y=df['y'], 
            name='Historical Sales',
            line=dict(color='lightgrey', width=1)
        ))
        
        predictions = []
        future_dates = []

        # --- PROPHET MODEL ---
        if "Prophet" in model_choice:
            m = Prophet(daily_seasonality=True, weekly_seasonality=True)
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=days_to_predict)
            forecast = m.predict(future)
            
            future_data = forecast.tail(days_to_predict)
            predictions = future_data['yhat'].values
            future_dates = future_data['ds']
            color = '#00a86b'
            
            # Add confidence interval
            fig_forecast.add_trace(go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat_upper'],
                fill=None,
                mode='lines',
                line_color='rgba(0,168,107,0.1)',
                showlegend=False
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_data['ds'],
                y=future_data['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,168,107,0.1)',
                name='95% Confidence'
            ))

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
            
            # Recursive forecasting
            future_preds = []
            curr_batch = X.iloc[-1].values.reshape(1, -1)
            
            for i in range(days_to_predict):
                pred = model.predict(curr_batch)[0]
                pred = max(0, pred)  # No negative sales
                future_preds.append(pred)
                
                new_batch = np.roll(curr_batch, 1)
                new_batch[0, 0] = pred
                curr_batch = new_batch
                
            last_date = df['ds'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
            predictions = future_preds

        # Plot forecast
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, 
            y=predictions, 
            name=f'{model_choice.split("(")[0].strip()} Forecast',
            line=dict(color=color, width=3)
        ))
        
        fig_forecast.update_layout(
            height=450,
            xaxis_title="Date",
            yaxis_title="Units",
            hovermode='x unified'
        )

        st.success(f"✅ Forecast complete using {model_choice}!")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast summary
        total_predicted = np.sum(predictions)
        avg_predicted = np.mean(predictions)
        max_predicted = np.max(predictions)
        
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            st.info(f"**Total Demand**\n\n{int(total_predicted)} units")
        with fc2:
            st.info(f"**Avg Daily**\n\n{avg_predicted:.1f} units/day")
        with fc3:
            st.info(f"**Peak Day**\n\n{int(max_predicted)} units")
        with fc4:
            revenue_est = total_predicted * df['Unit_Price_SGD'].iloc[0]
            st.info(f"**Est. Revenue**\n\n${revenue_est:,.0f}")
        
        # ==========================================
        # 7. EXPERT SYSTEM ANALYSIS
        # ==========================================
        if enable_expert_system:
            st.markdown("---")
            st.markdown("## 🧠 Expert System Analysis & Recommendations")
            
            with st.spinner("Running expert system inference engine..."):
                # Initialize and run expert system
                expert = HydraulicFilterExpertSystem(df, predictions, days_to_predict)
                report = expert.generate_report()
            
            # Alert banners
            if report['alerts']:
                for alert in report['alerts']:
                    if "STOCKOUT" in alert:
                        st.markdown(f'<div class="alert-banner banner-critical">{alert}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-banner banner-warning">{alert}</div>', unsafe_allow_html=True)
            
            # System metrics
            es_col1, es_col2, es_col3, es_col4 = st.columns(4)
            
            with es_col1:
                risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
                st.metric("Risk Assessment", f"{risk_colors.get(report['risk_level'], '⚪')} {report['risk_level']}")
            
            with es_col2:
                st.metric("System Confidence", f"{report['confidence']:.0f}%")
            
            with es_col3:
                st.metric("Rules Activated", f"{len(report['rules_fired'])}/9")
            
            with es_col4:
                st.metric("Stock Coverage", f"{report['metrics']['days_of_stock']:.1f} days")
            
            # Key metrics display
            st.markdown("#### 📊 Calculated Performance Metrics")
            
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Avg Daily Sales", f"{report['metrics']['avg_daily_sales']:.1f} units")
            with m2:
                st.metric("Sales Volatility", f"±{report['metrics']['sales_volatility']:.1f} units")
            with m3:
                st.metric("30-Day Trend", f"{report['metrics']['trend_30day']:+.1f}%")
            with m4:
                st.metric("Demand Variability (CV)", f"{report['metrics']['cv']:.3f}")
            with m5:
                st.metric("Current Inventory", f"{int(report['metrics']['current_stock'])} units")
            
            # Recommendations grouped by priority
            st.markdown("#### 💡 Actionable Recommendations")
            
            priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            
            for priority in priority_order:
                recs = [r for r in report['recommendations'] if r['priority'] == priority]
                
                if recs:
                    for rec in recs:
                        priority_class = priority.lower()
                        impact_text = f"<br/><em>Impact:</em> {rec['impact']}" if 'impact' in rec else ""
                        
                        st.markdown(f"""
                        <div class="expert-box {priority_class}">
                            <strong style="font-size: 1.1em;">[{rec['priority']}] {rec['category']}</strong><br/>
                            <strong>Action:</strong> {rec['action']}<br/>
                            <em>Reasoning:</em> {rec['reason']}{impact_text}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Inference chain viewer
            with st.expander("🔍 View Expert System Inference Chain"):
                st.markdown("**Forward Chaining Rules Fired:**")
                for i, rule in enumerate(report['rules_fired'], 1):
                    st.markdown(f"`{i}.` {rule}")
                
                st.markdown("---")
                st.markdown("""
                **Knowledge Base Rules:**
                - **R1:** Stockout risk assessment based on lead time coverage
                - **R2:** Demand volatility analysis (Coefficient of Variation)
                - **R3:** Growth trend detection (short-term & long-term)
                - **R4:** Optimal reorder point calculation (ABC classification)
                - **R5:** Economic Order Quantity with international logistics
                - **R6:** AI-predicted demand spike detection
                - **R7:** International supply chain risk mitigation
                - **R8:** Profitability & cash flow optimization
                - **R9:** Category-specific patterns (Heavy Machinery B2B)
                
                **Expert System Architecture:** Rule-based forward chaining with domain-specific knowledge for industrial hydraulic filters.
                """)
            
            # Product context
            with st.expander("📦 Product & Supplier Context"):
                st.markdown(f"""
                **Product:** {report['product_info']['name']}  
                **Supplier:** {report['product_info']['supplier']} (Germany)  
                **Lead Time:** {report['product_info']['lead_time']} days  
                **Unit Cost:** ${report['product_info']['unit_cost']}  
                **Unit Price:** ${report['product_info']['unit_price']}  
                **Profit per Unit:** ${report['product_info']['profit_margin']} ({report['product_info']['profit_margin']/report['product_info']['unit_cost']*100:.1f}% margin)
                
                **Category:** Heavy Machinery Parts (Fast-moving, Class A item)
                **Zone:** Z-FAST-MOVING, Aisle A-04, Bin B-12-L2
                """)
