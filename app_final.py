import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# --- AI & ML Libraries ---
from prophet import Prophet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# --- Deep Learning ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# ==========================================
# EXPERT SYSTEM (unchanged from your original)
# ==========================================
class SimpleExpertSystem:
    def __init__(self, df, predictions=None, forecast_days=30, damaged_units=0):
        self.df = df
        self.predictions = predictions
        self.forecast_days = forecast_days
        self.damaged_units = damaged_units  # NEW: fed from CV subsystem

        self.MIN_STOCK_DAYS = 21
        self.MAX_STOCK_DAYS = 45

        self.rules_fired = []
        self.recommendations = []
        self.alerts = []
        self.risk_level = "LOW"

    def apply_rules(self):
        recent_data = self.df.tail(30)

        if 'Quantity_Sold' in recent_data.columns:
            self.avg_daily_sales = recent_data['Quantity_Sold'].mean()
        elif 'y' in recent_data.columns:
            self.avg_daily_sales = recent_data['y'].mean()
        else:
            self.avg_daily_sales = 0

        raw_stock = recent_data['Opening_Stock'].iloc[-1] if 'Opening_Stock' in recent_data.columns else 0

        # NEW: subtract damaged units detected by CV subsystem
        self.current_stock = max(raw_stock - self.damaged_units, 0)
        self.days_of_stock = self.current_stock / self.avg_daily_sales if self.avg_daily_sales > 0 else 0

        # Rule 1: Low Stock
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

        # Rule 2: Excess Stock
        elif self.days_of_stock > self.MAX_STOCK_DAYS:
            self.rules_fired.append("Rule 2: Excess Stock Detected")
            self.risk_level = "MEDIUM"
            self.recommendations.append({
                "priority": "LOW",
                "category": "Overstock",
                "action": "Pause purchasing.",
                "reason": f"Inventory covers {self.days_of_stock:.1f} days (Exceeds {self.MAX_STOCK_DAYS} day maximum)."
            })

        # Rule 3: Healthy Stock
        else:
            self.rules_fired.append("Rule 3: Healthy Inventory Level")
            self.recommendations.append({
                "priority": "LOW",
                "category": "Optimization",
                "action": "Maintain current operations.",
                "reason": f"Stock is at an optimal {self.days_of_stock:.1f} days."
            })

        # Rule 4: AI Spike Detection
        if self.predictions is not None:
            max_predicted = np.max(self.predictions)
            if max_predicted > (self.avg_daily_sales * 1.5):
                self.rules_fired.append("Rule 4: AI Demand Spike Predicted")
                self.alerts.append("🔥 AI DEMAND SPIKE DETECTED")
                self.risk_level = "CRITICAL"
                self.recommendations.append({
                    "priority": "CRITICAL",
                    "category": "AI Forecast",
                    "action": "Prepare warehouse for incoming surge.",
                    "reason": f"AI predicts a peak of {int(max_predicted)} units/day (50%+ above normal)."
                })

        # NEW Rule 5: Damage Alert
        if self.damaged_units > 0:
            self.rules_fired.append("Rule 5: Damaged Stock Detected")
            self.alerts.append(f"🔍 DAMAGE ALERT: {self.damaged_units} units flagged as damaged")
            damage_pct = self.damaged_units / max(raw_stock, 1)
            if damage_pct > 0.10:
                self.risk_level = max(self.risk_level, "HIGH") if self.risk_level != "CRITICAL" else "CRITICAL"
                self.recommendations.append({
                    "priority": "HIGH",
                    "category": "Quality Control",
                    "action": f"Quarantine {self.damaged_units} damaged units. Contact supplier SUP-BOSCH-DE.",
                    "reason": f"{damage_pct:.0%} of stock flagged as damaged by AI vision system."
                })
            else:
                self.recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Quality Control",
                    "action": f"Review {self.damaged_units} flagged units before dispatch.",
                    "reason": "CV subsystem detected potential product damage."
                })

    def generate_report(self):
        return {
            "risk_level": self.risk_level,
            "alerts": self.alerts,
            "rules_fired": self.rules_fired,
            "recommendations": self.recommendations,
            "metrics": {
                "avg_daily_sales": self.avg_daily_sales,
                "current_stock": self.current_stock,
                "days_of_stock": self.days_of_stock,
                "damaged_units": self.damaged_units,
            }
        }


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="SmartStock AI",
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
    .high     { border-left-color: #fd7e14; background-color: #fff8f0; }
    .medium   { border-left-color: #0dcaf0; background-color: #f0fcff; }
    .low      { border-left-color: #198754; background-color: #f0fff4; }
    .alert-banner { padding: 12px 20px; border-radius: 6px; margin: 10px 0; font-weight: 600; text-align: center; }
    .banner-critical { background-color: #dc3545; color: white; }
    .banner-warning  { background-color: #ffc107; color: #000; }
    .product-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;
    }
    .damage-badge-damaged   { background:#dc3545; color:white; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.85em; }
    .damage-badge-undamaged { background:#198754; color:white; padding:3px 12px; border-radius:20px; font-weight:700; font-size:0.85em; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="product-header">
    <h1>📦 SmartStock AI — Intelligent Inventory System</h1>
    <p style="margin: 5px 0;"><strong>Product:</strong> Industrial Hydraulic Filter v2 &nbsp;|&nbsp;
    <strong>Warehouse:</strong> WH-SG-JURONG-01 &nbsp;|&nbsp;
    <strong>Supplier:</strong> Bosch Rexroth AG</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    for path in ["inventory_data.csv", "/mnt/user-data/uploads/inventory_data.csv"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.rename(columns={'Date': 'ds', 'Quantity_Sold': 'y'})
            df['ds'] = pd.to_datetime(df['ds'])
            return df

    # Fallback synthetic data matching the real schema
    dates = pd.date_range(end=datetime.today(), periods=730)
    y     = np.maximum(np.random.normal(55, 15, 730), 5).astype(int)
    stock = np.maximum(np.random.normal(450, 80, 730), 50).astype(int)
    price = 145.5
    return pd.DataFrame({
        'ds': dates, 'y': y,
        'Opening_Stock': stock,
        'Unit_Price_SGD': price,
        'Total_Revenue':  y * price,
        'Gross_Profit':   y * (price - 85.0),
        'Supplier_ID':    'SUP-BOSCH-DE',
        'Transaction_ID': [f'TRX-{100000+i}' for i in range(730)],
        'Warehouse_ID':   'WH-SG-JURONG-01',
    })

df = load_data()


# ==========================================
# SIDEBAR
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
    st.sidebar.success("🧠 Expert System: ACTIVE")

st.sidebar.markdown("---")
st.sidebar.subheader("4. CV Damage Detection")
cv_model_choice = st.sidebar.radio(
    "CV Model:",
    ("MobileNetV2", "YOLOv8n-cls"),
    help="MobileNetV2 = transfer learning baseline. YOLOv8 = faster, modern unified model."
)
st.sidebar.caption("Trained weights auto-loaded from models/ if available, else demo mode.")

st.sidebar.markdown("---")
st.sidebar.caption("ICT304 · SmartStock v2.0")
st.sidebar.caption("Vishnu · Sudheep · Ashik")


# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📈 Demand Forecasting",
    "🔍 Damage Detection (CV)",
    "📊 Model Comparison"
])


# ==========================================
# TAB 1 — FORECASTING (your original logic, unchanged)
# ==========================================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"SGD {df['Total_Revenue'].sum():,.0f}")
    with col2:
        st.metric("Unit Price", f"SGD {df['Unit_Price_SGD'].iloc[-1]:.2f}")
    with col3:
        st.metric("30-Day Avg Sales", f"{df.tail(30)['y'].mean():.1f} units")
    with col4:
        stock_val = int(df['Opening_Stock'].iloc[-1]) if 'Opening_Stock' in df.columns else "N/A"
        damaged_so_far = st.session_state.get("damaged_units", 0)
        effective = stock_val - damaged_so_far if isinstance(stock_val, int) else stock_val
        st.metric("Effective Stock", f"{effective} units",
                  delta=f"-{damaged_so_far} damaged" if damaged_so_far > 0 else None,
                  delta_color="inverse")

    st.subheader("📊 Historical Sales & Stock Trends")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Daily Sales',
                                  line=dict(color='#3366cc', width=2), yaxis='y'))
    if 'Opening_Stock' in df.columns:
        fig_hist.add_trace(go.Scatter(x=df['ds'], y=df['Opening_Stock'], name='Stock Level',
                                      line=dict(color='#dc3912', width=1, dash='dot'), yaxis='y2'))
    fig_hist.update_layout(
        height=400, xaxis_title="Date",
        yaxis=dict(title="Units Sold", side='left'),
        yaxis2=dict(title="Stock Level", side='right', overlaying='y'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 AI-Powered Demand Forecasting")

    if st.button("🚀 Run Forecast Model", type="primary", key="forecast_btn"):
        with st.spinner(f"Training {model_choice}..."):
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Historical Sales',
                                              line=dict(color='lightgrey', width=1)))
            predictions, future_dates = [], []

            # 1D CNN
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

                model_tf = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, 1)),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(50, activation='relu'),
                    Dense(1)
                ])
                model_tf.compile(optimizer='adam', loss='mse')
                model_tf.fit(X_cnn, y_cnn, epochs=50, batch_size=16, verbose=0)

                curr_batch = scaled_data[-look_back:].reshape((1, look_back, 1))
                future_preds_scaled = []
                for _ in range(days_to_predict):
                    pred = model_tf.predict(curr_batch, verbose=0)[0]
                    future_preds_scaled.append(pred)
                    curr_batch = np.append(curr_batch[:, 1:, :], [[pred]], axis=1)

                predictions = np.maximum(scaler.inverse_transform(future_preds_scaled).flatten(), 0)
                future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict)
                color = '#8a2be2'

            # Prophet
            elif "Prophet" in model_choice:
                m = Prophet(daily_seasonality=True, weekly_seasonality=True)
                m.fit(df[['ds', 'y']])
                future   = m.make_future_dataframe(periods=days_to_predict)
                forecast = m.predict(future)
                future_data  = forecast.tail(days_to_predict)
                predictions  = future_data['yhat'].values
                future_dates = future_data['ds']
                color = '#00a86b'

            # Bayesian / Random Forest
            else:
                df_ml = df[['y']].copy()
                look_back = 30
                for i in range(1, look_back + 1):
                    df_ml[f'lag_{i}'] = df_ml['y'].shift(i)
                df_ml = df_ml.dropna()
                X_ml, y_ml = df_ml.drop('y', axis=1), df_ml['y']

                if "Bayesian" in model_choice:
                    model_sk = BayesianRidge()
                    color = '#ff8c00'
                else:
                    model_sk = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    color = '#dc143c'

                model_sk.fit(X_ml, y_ml)
                curr_batch = X_ml.iloc[-1].values.reshape(1, -1)
                future_preds = []
                for _ in range(days_to_predict):
                    pred = max(0, model_sk.predict(curr_batch)[0])
                    future_preds.append(pred)
                    new_batch = np.roll(curr_batch, 1)
                    new_batch[0, 0] = pred
                    curr_batch = new_batch

                predictions  = future_preds
                future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict)

            # Store predictions in session state for expert system
            st.session_state["predictions"]  = predictions
            st.session_state["future_dates"] = future_dates

            fig_forecast.add_trace(go.Scatter(x=future_dates, y=predictions, name='Forecast',
                                              line=dict(color=color, width=3)))
            fig_forecast.update_layout(height=450, xaxis_title="Date", yaxis_title="Units", hovermode='x unified')

            st.success(f"✅ Forecast complete using {model_choice}!")
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Expert System
            if enable_expert_system:
                st.markdown("---")
                st.markdown("## 🧠 Expert System Recommendations")

                damaged_units = st.session_state.get("damaged_units", 0)
                expert = SimpleExpertSystem(df, predictions, days_to_predict, damaged_units)
                expert.apply_rules()
                report = expert.generate_report()

                for alert in report['alerts']:
                    st.markdown(f'<div class="alert-banner banner-critical">{alert}</div>',
                                unsafe_allow_html=True)

                es_col1, es_col2, es_col3, es_col4 = st.columns(4)
                risk_icons = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
                es_col1.metric("Risk Level",         f"{risk_icons.get(report['risk_level'], '⚪')} {report['risk_level']}")
                es_col2.metric("Days of Stock",      f"{report['metrics']['days_of_stock']:.1f}")
                es_col3.metric("Rules Triggered",    str(len(report['rules_fired'])))
                es_col4.metric("Damaged Units",      str(report['metrics']['damaged_units']))

                st.markdown("#### 💡 Actionable Recommendations")
                for rec in report['recommendations']:
                    st.markdown(f"""
                    <div class="expert-box {rec['priority'].lower()}">
                        <strong style="font-size:1.1em;">[{rec['priority']}] {rec['category']}</strong><br/>
                        <strong>Action:</strong> {rec['action']}<br/>
                        <em>Reasoning:</em> {rec['reason']}
                    </div>""", unsafe_allow_html=True)

                with st.expander("🔍 View Rule Engine"):
                    st.markdown("""
                    **Active Rules:**
                    1. **Low Stock:** Triggers if inventory < 21 days of supply.
                    2. **Excess Stock:** Triggers if inventory > 45 days of supply.
                    3. **Optimal Stock:** Triggers if between 21–45 days.
                    4. **AI Spike Detection:** Triggers if forecast predicts +50% surge.
                    5. **Damage Alert:** Triggers if CV subsystem detects damaged units (NEW).
                    """)


# ==========================================
# TAB 2 — CV DAMAGE DETECTION
# ==========================================
with tab2:
    st.header("🔍 Damaged Goods Detection")
    st.markdown("Upload product images. The AI classifies each as **damaged** or **undamaged**. "
                "Detected damaged units are automatically subtracted from effective stock in the Expert System.")

    # Load detector
    @st.cache_resource
    def get_detector(model_type):
        cv_dir = Path(__file__).parent / "cv_subsystem"
        if cv_dir.exists():
            sys.path.insert(0, str(cv_dir))
        try:
            from cv_inference import DamageDetector
        except ImportError:
            st.error("cv_subsystem/cv_inference.py not found. Place it next to app.py.")
            return None

        weights_map = {
            "mobilenet": "models/mobilenet_best.pth",
            "yolo":      "models/smartstock_yolo/weights/best.pt",
        }
        key = "mobilenet" if "MobileNet" in model_type else "yolo"
        wp  = weights_map[key]
        has_weights = Path(wp).exists()
        return DamageDetector(model_type=key,
                              weights_path=wp if has_weights else None,
                              demo_mode=not has_weights)

    detector = get_detector(cv_model_choice)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded_files = st.file_uploader(
            "Upload product images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="You can use images from dataset/test/damaged/ or dataset/test/undamaged/ to test"
        )
        batch_units = st.number_input(
            "How many units does this batch represent?",
            min_value=1, max_value=10000, value=100
        )

        run_cv = st.button("🔍 Analyse Images", type="primary",
                           disabled=(not uploaded_files or detector is None),
                           use_container_width=True)

    with col_right:
        if run_cv and uploaded_files:
            from PIL import Image

            results = []
            progress_bar = st.progress(0)
            for i, f in enumerate(uploaded_files):
                img = Image.open(f).convert("RGB")
                res = detector.predict(img)
                res["filename"] = f.name
                res["image"]    = img
                results.append(res)
                progress_bar.progress((i + 1) / len(uploaded_files))

            n_damaged   = sum(1 for r in results if r["label"] == "damaged")
            damage_rate = n_damaged / len(results)
            est_damaged = int(damage_rate * batch_units)

            # Push to session state → Expert System picks it up
            st.session_state["damaged_units"] = est_damaged
            st.session_state["cv_results"]    = results

            st.metric("Images analysed", len(results))
            c1, c2 = st.columns(2)
            c1.metric("Damaged",   n_damaged)
            c2.metric("Undamaged", len(results) - n_damaged)
            st.metric("Est. damaged units in batch", est_damaged,
                      delta=f"{damage_rate:.0%} damage rate")

            if est_damaged > 0:
                st.warning(f"⚠️ Effective stock reduced by {est_damaged} units. "
                           f"Re-run forecast to update Expert System.")

        elif "cv_results" in st.session_state:
            results = st.session_state["cv_results"]
            n_damaged = sum(1 for r in results if r["label"] == "damaged")
            st.info(f"Last scan: {len(results)} images — {n_damaged} damaged detected.")

    # Per-image results
    if "cv_results" in st.session_state:
        st.markdown("---")
        st.subheader("Per-image results")
        for r in st.session_state["cv_results"]:
            img_col, info_col = st.columns([1, 3])
            with img_col:
                st.image(r["image"], width=100)
            with info_col:
                badge = f"damage-badge-{r['label']}"
                st.markdown(f"**{r['filename']}** &nbsp; "
                            f"<span class='{badge}'>{r['label'].upper()}</span> &nbsp; "
                            f"Confidence: **{r['confidence']:.1%}** &nbsp; "
                            f"<small>Model: {r['model_used']}</small>",
                            unsafe_allow_html=True)

                bar = go.Figure(go.Bar(
                    x=[r["damaged_prob"], 1 - r["damaged_prob"]],
                    y=["damaged", "undamaged"],
                    orientation="h",
                    marker_color=["#dc3545", "#198754"],
                ))
                bar.update_layout(height=70, margin=dict(t=0, b=0, l=0, r=0),
                                  template="plotly_white", showlegend=False,
                                  xaxis=dict(range=[0, 1]))
                st.plotly_chart(bar, use_container_width=True)


# ==========================================
# TAB 3 — MODEL COMPARISON
# ==========================================
with tab3:
    st.header("📊 Model Comparison & Evaluation")

    import json
    mob_path  = Path("models/mobilenet_metrics.json")
    yolo_path = Path("models/yolo_metrics.json")

    if mob_path.exists() and yolo_path.exists():
        with open(mob_path)  as f: mob_m  = json.load(f)
        with open(yolo_path) as f: yolo_m = json.load(f)

        comparison = pd.DataFrame([
            {"Model": "MobileNetV2",  "Accuracy": mob_m["accuracy"],
             "Precision": mob_m["precision"], "Recall": mob_m["recall"], "F1": mob_m["f1"]},
            {"Model": "YOLOv8n-cls", "Accuracy": yolo_m["accuracy"],
             "Precision": yolo_m["precision"], "Recall": yolo_m["recall"], "F1": yolo_m["f1"]},
        ])
        st.dataframe(comparison.set_index("Model").style.highlight_max(axis=0),
                     use_container_width=True)

        fig_cmp = go.Figure()
        for _, row in comparison.iterrows():
            fig_cmp.add_trace(go.Bar(name=row["Model"],
                                     x=["Accuracy","Precision","Recall","F1"],
                                     y=[row["Accuracy"],row["Precision"],row["Recall"],row["F1"]]))
        fig_cmp.update_layout(barmode="group", height=350, yaxis=dict(range=[0,1.05]),
                              title="CV Model Performance — Test Set")
        st.plotly_chart(fig_cmp, use_container_width=True)

        if "history" in mob_m:
            h = mob_m["history"]
            ep = list(range(1, len(h["train_acc"]) + 1))
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=ep, y=h["train_acc"], name="Train Acc", mode="lines+markers"))
            fig_lc.add_trace(go.Scatter(x=ep, y=h["val_acc"],   name="Val Acc",   mode="lines+markers"))
            fig_lc.update_layout(height=300, title="MobileNetV2 Learning Curves",
                                 xaxis_title="Epoch", yaxis_title="Accuracy")
            st.plotly_chart(fig_lc, use_container_width=True)

    else:
        st.info("Train both models first to see live results here.")
        st.markdown("""
```bash
python cv_subsystem/train_mobilenet.py --data_dir dataset --epochs 15
python cv_subsystem/train_yolov8.py    --data_dir dataset --epochs 30
```
        """)
        # Literature-based expected results placeholder
        st.subheader("Expected performance (literature baseline)")
        st.dataframe(pd.DataFrame([
            {"Model": "MobileNetV2",  "Accuracy": "~92–95%", "F1": "~0.91–0.94", "Params": "3.4M", "Inference": "~20ms CPU"},
            {"Model": "YOLOv8n-cls", "Accuracy": "~93–96%", "F1": "~0.92–0.95", "Params": "1.9M", "Inference": "~8ms CPU"},
        ]).set_index("Model"), use_container_width=True)

    st.markdown("---")
    st.subheader("Technique Comparison")
    st.dataframe(pd.DataFrame([
        {"Criterion": "Architecture",   "MobileNetV2": "Inverted residuals, depthwise separable convolutions", "YOLOv8n-cls": "CSPNet backbone, unified head"},
        {"Criterion": "Pre-training",   "MobileNetV2": "ImageNet (1.28M images)",       "YOLOv8n-cls": "COCO + ImageNet composite"},
        {"Criterion": "Inference speed","MobileNetV2": "~20ms CPU",                      "YOLOv8n-cls": "~8ms CPU"},
        {"Criterion": "Model size",     "MobileNetV2": "~14MB",                          "YOLOv8n-cls": "~6MB"},
        {"Criterion": "Fine-tuning",    "MobileNetV2": "Standard PyTorch transfer",      "YOLOv8n-cls": "Ultralytics one-line API"},
        {"Criterion": "Best for",       "MobileNetV2": "Reliable baseline, documented",  "YOLOv8n-cls": "Production, speed-critical"},
    ]).set_index("Criterion"), use_container_width=True)
