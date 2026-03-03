"""
Sales Forecasting using Time Series Analysis
=============================================
A Streamlit application that predicts future sales using statistical
time-series models: Moving Average, ARIMA, and Exponential Smoothing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import io
import math

warnings.filterwarnings("ignore")

# ──────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
    .section-header {
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Helper Functions ───────────────────────
def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df


def clean_data(df: pd.DataFrame, date_col: str, sales_col: str) -> pd.DataFrame:
    """Clean and prepare the time-series data."""
    df = df[[date_col, sales_col]].copy()
    df.columns = ["Date", "Sales"]

    # Convert to datetime
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Convert sales to numeric
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    # Sort by date
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Handle missing values – forward fill then backward fill
    df["Sales"].fillna(method="ffill", inplace=True)
    df["Sales"].fillna(method="bfill", inplace=True)

    # Drop any remaining NaN rows
    df.dropna(inplace=True)

    return df


def aggregate_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate data to the desired frequency."""
    df = df.set_index("Date")
    df_agg = df.resample(freq).sum()
    df_agg = df_agg[df_agg["Sales"] > 0]  # remove zero-sum periods
    return df_agg


def decompose_series(series: pd.Series, period: int):
    """Perform seasonal decomposition."""
    if len(series) < 2 * period:
        return None
    try:
        result = seasonal_decompose(series, model="additive", period=period)
        return result
    except Exception:
        return None


def moving_average_forecast(train: pd.Series, test_len: int, window: int):
    """Generate forecasts using Moving Average."""
    history = list(train.values)
    predictions = []
    for _ in range(test_len):
        avg = np.mean(history[-window:])
        predictions.append(avg)
        history.append(avg)
    return np.array(predictions)


def arima_forecast(train: pd.Series, test_len: int, order: tuple):
    """Generate forecasts using ARIMA."""
    try:
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=test_len)
        return forecast.values
    except Exception as e:
        st.warning(f"ARIMA fitting issue: {e}")
        return None


def exp_smoothing_forecast(train: pd.Series, test_len: int, seasonal_periods: int):
    """Generate forecasts using Exponential Smoothing (Holt-Winters)."""
    try:
        if seasonal_periods and len(train) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=seasonal_periods,
            )
        else:
            model = ExponentialSmoothing(train, trend="add", seasonal=None)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(test_len)
        return forecast.values
    except Exception as e:
        st.warning(f"Exponential Smoothing issue: {e}")
        return None


def compute_metrics(actual: np.ndarray, predicted: np.ndarray):
    """Compute MAE and RMSE."""
    mae = mean_absolute_error(actual, predicted)
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse


# ──────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/combo-chart.png",
        width=80,
    )
    st.markdown("## Sales Forecasting")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📁 Upload Sales Data",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file with a date column and a sales column.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    freq_options = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Quarterly": "QS", "Yearly": "YS"}
    freq_label = st.selectbox("Aggregation Frequency", list(freq_options.keys()), index=2)
    freq = freq_options[freq_label]

    test_split = st.slider("Test Split (%)", 10, 40, 20, 5,
                           help="Percentage of data used for evaluating models.")

    forecast_periods = st.number_input("Forecast Periods Ahead", 1, 60, 12,
                                       help="Number of future periods to forecast.")

    st.markdown("---")
    st.markdown("### 🔧 Model Parameters")

    ma_window = st.slider("Moving Average Window", 2, 24, 3)
    arima_p = st.number_input("ARIMA p (AR order)", 0, 10, 5)
    arima_d = st.number_input("ARIMA d (Differencing)", 0, 3, 1)
    arima_q = st.number_input("ARIMA q (MA order)", 0, 10, 0)

    st.markdown("---")
    st.caption("Built with Streamlit • Time Series Analysis")


# ──────────────────────────── Main Content ───────────────────────────
st.markdown('<p class="main-header">📊 Sales Forecasting using Time Series Analysis</p>', unsafe_allow_html=True)

if uploaded_file is None:
    # Landing / instructions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 1️⃣ Upload Data
        Upload a **CSV** or **Excel** file containing historical sales data with at least a
        **date** column and a **sales** column.
        """)
    with col2:
        st.markdown("""
        ### 2️⃣ Configure
        Choose aggregation frequency, test split ratio, and tune model parameters
        in the **sidebar**.
        """)
    with col3:
        st.markdown("""
        ### 3️⃣ Forecast
        The system will automatically **analyze**, **model**, and **forecast**
        your future sales.
        """)

    st.info("👈 Upload your sales data file from the sidebar to get started!")

    # Show a sample dataset option
    st.markdown("---")
    st.markdown("### 🧪 No data? Try a sample dataset!")
    if st.button("Generate & Use Sample Data", type="primary"):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=72, freq="MS")
        base = 1000
        trend = np.linspace(0, 500, 72)
        seasonal = 200 * np.sin(np.linspace(0, 6 * np.pi, 72))
        noise = np.random.normal(0, 50, 72)
        sales = base + trend + seasonal + noise
        sample_df = pd.DataFrame({"Date": dates, "Sales": np.round(sales, 2)})

        # Store in session state
        st.session_state["sample_data"] = sample_df
        st.rerun()

    # Check if sample data was generated
    if "sample_data" in st.session_state:
        uploaded_file = "sample"

if uploaded_file is not None:
    # ─── Load Data ───
    if uploaded_file == "sample":
        raw_df = st.session_state["sample_data"]
        date_col = "Date"
        sales_col = "Sales"
    else:
        raw_df = load_data(uploaded_file)
        if raw_df is None:
            st.stop()

        st.markdown('<div class="section-header"><h2>📋 Raw Data Preview</h2></div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(20), use_container_width=True)

        # Column selection
        col_a, col_b = st.columns(2)
        with col_a:
            date_col = st.selectbox("Select Date Column", raw_df.columns.tolist())
        with col_b:
            sales_col = st.selectbox("Select Sales Column",
                                     [c for c in raw_df.columns if c != date_col])

    # ─── Clean Data ───
    df_clean = clean_data(raw_df, date_col, sales_col)

    if df_clean.empty or len(df_clean) < 6:
        st.error("Not enough valid data points after cleaning. Need at least 6.")
        st.stop()

    # ─── Aggregate ───
    df_agg = aggregate_data(df_clean, freq)

    if len(df_agg) < 6:
        st.error(f"Not enough data points at {freq_label} frequency ({len(df_agg)} points). Try a finer frequency.")
        st.stop()

    # ─── Quick Stats ───
    st.markdown('<div class="section-header"><h2>📈 Data Overview</h2></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(df_agg):,}")
    m2.metric("Date Range", f"{df_agg.index.min().strftime('%b %Y')} – {df_agg.index.max().strftime('%b %Y')}")
    m3.metric("Avg Sales", f"{df_agg['Sales'].mean():,.0f}")
    m4.metric("Total Sales", f"{df_agg['Sales'].sum():,.0f}")

    # ─── Sales Over Time Chart ───
    fig_overview = px.line(
        df_agg, x=df_agg.index, y="Sales",
        title="Sales Over Time",
        labels={"x": "Date", "Sales": "Sales"},
        template="plotly_white",
    )
    fig_overview.update_traces(line=dict(color="#667eea", width=2.5))
    fig_overview.update_layout(hovermode="x unified")
    st.plotly_chart(fig_overview, use_container_width=True)

    # ─── Seasonal Decomposition ───
    st.markdown('<div class="section-header"><h2>🔍 Time Series Decomposition</h2></div>', unsafe_allow_html=True)

    period_map = {"D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}
    seasonal_period = period_map.get(freq, 12)

    decomp = decompose_series(df_agg["Sales"], seasonal_period)

    if decomp is not None:
        fig_decomp = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            vertical_spacing=0.06,
        )
        fig_decomp.add_trace(go.Scatter(x=df_agg.index, y=decomp.observed, name="Observed",
                                        line=dict(color="#667eea")), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=df_agg.index, y=decomp.trend, name="Trend",
                                        line=dict(color="#f093fb")), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=df_agg.index, y=decomp.seasonal, name="Seasonal",
                                        line=dict(color="#4facfe")), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=df_agg.index, y=decomp.resid, name="Residual",
                                        line=dict(color="#43e97b")), row=4, col=1)
        fig_decomp.update_layout(height=700, template="plotly_white", showlegend=False,
                                 title_text="Seasonal Decomposition")
        st.plotly_chart(fig_decomp, use_container_width=True)
    else:
        st.warning("Not enough data for seasonal decomposition at this frequency. Try finer aggregation.")

    # ─── Train/Test Split ───
    st.markdown('<div class="section-header"><h2>🧪 Model Training & Evaluation</h2></div>', unsafe_allow_html=True)

    split_idx = int(len(df_agg) * (1 - test_split / 100))
    train = df_agg["Sales"].iloc[:split_idx]
    test = df_agg["Sales"].iloc[split_idx:]

    st.write(f"**Training set:** {len(train)} periods | **Test set:** {len(test)} periods")

    if len(test) < 2:
        st.error("Test set is too small. Upload more data or reduce the test split %.")
        st.stop()

    # ─── Run Models ───
    results = {}

    # 1) Moving Average
    ma_preds = moving_average_forecast(train, len(test), ma_window)
    ma_mae, ma_rmse = compute_metrics(test.values, ma_preds)
    results["Moving Average"] = {"preds": ma_preds, "mae": ma_mae, "rmse": ma_rmse}

    # 2) ARIMA
    arima_order = (arima_p, arima_d, arima_q)
    arima_preds = arima_forecast(train, len(test), arima_order)
    if arima_preds is not None:
        arima_mae, arima_rmse = compute_metrics(test.values, arima_preds)
        results["ARIMA"] = {"preds": arima_preds, "mae": arima_mae, "rmse": arima_rmse}

    # 3) Exponential Smoothing
    es_preds = exp_smoothing_forecast(train, len(test), seasonal_period)
    if es_preds is not None:
        es_mae, es_rmse = compute_metrics(test.values, es_preds)
        results["Exponential Smoothing"] = {"preds": es_preds, "mae": es_mae, "rmse": es_rmse}

    if not results:
        st.error("All models failed to produce forecasts. Please check your data or parameters.")
        st.stop()

    # ─── Evaluation Comparison ───
    st.markdown("### Model Performance Comparison")

    metrics_df = pd.DataFrame({
        model: {"MAE": r["mae"], "RMSE": r["rmse"]}
        for model, r in results.items()
    }).T
    metrics_df = metrics_df.round(2)

    best_model = metrics_df["RMSE"].idxmin()

    # Display metrics
    cols = st.columns(len(results))
    for i, (model, r) in enumerate(results.items()):
        with cols[i]:
            badge = " 🏆" if model == best_model else ""
            st.markdown(f"#### {model}{badge}")
            st.metric("MAE", f"{r['mae']:,.2f}")
            st.metric("RMSE", f"{r['rmse']:,.2f}")

    st.success(f"**Best Model: {best_model}** (lowest RMSE: {results[best_model]['rmse']:,.2f})")

    # ─── Actual vs Predicted Chart ───
    fig_eval = go.Figure()

    # Training data
    fig_eval.add_trace(go.Scatter(
        x=train.index, y=train.values, name="Train",
        line=dict(color="#667eea", width=2),
    ))
    # Actual test data
    fig_eval.add_trace(go.Scatter(
        x=test.index, y=test.values, name="Actual (Test)",
        line=dict(color="#2d3436", width=2, dash="dot"),
    ))

    colors = {"Moving Average": "#f093fb", "ARIMA": "#4facfe", "Exponential Smoothing": "#43e97b"}
    for model, r in results.items():
        fig_eval.add_trace(go.Scatter(
            x=test.index, y=r["preds"], name=f"{model} Prediction",
            line=dict(color=colors.get(model, "#ff6b6b"), width=2),
        ))

    fig_eval.update_layout(
        title="Actual vs Predicted Sales",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig_eval, use_container_width=True)

    # ─── Bar chart of metrics ───
    fig_bar = go.Figure()
    model_names = list(results.keys())
    fig_bar.add_trace(go.Bar(
        name="MAE", x=model_names,
        y=[results[m]["mae"] for m in model_names],
        marker_color="#667eea",
    ))
    fig_bar.add_trace(go.Bar(
        name="RMSE", x=model_names,
        y=[results[m]["rmse"] for m in model_names],
        marker_color="#f093fb",
    ))
    fig_bar.update_layout(
        barmode="group", title="Model Error Comparison",
        yaxis_title="Error Value", template="plotly_white", height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ─── Future Forecast ───
    st.markdown('<div class="section-header"><h2>🔮 Future Sales Forecast</h2></div>', unsafe_allow_html=True)
    st.markdown(f"Generating **{forecast_periods}** period forecast using the best model: **{best_model}**")

    full_series = df_agg["Sales"]

    if best_model == "Moving Average":
        future_preds = moving_average_forecast(full_series, forecast_periods, ma_window)
    elif best_model == "ARIMA":
        future_preds = arima_forecast(full_series, forecast_periods, arima_order)
    else:
        future_preds = exp_smoothing_forecast(full_series, forecast_periods, seasonal_period)

    if future_preds is None:
        # Fallback to Moving Average
        st.warning(f"{best_model} failed on full data. Falling back to Moving Average.")
        future_preds = moving_average_forecast(full_series, forecast_periods, ma_window)
        best_model = "Moving Average"

    # Build future date index
    last_date = df_agg.index[-1]
    future_index = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:]

    future_df = pd.DataFrame({"Date": future_index, "Forecasted Sales": np.round(future_preds, 2)})
    future_df.set_index("Date", inplace=True)

    # Forecast chart
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(
        x=df_agg.index, y=df_agg["Sales"].values,
        name="Historical Sales",
        line=dict(color="#667eea", width=2),
    ))
    fig_future.add_trace(go.Scatter(
        x=future_index, y=future_preds,
        name="Forecast",
        line=dict(color="#f093fb", width=3, dash="dash"),
        fill="tozeroy",
        fillcolor="rgba(240,147,251,0.1)",
    ))
    fig_future.update_layout(
        title=f"Future Sales Forecast ({best_model})",
        xaxis_title="Date", yaxis_title="Sales",
        template="plotly_white", hovermode="x unified", height=500,
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # Forecast table
    st.markdown("### 📅 Forecast Table")
    st.dataframe(future_df.style.format({"Forecasted Sales": "{:,.2f}"}), use_container_width=True)

    # Summary metrics for forecast
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Avg Forecasted Sales", f"{future_preds.mean():,.0f}")
    fc2.metric("Max Forecasted Sales", f"{future_preds.max():,.0f}")
    fc3.metric("Min Forecasted Sales", f"{future_preds.min():,.0f}")

    # ─── Download Forecast ───
    st.markdown("---")
    csv_buffer = io.StringIO()
    future_df.to_csv(csv_buffer)
    st.download_button(
        "📥 Download Forecast as CSV",
        data=csv_buffer.getvalue(),
        file_name="sales_forecast.csv",
        mime="text/csv",
        type="primary",
    )

    # ─── Footer ───
    st.markdown("---")
    st.markdown(
        "<center><sub>Sales Forecasting System • Built with Streamlit • "
        "Time Series Analysis using Moving Average, ARIMA & Exponential Smoothing</sub></center>",
        unsafe_allow_html=True,
    )
