# 📊 Yahi - Sales Forecasting System

Yahi is an interactive, web-based Sales Forecasting and Time Series Analysis application built using Streamlit. It enables users to upload historical transaction datasets, perform seasonal decompositions, and generate forecasts.

---

## ✨ Features

- **File Uploader**: Loads CSV, XLS, or XLSX sales data sheets directly from the sidebar.
- **Time Series Decomposition**: Decomposes raw sales lines into Observed, Trend, Seasonal, and Residual outputs using additive seasonal decomposition.
- **Model Evaluation**: Runs multiple mathematical and statistical forecasting models side-by-side:
  - **Moving Average**
  - **ARIMA** (Autoregressive Integrated Moving Average)
  - **Exponential Smoothing** (Holt-Winters additive model)
- **Error Metric Visualizer**: Computes MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) comparisons, highlighting the best performing model automatically.
- **Future Forecasting & Export**: Extrapolates historical sales trends to forecast future periods and download predictions as CSV.

---

## 🛠️ Tech Stack

- **Frontend / Dashboard**: Streamlit
- **Math / Stats Engine**: `statsmodels` (ARIMA, ExponentialSmoothing, seasonal_decompose), `numpy`, `pandas`, `scikit-learn`
- **Charts / Visualizations**: Plotly, Plotly Express

---

## 📂 Project Structure

- `app.py`: Main Streamlit application and forecast modeling logic.
- `generate_sample.py`: Local script to generate sales patterns for testing.
- `sample_sales.csv`: Sample sales dataset for immediate testing.
- `requirements.txt`: Python package requirements list.

---

## ⚙️ Running Locally

1. Install system dependencies:
   ```bash
   pip install streamlit pandas numpy plotly statsmodels scikit-learn
   ```

2. Start the Streamlit application server:
   ```bash
   streamlit run app.py
   ```
