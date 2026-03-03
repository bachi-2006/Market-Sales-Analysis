"""Generate a sample sales dataset for testing the forecasting app."""
import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range("2018-01-01", periods=72, freq="MS")

# Build realistic sales with trend + seasonality + noise
base = 5000
trend = np.linspace(0, 3000, 72)
seasonal = 1200 * np.sin(np.linspace(0, 6 * np.pi, 72))
noise = np.random.normal(0, 300, 72)
sales = base + trend + seasonal + noise
sales = np.maximum(sales, 500)  # floor at 500

df = pd.DataFrame({
    "Date": dates.strftime("%Y-%m-%d"),
    "Sales": np.round(sales, 2),
})

df.to_csv("sample_sales.csv", index=False)
print(f"Generated sample_sales.csv with {len(df)} rows")
print(df.head(12).to_string(index=False))
