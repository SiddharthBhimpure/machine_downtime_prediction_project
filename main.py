import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import warnings
import streamlit as st

warnings.filterwarnings('ignore')

# Streamlit UI
st.title("Downtime Prediction using ARIMA Model")
st.write("Predict the next machine downtime occurrence based on Loss Reason Code")

# Load the dataset
df = pd.read_csv('downtime_data.csv')

# Convert 'Loss Start Time' and 'Loss End Time' to datetime
df['Loss Start Time'] = pd.to_datetime(df['Loss Start Time'])
df['Loss End Time'] = pd.to_datetime(df['Loss End Time'])

# Calculate downtime duration in minutes
df['Downtime Duration'] = (df['Loss End Time'] - df['Loss Start Time']).dt.total_seconds() / 60

# Resample to hourly frequency
df.set_index('Loss Start Time', inplace=True)
hourly_downtimes = df.groupby(['Loss Reason Code']).resample('H').agg({'Downtime Duration': 'sum'}).unstack(level=0)
hourly_downtimes = hourly_downtimes.fillna(0)

# Function to fit ARIMA model and make predictions
def fit_arima_and_predict(series, n_periods=168):  # 168 hours = 7 days
    model = auto_arima(series, start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                       start_P=0, seasonal=False, d=1, D=1, trace=True,
                       error_action='ignore', suppress_warnings=True, stepwise=True)
    model.fit(series)
    forecast = model.predict(n_periods=n_periods)
    return forecast

# Function to predict next downtime in hours
def predict_next_downtime_in_hours(loss_reason_code, n_periods=168):
    series = hourly_downtimes[('Downtime Duration', loss_reason_code)]
    forecast = fit_arima_and_predict(series, n_periods=n_periods)
    next_downtime = next((i for i, x in enumerate(forecast) if x > 0), None)
    
    if next_downtime is not None:
        return f"The next downtime for Loss Reason Code {loss_reason_code} is predicted to occur in {next_downtime} hours."
    else:
        return f"No downtime predicted for Loss Reason Code {loss_reason_code} in the next {n_periods} hours."

# Get unique Loss Reason Codes for the dropdown
loss_reason_codes = hourly_downtimes.columns.get_level_values(1).unique()
loss_reason_code = st.selectbox("Select Loss Reason Code:", loss_reason_codes)

# Predict and display result on Streamlit
if st.button("Predict Next Downtime"):
    result = predict_next_downtime_in_hours(loss_reason_code)
    st.write(result)
