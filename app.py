# app.py
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
df_sales = pd.read_csv('coffee_shop_sales.csv')
df_sales['Date'] = pd.to_datetime(df_sales['Date'])
df_sales = df_sales.rename(columns={'Date': 'ds', 'Total_Sales': 'y'})

# Train the Prophet model
model = Prophet()
model.fit(df_sales)

# Make future predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Display forecast table
st.write("## Sales Forecast for the Next 30 Days")
st.write(forecast[['ds', 'yhat']].tail(30).rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales'}))

# Plot forecast
st.write("## Forecast Plot")
fig, ax = plt.subplots()
model.plot(forecast, ax=ax)
st.pyplot(fig)
