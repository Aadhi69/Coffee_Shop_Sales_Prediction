import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import base64

# Set page configuration
st.set_page_config(page_title="Coffee Shop Sales Prediction", layout="centered")

# Title and description
st.title("📊 Coffee Shop Sales Prediction App")
st.write("Upload your CSV file with sales data, and the app will forecast sales using the Prophet model.")

# Sidebar for user input options
st.sidebar.header("⚙️ Settings")
forecast_period = st.sidebar.slider("Forecast Period (days)", min_value=7, max_value=90, value=30, step=1)
plot_theme = st.sidebar.selectbox("Select Plot Theme", options=["default", "seaborn", "ggplot", "bmh"])

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Process the uploaded file
if uploaded_file is not None:
    try:
        # Load the uploaded file into a DataFrame
        df_sales = pd.read_csv(uploaded_file)
        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        df_sales = df_sales.rename(columns={'Date': 'ds', 'Total_Sales': 'y'})
        
        # Show a preview of the uploaded data
        st.write("### 📄 Preview of Uploaded Data")
        st.write(df_sales.head())
        
        # Display summary statistics with rupee symbol
        st.write("### 📈 Summary Statistics")
        total_sales = df_sales['y'].sum()
        avg_sales = df_sales['y'].mean()
        min_sales = df_sales['y'].min()
        max_sales = df_sales['y'].max()
        st.metric("Total Sales", f"₹{total_sales:,.2f}")
        st.metric("Average Daily Sales", f"₹{avg_sales:,.2f}")
        st.metric("Min Sales", f"₹{min_sales:,.2f}")
        st.metric("Max Sales", f"₹{max_sales:,.2f}")
        
        # Train the Prophet model
        with st.spinner("Training the model..."):
            model = Prophet()
            model.fit(df_sales)
        
        # Make future predictions based on the forecast period selected by the user
        future = model.make_future_dataframe(periods=forecast_p





