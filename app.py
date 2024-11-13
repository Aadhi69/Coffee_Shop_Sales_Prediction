import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Coffee Shop Sales Prediction", layout="centered")

# Title and description
st.title("Coffee Shop Sales Prediction App")
st.write("Upload your CSV file with sales data, and the app will forecast sales for the next 30 days using the Prophet model.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Process the uploaded file
if uploaded_file is not None:
    # Load the uploaded file into a DataFrame
    try:
        df_sales = pd.read_csv(uploaded_file)
        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        df_sales = df_sales.rename(columns={'Date': 'ds', 'Total_Sales': 'y'})
        
        # Show a preview of the uploaded data
        st.write("### Preview of Uploaded Data")
        st.write(df_sales.head())
        
        # Train the Prophet model
        with st.spinner("Training the model..."):
            model = Prophet()
            model.fit(df_sales)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Display forecast table
        st.write("### Sales Forecast for the Next 30 Days")
        forecast_table = forecast[['ds', 'yhat']].tail(30).rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales'})
        st.write(forecast_table)

        # Plot forecast
        st.write("### Forecast Plot")
        fig, ax = plt.subplots()
        model.plot(forecast, ax=ax)
        plt.title("Predicted Sales Forecast")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()
else:
    st.info("Please upload a CSV file to proceed.")

# App instructions
st.write("#### Instructions")
st.write("""
1. Upload a CSV file with at least two columns: **Date** (for dates) and **Total_Sales** (for sales data).
2. The Date column should contain date values, and Total_Sales should contain numerical sales values.
3. Once uploaded, the app will display a preview of the data and automatically forecast the next 30 days of sales.
""")

