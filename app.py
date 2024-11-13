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

@st.cache(allow_output_mutation=True)
def train_model(data):
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(data)
    return model

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Load and prepare the data
        df_sales = pd.read_csv(uploaded_file)
        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        df_sales = df_sales.rename(columns={'Date': 'ds', 'Total_Sales': 'y'})
        
        # Show a preview of the data
        st.write("### 📄 Preview of Uploaded Data")
        st.write(df_sales.head())
        
        # Display summary statistics
        st.write("### 📈 Summary Statistics")
        total_sales = df_sales['y'].sum()
        avg_sales = df_sales['y'].mean()
        min_sales = df_sales['y'].min()
        max_sales = df_sales['y'].max()
        st.metric("Total Sales", f"₹{total_sales:,.2f}")
        st.metric("Average Daily Sales", f"₹{avg_sales:,.2f}")
        st.metric("Min Sales", f"₹{min_sales:,.2f}")
        st.metric("Max Sales", f"₹{max_sales:,.2f}")

        # Train and cache the model
        with st.spinner("Training the model..."):
            model = train_model(df_sales)
        
        # Generate future predictions based on forecast period
        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)
        
        # Display forecast table with rupee symbol
        st.write(f"### 🔮 Sales Forecast for the Next {forecast_period} Days")
        forecast_table = forecast[['ds', 'yhat']].tail(forecast_period)
        forecast_table = forecast_table.rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales'})
        forecast_table['Predicted Sales'] = forecast_table['Predicted Sales'].apply(lambda x: f"₹{x:,.2f}")
        st.write(forecast_table)

        # Allow users to download forecasted data as CSV
        csv = forecast_table.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="forecasted_sales.csv">📥 Download Forecasted Data as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Set the selected plot theme
        plt.style.use(plot_theme)

        # Plot forecast
        st.write("### 📊 Forecast Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        model.plot(forecast, ax=ax)
        ax.set_title(f"Predicted Sales Forecast (Next {forecast_period} Days)", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales (₹)")
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# Instructions
st.write("#### ℹ️ Instructions")
st.write("""
1. Upload a CSV file with at least two columns: **Date** (for dates) and **Total_Sales** (for sales data).
2. The Date column should contain date values, and Total_Sales should contain numerical sales values.
3. Use the sidebar to adjust the forecast period and plot theme. 
4. Once uploaded, the app will display a preview of the data and automatically forecast the selected period of sales.
""")









