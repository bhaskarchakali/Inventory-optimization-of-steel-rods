import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# Load the trained model
model = joblib.load('best_model.joblib')

# Define the number of weeks to predict
NUM_WEEKS = 52

# Function to generate future dates
def generate_dates(start_date, num_weeks):
    dates = []
    current_date = start_date
    for _ in range(num_weeks):
        dates.append(current_date)
        current_date += timedelta(weeks=1)
    return dates

# Function to predict sales volume for the next 52 weeks
def predict_sales(start_date):
    dates = generate_dates(start_date, NUM_WEEKS)
    sales = []
    for date in dates:
        sales.append(model.predict(date))
    return pd.DataFrame({'Date': [str(date) for date in dates], 'Sales Volume/Steel (in Tonnes)': np.array(sales).flatten()})

# Streamlit app
def main():
    # Set app title and description
    st.title('Sales Volume Prediction/Steel')
    st.write('Predicting sales volume for the next 52 weeks')

    # Get user input for start date
    start_date = st.date_input('Enter the start date')

    if st.button('Predict'):
        # Perform prediction
        result = predict_sales(start_date)

        # Display the predicted sales volume
        st.subheader('Predicted Sales Volume in weekly wise up to 52 weeks')
        st.dataframe(result)

if __name__ == '__main__':
    main()
