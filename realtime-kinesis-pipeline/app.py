from urllib import response
import streamlit as st
import boto3
import pandas as pd
from decimal import Decimal
from datetime import datetime

# Connect to DynamoDB
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')  # Change region if needed
table = dynamodb.Table('ClickEvents')

@st.cache_data(show_spinner=False)
def fetch_dynamodb_data():
    response = table.scan()
    items = response.get('Items', [])
    
    # Convert items to usable format
    for item in items:
        item['user_id'] = int(item['user_id'])
        item['timestamp'] = float(item['timestamp']) if isinstance(item['timestamp'], Decimal) else item['timestamp']
    
    return pd.DataFrame(items)

st.title("📊 Real-Time Clickstream Analytics")

try:
    df = fetch_dynamodb_data()

    if df.empty:
        st.warning("No data found in DynamoDB table.")
    else:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        st.subheader("Raw Data from DynamoDB")
        st.dataframe(df)

        st.subheader("Action Distribution")
        st.bar_chart(df['action'].value_counts())

        st.subheader("Clicks Over Time")
        if 'click' in df['action'].unique():
            # Filter click actions
            click_df = df[df['action'] == 'click'].copy()  # Copy avoids chained assignment issues

            # Convert timestamp to datetime
            click_df.loc[:, 'datetime'] = pd.to_datetime(click_df['timestamp'], unit='s', errors='coerce')
            click_df = click_df.dropna(subset=['datetime'])

            # Plot safely
            try:
                click_series = click_df.set_index('datetime').resample('1min').count()['action']
                st.line_chart(click_series)
            except Exception as e:
                st.error(f"Error displaying click chart: {e}")


        else:   
            st.warning("No click actions found in the data.")                
except Exception as e:
    st.error(f"Error fetching data: {e}")
