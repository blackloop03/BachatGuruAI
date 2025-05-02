# fastAPI.py

from fastapi import FastAPI
from pymongo import MongoClient
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime, timedelta

# =================== Setup FastAPI App ===================

app = FastAPI()

# =================== MongoDB Setup ===================

mongo_client = MongoClient('mongodb://localhost:27017/')  # Adjust URI if needed
db = mongo_client['your_database_name']                   # Replace with your database name
collection = db['your_collection_name']                   # Replace with your collection name

# =================== Load Model and Scaler ===================

model = load_model('models/lstm_expense_predictor.keras', compile=False)  # Adjust path if needed

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# =================== Helper Functions ===================

def fetch_expense_data_from_mongo():
    today = datetime.today()
    start_date = today - timedelta(days=30)

    expenses_cursor = collection.find(
        {"Date": {"$gte": start_date}},
        {"_id": 0, "Date": 1, "amount": 1}
    )

    expenses = list(expenses_cursor)
    if not expenses:
        raise Exception("No transactions found in the last 30 days.")

    df = pd.DataFrame(expenses)
    df['Date'] = pd.to_datetime(df['Date'])

    return df

def clean_and_prepare_data(df):
    daily_totals = df.groupby(df['Date'].dt.date)['amount'].sum().reset_index()
    daily_totals.rename(columns={'Date': 'date', 'amount': 'amount'}, inplace=True)
    daily_totals['date'] = pd.to_datetime(daily_totals['date'])

    # Create full date range
    full_dates = pd.date_range(start=daily_totals['date'].min(), end=daily_totals['date'].max())
    full_data = pd.DataFrame({'date': full_dates})
    full_data = full_data.merge(daily_totals, how='left', on='date')

    # Fill missing dates
    average_expense = daily_totals['amount'].mean()
    full_data['amount'] = full_data['amount'].fillna(average_expense)

    return full_data

def prepare_last_30_days(full_data):
    amounts = full_data['amount'].values.reshape(-1, 1)
    scaled_amounts = scaler.transform(amounts)

    if len(scaled_amounts) < 30:
        average_value = np.mean(scaled_amounts)
        padding = np.full((30 - len(scaled_amounts), 1), average_value)
        scaled_amounts = np.vstack([padding, scaled_amounts])

    last_30_days = scaled_amounts[-30:]
    last_30_days = last_30_days.reshape((1, 30, 1))
    return last_30_days

def predict_next_7_days(last_30_days):
    future_predictions = []
    current_sequence = last_30_days.copy()

    for _ in range(7):
        predicted_scaled = model.predict(current_sequence, verbose=0)
        new_value = np.reshape(predicted_scaled, (1, 1, 1))
        future_predictions.append(predicted_scaled[0, 0])
        current_sequence = np.concatenate((current_sequence[:, 1:, :], new_value), axis=1)

    future_predictions_real = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions_real.flatten().tolist()

# =================== FastAPI Endpoint ===================

@app.get("/predict-7-days-from-mongo")
def predict_from_mongo():
    try:
        raw_data = fetch_expense_data_from_mongo()
        full_data = clean_and_prepare_data(raw_data)
        last_30_days = prepare_last_30_days(full_data)
        predictions = predict_next_7_days(last_30_days)

        return {
            "future_expenses": predictions
        }
    except Exception as e:
        return {"error": str(e)}
