from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# ================== Load Model and Scaler ==================

# Load the trained LSTM model
model = load_model('lstm_expense_predictor.keras')

# Load the saved scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ================== Helper Functions ==================

def clean_and_prepare_data(user_expenses):
    df = pd.DataFrame(user_expenses)

    # Ensure 'Date' is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Group by day if multiple expenses in a day
    daily_totals = df.groupby(df['Date'].dt.date)['amount'].sum().reset_index()
    daily_totals.rename(columns={'Date': 'date', 'amount': 'amount'}, inplace=True)
    daily_totals['date'] = pd.to_datetime(daily_totals['date'])

    # Create full daily range
    full_dates = pd.date_range(start=daily_totals['date'].min(), end=daily_totals['date'].max())
    full_data = pd.DataFrame({'date': full_dates})
    full_data = full_data.merge(daily_totals, how='left', on='date')

    # Calculate average of available expenses
    average_expense = daily_totals['amount'].mean()

    # Fill missing dates with the average expense
    full_data['amount'] = full_data['amount'].fillna(average_expense)

    return full_data

def prepare_last_30_days(full_data):
    amounts = full_data['amount'].values.reshape(-1, 1)

    # Scale amounts
    scaled_amounts = scaler.transform(amounts)

    # If less than 30 days, pad with average expensea
    if len(scaled_amounts) < 30:a
        average_value = np.mean(scaled_amounts)
        padding = np.full((30 - len(scaled_amounts), 1), average_value)
        scaled_amounts = np.vstack([padding, scaled_amounts])

    # Take last 30 days
    last_30_days = scaled_amounts[-30:]
    last_30_days = last_30_days.reshape((1, 30, 1))

    return last_30_days


def predict_next_7_days(last_30_days):
    future_predictions = []
    current_sequence = last_30_days.copy()

    for _ in range(7):
        predicted_scaled = model.predict(current_sequence, verbose=0)

        # Correct reshaping
        new_value = np.reshape(predicted_scaled, (1, 1, 1))

        future_predictions.append(predicted_scaled[0, 0])

        # Correct sequence update
        current_sequence = np.concatenate((current_sequence[:, 1:, :], new_value), axis=1)

    # Inverse scale to get real money values
    future_predictions_real = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    return future_predictions_real.flatten().tolist()

# ================== Flask API Endpoint ==================

@app.route('/predict-7-days', methods=['POST'])
def predict():
    try:
        user_expenses = request.get_json()

        full_data = clean_and_prepare_data(user_expenses)
        last_30_days = prepare_last_30_days(full_data)
        predictions = predict_next_7_days(last_30_days)

        return jsonify({
            'future_expenses': predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ================== Run Server ==================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
