import requests

# Test data
data = [
    { "Date": "2025-01-01", "amount": 400 },
    { "Date": "2025-01-02", "amount": 500 },
    { "Date": "2025-01-04", "amount": 450 },
    { "Date": "2025-01-05", "amount": 600 },
    { "Date": "2025-01-06", "amount": 550 }
]

response = requests.post('http://localhost:5000/predict-7-days', json=data)

if response.status_code == 200:
    print("Prediction Success:")
    print(response.json())
else:
    print("Error occurred:", response.text)
