from flask import Flask, request, jsonify
import joblib, os
import pandas as pd

app = Flask(__name__)
MODEL_PATH = "/app/artifacts/model.joblib"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Model not found. Train it first using train.py.")
    model = None

@app.route('/')
def home():
    return jsonify({'status': 'running', 'message': 'Flask ML API is ready!'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    X = data.get('instances')
    df = pd.DataFrame(X)
    preds = model.predict(df)
    return jsonify({'predictions': preds.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
