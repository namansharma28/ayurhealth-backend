from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
# DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, 'disease_model.pkl')
DOSHA_MODEL_PATH = os.path.join(MODEL_DIR, 'dosha_model.pkl')
DOSHA_ENCODER_PATH = os.path.join(MODEL_DIR, 'dosha_label_encoder.pkl')

# Load models
# disease_model = joblib.load(DISEASE_MODEL_PATH)
dosha_model = joblib.load(DOSHA_MODEL_PATH)
dosha_label_encoder = joblib.load(DOSHA_ENCODER_PATH)

# @app.route('/predict-disease', methods=['POST'])
# def predict_disease():
#     try:
#         data = request.get_json()
#         features = np.array(data['features']).reshape(1, -1)
#         prediction = disease_model.predict(features)
#         return jsonify({'prediction': prediction[0]})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/predict-dosha', methods=['POST'])
def predict_dosha():
    try:
        data = request.get_json()
        print("Raw input:", data)

        # Extract fields
        age = data.get('age')
        gender = data.get('gender')
        weight = data.get('weight')
        height = data.get('height')
        food_pref = data.get('food_preference')
        # mood = data.get('mood', 'neutral')/
        sleep_quality = data.get('sleep_quality')
        # symptoms = data.get('symptoms', [])

        # # Manual encoding (same as training!)
        gender_encoded = 0 if gender == 'male' else 1
        food_encoded = {'veg': 0, 'non-veg': 1, 'vegan': 2}.get(food_pref)
        # # mood_encoded = {'calm': 0, 'neutral': 1, 'anxious': 2}.get(mood, 1)
        sleep_encoded = {'good': 0, 'average': 1, 'bad': 2}.get(sleep_quality)

        # Symptom encoding (simple version for now)
        # all_symptoms = ['fatigue', 'dry skin', 'acne', 'headache', 'joint pain']  # same as training!
        # symptom_vector = [1 if sym in symptoms else 0 for sym in all_symptoms]

        # Combine all features
        input_vector = [age, gender_encoded, weight, height, food_encoded, sleep_encoded]
        input_vector = np.array(input_vector).reshape(1, -1)

        print("Final input vector:", input_vector)

        # Predict
        prediction = dosha_model.predict(input_vector)
        decoded_prediction = dosha_label_encoder.inverse_transform(prediction)

        return jsonify({'prediction': decoded_prediction[0]})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    return "AyurHealth.AI Flask API is running 🚀"

if __name__ == '__main__':
    app.run(debug=True)
