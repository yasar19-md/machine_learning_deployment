import joblib
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

import joblib

# Assume model is already trained before this step
# Example: model.fit(X_train, y_train)

model_filename = 'logistic_regression_model.pkl'

# Save the trained model
joblib.dump(model, model_filename)

print(f'Model successfully exported to {model_filename}')
print('You can download this file from the Colab environment.')
# Define the filename for the exported model
model_filename = 'logistic_regression_model.pkl'

# Load the trained model
try:
    model = joblib.load(model_filename)
    print(f'Model loaded successfully from {model_filename}')
except FileNotFoundError:
    print(f'Error: Model file {model_filename} not found. Please ensure the model is exported.')
    exit()


@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)

    # Required features
    required_features = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]
    
    input_values = []
    for feature in required_features:
        if feature not in data:
            return jsonify({"error": f"Missing feature: {feature}"}), 400
        input_values.append(data[feature])

    # Convert input to numpy array
    features_array = np.array(input_values).reshape(1, -1)

    try:
        prediction = model.predict(features_array)
        prediction_proba = model.predict_proba(features_array)

        outcome = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
        probability_diabetic = prediction_proba[0][1]

        return jsonify({
            'prediction': outcome,
            'probability_diabetic': float(probability_diabetic)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
