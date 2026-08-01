import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained Random Forest model
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("assignment-10", "model.pkl")

print(f"Flask loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Features in the exact order the model expects
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/')
def home():
    # Render the interactive web form dashboard
    return render_template('index.html', prediction_text=None, prediction_class=None, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains JSON data (REST API client)
        if request.is_json:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Missing JSON request body"}), 400
            
            # Extract features in the correct order
            feature_values = []
            for feat in FEATURE_NAMES:
                if feat not in data:
                    return jsonify({"error": f"Missing required feature: {feat}"}), 400
                feature_values.append(float(data[feat]))
            
            # Convert to DataFrame with feature names (to avoid scikit-learn warnings)
            features_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
            prediction = int(model.predict(features_df)[0])
            
            # Determine prediction text
            result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
            return jsonify({"prediction": result})
            
        else:
            # HTML Form submission Mode
            form_data = {}
            feature_values = []
            for feat in FEATURE_NAMES:
                val = request.form.get(feat)
                if val is None or val == "":
                    return render_template(
                        'index.html', 
                        prediction_text=f"Error: Parameter '{feat}' is required.",
                        prediction_class="error",
                        form_data=request.form
                    )
                # Parse to float or int
                form_data[feat] = val
                feature_values.append(float(val))
            
            # Convert to DataFrame
            features_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
            prediction = int(model.predict(features_df)[0])
            
            # Determine prediction text
            if prediction == 1:
                result = "Heart Disease Detected"
                prediction_class = "danger"
            else:
                result = "No Heart Disease Detected"
                prediction_class = "success"
                
            return render_template(
                'index.html', 
                prediction_text=result,
                prediction_class=prediction_class,
                form_data=form_data
            )
            
    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        else:
            return render_template(
                'index.html', 
                prediction_text=f"Server Error: {str(e)}", 
                prediction_class="error",
                form_data=None
            )

if __name__ == '__main__':
    # Run Flask server locally
    app.run(host='0.0.0.0', port=5000, debug=True)
