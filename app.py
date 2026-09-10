import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML/CSS Template included directly in app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Status Risk Predictor</title>
    <style>
        * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: #f4f7f6; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; padding: 20px; }
        .card { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); width: 100%; max-width: 600px; }
        h2 { margin-top: 0; color: #2c3e50; text-align: center; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { margin-bottom: 12px; }
        .form-group.full { grid-column: span 2; }
        label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 5px; color: #34495e; }
        input, select { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; font-size: 0.95rem; }
        button { width: 100%; padding: 12px; background-color: #3498db; color: white; border: none; border-radius: 6px; font-size: 1rem; font-weight: bold; cursor: pointer; margin-top: 15px; }
        button:hover { background-color: #2980b9; }
        .result { margin-top: 20px; padding: 15px; border-radius: 6px; text-align: center; font-size: 1.2rem; font-weight: bold; }
        .At-Risk { background-color: #ffeaa7; color: #d63031; }
        .High-Risk { background-color: #ff7675; color: #white; }
        .Safe { background-color: #55efc4; color: #00b894; }
    </style>
</head>
<body>
    <div class="card">
        <h2>Risk Prediction Form</h2>
        <form action="/predict" method="post">
            <div class="grid">
                <!-- Numeric Inputs -->
                <div class="form-group">
                    <label>Attendance (%)</label>
                    <input type="number" step="any" name="attendance" required>
                </div>
                <div class="form-group">
                    <label>Study Hours / Week</label>
                    <input type="number" step="any" name="study_hours" required>
                </div>
                <div class="form-group">
                    <label>Past Failures</label>
                    <input type="number" name="past_failures" required>
                </div>
                <div class="form-group">
                    <label>Assignments Completed (%)</label>
                    <input type="number" step="any" name="assignments_completed_pct" required>
                </div>
                <div class="form-group">
                    <label>Previous Grade</label>
                    <input type="number" step="any" name="previous_grade" required>
                </div>
                <div class="form-group">
                    <label>Final Score</label>
                    <input type="number" step="any" name="final_score" required>
                </div>

                <!-- Categorical Inputs (Dropdowns) -->
                <div class="form-group">
                    <label>Parental Education</label>
                    <select name="parental_education" required>
                        <option value="0">High School</option>
                        <option value="1">Associate Degree</option>
                        <option value="2">Bachelor's Degree</option>
                        <option value="3">Master's / Ph.D.</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Family Income Level</label>
                    <select name="family_income" required>
                        <option value="0">Low</option>
                        <option value="1">Medium</option>
                        <option value="2">High</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Extracurricular Activities</label>
                    <select name="extracurricular" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Internet Access</label>
                    <select name="internet_access" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
            </div>

            <button type="submit">Predict Risk Status</button>
        </form>

        {% if prediction %}
            <div class="result {{ prediction }}">
                Result: {{ prediction }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features matching the original model's feature order
    features = [
        float(request.form['attendance']),
        float(request.form['study_hours']),
        float(request.form['past_failures']),
        float(request.form['assignments_completed_pct']),
        float(request.form['parental_education']),
        float(request.form['family_income']),
        float(request.form['extracurricular']),
        float(request.form['internet_access']),
        float(request.form['previous_grade']),
        float(request.form['final_score'])
    ]

    # Convert to array for prediction
    input_data = np.array([features])
    prediction_class = model.predict(input_data)[0]

    return render_template_string(HTML_TEMPLATE, prediction=prediction_class)

# Required for local testing
if __name__ == '__main__':
    app.run(debug=True)
