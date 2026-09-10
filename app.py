import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the logistic regression model
with open("logistic.pkl", "rb") as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Risk Assessment Predictor</title>
    <style>
        * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        body { background-color: #f4f6f9; margin: 0; padding: 40px 20px; display: flex; justify-content: center; }
        .container { background: #ffffff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08); max-width: 650px; width: 100%; }
        h2 { text-align: center; color: #2c3e50; margin-bottom: 25px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .form-group { display: flex; flex-direction: column; }
        label { font-size: 14px; font-weight: 600; color: #34495e; margin-bottom: 6px; }
        input, select { padding: 10px; border: 1px solid #cccccc; border-radius: 6px; font-size: 14px; }
        input:focus, select:focus { border-color: #3498db; outline: none; }
        .full-width { grid-column: span 2; }
        button { background-color: #3498db; color: white; padding: 12px; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 15px; transition: background 0.2s; }
        button:hover { background-color: #2980b9; }
        .result-box { margin-top: 25px; padding: 15px; border-radius: 6px; text-align: center; font-size: 18px; font-weight: bold; }
        .At-Risk { background-color: #fce4e4; color: #c0392b; border: 1px solid #f5c6cb; }
        .High-Risk { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        .Safe { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Student Performance Predictor</h2>
        <form method="POST" action="/">
            <div class="grid">
                <div class="form-group">
                    <label>Attendance (%)</label>
                    <input type="number" step="any" name="attendance" required placeholder="e.g. 85">
                </div>
                <div class="form-group">
                    <label>Study Hours (Weekly)</label>
                    <input type="number" step="any" name="study_hours" required placeholder="e.g. 15">
                </div>
                <div class="form-group">
                    <label>Past Failures</label>
                    <input type="number" name="past_failures" required placeholder="e.g. 0">
                </div>
                <div class="form-group">
                    <label>Assignments Completed (%)</label>
                    <input type="number" step="any" name="assignments_completed_pct" required placeholder="e.g. 90">
                </div>
                <div class="form-group">
                    <label>Parental Education</label>
                    <select name="parental_education" required>
                        <option value="0">High School</option>
                        <option value="1">Associate Degree</option>
                        <option value="2">Bachelor's Degree</option>
                        <option value="3">Master's / Higher</option>
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
                <div class="form-group">
                    <label>Previous Grade</label>
                    <input type="number" step="any" name="previous_grade" required placeholder="e.g. 75">
                </div>
                <div class="form-group">
                    <label>Final Score</label>
                    <input type="number" step="any" name="final_score" required placeholder="e.g. 80">
                </div>
            </div>
            <button type="submit" class="full-width">Predict Status</button>
        </form>

        {% if prediction %}
        <div class="result-box {{ prediction }}">
            Predicted Status: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Extract features in the exact order expected by feature_names_in_
        features = [
            float(request.form["attendance"]),
            float(request.form["study_hours"]),
            float(request.form["past_failures"]),
            float(request.form["assignments_completed_pct"]),
            float(request.form["parental_education"]),
            float(request.form["family_income"]),
            float(request.form["extracurricular"]),
            float(request.form["internet_access"]),
            float(request.form["previous_grade"]),
            float(request.form["final_score"])
        ]
        
        final_features = np.array([features])
        prediction_result = model.predict(final_features)[0]
        prediction = str(prediction_result)

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
