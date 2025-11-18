from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# Flask Application Initialization
# -------------------------------------------------------------
application = Flask(__name__)
app = application


# -------------------------------------------------------------
# Route: Home Page
# -------------------------------------------------------------
@app.route('/')
def index():
    """
    Renders the landing/index page of the web application.

    Returns
    -------
    HTML template : index.html
    """
    return render_template('index.html')


# -------------------------------------------------------------
# Route: Prediction Page (GET: show form, POST: make prediction)
# -------------------------------------------------------------
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles the prediction workflow.

    GET  -> Renders the input form page.
    POST -> Collects form inputs, converts them into a DataFrame,
            sends them through the ML pipeline, and returns prediction.

    Returns
    -------
    HTML template : home.html containing prediction results.
    """
    if request.method == 'GET':
        # User is requesting the prediction form
        return render_template('home.html')

    else:
        # Extract form input values
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert user input into a clean DataFrame for prediction
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # Useful for debugging/logging

        # Create prediction pipeline instance
        predict_pipeline = PredictPipeline()

        # Get model prediction
        results = predict_pipeline.predict(pred_df)

        # Display prediction result in the same page
        return render_template('home.html', results=results[0])


# -------------------------------------------------------------
# Application Runner
# -------------------------------------------------------------
if __name__ == "__main__":
    """
    Starts the Flask development server.

    Access URLs:
    - Home page:          http://127.0.0.1:5000/
    - Prediction page:    http://127.0.0.1:5000/predictdata
    """
    app.run(host="0.0.0.0")
