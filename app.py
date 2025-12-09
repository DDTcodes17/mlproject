from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predicting_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

#Route for Home Page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethenicity = request.form.get('ethinicity'),
            parental_level_of_education=request.form.get('parental level of education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test preparation course'),
            reading_score = request.form.get('reading score'),
            writing_score = request.form.get('writing score'),    
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        pred_pipeline = PredictPipeline()
        results = pred_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug = True)


