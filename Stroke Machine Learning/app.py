import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

app= Flask(__name__)

stroke_data_df = pd.read_csv('data\stroke_data.csv')

model=pickle.load(open('model\stroke_model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        age = float(request.form["age"])
        hypertension = (request.form["hyperTension"])
        heartdisease = (request.form["heartdisease"])
        glucose = float(request.form["glucose"])
        bmi = float(request.form["bmi"])
        data = [[age, hypertension, heartdisease, glucose,bmi]]
        
        df=pd.DataFrame(data, columns = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"])

        features = stroke_data_df[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]]
        original_df = features.copy()
        reference = original_df.append(df).reset_index()

        new_df = reference[["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]]
        scaler = MinMaxScaler()
        scaled_df = scaler.fit_transform(new_df)

        print("Scaled Weight", scaled_df)
        output = model.predict([list(scaled_df[len(scaled_df)-1])])

        if(output[0] == 1):
            classify = "Risk of stroke"
        elif(output[0] == 0):
            classify = "No risk of stroke"

        return render_template("results.html", classify=classify)


@app.route('/test')
def test():
  return('test')


if __name__ == '__main__':
    app.run(debug=True)