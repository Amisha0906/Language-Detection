from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd
import re
import numpy as np

app = Flask(__name__)
@app.route('/')

def home():
    return render_template("home.html")

@app.route("/predict", methods = ["POST"])

def predict():

    # url_dataset = 'https://drive.google.com/file/d/1V3WrHKzRewBwvo9AOL7Qz6V3PnwuHtD9/view?usp=drive_link'
    # file_id = url_dataset.split('/')[-2]
    
    # dwn_url = 'https://drive.google.com/uc?id=' + file_id
    # data = pd.read_csv(dwn_url)
    # # data = pd.read_csv("language_detection.csv")
    # # y = data["Language"]

    # X = df['sentence']
    # languyage = df['language']

    #loading the model
    model = pickle.load(open("language_detector_model.pkl", "rb"))

    if request.method == "POST":
        # taking the input
        text = request.form["text"]
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', '', text)
        dat = [text]

        # prediction
        my_pred = model.predict(dat)

    return render_template("home.html", pred=" The above text is in {}".format(my_pred[0]))


if __name__ =="__main__":
    app.run(debug=True)