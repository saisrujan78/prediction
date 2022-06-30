
from flask import Flask, request, render_template, jsonify

from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("predict_lr.pkl",'rb'))

@app.route("/")
@cross_origin()
def home():
    return render_template("cpp.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():

    if request.method == "POST":

        gre = request.form["gre"]

        toefl = request.form["toefl"]

        cgpa = float(request.form["cgpa"])

        sop = float(request.form["sop"])

        lor = float(request.form["lor"])

        ur = request.form["ur"]




        prediction = model.predict(np.array([[gre,toefl,ur,sop,lor,cgpa,1]]))

        output = round(prediction[0]*100,2)

        return render_template("cpp.html", prediction_text= "chance of admit is {}%".format(output))
    
    return render_template("cpp.html")


if __name__ == "__main__":
    app.run(debug=True)

