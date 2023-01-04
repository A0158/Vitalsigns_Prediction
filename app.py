from flask import Flask,request
import numpy as np
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger
import opencv as cv2
from inference_preprocess import preprocess_raw_video, detrend
from predict_video import predict_vitals
app = Flask(__name__)
Swagger(app)
pickle_in = open('classifier.pkl','rb')
classifier= pickle.load(pickle_in)

## root page
@app.route('/')
def welcome():
    return "Welcome to my world"

##
@app.route('/predict')
def predict_vitals():
        
   return

@app.route('/predicting_files',methods=["POST"])
def predict_vitals():

    df = preprocess_raw_video(request.files.get("file"))
    prediction = predict_vitals(df)
    return "The prediction is "+str(list(prediction))
    
    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, threaded=True)
    
