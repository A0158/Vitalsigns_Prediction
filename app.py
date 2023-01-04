from flask import Flask,request
import numpy as np
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger

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
def predict_price():
        
    datas = request.args.get("datas")
    prediction = classifier.predict([[datas]*200])
    return "The prediction is "+str(prediction)

@app.route('/predicting_files',methods=["POST"])
def predict_pricedes():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
         
    """
    df = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df)
    return "The prediction is "+str(list(prediction))
    
    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, threaded=True)
    