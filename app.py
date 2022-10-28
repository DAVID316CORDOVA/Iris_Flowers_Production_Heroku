import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for
from keras.models import load_model
import joblib

diccionario=joblib.load(open("indice_diccionario","rb"))
app = Flask(__name__)
model = load_model("final_iris_model.h5")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    result=request.form
    sepal_le=result["sepal_length"]
    sepal_wi=result["sepal_width"]
    petal_le=result["petal_length"]
    petal_wi=result["petal_width"]
    
    vector_zeros=np.zeros(len(diccionario))

    vector_zeros[diccionario["sepal_length"]]=sepal_le
    
    vector_zeros[diccionario["sepal_width"]]=sepal_wi

    vector_zeros[diccionario["petal_length"]]=petal_le
    
    vector_zeros[diccionario["petal_width"]]=petal_wi
   
    escalador_x=joblib.load(open("iris_scaler.pkl","rb"))

    data=pd.DataFrame(vector_zeros).T
    
    data_transformada=escalador_x.transform(data)
    
    prediction = model.predict(data_transformada,verbose=0)

    prediccion_real=np.argmax(prediction,axis=1)[0]

    return render_template('result.html', data=prediccion_real)

if __name__ == '__main__':
    app.run(debug=True)
