from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

flower_model=load_model("final_iris_model.h5")

flower_scaler=joblib.load("iris_scaler.pkl")

flower_example={"sepal_length":[5.1],
               "sepal_width":[3.5],
               "petal_length":[1.4],
               "petal_width": [0.2] }

def return_prediction(model,scaler,flower_data):
    
    flower_data=pd.DataFrame(flower_data)
    
    classes=np.array(['Setosa', 'Versicolor', 'Virginica'])
    
    flower=scaler.transform(flower_data)
    
    class_ind=np.argmax(model.predict(pd.DataFrame(flower),verbose=0) ,axis=1)[0]
    
    return classes[class_ind]

print("Tipo de Flor")
print(return_prediction(flower_model,flower_scaler,flower_example))