import pandas as pd
import joblib

df=pd.read_csv("iris.csv")

X=df.drop("species",axis=1)
y=df["species"]

from sklearn.preprocessing import LabelBinarizer
encoder=LabelBinarizer()
y=encoder.fit_transform(y)

from sklearn.preprocessing import MinMaxScaler
full_scaler=MinMaxScaler()
X_scaled=full_scaler.fit_transform(X)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(400,activation="relu",input_shape=[4,]))
model.add(Dense(200,activation="relu"))
model.add(Dense(3,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy" ,metrics=["accuracy"])
model.fit(X_scaled,y,epochs=300)

model.save("final_iris_model.h5")
joblib.dump(full_scaler,"iris_scaler.pkl")


#Saving a dictionary with all the columns

diccionario=dict(zip(X.columns,range(X.shape[1])))

joblib.dump(diccionario,open("indice_diccionario","wb"))

print("Diccionario de columnas:")
print(diccionario)