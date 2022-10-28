import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("iris.csv")

X=df.drop("species",axis=1)
y=df["species"]

from sklearn.preprocessing import LabelBinarizer
encoder=LabelBinarizer()

y=encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)

scaler_X_train=scaler.transform(X_train)
scaler_X_test=scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping

model=Sequential()
model.add(Dense(400,activation="relu",input_shape=[4,]))
model.add(Dense(200,activation="relu"))
model.add(Dense(3,activation="softmax"))
model.compile(optimizer="adam",loss="categorical_crossentropy" ,metrics=["accuracy"]) 

early=EarlyStopping(monitor="val_accuracy" ,patience=7)

model.fit(scaler_X_train,y_train,epochs=300,validation_data=(scaler_X_test,y_test),callbacks=early)

metrics=pd.DataFrame(model.history.history)
plt.show()

metrics[["accuracy","val_accuracy"]].plot();
plt.show()