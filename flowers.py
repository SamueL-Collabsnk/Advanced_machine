from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#load the dataset
iris = load_iris()

print(iris.data.shape)

#Seperate the features and targets
X = iris.data
y = iris.target

#Split the data into training and testing sets
X_train, X_test, y_train, y_test =  train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

#Normalize the data 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)

#One hot Encoding
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#build the Neural Network Model
model = keras.Sequential([
    layers.Dense(32, activation = 'relu', input_shape = (4,)),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(8, activation = 'relu'),
    layers.Dense(3, activation = 'softmax')
])

#Model Complilation
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
#train the model
history = model.fit(
    X_train,
    y_train,
    epochs = 10,
    batch_size = 16,
    validation_split = 0.2,
)

#Evaluation of the model 
loss, accuracy = model.evaluate(
    X_test,
    y_test
)
print("Accuracy:", accuracy)
#Make predictions
predictions = model.predict(X_test)
print(predictions[0])

model.save('iris_model.keras')









