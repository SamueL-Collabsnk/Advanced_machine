import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from  tensorflow import  keras
from  tensorflow.keras import layers

#load the Dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(X_train.shape)
print(X_test.shape)
 
#visualize the data
plt.imshow(X_train[0], cmap = 'gray')
plt.show()
 
#Normalize the data
X_train = X_train/255.0
X_test = X_test/255.0

#Build and Flattening the data using keras flatten layer
model = keras.Sequential([
    layers.Flatten(input_shape = (28,28)),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
]) 

#Model Compilation 
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
#View The architecture of the model
model.summary()

#Train the model
history = model.fit(
    X_train,
    y_train,
    epochs = 10,
    batch_size = 32,
    validation_split = 0.2
)
#Evaluate the model
loss,accuracy = model.evaluate(
    X_test,
    y_test,
)
print("Accuracy:", accuracy)

#Make predictions
predictions = model.predict(X_test)
print(predictions[0])
#Highest probability class
print("Predicted class:", np.argmax(predictions[0]))

#Save the model
model.save('mnist_model.h5')
