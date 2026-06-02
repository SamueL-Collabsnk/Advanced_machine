import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load the Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

# Class Names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
#  Visualize the Data
plt.figure(figsize = (10,10))

for i in range(9):
    plt.subplot(3,3,i+1),
    plt.imshow(X_train[i]),
    plt.title(class_names[y_train[i][0]]),
    plt.axis("off")
plt.show()    

# Normalize the Data
X_train = X_train/255.0
X_test = X_test/255.0

#Build model
model = keras.Sequential([
    layers.Conv2D(
        32,
        (3,3), 
        activation = "relu",
        input_shape = (32,32,3)
        ),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(
        64,
        (3,3),
        activation = "relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation = "softmax")
])

#Model Compile
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

#Train the Model
model.fit(
    X_train,
    y_train,
    epochs= 10,
    batch_size = 64,
    validation_split = 0.2
)

#Evaluate the model
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Accuracy:", accuracy)

#Make Predictions
predictions = model.predict(X_test)

#first prediction
predicted_class = np.argmax(predictions[0])
print("Predictions:", class_names[predicted_class])

#Actual Label
print("Actual:", class_names[y_test[0][0]])

#Visualize the Prediction
plt.imshow(X_test[0])
plt.title(f"Predicted: {class_names[predicted_class]}")
plt.axis("off")
plt.show()