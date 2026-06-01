import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist

#load the dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print(X_train.shape)
print(X_test.shape)

#Visualize the data
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap=plt.cm.binary)
plt.show()

#Normalize Data
X_train = X_train/255.0
X_test = X_test/255.0

#build the model
model = keras.Sequential([
    layers.Conv2D(
        32, 
        (3,3),
        activation = 'relu',
        input_shape = (28,28,1)
    ),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation = 'softmax')
])

model.compile(
    optimizer = 'adam', 
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

#train the model
model.fit(
    X_train,
    y_train,
    epochs = 10,
    batch_size = 32,
    validation_split = 0.2
)
#Evaluate the model
loss, accuracy = model.evaluate(
    X_test,
    y_test
)
print("Accuracy:", accuracy)
print("Loss:", loss)

#Make predictions
predictions = model.predict(X_test)
print(predictions[5])

#Save the model 
model.save('Fashion_cnn_model.keras')

