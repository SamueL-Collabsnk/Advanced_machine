import numpy as  np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100

# Load the Dataset
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',
    'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew',
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
    'table', 'tank', 'telephone', 'television',
    'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman', 'worm'
]
print(class_names[70])
# Visualize the data
plt.figure(figsize=(6,6))

for i in range(9):
    plt.subplot(3,3,i+1),
    plt.imshow(X_train[i]),
    plt.title(class_names[y_train[i][0]]),
    plt.axis("off")
plt.show()

# Normalize Data 
X_train = X_train/255.0
X_test  = X_test/255.0

#build the model
model = keras.Sequential([
    layers.Conv2D(
        32,
        (3,3),
        activation = "relu",
        input_shape = (32,32,3)     
    ),
    layers.BatchNormalization(),
    layers.Conv2D(
        64,
        (3,3),
        activation = "relu"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    
    layers.Dropout(0.25),
    layers.Conv2D(128, (3,3), activation = "relu"),
    
    layers.Flatten(),
    layers.Dense(256, activation = "relu"),
    layers.Dropout(0.5),
    layers.Dense(100, activation = "softmax")
])
    
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_entropy",
    metrics = ["accuracy"]
)    
#train the model
model.fit(
    X_train,
    y_train,
    epochs = 10,
    batch_size = 64,
    validation_split = 0.2
)    
loss, accuracy = model.evaluate(
    X_test,
    y_test
)  
print("Accuracy:", accuracy)

#make Predictions
predictions = model.predict(X_test)

#Predict the class of the first image in the test set
predicted_class =  np.argmax(class_names[predictions[0]])
print("First class Prediction:", class_names[predicted_class])

#Actual class of the first image in the test set
actual_class = class_names[y_test[0][0]]
print("Actual Class:", class_names[actual_class])

#Visualize the predicted class

plt.imshow(X_test[0])
plt.title(f"Predicted Class:{class_names[predicted_class]}")
plt.axis("off")
plt.show()

#save the model
model.save("Cifar100_cnn_model.keras")