import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras import layers

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

X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0

#Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

model = keras.Sequential([
    layers.Input(shape=(32,32,3)),
    data_augmentation,
    
    layers.Conv2D(32, (3,3), activation = "relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation = "relu"),
    layers.BatchNormalization(),
    
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3,3), activation = "relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64,(3,3), activation = "relu"),
    layers.BatchNormalization(),
    
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(256, activation = "relu"), 
    layers.Dropout(0.5),
    layers.Dense(100, activation = "softmax")
    
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

#Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 5,
    restore_best_weights = True
)

#Checkpoints 
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_cifar100.keras",
    monitor = "val_accuracy",
    save_best_only = True
)

model.fit(
    X_train,
    y_train,
    epochs = 30,
    batch_size = 64,
    validation_split = 0.2,
    callbacks = [early_stopping, checkpoint]
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)
print("Accuracy:", accuracy)

predictions = model.predict(X_test)

predicted_class = np.argmax(predictions[50])
actual_class = y_test[50][0]

print(f"Predicted Label:, {class_names[predicted_class]}" )
print(f"Actual Class:, {class_names[actual_class]}")