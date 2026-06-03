import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import ResNet50


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

X_train = X_train.astype("float32")/ 255.0
X_test = X_test.astype("float32")/ 255.0

print(X_train.shape)

#Feature Extraction
base_model = ResNet50(
    weights = "imagenet",
    include_top = False,
    input_shape = (224,224,3)
)

#freeze the Resnet trained model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

#Build model
model = keras.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Resizing(224,224),
    base_model,
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation = "relu"),
    layers.Dropout(0.5),
    layers.Dense(100, activation = "softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate = 1e-5),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)
#EarlyStopping
early_stop = keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 5,
    restore_best_weights = True
)

#Checkpoints
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_resnet.keras",
    monitor = "val_accuracy",
    save_best_only = True
)

#Training 
model.fit(
    X_train,
    y_train,
    epochs = 20,
    batch_size = 64,
    validation_split = 0.2,
    callbacks = [early_stop, checkpoint]
)

loss, accuracy = model.evaluate(
    X_test,
    y_test
)
print("Accuracy:", accuracy)

predictions = model.predict(X_test)

#to predict class the 30th label
predicted_class = np.argmax(predictions[30])
#The actual class in 30th label
actual_class = y_test[30][0]

print(f"Predicted Label: {class_names[predicted_class]}")
print(f"Actual Label: {class_names[actual_class]}")

#Save the model
model.save("transferlearning_model.keras")

