import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import EfficientNetB0
import keras_tuner as kt

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


data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1,0.1),
    layers.RandomContrast(0.1),
])

basemodel = EfficientNetB0(
    weights = "imagenet",
    include_top = False,
    input_shape = (224,224,3)
)
basemodel.trainable = True

for layer in basemodel.layers[:-20]:
    layer.trainable = False

#Build model
def build_model(hp):
    lr = hp.Choice("learning_rate", [1e-4, 5e-4,1e-3])
    dense_units = hp.Int("dense_units", min_value = 128, max_value=256, step=64)
    wd = hp.Choice("weight_decay", [0.0,1e-4,1e-3])
    model = keras.Sequential([
        layers.Input(shape=(32,32,3), dtype = "float32"),
        layers.Resizing(224,224),
        layers.Rescaling(1.0/255.0),
        data_augmentation,
        basemodel,
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation = "relu",
                     kernel_regularizer = keras.regularizers.l2(wd)),
        layers.Dropout(0.5),
        layers.Dense(10, activation = "softmax")
    ])
    
    model.compile(
    optimizer = keras.optimizers.Adam(
    learning_rate= lr),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = ["accuracy"]
)
    return model
#Hyperparameter tuner
tuner = kt.RandomSearch(
    build_model,
    objective ="val_accuracy",
    max_trials= 10,
    overwrite = True,
    directory = "tuning",
    project_name = "cifar10cnn_project"
)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(10000).batch(64).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(AUTOTUNE)

#Search for the best parameters
tuner.search(
    train_ds,
    epochs = 5,
    validation_data = val_ds,
    batch_size = 32
)
best_hp = tuner.get_best_hyperparameters(1)[0]
print("Best Hyperparameters:")
print(best_hp)

#Build Best Model
best_model = tuner.hypermodel.build(
    best_hp
)
#Callbacks

tensorboard = keras.callbacks.TensorBoard(
    log_dir = "logs",
    histogram_freq = 1
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor ="val_loss",
    factor = 0.5,
    patience = 3,
    min_lr = 1e-6
)
early_stop = keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 8,
    restore_best_weights = True
)
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_cifar10.keras",
    monitor = "val_accuracy",
    save_best_only = True
)
#Train the Model
history = best_model.fit(
    train_ds,
    epochs= 30,
    batch_size = 64,
    validation_data = val_ds,
    callbacks = [tensorboard,
                 reduce_lr,
                 early_stop,
                 checkpoint,
                 ]
)

#Evaluate the model
loss, accuracy = best_model.evaluate(
    X_test,
    y_test
)

print("Accuracy:", accuracy)

import pandas as pd
history_df = pd.DataFrame(
    history.history
)
history_df.to_csv("training_history.csv", index=False)

#Make Predictions
predictions = best_model.predict(X_test)

#first prediction
predicted_classes = np.argmax(predictions, axis=1)
first_predicted = predicted_classes[0]
#Save the model
best_model.save("cifar_cnn_model.keras")