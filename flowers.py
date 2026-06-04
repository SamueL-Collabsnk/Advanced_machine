from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns 

import pandas as pd
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
#creating log directory
log_dir = "logs"

tensorboard = keras.callbacks.TensorBoard(
    log_dir = log_dir,
    histogram_freq = 1
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor = "val_loss",
    factor = 0.1,
    patience = 3,
    min_lr = 1e-7
)

early_stop = keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 8,
    restore_best_weights = True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    "best_iris.keras",
    monitor = "val_accuracy",
    save_best_only = True
)


#train the model
history = model.fit(
    X_train,
    y_train,
    epochs = 30,
    batch_size = 16,
    validation_split = 0.2,
    callbacks = [tensorboard, reduce_lr, early_stop,checkpoint]
)

#Evaluation of the model 
loss, accuracy = model.evaluate(
    X_test,
    y_test
)
print("Accuracy:", accuracy)

history_df = pd.DataFrame(history.history)
print(history_df.head())
#Make predictions
predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis = 1)

y_actual = np.argmax(y_test,axis = 1)

#Evaluation of the metrics
cm = confusion_matrix(y_actual, y_pred)
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(cm, annot = True, fmt= "d")
plt.xlabel("y_actual")
plt.ylabel("y_pred")
plt.show()
print("Classifcation Reports for the Data:")
print (classification_report(y_actual, y_pred))

wrong = np.where(
    y_pred != y_actual
)[0]
print(wrong)
print("Misclassified indices:", wrong)

# Show details of each wrong prediction
for idx in wrong:
    print(f"\nSample {idx}:")
    print(f"  Predicted: {idx} (Class {y_pred[idx]})")
    print(f"  Actual: {idx} (Class {y_actual[idx]})")
    print(f"  Features: {X_test[idx]}")
 

print("\n" ,history.history.keys())  


model.save('iris_model.keras')









