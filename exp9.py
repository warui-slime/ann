# 9. Detecting credit card fraud with neural network

import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv('creditcard.csv')
data.head()

data.drop("Time",axis=1,inplace=True)

data.isnull().any().describe()

## Training Model

limit = int(0.9*len(data))
train = data.loc[:limit]
val_test = data.loc[limit:]
val_test.reset_index(drop=True, inplace=True)
val_test_limit = int(0.5*len(val_test))
val = val_test.loc[:val_test_limit]
test = val_test.loc[val_test_limit:]

Balancing Data

train_positive = train[train["Class"] == 1]
train_positive = pd.concat([train_positive] * int(len(train) / len(train_positive)), ignore_index=True)
noise = np.random.uniform(0.9, 1.1, train_positive.shape)
train_positive = train_positive.multiply(noise)
train_positive["Class"] = 1
train_extended = pd.concat([train, train_positive], ignore_index=True)
train_shuffled = train_extended.sample(frac=1, random_state=0).reset_index(drop=True)

X_train = train_shuffled.drop(labels=["Class"], axis=1)
Y_train = train_shuffled["Class"]
X_val = val.drop(labels=["Class"], axis=1)
Y_val = val["Class"]
X_test = test.drop(labels=["Class"], axis=1)
Y_test = test["Class"]

# Feature Scaling
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train)
X_val[X_val.columns] = scaler.transform(X_val)
X_test[X_test.columns] = scaler.transform(X_test)

# Model Architecture
model = Sequential()
model.add(Dense(64, activation="relu", input_dim=X_train.shape[1]))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# Model Compilation
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# Model Training
history = model.fit(X_train,
                   Y_train,
                   epochs=10,
                   validation_data=(X_val, Y_val),
                   callbacks=[ReduceLROnPlateau(patience=3, verbose=1, min_lr=1e-6),
                             EarlyStopping(patience=5, verbose=1)])

num_epochs = len(history.history["loss"])

fig, axarr = plt.subplots(1, 2, figsize=(24, 8))

# Loss Plot
axarr[0].set_xlabel("Number of Epochs")
axarr[0].set_ylabel("Loss")
sns.lineplot(x=range(1, num_epochs+1),
             y=history.history["loss"],
             label="Train",
             ax=axarr[0])
sns.lineplot(x=range(1, num_epochs+1),
             y=history.history["val_loss"],
             label="Validation",
             ax=axarr[0])

# Accuracy Plot
axarr[1].set_xlabel("Number of Epochs")
axarr[1].set_ylabel("Accuracy")
axarr[1].set_ylim(0, 1)
sns.lineplot(x=range(1, num_epochs+1),
             y=history.history["accuracy"],
             label="Train",
             ax=axarr[1])
sns.lineplot(x=range(1, num_epochs+1),
             y=history.history["val_accuracy"],
             label="Validation",
             ax=axarr[1])

plt.show()

# Evaluate model on test set
test_results = model.evaluate(X_test, Y_test)
print("The model test accuracy is {}.".format(test_results[1]))

# Make predictions and calculate average precision

predictions = (model.predict(X_test) > 0.5).astype("int32")
ap_score = average_precision_score(Y_test, predictions)
print("The model test average precision score is {}.".format(ap_score))


# Create and plot confusion matrix
confusion = pd.DataFrame(confusion_matrix(Y_test, predictions))
confusion.columns = ["Predicted Negative", "Predicted Positive"]
confusion.index = ["Actual Negative", "Actual Positive"]
sns.heatmap(confusion, annot=True, fmt='d')
plt.yticks(rotation=0)
plt.show()
