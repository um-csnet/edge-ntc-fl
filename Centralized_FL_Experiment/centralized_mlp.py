#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Model Training for MLP_v2 using ISCX2016 dataset with 740 features

import numpy as np
import pandas as pd
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import time as timex

#load dataset
x_train = np.load("x_train-bal-ISCX-740features.npy")
y_train = np.load("y_train-bal-ISCX-740features.npy")

x_test = np.load("x_test-bal-ISCX-740features.npy")
y_test = np.load("y_test-bal-ISCX-740features.npy")

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

display_name = "centralized_bal_mlp_gpu"

wandb.init(
    # set the wandb project where this run will be logged
    project="ntcfl",
    name=display_name,

    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 6,
        "activation_1": "relu",
        "layer_2": 6,
        "activation_2": "relu",
        "layer_3": 10,
        "activation_3": "softmax",
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 36,
        "batch_size": 64
    }
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
#MLP Model
model = Sequential()
model.add(InputLayer(input_shape = (740,))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer

model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Begin counting Time
startTime = timex.time()

# fit the keras model on the dataset
model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 64, epochs = 36, verbose = True, shuffle = True, callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])

#End couting time
executionTime = (timex.time() - startTime)
executionTime = executionTime / 60
print('Execution time in minutes: ' + str(executionTime))

wandb.finish()

#Store Model
model.save("centralized_model_bal_mlp_gpu.h5")