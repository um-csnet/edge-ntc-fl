#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Client Program for ISCX 2016

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import flwr as fl
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import time as timex

#load dataset
x_train = np.load("x_bal-train-client 3.npy")
y_train = np.load("y_bal-train-client 3.npy")

x_test = np.load("x_bal-test-client 3.npy")
y_test = np.load("y_bal-test-client 3.npy")

#load dataset
#x_train = np.load("x_bal-twotrain-client 3.npy")
#y_train = np.load("y_bal-twotrain-client 3.npy")

#x_test = np.load("x_bal-twotest-client 3.npy")
#y_test = np.load("y_bal-twotest-client 3.npy")

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

display_name = "fl_client3_mlp_3client_36epochs_1round_gpu"

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

#MLP Model
model = Sequential()
model.add(InputLayer(input_shape = (740,))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class ntcClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=36, batch_size=64, shuffle = True, callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])
        return model.get_weights(), len(x_train), {'train_loss':history.history['loss'][0]}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        model.save("model_bal_client3_mlp_3client_36epochs_1round_gpu.h5")
        return loss, len(x_test), {"accuracy": float(accuracy)}

#Begin counting Time
startTime = timex.time()
 
fl.client.start_numpy_client(server_address="192.168.0.74:8080", client=ntcClient())

#End couting time
executionTime = (timex.time() - startTime)
executionTime = executionTime / 60
print('Execution time in minutes: ' + str(executionTime))

wandb.finish()