#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Client Program XAI for ISCX 2016

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import flwr as fl
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import time as timex

#load dataset
x_train = np.load("x_train-client 2.npy")
y_train = np.load("y_train-client 2.npy")

x_test = np.load("x_test-client 2.npy")
y_test = np.load("y_test-client 2.npy")

#load dataset
#x_train = np.load("x_twotrain-client 2.npy")
#y_train = np.load("y_twotrain-client 2.npy")

#x_test = np.load("x_twotest-client 2.npy")
#y_test = np.load("y_twotest-client 2.npy")

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

#x_train = x_train[:,[2,3,6,8,9,10,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,54,55,64,66,71,79,92]]
#x_test = x_test[:,[2,3,6,8,9,10,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,52,53,54,55,64,66,71,79,92]]

#fl 100 bytes 2 client
#x_train = x_train[:,[2,3,4,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,58,59,60,61,62,63,64,65,66,68,69,70,71,72,74,75,78,79,81,87,88,90,92,98,99,100,101,102,104,106,107,110,111,114,117,119,120,121,129,136,137,148,153,183,185,208,240,243]]
#x_test = x_test[:,[2,3,4,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,58,59,60,61,62,63,64,65,66,68,69,70,71,72,74,75,78,79,81,87,88,90,92,98,99,100,101,102,104,106,107,110,111,114,117,119,120,121,129,136,137,148,153,183,185,208,240,243]]

#100 bytes 3 client
#x_train = x_train[:,[2,3,4,6,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,64,66,67,68,70,71,72,73,74,76,78,79,80,81,82,83,84,87,88,89,90,92,93,97,98,99,100,102,104,106,109,110,115,116,119,120,121,123,124,127,128,129,137,175,245]]
#x_test = x_test[:,[2,3,4,6,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,64,66,67,68,70,71,72,73,74,76,78,79,80,81,82,83,84,87,88,89,90,92,93,97,98,99,100,102,104,106,109,110,115,116,119,120,121,123,124,127,128,129,137,175,245]]

#100 bytes 3 client
x_train = x_train[:,[2,3,4,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,62,64,67,70,71,75,77,78,79,81,82,83,88,89,90,91,92,96,97,98,99,100,104,106,109,110,113,114,115,118,119,120,123,124,127,129,137,183,185,238,240,243,244,249]]
x_test = x_test[:,[2,3,4,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,57,58,59,60,61,62,64,67,70,71,75,77,78,79,81,82,83,88,89,90,91,92,96,97,98,99,100,104,106,109,110,113,114,115,118,119,120,123,124,127,129,137,183,185,238,240,243,244,249]]


print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

display_name = "fl_xai_client2_mlp_3client_1epochs_36round_100b_gpu"

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
        "epoch": 1,
        "batch_size": 64
    }
)

#MLP Model
model = Sequential()
model.add(InputLayer(input_shape = (100,))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class ntcClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=1, batch_size=64, shuffle = True, callbacks=[WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])
        return model.get_weights(), len(x_train), {'train_loss':history.history['loss'][0]}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        model.save("model_xai_client2_mlp_3client_1epochs_36round_100b_gpu.h5")
        return loss, len(x_test), {"accuracy": float(accuracy)}

#Begin counting Time
startTime = timex.time()

#fl.client.start_numpy_client(server_address="192.168.0.74:8080", client=ntcClient())
fl.client.start_numpy_client(server_address="192.168.0.128:8080", client=ntcClient())

#End couting time
executionTime = (timex.time() - startTime)
executionTime = executionTime / 60
print('Execution time in minutes: ' + str(executionTime))

wandb.finish()