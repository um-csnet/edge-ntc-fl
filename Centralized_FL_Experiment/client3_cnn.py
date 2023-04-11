#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Client Program for CNN ISCX 2016

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
import flwr as fl
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import time as timex

display_name = "fl_client3_cnn_3client_36epochs_1round_gpu"

wandb.init(
    # set the wandb project where this run will be logged
    project="ntcfl",
    name=display_name,

    # track hyperparameters and run metadata with wandb.config
    config={
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 36,
        "batch_size": 64,
        "learning_rate": 0.0001
    }
)

timestep = 36
num_class = 10
features = 20

#load dataset
x_train = np.load("x_train-client 3.npy")
y_train = np.load("y_train-client 3.npy")

x_test = np.load("x_test-client 3.npy")
y_test = np.load("y_test-client 3.npy")

#load dataset
#x_train = np.load("x_bal-twotrain-client 1.npy")
#y_train = np.load("y_bal-twotrain-client 1.npy")

#x_test = np.load("x_bal-twotest-client 1.npy")
#y_test = np.load("y_bal-twotest-client 1.npy")

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

x_train = x_train[:,:720]
x_test = x_test[:,:720]

x_train = x_train.reshape((x_train.shape[0], timestep , features))
x_test = x_test.reshape((x_test.shape[0], timestep, features))

def base_model():

    model_input = Input(shape=(timestep,features))

    x = Conv1D(32, 3, activation='relu')(model_input)
    x = Conv1D(32, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(64, 3, activation='relu')(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(128, 3, activation='relu')(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Flatten()(x)

    x3 = Dense(256)(x)
    x3 = Activation('relu')(x3)
    x3 = Dense(256)(x3)
    x3 = Activation('relu')(x3)
    output3 = Dense(num_class, activation='softmax', name='Class')(x3)

    model = Model(inputs=model_input, outputs=[output3])
    opt = Adam(clipnorm = 1., learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

model = base_model()

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
        model.save("model_bal_client3_cnn_3client_36epochs_1round_gpu.h5")
        return loss, len(x_test), {"accuracy": float(accuracy)}
        
#Begin counting Time
startTime = timex.time()
    
fl.client.start_numpy_client(server_address="192.168.0.74:8080", client=ntcClient())

#End couting time
executionTime = (timex.time() - startTime)
executionTime = executionTime / 60
print('Execution time in minutes: ' + str(executionTime))

wandb.finish()