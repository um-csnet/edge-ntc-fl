#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Server Program for ISCX 2016

import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
#import wandb
#from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import time as timex

#display_name = "federated_mlp_server_2client_36epochs_3round_gpu"

'''
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
'''

MAX_ROUNDS = 3

model = Sequential()
model.add(InputLayer(input_shape = (740,))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class SaveKerasModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        agg_weights = super().aggregate_fit(server_round, results, failures)

        if (server_round == MAX_ROUNDS):
            
            model.set_weights(fl.common.parameters_to_ndarrays(agg_weights[0]))
            model.save('global_model_mlp_2client_36epochs_3round_cpu.h5')

        return agg_weights

strategy = SaveKerasModelStrategy(min_available_clients=2, min_fit_clients=2, min_evaluate_clients=2)

#Begin counting Time
startTime = timex.time()

fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS))

#End couting time
executionTime = (timex.time() - startTime)
executionTime = executionTime / 60
print('Execution time in minutes: ' + str(executionTime))

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from tensorflow import keras

model = keras.models.load_model("global_model_mlp_2client_36epochs_3round_cpu.h5")

x_test = np.load("x_test-MLP-Multiclass-ISCX-740features.npy")
y_test = np.load("y_test-MLP-Multiclass-ISCX-740features.npy")

y_pred_class = np.argmax(model.predict(x_test),axis=1)
y_test_class = np.argmax(y_test, axis=1)
print(confusion_matrix(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class, digits=4))

#wandb.finish()

print("Done")
