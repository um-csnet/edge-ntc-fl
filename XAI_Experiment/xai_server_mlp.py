#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Server Program for XAI ISCX 2016

import flwr as fl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import time as timex

display_name = "federated_xai_mlp_server_3client_1epochs_36round_100b_gpu"

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

MAX_ROUNDS = 36

model = Sequential()
model.add(InputLayer(input_shape = (100,))) # input layer
model.add(Dense(6, activation='relu')) # hidden layer 1
model.add(Dense(6, activation='relu')) # hidden layer 2
model.add(Dense(10, activation='softmax')) # output layer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class SaveKerasModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        agg_weights = super().aggregate_fit(server_round, results, failures)

        if (server_round == MAX_ROUNDS):
            
            model.set_weights(fl.common.parameters_to_ndarrays(agg_weights[0]))
            model.save('global_model_xai_mlp_3client_1epochs_36round_100b_gpu.h5')

        return agg_weights

strategy = SaveKerasModelStrategy(min_available_clients=3, min_fit_clients=3, min_evaluate_clients=3)

#Begin counting Time
startTime = timex.time()

fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=MAX_ROUNDS))

#End couting time
executionTime = (timex.time() - startTime)
executionTime = executionTime / 60
print('Execution time in minutes: ' + str(executionTime))

wandb.finish()

print("Done")
