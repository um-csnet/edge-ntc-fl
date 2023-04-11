#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: FL Server Program for CNN ISCX 2016

import flwr as fl
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Activation
from keras.optimizers import Adam
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import time as timex

display_name = "federated_xai_cnn_server_3client_1epochs_36round_gpu"

wandb.init(
    # set the wandb project where this run will be logged
    project="ntcfl",
    name=display_name,

    # track hyperparameters and run metadata with wandb.config
    config={
        "optimizer": "adam",
        "loss": "categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 1,
        "batch_size": 64,
        "learning_rate": 0.0001
    }
)

num_class = 10
timestep = 5
features = 20

MAX_ROUNDS = 36

def base_model():

    model_input = Input(shape=(timestep,features))

    x = Conv1D(32, 3, activation='relu', padding='same')(model_input)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(2), padding='same')(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(2), padding='same')(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=(2), padding='same')(x)

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

class SaveKerasModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        agg_weights = super().aggregate_fit(server_round, results, failures)

        if (server_round == MAX_ROUNDS):
            
            model.set_weights(fl.common.parameters_to_ndarrays(agg_weights[0]))
            model.save('global_model_xai_3client_1epochs_36round_100b_cnn.h5')

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
