#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Model Training for CNN_v2 using ISCX2016 dataset with 740 features

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import Adam
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import time as timex

timestep = 36
num_class = 10
features = 20

#load dataset
x_train = np.load("x_train-bal-ISCX-740features.npy")
y_train = np.load("y_train-bal-ISCX-740features.npy")

x_test = np.load("x_test-bal-ISCX-740features.npy")
y_test = np.load("y_test-bal-ISCX-740features.npy")


print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

x_train = x_train[:,:720]
x_test = x_test[:,:720]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

x_train = x_train.reshape((x_train.shape[0], timestep , features))
x_test = x_test.reshape((x_test.shape[0], timestep, features))

print(x_train.shape)
print(x_test.shape)

display_name = "centralized_bal_CNN_gpu"

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
model.summary()

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
model.save("centralized_model_bal_cnn_gpu.h5")