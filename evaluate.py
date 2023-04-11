#Load Trained Model
from tensorflow import keras
model = keras.models.load_model("global_model_3client_cnn_gpu.h5")

#Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

x_test = np.load("x_test-MLP-Multiclass-ISCX-740features.npy")
y_test = np.load("y_test-MLP-Multiclass-ISCX-740features.npy")

#For CNN & GRU
timestep = 36
num_class = 10
features = 20
x_test = x_test[:,:720]
x_test = x_test.reshape((x_test.shape[0], timestep, features))

y_pred_class = np.argmax(model.predict(x_test),axis=1)
y_test_class = np.argmax(y_test, axis=1)
print(confusion_matrix(y_test_class, y_pred_class))
print(classification_report(y_test_class, y_pred_class, digits=4))