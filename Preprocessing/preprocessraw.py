#Author: Muhammad Azizi Bin Mohd Ariffin
#Email: mazizi@fskm.uitm.edu.my
#Description: Raw Data Preprocessing for FL Client ISCX 2016

# libraries 
import pandas as pd
from pathlib import Path

filePath = "D:\\repo\\ntcfl-dataset\\DATA\\"

# combine dataset and store in Pandas Dataframe
count = 0
entriesAll = Path(filePath)
for entryAll in sorted(entriesAll.iterdir()):
  fileName = entryAll.name
  print(fileName)
  if count == 0 :
    df = pd.read_csv(filePath + fileName)
    count += 1
  else :
    df2 = pd.read_csv(filePath + fileName)
    df = pd.concat([df, df2])

# reset dataframe index after combining dataset
df = df.reset_index(drop=True)
print(df)

print(df['label'].value_counts())
label = df['label']
print(label)

df = df.iloc[:, 0:740]
print(df)

df['label'] = label
print(df)

print(df['label'].value_counts(sort=False))

#exit()

import xai.data
df.head()

df = xai.balance(df, "label", upsample=0.2)

print(df['label'].value_counts(sort=False))

print(df)

#split dataset into data and label
y = df['label']
print(y)
print(y.value_counts())
x = df.iloc[:, 0:740]
print(x)

del df
del df2
import gc
gc.collect()

#Normalized dataset by column
from sklearn import preprocessing
d = preprocessing.normalize(x, axis=0)
scaled_x = pd.DataFrame(d, columns=x.columns)
print(scaled_x)

#convert label to categorical label
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
print(y)
print(y.shape)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
label_y = np_utils.to_categorical(encoded_y)

#Split the train and test set
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test =train_test_split(scaled_x,label_y,test_size=0.3)

#convert dataset to numpy array
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

#save dataset
import numpy as np
np.save('x_train-bal-ISCX-740features', x_train)
np.save('y_train-bal-Multiclass-ISCX-740features', y_train)

np.save('x_test-bal-ISCX-740features', x_test)
np.save('y_test-bal-ISCX-740features', y_test)