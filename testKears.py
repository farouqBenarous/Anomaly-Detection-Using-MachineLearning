from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf

# fix random seed for reproducibility
numpy.random.seed(7)

df = pd.read_csv('Data/demo2/AllTrafficFinal.csv')
print("Data is ready  , the training will start after a while")

inputX = df.loc[:,
         ['http', 'Agent', 'Pragma', 'Cache', 'Accept', 'Encoding', ' Charset', 'Language', 'Host', 'Cookie',
          'Connection']].values

inputY = df.loc[:, ["Target"]].values


X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2)

model = Sequential()
model.add(Dense(11, input_dim=11, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Fit the model
model.fit(inputX, inputY, epochs=10000, batch_size=10)

# evaluate the model
scores = model.evaluate(inputX, inputY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
