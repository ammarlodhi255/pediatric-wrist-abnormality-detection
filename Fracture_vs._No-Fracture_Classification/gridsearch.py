import numpy as np # For array operations
import matplotlib.pyplot as plt
import cv2 as cv # Mostly for showing and normalization
import os #To iterate through directories and join paths
from sklearn import preprocessing
from random import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from random import randint
import h5py
from keras.layers import BatchNormalization
from keras.utils import img_to_array
from keras.models import Sequential, model_from_json
from keras.callbacks import TensorBoard
import time
import pickle

import random
seed = 42
random.seed(seed)
saved_datadir = r"D:\Ammar's\FYP_DATA\Saved Data (Classification)\fr-aug-binary-classification"

# Load data

with open(os.path.join(saved_datadir, "X_train.pkl"), "rb") as f:
  X_train = pickle.load(f)

with open(os.path.join(saved_datadir, "y_train.pkl"), "rb") as f:
  y_train = pickle.load(f)

with open(os.path.join(saved_datadir, "X_test.pkl"), "rb") as f:
  X_test = pickle.load(f)

with open(os.path.join(saved_datadir, "y_test.pkl"), "rb") as f:
  y_test = pickle.load(f)


from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, RMSprop, SGD

counter = 1
def define_model(neurons, dense_layers, bn, dropouts, opt):
  global counter
  model = Sequential()

  for i, nodes in enumerate(neurons):
    if i == 0:
      model.add(Conv2D(nodes, (3, 3), input_shape=X_train.shape[1:], activation='relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      if bn:
        model.add(BatchNormalization())
    else:
      model.add(Conv2D(nodes, (3, 3), activation='relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())

  if len(dense_layers) == len(dropouts) or len(dense_layers) < len(dropouts):
    for i, nodes in enumerate(dense_layers):
      model.add(Dense(nodes, activation='relu'))
      model.add(Dropout(dropouts[i]))
  elif len(dense_layers) > len(dropouts):
    for i, dropout in enumerate(dropouts):
      model.add(Dense(dense_layers[i], activation='relu'))
      model.add(Dropout(dropout))

  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss="binary_crossentropy", optimizer=opt(), metrics=["accuracy"])
  print(f'\nConv Neurons: {neurons}, Dense Neurons: {dense_layers}, BN: {bn}, Dropouts: {dropouts}, Optimizer: {opt}')
  print(f'Experiment No: {counter}\n\n')
  counter += 1
  return model


tf_seed = 42
tf.random.set_seed(tf_seed)

model = KerasClassifier(build_fn=define_model, verbose=1) 

neurons = [[32, 64], [32, 64, 128], [32, 64, 128, 128], [32, 64, 128, 256], [64, 128, 256, 256]]
# neurons = [[32, 64, 128, 128], [32, 64, 128, 256], [64, 128, 256, 256]]
opt = [SGD, RMSprop, Adam]
dense_layers = [[], [512], [256], [256, 256], [512, 256], [512, 512]]
dropouts = [[0.2], [0.5], [0.2, 0.2], [0.2, 0.5], [0.5, 0.5]]
bn = [True, False]

param_grid = dict(neurons=neurons, dense_layers=dense_layers, bn=bn, dropouts=dropouts, opt=opt, batch_size=[132], epochs=[20])
grid = GridSearchCV(estimator=model, param_grid=param_grid, return_train_score=True, n_jobs=1, cv=3, error_score='raise')

grid_result = grid.fit(X_train, y_train)


# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean = %f (std=%f) with: %r" % (mean, stdev, param))