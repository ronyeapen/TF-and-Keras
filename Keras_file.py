import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#for file name, using timestamps
import time
import pickle
import keras as k
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras import optimizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#%%
start_time = time.clock()
unique_code='onlyAUS'
train_data_set='Nx&y_train_'+unique_code+'.npz'
test_data_set='Nx&y_test_'+unique_code+'.npz'
train_files=np.load(train_data_set)
train_x=train_files['x']
train_y=train_files['y']
dim=train_x.shape[1]



def loss_1(train_y,pred):
        return k.backend.mean(k.backend.square((pred-train_y)))


model=Sequential()

def build_model():
  model = keras.Sequential([
    layers.Dense(600, activation=tf.nn.relu, input_shape=(dim,)),
    layers.Dense(300, activation=tf.nn.relu),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dense(25, activation=tf.nn.relu),
    layers.Dense(12, activation=tf.nn.relu),
    layers.Dense(1)
  ])


  optimizer=tf.train.AdamOptimizer(learning_rate=0.0001)

  model.compile(loss=loss_1,
                optimizer=optimizer,
                metrics=['mean_absolute_error'])
  return model

model=build_model()

history=model.fit(train_x,train_y,epochs=1000,batch_size=5000,verbose=1)
