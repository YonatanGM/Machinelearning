# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:41:41 2019

@author: Yonatan
"""

from keras.datasets import cifar10
#from sklearn.preprocessing import LabelBinarizer   #one hot encode
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential   #feed forward network, layers added sequentially 
from keras.layers.core import Dense   #fully connected layers
from keras.optimizers import SGD
from keras.utils import to_categorical
#import matplotlib.pyplot as plt
#import numpy as np
#import argparse


#ap = argparse.ArgumentParser()
#ap.add_argument("-o", "--output", required=True, 
#                help="path to output loss/accuracy plot")
#args = vars(ap.parse_args())


#Load the MNIST data set

print("loading CIFAR-10 dataset...")
(trainX, testX), (trainY, testY) = cifar10.load_data() #default split
input_shape = trainX.shape[1] * trainX.shape[2]
trainX.reshape((trainX.shape[0], input_shape))
trainX = trainX.astype('float') / 255  #scaling 
testX.reshape((testX.shape[0], input_shape))
testX = testX.astype('float') / 255

trainY = to_categorical(trainX)
testY = to_categorical(testY)
 
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
model = Sequential()
model.add(Dense(1024, input_shape=(input_shape,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))   #for normalized class probabilities

print("training network...")
model.compile(loss="categorical_crossentropy", optimizer=SGD(0.01), metrics=["accuracy"])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

print("evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))


