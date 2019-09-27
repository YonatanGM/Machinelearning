# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:41:41 2019

@author: Yonatan
"""

from keras.datasets import mnist
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

print("loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data() #default split

trainX = trainX.reshape((trainX.shape[0], 784))
trainX = trainX.astype('float') / 255  #scaling 
testX = testX.reshape((testX.shape[0], 784))
testX = testX.astype('float') / 255

print(trainX.shape, testX.shape)


trainY = to_categorical(trainY)
testY = to_categorical(testY)


print(trainY.shape, testY.shape)
model = Sequential()
model.add(Dense(512,  activation="relu", input_shape=(784,)))
model.add(Dense(10, activation="softmax"))   #for normalized class probabilities

print("training network...")
model.compile(loss="categorical_crossentropy", optimizer=SGD(0.01), metrics=["accuracy"])
model.fit(trainX, trainY, epochs=5, batch_size=128)

print("evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))
