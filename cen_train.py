# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:13:59 2020

@author: yzy86
"""


# matplot

from datetime import datetime
import pickle, keras
import functools
# sys
import sys, os
# keras
import numpy as np
import keras.models as models
from keras.models import Model
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers import Input
from keras.optimizers import adam
# mine
from dataset import DataSet


# Params Section
###################################################
# dataset file
path_dataset = './datasets/RML2016.10a_dict.pkl'



device = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = device


# 1. loadDataSet
dataset = DataSet(path_dataset)
# X(220000 * (2, 128)) and lbl(220000 * 1) is whole dataset
# snrs(20) = -20 -> 18, mods(11) = ['8PSK', 'AM-DSB', ...]
X, lbl, snrs, mods = dataset.getX()
# X_train(176000) Y_train(176000 * 11) classes(11)=mods
X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()
# print(X_train.shape)
X_train = X_train.transpose(0,2,3,1)
X_test = X_test.transpose(0,2,3,1)
in_shp = list(X_train.shape[1:]) # (2, 128)
# print(in_shp)


# 2. build VT-CNN2 Neural Net model
dr = 0.5 # dropout rate
model = models.Sequential()
model.add(Reshape([2,128,4], input_shape=in_shp, name='reshape1'))
model.add(ZeroPadding2D((0, 2), name='padding1'))
model.add(Conv2D(256, (1, 3), strides=1, padding='valid', activation='relu', name='conv1', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr, name='drop1'))
model.add(ZeroPadding2D((0, 2), name='padding2'))
model.add(Conv2D(80, (2, 3), padding='valid', activation='relu', name='conv2', kernel_initializer='glorot_uniform'))
model.add(Dropout(dr, name='drop2'))
model.add(Flatten(name='flatten1'))
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1'))
model.add(Dropout(dr, name='drop3'))
model.add(Dense(len(classes), kernel_initializer='he_normal', name='dense2'))
model.add(Activation('softmax', name='softmax1'))
model.add(Reshape([len(classes)], name='reshape2'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# Set up some params
# nb_epochs = 100
# batch_size = 1024
# verbose = 1



# # 3. train or evaluate
# def format_time(second):
#     tmp = second
#     hour = second // 3600
#     second = second - hour * 3600
#     minute = second // 60
#     second = second - minute * 60
#     return '%s:%s:%s, total seconds: %s'%(hour, minute, second, tmp)

# filepath = 'Centralized_CNN2_8channel.wts.h5'

# # 3.1 train
# train_start = datetime.now()
# history = model.fit(
#             X_train,
#             Y_train,
#             batch_size=batch_size,
#             epochs=nb_epochs,
#             verbose=verbose,
#             validation_data=(X_test, Y_test),
#             callbacks= [
#                 keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
#                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
#             ])
# train_end = datetime.now()
# print('total train time: %s' % (format_time((train_end - train_start).seconds)))

# # simple version of performance
# # model.load_weights(filepath)
# # score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
# # print("score is: ", score)

# # if train is True, program finished
# sys.exit(0)

