# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:49:49 2020

@author: yzy86
"""
import keras.models as models
from keras.models import Model
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers import Input

d_model1_1_1 = models.Sequential()
d_model1_1_1.add(Reshape([2,32,1], input_shape=[2,32,1], name='d_reshape1'))
d_model1_1_1.add(ZeroPadding2D((0,2), name='padding1_1'))
d_model1_1_1.add(Conv2D(256, (1, 3), strides=1, input_shape = [2,36,1], padding='valid', activation='relu', name='d_conv1_1', kernel_initializer='glorot_uniform'))
# d_model1_1_1.summary()
d_model1_1_1.load_weights('./model/d_model1_1_1.h5')

n_input1 = Input(shape=[2,32,1])
n_reshape1 = d_model1_1_1.get_layer('d_reshape1')
n_padding1 = d_model1_1_1.get_layer('padding1_1')
n_conv1 = d_model1_1_1.get_layer('d_conv1_1')
print(n_conv1.weights[0])