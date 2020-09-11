# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:24:44 2020

@author: yzy86
"""


# matplot

from datetime import datetime
# import pickle, keras
import functools
# sys
import sys, os
# keras
import numpy as np
import keras.models as models
import tensorflow as tf
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
###################################################

# 1. loadDataSet
dataset = DataSet(path_dataset)
# X(220000 * (2, 128)) and lbl(220000 * 1) is whole dataset
# snrs(20) = -20 -> 18, mods(11) = ['8PSK', 'AM-DSB', ...]
X, lbl, snrs, mods = dataset.getX()
# X_train(176000) Y_train(176000 * 11) classes(11)=mods
X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()
# print("the shape is:")
# print(X_train[0].shape)
X_train = X_train.transpose(0,2,3,1)
X_test = X_test.transpose(0,2,3,1)
in_shp = list(X_train.shape[1:]) # (2, 128)
#print(in_shp)


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
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# model.summary()

# Set up some params
nb_epochs = 100
batch_size = 1024
verbose = 1

# filepath = './model/Centralized_CNN2_0.5.wts.h5'
filepath = './model/Centralized_CNN2_8channel.wts.h5'

# evaluate
# model.load_weights(filepath)
# print(model.metrics_names)
# score, acc = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
# print("score is: ", score)
# print("acc is: ", acc)

# Load model
model.load_weights(filepath)
n_input1 = Input(shape=in_shp)
n_reshape1 = model.get_layer('reshape1')
n_padding1 = model.get_layer('padding1')
n_conv1 = model.get_layer('conv1')
# print(n_conv1.input_shape)
# print(n_conv1.output_shape)
# print(len(n_conv1.get_weights()[1]))
d_conv_weights1_1 = tf.slice(n_conv1.get_weights()[0], [0,0,0,0],[1,3,1,256])
d_conv_weights1_2 = tf.slice(n_conv1.get_weights()[0], [0,0,1,0],[1,3,1,256])
d_conv_weights1_3 = tf.slice(n_conv1.get_weights()[0], [0,0,2,0],[1,3,1,256])
d_conv_weights1_4 = tf.slice(n_conv1.get_weights()[0], [0,0,3,0],[1,3,1,256])
bias1 = n_conv1.get_weights()[1]/4
# print(n_conv1.get_weights()[0])
# print(n_conv1.get_weights())
n_drop1 = model.get_layer('drop1')
n_padding2 = model.get_layer('padding2')
n_conv2 = model.get_layer('conv2')
d_conv_weights2_1 = tf.slice(n_conv2.get_weights()[0], [0,0,0,0],[2,3,64,80])
d_conv_weights2_2 = tf.slice(n_conv2.get_weights()[0], [0,0,64,0],[2,3,64,80])
d_conv_weights2_3 = tf.slice(n_conv2.get_weights()[0], [0,0,128,0],[2,3,64,80])
d_conv_weights2_4 = tf.slice(n_conv2.get_weights()[0], [0,0,192,0],[2,3,64,80])
bias2 = n_conv2.get_weights()[1]/4
n_drop2 = model.get_layer('drop2')
n_flatten1 = model.get_layer('flatten1')
n_dense1 = model.get_layer('dense1')
# print(n_dense1.get_weights()[0])
d_fc_weights1_1 = tf.slice(n_dense1.get_weights()[0], [0,0],[2640,256])
d_fc_weights1_2 = tf.slice(n_dense1.get_weights()[0], [2640,0],[2640,256])
d_fc_weights1_3 = tf.slice(n_dense1.get_weights()[0], [5280,0],[2640,256])
d_fc_weights1_4 = tf.slice(n_dense1.get_weights()[0], [7920,0],[2640,256])
bias3 = n_dense1.get_weights()[1]/4
n_drop3 = model.get_layer('drop3')
n_dense2 = model.get_layer('dense2')
d_fc_weights2_1 = tf.slice(n_dense2.get_weights()[0], [0,0],[64,11])
d_fc_weights2_2 = tf.slice(n_dense2.get_weights()[0], [64,0],[64,11])
d_fc_weights2_3 = tf.slice(n_dense2.get_weights()[0], [128,0],[64,11])
d_fc_weights2_4 = tf.slice(n_dense2.get_weights()[0], [192,0],[64,11])
bias4 = n_dense2.get_weights()[1]/4
n_softmax1 = model.get_layer('softmax1')
n_reshape2 = model.get_layer('reshape2')
#model1 = Model(n_input1, n_drop3(n_dense1(n_flatten1(n_drop2(n_conv2(n_padding2(n_drop1(n_conv1(n_padding1(n_reshape1(n_input1)))))))))))
# n_input2 = Input(shape=(256, ))
# model2 = Model(n_input2, n_reshape2(n_softmax1(n_dense2(n_input2))))


###
n_input1_1 = Input(shape=in_shp)
model1 = Model(n_input1_1, n_reshape1(n_input1_1))
# n_input2_1 = Input(shape=[2,32,4])
# model2 = Model(n_input2_1, n_padding1(n_input2_1))
# n_input3_1 = Input(shape=[2,36,4])
# model3 = Model(n_input3_1, n_drop1(n_conv1(n_input3_1)))
# n_input4_1 = Input(shape=[2,34,256])
# model4 = Model(n_input4_1, n_padding2(n_input4_1))
# n_input5_1 = Input(shape=[2,38,256])
# model5 = Model(n_input5_1, n_drop2(n_conv2(n_input5_1)))
# n_input6_1 = Input(shape=[1,36,80])
# model6 = Model(n_input6_1, n_flatten1(n_input6_1))
# n_input7_1 = Input(shape=[2880])
# model7 = Model(n_input7_1, n_drop3(n_dense1(n_input7_1)))
# n_input8_1 = Input(shape=[256])
# model8 = Model(n_input8_1, n_reshape2(n_softmax1(n_dense2(n_input8_1))))
# n_input9_1 = Input(shape=[11])
# model9_1 = Model(n_input9_1, n_reshape1(n_input9_1))


# model1.summary()
# model2.summary()

# print(X_test[0])
output1 = model1.predict(X_test[0:1])
# print(output1)
output1_1 = output1[:, :2, :128, :1]
output1_2 = output1[:, :2, :128, 1:2]
output1_3 = output1[:, :2, :128, 2:3]
output1_4 = output1[:, :2, :128, 3:]
# print(output1_1)
# d_padding1 = ZeroPadding2D((0, 2), name='padding1_1')
# output2_1 = d_padding1(tf.convert_to_tensor(output1_1))
# print(output2_1)
# output2_2 = d_padding1(tf.convert_to_tensor(output1_2))
# output2_3 = d_padding1(tf.convert_to_tensor(output1_3))
# output2_4 = d_padding1(tf.convert_to_tensor(output1_4))
input2 = Input(shape=(2,128,1))
d_padding1 = ZeroPadding2D((0,2), name='padding1_1')
output2 = d_padding1(input2)
model2 = Model(input2, output2)
output2_1 = model2.predict(output1_1)
output2_2 = model2.predict(output1_2)
output2_3 = model2.predict(output1_3)
output2_4 = model2.predict(output1_4)
# print(output2_1.shape)

d_conv1_1 = Conv2D(256, (1, 3), strides=1, padding='valid', activation='relu', name='d_conv1_1', kernel_initializer='glorot_uniform')
d_conv1_2 = Conv2D(256, (1, 3), strides=1, padding='valid', activation='relu', name='d_conv1_2', kernel_initializer='glorot_uniform')
d_conv1_3 = Conv2D(256, (1, 3), strides=1, padding='valid', activation='relu', name='d_conv1_3', kernel_initializer='glorot_uniform')
d_conv1_4 = Conv2D(256, (1, 3), strides=1, padding='valid', activation='relu', name='d_conv1_4', kernel_initializer='glorot_uniform')
d_conv1_1_output = d_conv1_1(Input(shape=(2,132,1)))
d_conv1_2_output = d_conv1_2(Input(shape=(2,132,1)))
d_conv1_3_output = d_conv1_3(Input(shape=(2,132,1)))
d_conv1_4_output = d_conv1_4(Input(shape=(2,132,1)))
# d_conv1_1_output = d_conv1_1(tf.convert_to_tensor(output2_1))
# d_conv1_2_output = d_conv1_2(tf.convert_to_tensor(output2_2))
# d_conv1_3_output = d_conv1_3(tf.convert_to_tensor(output2_3))
# d_conv1_4_output = d_conv1_4(tf.convert_to_tensor(output2_4))
# print(d_conv1_1_output)
d_conv1_1.set_weights([d_conv_weights1_1, bias1])
d_conv1_2.set_weights([d_conv_weights1_2, bias1])
d_conv1_3.set_weights([d_conv_weights1_3, bias1])
d_conv1_4.set_weights([d_conv_weights1_4, bias1])
input3 = Input(shape=(2,132,1))
output3_1 = d_conv1_1(input3)
output3_2 = d_conv1_2(input3)
output3_3 = d_conv1_3(input3)
output3_4 = d_conv1_4(input3)
model3_1 = Model(input3, output3_1)
model3_2 = Model(input3, output3_2)
model3_3 = Model(input3, output3_3)
model3_4 = Model(input3, output3_4)
d_conv1_1_output = model3_1.predict(output2_1)
d_conv1_2_output = model3_2.predict(output2_2)
d_conv1_3_output = model3_3.predict(output2_3)
d_conv1_4_output = model3_4.predict(output2_4)
d_conv1_output = d_conv1_1_output + d_conv1_2_output + d_conv1_3_output + d_conv1_4_output
# print(d_conv1_output)
output3_1 = d_conv1_output[:, :2, :130, :64]
output3_2 = d_conv1_output[:, :2, :130, 64:128]
output3_3 = d_conv1_output[:, :2, :130, 128:192]
output3_4 = d_conv1_output[:, :2, :130, 192:]
input4 = Input(shape=(2,130,64))
d_padding2 = ZeroPadding2D((0,2), name='padding2_1')
output4 = d_padding2(input4)
model4 = Model(input4, output4)
output4_1 = model4.predict(output3_1)
output4_2 = model4.predict(output3_2)
output4_3 = model4.predict(output3_3)
output4_4 = model4.predict(output3_4)
# print(output4_1)
# d_padding2 = ZeroPadding2D((0, 2), name='padding2_1')
# output4_1 = d_padding2(tf.convert_to_tensor(output3_1))
# output4_2 = d_padding2(tf.convert_to_tensor(output3_2))
# output4_3 = d_padding2(tf.convert_to_tensor(output3_3))
# output4_4 = d_padding2(tf.convert_to_tensor(output3_4))
d_conv2_1 = Conv2D(80, (2, 3), padding='valid', activation='relu', name='d_conv2_1', kernel_initializer='glorot_uniform')
d_conv2_2 = Conv2D(80, (2, 3), padding='valid', activation='relu', name='d_conv2_2', kernel_initializer='glorot_uniform')
d_conv2_3 = Conv2D(80, (2, 3), padding='valid', activation='relu', name='d_conv2_3', kernel_initializer='glorot_uniform')
d_conv2_4 = Conv2D(80, (2, 3), padding='valid', activation='relu', name='d_conv2_4', kernel_initializer='glorot_uniform')
d_conv2_1_output = d_conv2_1(Input(shape=(2,134,64)))
d_conv2_2_output = d_conv2_2(Input(shape=(2,134,64)))
d_conv2_3_output = d_conv2_3(Input(shape=(2,134,64)))
d_conv2_4_output = d_conv2_4(Input(shape=(2,134,64)))
# print(d_conv2_1_output.shape)
d_conv2_1.set_weights([d_conv_weights2_1, bias2])
d_conv2_2.set_weights([d_conv_weights2_2, bias2])
d_conv2_3.set_weights([d_conv_weights2_3, bias2])
d_conv2_4.set_weights([d_conv_weights2_4, bias2])
input5 = Input(shape=(2,134,64))
output5_1 = d_conv2_1(input5)
output5_2 = d_conv2_2(input5)
output5_3 = d_conv2_3(input5)
output5_4 = d_conv2_4(input5)
model5_1 = Model(input5, output5_1)
model5_2 = Model(input5, output5_2)
model5_3 = Model(input5, output5_3)
model5_4 = Model(input5, output5_4)
d_conv2_1_output = model5_1.predict(output4_1)
d_conv2_2_output = model5_2.predict(output4_2)
d_conv2_3_output = model5_3.predict(output4_3)
d_conv2_4_output = model5_4.predict(output4_4)
d_conv2_output = d_conv2_1_output + d_conv2_2_output + d_conv2_3_output + d_conv2_4_output
# print(d_conv2_output.shape)
input6 = Input(shape=(1,132,80))
d_flatten = Flatten(name='d_flatten1')
output6 = d_flatten(input6)
model6 = Model(input6, output6)
d_output6 = model6.predict(d_conv2_output)
# print(d_output6.shape)
# d_flatten1 = Flatten(name='d_flatten1')
# output5 = d_flatten1(tf.convert_to_tensor(d_conv2_output))
# print(output5.shape)
output6_1 = d_output6[:, :2640]
output6_2 = d_output6[:, 2640:5280]
output6_3 = d_output6[:, 5280:7920]
output6_4 = d_output6[:, 7920:]
d_fc1_1 = Dense(256, activation='relu', kernel_initializer='he_normal', name='d_fc1_1')
d_fc1_2 = Dense(256, activation='relu', kernel_initializer='he_normal', name='d_fc1_2')
d_fc1_3 = Dense(256, activation='relu', kernel_initializer='he_normal', name='d_fc1_3')
d_fc1_4 = Dense(256, activation='relu', kernel_initializer='he_normal', name='d_fc1_4')
d_fc1_1_output = d_fc1_1(Input(shape=(2640,)))
d_fc1_2_output = d_fc1_2(Input(shape=(2640,)))
d_fc1_3_output = d_fc1_3(Input(shape=(2640,)))
d_fc1_4_output = d_fc1_4(Input(shape=(2640,)))
# print(d_conv2_1_output.shape)
d_fc1_1.set_weights([d_fc_weights1_1, bias3])
d_fc1_2.set_weights([d_fc_weights1_2, bias3])
d_fc1_3.set_weights([d_fc_weights1_3, bias3])
d_fc1_4.set_weights([d_fc_weights1_4, bias3])
input7 = Input(shape=(2640,))
output7_1 = d_fc1_1(input7)
output7_2 = d_fc1_2(input7)
output7_3 = d_fc1_3(input7)
output7_4 = d_fc1_4(input7)
model7_1 = Model(input7, output7_1)
model7_2 = Model(input7, output7_2)
model7_3 = Model(input7, output7_3)
model7_4 = Model(input7, output7_4)
d_fc1_1_output = model7_1.predict(output6_1)
d_fc1_2_output = model7_2.predict(output6_2)
d_fc1_3_output = model7_3.predict(output6_3)
d_fc1_4_output = model7_4.predict(output6_4)
d_fc1_output = d_fc1_1_output + d_fc1_2_output + d_fc1_3_output + d_fc1_4_output
# print(d_fc1_output)
output8_1 = d_fc1_output[:, :64]
output8_2 = d_fc1_output[:, 64:128]
output8_3 = d_fc1_output[:, 128:192]
output8_4 = d_fc1_output[:, 192:]
d_fc2_1 = Dense(11, activation='softmax', kernel_initializer='he_normal', name='d_fc2_1')
d_fc2_2 = Dense(11, activation='softmax', kernel_initializer='he_normal', name='d_fc2_2')
d_fc2_3 = Dense(11, activation='softmax', kernel_initializer='he_normal', name='d_fc2_3')
d_fc2_4 = Dense(11, activation='softmax', kernel_initializer='he_normal', name='d_fc2_4')
d_fc2_1_output = d_fc2_1(Input(shape=(64,)))
d_fc2_2_output = d_fc2_2(Input(shape=(64,)))
d_fc2_3_output = d_fc2_3(Input(shape=(64,)))
d_fc2_4_output = d_fc2_4(Input(shape=(64,)))
# print(d_fc2_1_output.shape)
d_fc2_1.set_weights([d_fc_weights2_1, bias4])
d_fc2_2.set_weights([d_fc_weights2_2, bias4])
d_fc2_3.set_weights([d_fc_weights2_3, bias4])
d_fc2_4.set_weights([d_fc_weights2_4, bias4])
input9 = Input(shape=(64,))
output9_1 = d_fc2_1(input9)
output9_2 = d_fc2_2(input9)
output9_3 = d_fc2_3(input9)
output9_4 = d_fc2_4(input9)
model9_1 = Model(input9, output9_1)
model9_2 = Model(input9, output9_2)
model9_3 = Model(input9, output9_3)
model9_4 = Model(input9, output9_4)
d_fc2_1_output = model9_1.predict(output8_1)
d_fc2_2_output = model9_2.predict(output8_2)
d_fc2_3_output = model9_3.predict(output8_3)
d_fc2_4_output = model9_4.predict(output8_4)
d_fc2_output = d_fc2_1_output + d_fc2_2_output + d_fc2_3_output + d_fc2_4_output
# print(d_fc2_output)
###生成模型
######
d_model1_1_1 = models.Sequential()
d_model1_1_1.add(Reshape([2,128,1], input_shape=[2,128,1], name='d_reshape1'))
d_model1_1_1.add(d_padding1)
d_model1_1_1.add(d_conv1_1)
# d_model1_1_1.summary()

d_model1_1_2 = models.Sequential()
d_model1_1_2.add(Reshape([2,130,64], input_shape=[2,130,64], name='d_reshape2'))
d_model1_1_2.add(d_padding2)
d_model1_1_2.add(d_conv2_1)
d_model1_1_2.add(d_flatten)
# d_model1_1_2.summary()

d_model1_1_3 = models.Sequential()
d_model1_1_3.add(Reshape([2640,], input_shape=[2640,], name='d_reshape3'))
d_model1_1_3.add(d_fc1_1)
# d_model1_1_3.summary()

d_model1_1_4 = models.Sequential()
d_model1_1_4.add(Reshape([64,], input_shape=[64,], name='d_reshape4'))
d_model1_1_4.add(d_fc2_1)
# d_model1_1_4.summary()

d_model1_2_1 = models.Sequential()
d_model1_2_1.add(Reshape([2,128,1], input_shape=[2,128,1], name='d_reshape1'))
d_model1_2_1.add(d_padding1)
d_model1_2_1.add(d_conv1_2)

d_model1_2_2 = models.Sequential()
d_model1_2_2.add(Reshape([2,130,64], input_shape=[2,130,64], name='d_reshape2'))
d_model1_2_2.add(d_padding2)
d_model1_2_2.add(d_conv2_2)
d_model1_2_2.add(d_flatten)

d_model1_2_3 = models.Sequential()
d_model1_2_3.add(Reshape([2640,], input_shape=[2640,], name='d_reshape3'))
d_model1_2_3.add(d_fc1_2)

d_model1_2_4 = models.Sequential()
d_model1_2_4.add(Reshape([64,], input_shape=[64,], name='d_reshape4'))
d_model1_2_4.add(d_fc2_2)

d_model1_3_1 = models.Sequential()
d_model1_3_1.add(Reshape([2,128,1], input_shape=[2,128,1], name='d_reshape1'))
d_model1_3_1.add(d_padding1)
d_model1_3_1.add(d_conv1_3)

d_model1_3_2 = models.Sequential()
d_model1_3_2.add(Reshape([2,130,64], input_shape=[2,130,64], name='d_reshape2'))
d_model1_3_2.add(d_padding2)
d_model1_3_2.add(d_conv2_3)
d_model1_3_2.add(d_flatten)

d_model1_3_3 = models.Sequential()
d_model1_3_3.add(Reshape([2640,], input_shape=[2640,], name='d_reshape3'))
d_model1_3_3.add(d_fc1_3)

d_model1_3_4 = models.Sequential()
d_model1_3_4.add(Reshape([64,], input_shape=[64,], name='d_reshape4'))
d_model1_3_4.add(d_fc2_3)

d_model1_4_1 = models.Sequential()
d_model1_4_1.add(Reshape([2,128,1], input_shape=[2,128,1], name='d_reshape1'))
d_model1_4_1.add(d_padding1)
d_model1_4_1.add(d_conv1_4)

d_model1_4_2 = models.Sequential()
d_model1_4_2.add(Reshape([2,130,64], input_shape=[2,130,64], name='d_reshape2'))
d_model1_4_2.add(d_padding2)
d_model1_4_2.add(d_conv2_4)
d_model1_4_2.add(d_flatten)

d_model1_4_3 = models.Sequential()
d_model1_4_3.add(Reshape([2640,], input_shape=[2640,], name='d_reshape3'))
d_model1_4_3.add(d_fc1_4)

d_model1_4_4 = models.Sequential()
d_model1_4_4.add(Reshape([64,], input_shape=[64,], name='d_reshape4'))
d_model1_4_4.add(d_fc2_4)
# 第一个节点模型
d_model1_1_1.save('./model/d_model1_1_1.h5')
d_model1_1_2.save('./model/d_model1_1_2.h5')
d_model1_1_3.save('./model/d_model1_1_3.h5')
d_model1_1_4.save('./model/d_model1_1_4.h5')
# 第二个节点模型
d_model1_2_1.save('./model/d_model1_2_1.h5')
d_model1_2_2.save('./model/d_model1_2_2.h5')
d_model1_2_3.save('./model/d_model1_2_3.h5')
d_model1_2_4.save('./model/d_model1_2_4.h5')
# 第三个节点模型
d_model1_3_1.save('./model/d_model1_3_1.h5')
d_model1_3_2.save('./model/d_model1_3_2.h5')
d_model1_3_3.save('./model/d_model1_3_3.h5')
d_model1_3_4.save('./model/d_model1_3_4.h5')
# 第四个节点模型
d_model1_4_1.save('./model/d_model1_4_1.h5')
d_model1_4_2.save('./model/d_model1_4_2.h5')
d_model1_4_3.save('./model/d_model1_4_3.h5')
d_model1_4_4.save('./model/d_model1_4_4.h5')



# print(Y_test[0:1])
# print(output1.shape)
# if batch_size == 0:
#     output1 = model1.predict(X_test)
# else:
#     output1 = model1.predict(X_test, batch_size=batch_size)
# output1 = kmeanMethods.cluster(output1, k)
# return model2.predict(output1)