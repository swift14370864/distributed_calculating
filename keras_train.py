# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:09:23 2020

@author: yzy86
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt
import  io
import  datetime
(x, y), (test_x, test_y) = datasets.cifar10.load_data()
 
# 数据预处理
def progress (x ,y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    x = tf.reshape(x, [-1, 32*32*3])
    y = tf.one_hot(y, depth=10, dtype=tf.int32)
    return x, y
# 构建dataset对象 方便对数据的管理
db_train = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(1000)
db_test = tf.data.Dataset.from_tensor_slices((test_x,test_y))
# deal with data
train_db = db_train.map(progress).batch(128)
test_db = db_test.map(progress).batch(128)
train_iter = iter(train_db)
train_next = next(train_iter)
print(train_next[0].shape)
 
# 利用Tensorflow的Sequential容器去构建model
model = Sequential([
    # layers.Dense(256, activation=tf.nn.relu),
    # layers.Dense(128, activation=tf.nn.relu),
    # layers.Dense(64, activation=tf.nn.relu),
    # layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=[None, 32*32*3])
model.summary()
 
 
# 利用Sequential的compile方法，简化损失函数，梯度优化，计算准确率等操作
model.compile(optimizer=optimizers.Adam(lr = 0.01),
              loss = tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
# 设置模型对整体数据重复训练5次，每训练2次打印一次正确率
model.fit(train_db, epochs=2, validation_data = test_db,
              validation_freq=2)
# 另一种打印模型正确率的接口方法

model.evaluate(test_db)
