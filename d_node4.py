# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:18:05 2020
客户端
@author: yzy86
"""

import socket
import threading
import os
import get
import time
import numpy as np
import json
import keras.models as models
import tensorflow as tf
from keras.models import Model
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers import Input

from dataset_test import DataSet

path_dataset = './datasets/RML2016.10a_dict.pkl'
dataset = DataSet(path_dataset)
X, lbl, snrs, mods = dataset.getX()
X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()

in_shp = list(X_train.shape[1:])# (2, 128)
# print(in_shp)


d_model1_3_1 = models.Sequential()
d_model1_3_1.add(Reshape([2,128,1], input_shape=[2,128], name='d_reshape1'))
d_model1_3_1.add(ZeroPadding2D((0,2), name='padding1_1'))
d_model1_3_1.add(Conv2D(256, (1, 3), strides=1, input_shape = [2,132,1], padding='valid', activation='relu', name='d_conv1_1', kernel_initializer='glorot_uniform'))
# d_model1_3_1.summary()
d_model1_3_1.load_weights('./model/d_model1_3_1.h5')

d_model1_3_2 = models.Sequential()
d_model1_3_2.add(Reshape([2,130,64], input_shape=[2,130,64], name='d_reshape1'))
d_model1_3_2.add(ZeroPadding2D((0,2), name='padding1_2'))
d_model1_3_2.add(Conv2D(80, (2, 3), strides=1, input_shape = [2,134,64], padding='valid', activation='relu', name='d_conv2_1', kernel_initializer='glorot_uniform'))
d_model1_3_2.add(Flatten(name='flatten1'))
# d_model1_3_2.summary()
d_model1_3_2.load_weights('./model/d_model1_3_2.h5')

d_model1_3_3 = models.Sequential()
d_model1_3_3.add(Reshape([2640,], input_shape=[2640,], name='d_reshape1'))
d_model1_3_3.add((Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')))
# d_model1_3_3.summary()
d_model1_3_3.load_weights('./model/d_model1_3_3.h5')

d_model1_3_4 = models.Sequential()
d_model1_3_4.add(Reshape([64,], input_shape=[64,], name='d_reshape1'))
d_model1_3_4.add((Dense(11, activation='softmax', kernel_initializer='he_normal', name='dense2')))
# d_model1_3_4.summary()
d_model1_3_4.load_weights('./model/d_model1_3_4.h5')

# n_input1 = Input(shape=[2,32,1])
# n_reshape1 = d_model1_3_1.get_layer('d_reshape1')
# n_padding1 = d_model1_3_1.get_layer('padding1_1')
# n_conv1 = d_model1_3_1.get_layer('d_conv1_1')
# print(n_conv1.weights[0])

# model1 = Model(n_input1_1, n_reshape1(n_input1_1))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# addr = # 服务器端的IP和端口
addr = (get.get_ip(), 10000)#。只有自己一台电脑做测试时，可以直接用左边的
s.connect(addr)
i=0

def recv_msg():  #
    print("连接成功！现在可以接收消息！\n")
    while True:
        try:  # 测试发现，当服务器率先关闭时，这边也会报ConnectionResetError
            response = s.recv(40960000).decode("utf8")
            list_receive = json.loads(response)
            
            print('receive list:', len(list_receive))
            time.sleep(1)
            global i
            if(i==0):
                matrix = np.array(list_receive).reshape(1,2,130,64)
                matrix = cal1(matrix)
                trans_list = matrix.tolist()
                json_string = json.dumps(trans_list)
            elif(i==1):
                matrix = np.array(list_receive).reshape(1,2640,)
                matrix = cal2(matrix)
                trans_list = matrix.tolist()
                json_string = json.dumps(trans_list)
            elif(i==2):
                matrix = np.array(list_receive).reshape(1,64,)
                matrix = cal3(matrix)
                trans_list = matrix.tolist()
                json_string = json.dumps(trans_list)
            else:
                print(list_receive)
                print('now is the end.')
            i+=1
            s.send(json_string.encode("utf8"))
            
        except ConnectionResetError:
            print("服务器关闭，聊天已结束！")
            s.close()
            break
    os._exit(0)


def send_msg():
    print("连接成功！现在可以发送消息！\n")
    ##发送一个消息
    # msg = '1.0'        
    # s.send(msg.encode("utf8"))
    ##发送一个列表
    matrix = output2
    trans_list = matrix.tolist()
    json_string = json.dumps(trans_list)
    s.send(json_string.encode())
    
def cal1(matrix):
    matrix = tf.convert_to_tensor(matrix)
    global d_model1_3_2
    output = d_model1_3_2.predict(matrix, steps = 1)
    # output = output.numpy()
    return output

def cal2(matrix):
    matrix = tf.convert_to_tensor(matrix)
    global d_model1_3_3
    output = d_model1_3_3.predict(matrix, steps = 1)
    # output = output.numpy()
    return output

def cal3(matrix):
    matrix = tf.convert_to_tensor(matrix)
    global d_model1_3_4
    output = d_model1_3_4.predict(matrix, steps = 1)
    # output = output.numpy()
    return output


output1 = X_test[0:1]
output2 = d_model1_3_1.predict(output1)
##对output2的通信操作
threads = [threading.Thread(target=recv_msg), threading.Thread(target=send_msg)]
for t in threads:
    t.start()
# tele_output2 = [0,0,0,0,0]
# ##
# output3 = d_model1_3_2.predict(tele_output2)
# ##对output3的通信操作

# tele_output3 = [0,0,0,0,0]
# ##
# output4 = d_model1_3_3.predict(tele_output3)
# ##对output4的通信操作

# tele_output4 = [0,0,0,0,0]
# ##
# output5 = d_model1_3_4.predict(tele_output4)
##对output5的通信操作

##
