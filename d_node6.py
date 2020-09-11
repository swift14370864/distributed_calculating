# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:23:53 2020
这个节点作为服务端和客户端
@author: yzy86
"""
import socket
import get  # 自己写的
import threading
import os
import json
import numpy as np
import time
import keras.models as models
import tensorflow as tf
from keras.models import Model
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers import Input

from dataset_test import DataSet

class ChatSever:

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addr = (get.get_ip(), 10000)
        self.users = {}
        self.res = []
        self.address = []
        self.tmp = []
        self.i = 0
        # self.index = 0
        

    def start_sever(self):
       	try:
        	self.sock.bind(self.addr)
        except Exception as e: print(e)
        self.sock.listen(5)
        print("服务器已开启，等待连接...")
        print("在空白处输入stop sever并回车，来关闭服务器")

        self.accept_cont()

    def accept_cont(self):
        while True:
            s, addr = self.sock.accept()
            self.users[addr] = s
            self.res.append(s)
            self.address.append(addr)
            number = len(self.users)
            print("用户{}连接成功！现在共有{}位用户".format(addr, number))

            threading.Thread(target=self.recv_send, args=(s, addr)).start()
            
    def recv_send(self, sock, addr):
        while True:
            try:  # 测试后发现，当用户率先选择退出时，这边就会报ConnectionResetError
                response = sock.recv(40960000).decode('utf8')
                mylist = json.loads(response)
                print('收到消息：', len(mylist))
                self.tmp.append(mylist)
                # print('address list is:',self.address)
                if len(self.tmp)>=3:
                    self.i += 1
                    print('calculating...')
                    # sp = np.array(self.tmp[0]).shape
                    # l=1
                    # for s in sp:
                    #     l *= s
                    if self.i == 1: selfdata = output2
                    self.tmp.append(selfdata.tolist())
                    addresult = np.array(self.tmp[0]) + np.array(self.tmp[1]) + np.array(self.tmp[2]) + np.array(self.tmp[3])
                    sp = addresult.shape
                    l=1
                    for s in sp:
                        l *= s
                    addresult = addresult.reshape(l)
                    print(addresult.shape)
                    addresult = addresult.tolist()
                    print('...done.')
                    if self.i != 4:
                        cut = int(len(addresult)/4)
                        print(cut)
                        print('slicing...')
                        result1 = addresult[:cut]
                        result2 = addresult[cut:2*cut]
                        result3 = addresult[2*cut:3*cut]
                        result4 = addresult[3*cut:]
                        json_result1 = json.dumps(result1)
                        # json_result2 = json.dumps(result2)
                        json_result3 = json.dumps(result3)
                        json_result4 = json.dumps(result4)
                        result = [json_result1, json_result3, json_result4]
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        print('sending...')
                        for i in range(len(self.res)):
                            self.res[i].send(result[i].encode('utf8'))
                            print('send the result...done', i)
                        if self.i == 1:
                            matrix = np.array(result2).reshape(1,2,130,64)
                            matrix = cal1(matrix)
                            selfdata = matrix.tolist()
                        elif self.i ==2:
                            matrix = np.array(result2).reshape(1,2640,)
                            matrix = cal2(matrix)
                            selfdata = matrix.tolist()
                        elif self.i ==3:
                            matrix = np.array(result2).reshape(1,64,)
                            matrix = cal3(matrix)
                            selfdata = matrix.tolist()
                    else:
                        result1 = addresult
                        result2 = addresult
                        result3 = addresult
                        result4 = addresult
                        json_result1 = json.dumps(result1)
                        # json_result2 = json.dumps(result2)
                        json_result3 = json.dumps(result3)
                        json_result4 = json.dumps(result4)
                        result = [json_result1, json_result3, json_result4]
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        self.tmp.pop(0)
                        print('sending...')
                        for i in range(len(self.res)):
                            self.res[i].send(result[i].encode('utf8'))
                            # time.sleep(2)
                            print('send the result...done', i)
               
            except ConnectionResetError:
                print("用户{}已经退出聊天！".format(addr))
                self.users.pop(addr)
                index = self.address.index(addr)
                self.address.remove(addr)
                del(self.res[index])
                break

    def close_sever(self):
        for client in self.users.values():
            client.close()
        self.sock.close()
        os._exit(0)
        
def cal1(matrix):
    matrix = tf.convert_to_tensor(matrix)
    global d_model1_2_2
    print('matrix:', matrix.shape)
    output = d_model1_2_2.predict(matrix, steps = 1)
    # output = output.numpy()
    return output

def cal2(matrix):
    matrix = tf.convert_to_tensor(matrix)
    global d_model1_2_3
    output = d_model1_2_3.predict(matrix, steps = 1)
    # output = output.numpy()
    return output

def cal3(matrix):
    matrix = tf.convert_to_tensor(matrix)
    global d_model1_2_4
    output = d_model1_2_4.predict(matrix, steps = 1)
    # output = output.numpy()
    return output

if __name__ == "__main__":
    path_dataset = './datasets/RML2016.10a_dict.pkl'
    dataset = DataSet(path_dataset)
    X, lbl, snrs, mods = dataset.getX()
    X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()
    
    in_shp = list(X_train.shape[1:])# (2, 128)
    # print(in_shp)
    
    
    d_model1_2_1 = models.Sequential()
    d_model1_2_1.add(Reshape([2,128,1], input_shape=[2,128], name='d_reshape1'))
    d_model1_2_1.add(ZeroPadding2D((0,2), name='padding1_1'))
    d_model1_2_1.add(Conv2D(256, (1, 3), strides=1, input_shape = [2,132,1], padding='valid', activation='relu', name='d_conv1_1', kernel_initializer='glorot_uniform'))
    # d_model1_2_1.summary()
    d_model1_2_1.load_weights('./model/d_model1_2_1.h5')
    
    d_model1_2_2 = models.Sequential()
    d_model1_2_2.add(Reshape([2,130,64], input_shape=[2,130,64], name='d_reshape1'))
    d_model1_2_2.add(ZeroPadding2D((0,2), name='padding1_2'))
    d_model1_2_2.add(Conv2D(80, (2, 3), strides=1, input_shape = [2,134,64], padding='valid', activation='relu', name='d_conv2_1', kernel_initializer='glorot_uniform'))
    d_model1_2_2.add(Flatten(name='flatten1'))
    # d_model1_2_2.summary()
    d_model1_2_2.load_weights('./model/d_model1_2_2.h5')
    
    d_model1_2_3 = models.Sequential()
    d_model1_2_3.add(Reshape([2640,], input_shape=[2640,], name='d_reshape1'))
    d_model1_2_3.add((Dense(256, activation='relu', kernel_initializer='he_normal', name='dense1')))
    # d_model1_2_3.summary()
    d_model1_2_3.load_weights('./model/d_model1_2_3.h5')
    
    d_model1_2_4 = models.Sequential()
    d_model1_2_4.add(Reshape([64,], input_shape=[64,], name='d_reshape1'))
    d_model1_2_4.add((Dense(11, activation='softmax', kernel_initializer='he_normal', name='dense2')))
    # d_model1_2_4.summary()
    d_model1_2_4.load_weights('./model/d_model1_2_4.h5')
    output1 = X_test[0:1]
    print('output1',output1.shape)
    output2 = d_model1_2_1.predict(output1)
    print('output2',output2.shape)
    
    sever = ChatSever()
    sever.start_sever()
    while True:
        cmd = input()
        if cmd == "stop sever":
            sever.close_sever()
        else:
            print("输入命令无效，请重新输入！")


