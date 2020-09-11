# -*- coding: utf-8 -*-
"""
Created on Thu May 21 00:00:37 2020

@author: yzy86
"""


import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets
import  numpy as np
import  os
 
# 设置后台打印日志等级 避免后台打印一些无用的信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 

w0 = tf.Variable(tf.random.truncated_normal([4, 4], stddev=0.1))
w0_np=w0.numpy()

###转回来
#data_tensor= tf.convert_to_tensor(data_numpy)

pf = open('result.txt', 'w+')
[h,w]=w0_np.shape
pf.write(str(h)+' '+str(w)+'\n')
for h1 in range(h):
    for w1 in range(w):
        pf.write('%f ' %w0_np[h1][w1])
    pf.write('\n')
# try:   
#     bias = reader.get_tensor(quantized_conv_name+"/b")
#     n2=bias.shape
#     print bias.shape
#     print n2
#     print '***************************************'
#     pf.write('\n')
#     pf.write('**************************bias:')
#     pf.write('\n')
#     pf.write(str(n)+'\n')
#     #for n1 in range(n2):
#     #    pf.write('%f, ' %bias[n1]) 
#     #pf.write('\n')
#     for b in bias:
#        pf.write('%f '%b)
# except:
#     print 'no bias'
pf.write('\n')
pf.close()