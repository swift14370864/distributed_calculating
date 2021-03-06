# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:37:22 2020

@author: yzy86
input_split在合并时做sum操作
"""

import  tensorflow as tf
from    tensorflow.keras import datasets
import  os
import  time
 
# 设置后台打印日志等级 避免后台打印一些无用的信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
# 利用Tensorflow2中的接口加载mnist数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
#print(x.shape, y.shape)
 
# 对数据进行预处理
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x,y
 
# 构建dataset对象，方便对数据的打乱，批处理等超操作
train_db = tf.data.Dataset.from_tensor_slices((x,y)).shuffle(1000).batch(128)
train_db = train_db.map(preprocess)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)
test_db = test_db.map(preprocess)

w = tf.Variable(tf.random.truncated_normal([784, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))

lr = 1e-3
 
# epoch表示整个训练集循环的次数 这里循环100次
for epoch in range(2):
    # step表示当前训练到了第几个Batch
    for step, (x, y) in enumerate(train_db):
        # 把训练集进行打平操作
        x = tf.reshape(x, [-1, 28*28])
        # 构建模型并计算梯度
        with tf.GradientTape() as tape: # tf.Variable
            # 三层非线性模型搭建
            h = x@w + tf.broadcast_to(b, [x.shape[0], 10])
            out = h
 
            # 把标签转化成one_hot编码 
            y_onehot = tf.one_hot(y, depth=10)
 
            # 计算MSE
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)
 
        # 计算梯度
        grads = tape.gradient(loss, [w, b])
        
        # w = w - lr * w_grad
        # 利用上述公式进行权重的更新
        w.assign_sub(lr * grads[0])
        b.assign_sub(lr * grads[1])
        
 
        # 每训练100个Batch 打印一下当前的loss
        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))
            

w1,w2,w3,w4=tf.split(w,[196,196,196,196],0)

# ww1,ww2=tf.split(w,[5,5],1)
# bb1,bb2=tf.split(b,[5,5])
# 每训练完一次数据集 测试一下啊准确率
total_correct, total_num = 0, 0
st=time.time()
# total_correctc, total_numt = 0, 0
for step, (x,y) in enumerate(test_db):
 
    x = tf.reshape(x, [-1, 28*28])
    x1,x2,x3,x4=tf.split(x,[196,196,196,196],1)
    out1=x1@w1
    out2=x2@w2
    out3=x3@w3
    out4=x4@w4
    out=tf.add(tf.add(tf.add(tf.add(out1,out2),out3),out4),b)
    
    # outout1=x@ww1+bb1
    # outout2=x@ww2+bb2
    # outout=tf.concat([outout1,outout2],1)
    
    # print(out==outout)
    
 
    # 把输出值映射到[0~1]之间
    prob = tf.nn.softmax(out, axis=1)
    # 获取概率最大值得索引位置
    pred = tf.argmax(prob, axis=1)
    pred = tf.cast(pred, dtype=tf.int32)
    
    correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
    correct = tf.reduce_sum(correct)
    # 获取每一个batch中的正确率和batch大小
    total_correct += int(correct)
    total_num += x.shape[0]
    
    # probprob = tf.nn.softmax(outout, axis=1)
    # # 获取概率最大值得索引位置
    # predpred = tf.argmax(probprob, axis=1)
    # predpred = tf.cast(predpred, dtype=tf.int32)
    
    # correctc = tf.cast(tf.equal(predpred, y), dtype=tf.int32)
    # correctc = tf.reduce_sum(correctc)
    # # 获取每一个batch中的正确率和batch大小
    # total_correctc += int(correctc)
    # total_numt += x.shape[0]
# 计算总的正确率
acc = total_correct / total_num
et=time.time()
# accacc = total_correctc / total_numt
print('the total test acc:', acc)
print('the time is:', et-st)
# print('the total test acc:', accacc)
