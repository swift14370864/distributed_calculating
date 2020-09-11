# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:51:24 2020

@author: yzy86
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
 
import os
import matplotlib.pyplot as plt
 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
tf.random.set_seed(2345)
# load dataset
(x, y), (test_x, test_y) = datasets.cifar100.load_data()
print(y.shape)
 
# 数据预处理
def progress(x, y):
    x =  2 * tf.cast(x, dtype=tf.float32) / 255. -1
    y = tf.cast(y, dtype=tf.int32)
    return x, y
# 构建dataset对象
db_train = tf.data.Dataset.from_tensor_slices((x, y))
db_train = db_train.shuffle(1000).map(progress).batch(128)
db_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
db_test = db_test.map(progress).batch(64)
 
train_next = next(iter(db_test))
print(train_next[0].shape, train_next[1].shape)
# show images
#画图
# plt.close('all')
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid('off')
#     plt.imshow(train_next[0][i],cmap=plt.cm.binary)
# plt.show()
 
# 构建前半部分的卷积网络
cov_network = [
    # unit 1
    # 64代表着 利用64个卷积核把原先3通道的图片卷积成64通道
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
 
    # unit 2
    # 128代表着 利用128个卷积核把原先64通道的图片卷积成128通道
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
 
    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
 
    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
 
    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]
 
# 自定义层
class myDense (layers.Layer):
    # 实现__init__()方法
    def __init__(self, in_dim, out_dim):
        # 调用母类中的__init__()
        super(myDense, self).__init__()
 
        self.kernel = self.add_variable('w', [in_dim, out_dim])
        self.bias = self.add_variable('b', [out_dim])
    # 实现call()方法
    def call(self, inputs, training = None):
        # 构建模型结构
        out = inputs @ self.kernel + self.bias
        return out
#自定义网络模型
class myModel(keras.Model):
 
    def __init__(self):
        # 调用母类中的__init__()方法
        super(myModel, self).__init__()
        # 调用自定义层类 并构建每一层的连接数
        self.fc1 = myDense(512, 256)
        self.fc2 = myDense(256, 128)
        self.fc3 = myDense(128, 100)
 
    # 构建一个三层的全连接网络
    def call(self, inputs, training=None):
 
        # 把训练数据输入到自定义层中
        x = self.fc1(inputs)
        # 利用relu函数进行非线性激活操作
        out = tf.nn.relu(x)
        x = self.fc2(out)
        out = tf.nn.relu(x)
        x = self.fc3(out)
        return x
 
 
def main():
    # 构建卷积网络对象
    cov_model = Sequential(cov_network)
 
    # 构建全连接网络对象
    line_mode = myModel()
    # 设置模型的输入tensor形状
    cov_model.build(input_shape=[None, 32, 32, 3])
    line_mode.build(input_shape=[None, 512])
    # 把两个模型中的权重用一个变量表示，方便后面的权重更新
    all_trainable = cov_model.trainable_variables + line_mode.trainable_variables
    # 设置优化器
    optimizer = optimizers.Adam(lr = 1e-4)
    # 开始训练
    for epoch in range(50):
        for step, (x, y) in enumerate(db_train):
 
            with tf.GradientTape() as tape:
                # 把卷积网络和全连接网络进行组合
                out = cov_model(x)
                out = tf.reshape(out, [-1, 512])
                logits = line_mode(out)
 
                y_onehot =  tf.one_hot(y, depth=100)
                # 利用交叉熵计算loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            # 计算梯度
            grad = tape.gradient(loss, all_trainable)
            # 梯度更新
            optimizer.apply_gradients(zip(grad, all_trainable))
 
            if step%100 == 0:
                print('epoch' , epoch ,'step: ',step,'loss:' , float(loss))
        # test
        total_num = 0
        total_correct = 0
        for x, y in db_test:
            out = cov_model(x)
            out = tf.reshape(out, [-1, 512])
            logits = line_mode(out)
            prob = tf.nn.softmax(logits, axis=1)
            pre = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            total_correct = total_correct + tf.reduce_sum(tf.cast(tf.equal(pre, y), dtype=tf.int32))
            total_num = total_num + x.shape[0]
        last_correct = int(total_correct) / total_num
        print("epoch:", epoch, last_correct)
 
if __name__ == '__main__':
    main()