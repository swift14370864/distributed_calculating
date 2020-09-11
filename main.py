# matplot
import matplotlib.pyplot as plt
import matplotlib
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


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Params Section
###################################################
# dataset file
path_dataset = './datasets/RML2016.10a_dict.pkl'
matplotlib.use('Agg')

fig_prefix = 'img/'
fig_prefix_default = fig_prefix + 'radio_kmeans('
fig_training = fig_prefix_default + 'training_performance).png'
fig_accuracy_nokmeans = fig_prefix_default + 'classification_accuracy_nokmeans).png'
fig_accuracy_kmeans = fig_prefix_default + 'classification_accuracy_kmeans_cluster_%s).png'

fig_prefix_conf = fig_prefix + 'confusion/'
fig_prefix_conf_all = fig_prefix_conf + 'all/'
fig_confusion = fig_prefix_conf_all + 'radio_kmeans(confusion_matrix).svg'
fig_confusion_nokmeans = fig_prefix_conf_all + 'radio_kmeans(nokmeans_confusion_matrix).svg'
fig_confusion_kmeans = fig_prefix_conf_all + 'radio_kmeans(kmeans_confusion_matrix_k_%s).svg'
fig_prefix_conf_snr = fig_prefix_conf + 'snr/'
fig_conf_snr = fig_prefix_conf_snr + '/default/radio_kmeans(confusion_matrix_snr_%s).svg'
fig_conf_snr_nokmeans = fig_prefix_conf_snr + 'nokmeans/radio_kmeans(nokmeans_confusion_matrix_snr_%s).svg'
fig_prefix_conf_snr_kmeans = fig_prefix_conf_snr + '/kmeans/%s/'
fig_postfix_conf_snr_kmeans = 'radio_kmeans(nokmeans_confusion_matrix_snr_%s).svg'

accFile_prefix = 'txt/radio_kmeans('
data_accuracy_nokmeans = accFile_prefix + 'accuracy_no_kmeans).txt'
data_accuracy_kmeans = accFile_prefix + 'accuracy_with_k_%s_).txt'
###################################################

# 0. get some arguments from command line
# if len(sys.argv) == 1:
#     print('\n\nparams information\n',
#           '\nparamter 1 and 2 are required:\n',
#           '\t1. gpu: gpu/cpu \t(cpu)\n',
#           '\t2. train: train/evaluate \t(train)\n',
#           '\nif train is "evaluate", parameters below are needed.\n',
#           '\t3. evaluate_mode: only accuracy=1, only confusion matrix=2, both=3 \t(1)\n',
#           '\t4. kmeans: kmeans/nokmeans \t(nokmeans)\n',
#           '\t5. cluster_k_from: >=2 (2)\n',
#           '\t6: cluster_k_to: >=3 (3)\n',
#           '\t7. confusion_snrs: detail\ignore (ignore)\n\n')
#     #sys.exit(0)

# # default cpu
# device = '-1'
# if len(sys.argv) > 1 and sys.argv[1] == 'gpu':
#     device = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = device

# # train or evaluate
# train = True
# if len(sys.argv) > 2 and sys.argv[2] == 'evaluate':
#     train = False

# # evaluate mode: 1. only accuracy 2. only confusion matrix 3. both
# evaluate_mode = 1
# # kmeans transformation
# use_kmeans = False  # ignore kmeans transformation
# min_k = 2           # minimum num for range_of_k
# range_of_k = range(min_k, min_k + 1)    # do kmeans transformation for a range of k
# k_from = ''
# k_to = ''
# confusion_snrs = False

# evaluate stage and argvs is available
# 1. loadDataSet
dataset = DataSet(path_dataset)
# X(220000 * (2, 128)) and lbl(220000 * 1) is whole dataset
# snrs(20) = -20 -> 18, mods(11) = ['8PSK', 'AM-DSB', ...]
X, lbl, snrs, mods = dataset.getX()
# X_train(176000) Y_train(176000 * 11) classes(11)=mods
X_train, Y_train, X_test, Y_test, classes = dataset.getTrainAndTest()
print("X_train is:")
print(X_train)
in_shp = list(X_train.shape[1:]) # (2, 128)



# 2. build VT-CNN2 Neural Net model
dr = 0.5 # dropout rate
model = models.Sequential()
model.add(Reshape(in_shp+[1], input_shape=in_shp, name='reshape1'))
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
nb_epochs = 100
batch_size = 1024
verbose = 1



# 3. train or evaluate
def format_time(second):
    tmp = second
    hour = second // 3600
    second = second - hour * 3600
    minute = second // 60
    second = second - minute * 60
    return '%s:%s:%s, total seconds: %s'%(hour, minute, second, tmp)

filepath = './model/convmodrecnets_CNN2_0.5.wts.h5'

# 3.1 train
model.load_weights(filepath)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print("score is: ", score)
# if train is True, program finished
# if train is False, the program go ahead
# 3.2 evaluate

# get kmeans model
model.load_weights(filepath)
n_input1 = Input(shape=in_shp)
n_reshape1 = model.get_layer('reshape1')
n_padding1 = model.get_layer('padding1')
n_conv1 = model.get_layer('conv1')
n_drop1 = model.get_layer('drop1')
n_padding2 = model.get_layer('padding2')
n_conv2 = model.get_layer('conv2')
n_drop2 = model.get_layer('drop2')
n_flatten1 = model.get_layer('flatten1')
n_dense1 = model.get_layer('dense1')
n_drop3 = model.get_layer('drop3')
n_dense2 = model.get_layer('dense2')
n_softmax1 = model.get_layer('softmax1')
n_reshape2 = model.get_layer('reshape2')
model1 = Model(n_input1, n_drop3(n_dense1(n_flatten1(n_drop2(n_conv2(
                            n_padding2(n_drop1(n_conv1(n_padding1(n_reshape1(n_input1)))))))))))
n_input2 = Input(shape=(256, ))
model2 = Model(n_input2, n_reshape2(n_softmax1(n_dense2(n_input2))))


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[], fig_name=''):
    # plt.figure(figsize=(6.4, 4.7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    if fig_name == '':
        fig_name = fig_confusion
    plt.savefig(fig_name)

def computeAccuracy(test_SNRs, func_pref, confusion_plot=False, fig_name=''):
    acc = {}
    for snr in snrs:
        # arr_test_SNRs = np.array(test_SNRs)
        # arr_res = arr_test_SNRs == snr
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes

        print('snr %s predict:' % (snr))

        # if not use_kmeans:
        #     test_Y_i_hat = model.predict(test_X_i)
        # else:
        #     test_Y_i_hat = KmeansPredict(test_X_i, k=cluster_k_elem)

        test_Y_i_hat = func_pref(test_X_i)

        conf = np.zeros([len(classes), len(classes)])
        confnorm = np.zeros([len(classes), len(classes)])

        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1
        for i in range(0, len(classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        if confusion_plot:
            plt.figure()
            if fig_name == '':
                fig_name = fig_conf_snr%(snr)
            plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr),
                                  fig_name=fig_name)

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("Overall Accuracy: ", cor / (cor + ncor))
        acc[snr] = 1.0 * cor / (cor + ncor)

    return acc

def saveAccuracyOfSNRs(acc, fig_name, data_file):
    # 1. Save results to a pickle file for plotting later
    print(acc)
    fd = open('results_cnn2_d0.5.dat', 'wb')
    pickle.dump(("CNN2", 0.5, acc), fd)

    # 2. Plot accuracy curve
    snrs_val = []
    for snr in snrs:
        snrs_val.append(acc[snr])
    print(snrs_val)

    plt.figure()
    plt.plot(snrs, snrs_val)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
    plt.savefig(fig_name)

    # save accuracy over snrs
    with open(data_file, 'w') as f:
        f.write(str(acc))

def accracyMode():
    acc = {}
    test_SNRs = []
    test_idx = dataset.getTestIndex()
    for idx in test_idx:
        test_SNRs.append(lbl[idx][1])

    # use origin predict function of VT-CNN2 model
    # acc = computeAccuracy(test_SNRs, model.predict, confusion_plot=True)
    acc = computeAccuracy(test_SNRs, model.predict, True, fig_name=fig_conf_snr_nokmeans)

    saveAccuracyOfSNRs(acc, fig_accuracy_nokmeans, data_accuracy_nokmeans)
    

def plot_confusion_for_test_data(test_Y_hat, name='', title=''):
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, X_test.shape[0]):
        j = list(Y_test[i, :]).index(1)
        k = int(np.argmax(test_Y_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])

    plot_confusion_matrix(confnorm, title=title, labels=classes, fig_name=name)

def confusionMode():
    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    plot_confusion_for_test_data(test_Y_hat, name=fig_confusion_nokmeans, title='VT-CNN2 混淆矩阵')
   
# 3.2.1 print accuracy/snr figure
accracyMode()
# 3.2.2 print confusion matrix
confusionMode()


print('end')