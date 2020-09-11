import numpy as np
import pickle
import random


class DataSet(object):
    def __init__(self, path, unit_num=1000, rate=0.8, seed=2019):
        self.path = path
        self.unit_num = unit_num
        self.rate = rate
        self.seed = seed

        self.loadDataSet(path, self.unit_num)
        self.splitData(rate, seed)

    # public methods
    def getX(self):
        return self.X, self.lbl, self.snrs, self.mods
    def getTrainAndTest(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test, self.mods
    def getTrainIndex(self):
        return self.train_idx
    def getTestIndex(self):
        return self.test_idx

    def loadDataSet(self, path, unit_num):
        # 1. load dataset
        Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')
        # print(Xd.keys())
        # print(len(set(map(lambda x: x[0], Xd.keys()))))

        # snrs(20) = -20 -> 18  mods(11) = ['8PSK', 'AM-DSB', ...]
        self.snrs, self.mods = map(
            lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

        self.X = []
        self.lbl = []

        for mod in self.mods:
            for snr in self.snrs:
                # 1. all samples
                # X.append(Xd[(mod, snr)])
                # for i in range(Xd[(mod, snr)].shape[0]):
                #     lbl.append((mod, snr))

                # 2. only unit_num samples
                for i in range(unit_num):
                    tmp1 = Xd[(mod, snr)][i:i+1]
                    # print(tmp1.shape)
                    tmp2 = Xd[(mod, snr)][i:i+1]
                    tmp3 = Xd[(mod, snr)][i:i+1]
                    tmp4 = Xd[(mod, snr)][i:i+1]
                    tmp = np.vstack((tmp1, tmp2, tmp3, tmp4))
                    # print(tmp.shape)
                    self.X.append(tmp)
                # self.X.append(Xd[(mod, snr)][0:(unit_num)])
                # for i in range(unit_num):
                    self.lbl.append((mod, snr))
        # print(len(self.X[0][0]))
        # self.X = np.vstack(self.X)
        self.X = np.stack(self.X)
        # print(len(self.X))
        # print(len(self.X[0]))
        # print(len(self.X[0][0]))

    def to_onehot(self, yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    def splitData(self, rate = 0.8, seed=2019):
        np.random.seed(seed)
        n_examples = self.X.shape[0]
        n_train = int(n_examples * rate)
        self.train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
        self.test_idx = list(set(range(0, n_examples)) - set(self.train_idx))
        random.shuffle(self.test_idx)
        self.X_train = self.X[self.train_idx]
        # print(self.X_train.shape)
        self.X_test = self.X[self.test_idx]

        self.Y_train = self.to_onehot(list(map(lambda x: self.mods.index(self.lbl[x][0]), self.train_idx)))
        self.Y_test = self.to_onehot(list(map(lambda x: self.mods.index(self.lbl[x][0]), self.test_idx)))

        # type(X_train.shape[1:])
        # in_shp = list(X_train.shape[1:])
        # print(X_train.shape, in_shp)
        classes = self.mods

        return self.X_train, self.Y_train, self.X_test, self.Y_test, classes
