# -*- coding: utf-8 -*-
"""
KnnDtw_SingleVar_SmallData —— 1 近邻 + 动态时间规整（单变量，小样本）

适用条件:
    - **单变量**时间序列分类；
    - 样本量小（几十~几百），类别边界非线性，序列**长度可不一致**或存在时间错位；
    - 需要强、稳健的经典基线（1NN-DTW 是公认的传统强基线）。

数据要求:
    - 训练/测试各为样本列表：每个样本一维数组 [T_i]（长度可不同）；
    - 标签为整数数组。

如何使用:
    python tools/run_traditional.py --task classification --method KnnDtw_SingleVar_SmallData \
        --data UEA --dataset UWaveGestureLibrary
"""
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


class KnnDtw_SingleVar_SmallData:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = int(n_neighbors)
        self.clf = None

    def fit(self, Xs, y):
        X = np.array([np.asarray(x, dtype=np.float64) for x in Xs])
        X = X.reshape(X.shape[0], X.shape[1], 1)   # [N, T, 1]
        self.clf = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors, metric="dtw")
        self.clf.fit(X, np.asarray(y))
        return self

    def predict(self, Xs):
        X = np.array([np.asarray(x, dtype=np.float64) for x in Xs])
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return self.clf.predict(X)
