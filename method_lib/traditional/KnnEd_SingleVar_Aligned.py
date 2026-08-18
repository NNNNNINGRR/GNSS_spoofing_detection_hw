# -*- coding: utf-8 -*-
"""
KnnEd_SingleVar_Aligned —— 1 近邻 + 欧氏距离（单变量，已对齐等长）

适用条件:
    - **单变量**分类，且所有样本**等长、已对齐**（无时间错位）；
    - 样本量小到中等；比 DTW 快得多，适合作为快速基线。

数据要求:
    - 每个样本一维数组且长度全部相同；标签整数数组。

如何使用:
    python tools/run_traditional.py --task classification --method KnnEd_SingleVar_Aligned \
        --data UEA --dataset UWaveGestureLibrary
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KnnEd_SingleVar_Aligned:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = int(n_neighbors)
        self.clf = None

    def fit(self, Xs, y):
        X = np.array([np.asarray(x, dtype=np.float64) for x in Xs])   # [N, T]
        self.clf = KNeighborsClassifier(n_neighbors=self.n_neighbors, metric="euclidean")
        self.clf.fit(X, np.asarray(y))
        return self

    def predict(self, Xs):
        X = np.array([np.asarray(x, dtype=np.float64) for x in Xs])
        return self.clf.predict(X)
