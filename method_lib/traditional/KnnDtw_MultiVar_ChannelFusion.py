# -*- coding: utf-8 -*-
"""
KnnDtw_MultiVar_ChannelFusion —— 多通道 DTW 距离融合 + 1 近邻（多变量，小样本）

适用条件:
    - **多变量**分类，各通道长度可不同/存在时间错位；
    - 样本量小到中等；希望直接利用通道间时间形态；
    - 计算量随样本数上升，大数据集慎用。

数据要求:
    - 每个样本二维数组 [T_i, M]（M 为通道数）；标签整数数组。

如何使用:
    python tools/run_traditional.py --task classification --method KnnDtw_MultiVar_ChannelFusion \
        --data UEA --dataset Heartbeat
"""
import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


class KnnDtw_MultiVar_ChannelFusion:
    def __init__(self, n_neighbors=1):
        self.n_neighbors = int(n_neighbors)
        self.clf = None

    def fit(self, Xs, y):
        X = np.array([np.asarray(x, dtype=np.float64) for x in Xs])   # [N, T, M]
        self.clf = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors, metric="dtw")
        self.clf.fit(X, np.asarray(y))
        return self

    def predict(self, Xs):
        X = np.array([np.asarray(x, dtype=np.float64) for x in Xs])
        return self.clf.predict(X)
