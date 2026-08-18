# -*- coding: utf-8 -*-
"""
HotellingT2_MultiVar_ProcessMonitoring —— Hotelling T² + SPE（多变量，过程监控）

适用条件:
    - **多变量**工业/过程监控的标准方法；
    - 结合主成分空间内的 T²（均值/相关结构偏移）与残差空间 SPE（新模式出现）；
    - 近似多元正态假设。

数据要求:
    - 二维数组 [N, M]；训练集为正常数据。

如何使用:
    python tools/run_traditional.py --task anomaly --method HotellingT2_MultiVar_ProcessMonitoring \
        --data MSL --anomaly_ratio 1 --pca_components 0.9
"""
import numpy as np
from sklearn.decomposition import PCA


class HotellingT2_MultiVar_ProcessMonitoring:
    def __init__(self, components=0.9):
        self.components = components
        self.pca = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        Z = (X - self.mean_) / self.std_
        self.pca = PCA(n_components=self.components)
        self.pca.fit(Z)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.std_
        T = self.pca.transform(Z)
        rec = self.pca.inverse_transform(T)
        t2 = np.sum((T / (np.std(T, axis=0) + 1e-9)) ** 2, axis=1)
        spe = np.linalg.norm(Z - rec, axis=1)
        return t2 / (np.max(t2) + 1e-9) + spe / (np.max(spe) + 1e-9)
