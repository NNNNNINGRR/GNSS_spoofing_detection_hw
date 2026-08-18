# -*- coding: utf-8 -*-
"""
Lof_MultiVar_LocalDensity —— 局部离群因子 LOF（多变量，局部密度）

适用条件:
    - **多变量**、密度不均匀（存在多个正常簇）的数据；
    - 异常表现为“局部密度远低于邻居”；
    - 样本量中等（LOF 距离计算开销随 N 增大）。

数据要求:
    - 二维数组 [N, M]；训练集为正常数据；建议标准化。

如何使用:
    python tools/run_traditional.py --task anomaly --method Lof_MultiVar_LocalDensity \
        --data MSL --anomaly_ratio 1
"""
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class Lof_MultiVar_LocalDensity:
    def __init__(self, contamination=0.01, n_neighbors=20):
        self.contamination = float(contamination)
        self.n_neighbors = int(n_neighbors)
        self.clf = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        Z = (X - self.mean_) / self.std_
        self.clf = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, contamination=self.contamination, novelty=True)
        self.clf.fit(Z)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.std_
        # decision_function 越大越正常；取负使分数越大越异常
        return -self.clf.decision_function(Z)
