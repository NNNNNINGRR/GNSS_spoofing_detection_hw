# -*- coding: utf-8 -*-
"""
IsolationForest_MultiVar_NonGaussian —— 孤立森林（多变量，非高斯/非线性）

适用条件:
    - **多变量**、分布复杂/非高斯、维度较高；
    - 异常“少而不同”，不需要分布假设；
    - 训练快、对大数据友好（随机森林式抽样）。

数据要求:
    - 二维数组 [N, M]；训练集为正常数据（可含少量噪声）。

如何使用:
    python tools/run_traditional.py --task anomaly --method IsolationForest_MultiVar_NonGaussian \
        --data MSL --anomaly_ratio 1
"""
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForest_MultiVar_NonGaussian:
    def __init__(self, contamination=0.01, n_estimators=200, seed=0):
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.clf = None

    def fit(self, X):
        self.clf = IsolationForest(
            n_estimators=self.n_estimators, contamination=self.contamination,
            random_state=self.seed, n_jobs=-1)
        self.clf.fit(np.asarray(X, dtype=np.float64))
        return self

    def score(self, X):
        # score_samples 越大越正常，取负使其“越大越异常”
        return -self.clf.score_samples(np.asarray(X, dtype=np.float64))
