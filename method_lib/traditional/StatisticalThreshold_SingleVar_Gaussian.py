# -*- coding: utf-8 -*-
"""
StatisticalThreshold_SingleVar_Gaussian —— 统计阈值法（3σ / IQR，单变量）

适用条件:
    - **单变量**序列近似正态、无强趋势（有趋势请先差分或去趋势）；
    - 快速粗糙检测、可解释；对小幅漂移不敏感。

数据要求:
    - 二维数组 [N, M]（M 列会逐列独立检测，分数取各列平均）；
    - 训练集应为正常数据。

如何使用:
    python tools/run_traditional.py --task anomaly --method StatisticalThreshold_SingleVar_Gaussian \
        --data PSM --anomaly_ratio 1
"""
import numpy as np


class StatisticalThreshold_SingleVar_Gaussian:
    def __init__(self, method="zscore", k=3.0):
        self.method = method          # zscore 或 iqr
        self.k = float(k)
        self.mean_ = None
        self.std_ = None
        self.q1_ = None
        self.q3_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.method == "zscore":
            s = np.abs((X - self.mean_) / self.std_)
        else:
            iqr = (self.q3_ - self.q1_) + 1e-9
            s = np.maximum((self.q1_ - X) / iqr, (X - self.q3_) / iqr)
        return s.mean(axis=1) if X.ndim == 2 else s
