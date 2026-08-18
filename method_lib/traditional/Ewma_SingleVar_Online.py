# -*- coding: utf-8 -*-
"""
Ewma_SingleVar_Online —— 指数加权移动平均控制图 EWMA（单变量，在线）

适用条件:
    - **单变量**在线监控，对小幅漂移比 Shewhart 控制图更敏感；
    - 需要平滑噪声、逐点更新；与 CUSUM 相比更重视“近期”信息。

数据要求:
    - 二维数组 [N, M]（逐列独立，分数取平均）；训练集为正常基线。

如何使用:
    python tools/run_traditional.py --task anomaly --method Ewma_SingleVar_Online \
        --data PSM --anomaly_ratio 1
"""
import numpy as np


class Ewma_SingleVar_Online:
    def __init__(self, lam=0.2):
        self.lam = float(lam)
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = (X - self.mean_) / self.std_
        ewma = np.zeros_like(z)
        acc = np.zeros(z.shape[1])
        for i in range(z.shape[0]):
            acc = self.lam * z[i] + (1 - self.lam) * acc
            ewma[i] = acc
        return np.abs(ewma).mean(axis=1)
