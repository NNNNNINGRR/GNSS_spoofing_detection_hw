# -*- coding: utf-8 -*-
"""
Cusum_SingleVar_Online —— 累积和控制图 CUSUM（单变量，在线/小偏移）

适用条件:
    - **单变量**在线监控，擅长捕捉**持续的小幅偏移**（均值漂移）；
    - 需要逐个新点即时更新，不依赖离线批处理。

数据要求:
    - 二维数组 [N, M]（逐列独立，分数取平均）；训练集为正常基线。

如何使用:
    python tools/run_traditional.py --task anomaly --method Cusum_SingleVar_Online \
        --data PSM --anomaly_ratio 1
"""
import numpy as np


class Cusum_SingleVar_Online:
    def __init__(self, k=0.5, h=5.0):
        self.k = float(k)      # 允许漂移（单位：标准差）
        self.h = float(h)      # 报警阈值（单位：标准差）
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
        s_plus = np.zeros(z.shape[0] + 1)
        s_minus = np.zeros(z.shape[0] + 1)
        for i in range(z.shape[0]):
            s_plus[i + 1] = max(0.0, s_plus[i] + z[i].mean() - self.k)
            s_minus[i + 1] = max(0.0, s_minus[i] - z[i].mean() - self.k)
        return np.maximum(s_plus[1:], s_minus[1:]) / self.h
