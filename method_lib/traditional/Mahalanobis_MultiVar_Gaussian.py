# -*- coding: utf-8 -*-
"""
Mahalanobis_MultiVar_Gaussian —— 马氏距离（多变量，近似多元正态）

适用条件:
    - **多变量**点级异常检测，变量间存在相关性且近似多元正态；
    - 维度不宜过高（几十以内）；需要考虑协方差的标准化距离。

数据要求:
    - 二维数组 [N, M]；训练集为正常数据；不允许缺失值。

如何使用:
    python tools/run_traditional.py --task anomaly --method Mahalanobis_MultiVar_Gaussian \
        --data MSL --anomaly_ratio 1
"""
import numpy as np


class Mahalanobis_MultiVar_Gaussian:
    def __init__(self, reg=1e-6):
        self.reg = float(reg)
        self.mean_ = None
        self.cov_inv_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        self.cov_inv_ = np.linalg.pinv(cov + self.reg * np.eye(cov.shape[0]))
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        d = X - self.mean_
        return np.einsum("ni,ij,nj->n", d, self.cov_inv_, d)
