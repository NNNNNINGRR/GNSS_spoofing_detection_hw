# -*- coding: utf-8 -*-
"""
PcaReconstruction_MultiVar_Correlated —— PCA 重构误差（多变量，强相关/高维）

适用条件:
    - **多变量**且变量间强相关、维度较高；
    - 异常表现为“不符合主成分结构的点”（重构误差大）；
    - 可同时用于降维可视化。

数据要求:
    - 二维数组 [N, M]；训练集为正常数据；建议先标准化（内部自动做）。

如何使用:
    python tools/run_traditional.py --task anomaly --method PcaReconstruction_MultiVar_Correlated \
        --data MSL --anomaly_ratio 1 --pca_components 0.9
"""
import numpy as np
from sklearn.decomposition import PCA


class PcaReconstruction_MultiVar_Correlated:
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
        rec = self.pca.inverse_transform(self.pca.transform(Z))
        return np.linalg.norm(Z - rec, axis=1)
