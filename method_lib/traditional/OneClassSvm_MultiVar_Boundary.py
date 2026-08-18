# -*- coding: utf-8 -*-
"""
OneClassSvm_MultiVar_Boundary —— 一类支持向量机（多变量，非线性边界）

适用条件:
    - **多变量**、正常数据在特征空间中可被一个（核）区域包围；
    - 适合小到中等样本；对高维数据核方法更灵活但更慢；
    - 需要把“正常”定义成边界而非密度。

数据要求:
    - 二维数组 [N, M]；训练集基本为正常数据。

如何使用:
    python tools/run_traditional.py --task anomaly --method OneClassSvm_MultiVar_Boundary \
        --data MSL --anomaly_ratio 1
"""
import numpy as np
from sklearn.svm import OneClassSVM


class OneClassSvm_MultiVar_Boundary:
    def __init__(self, nu=0.01, kernel="rbf", gamma="scale"):
        self.nu = float(nu)
        self.kernel = kernel
        self.gamma = gamma
        self.clf = None

    def fit(self, X):
        self.clf = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.clf.fit(np.asarray(X, dtype=np.float64))
        return self

    def score(self, X):
        # decision_function 越大越正常；取负使分数越大越异常
        return -self.clf.decision_function(np.asarray(X, dtype=np.float64))
