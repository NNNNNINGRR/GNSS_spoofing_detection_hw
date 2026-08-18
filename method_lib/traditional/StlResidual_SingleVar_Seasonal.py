# -*- coding: utf-8 -*-
"""
StlResidual_SingleVar_Seasonal —— STL 分解残差法（单变量，强季节）

适用条件:
    - **单变量**序列有强趋势+稳定季节（如日/周周期）；
    - 异常表现为“去趋势去季节后的尖峰”；需要可解释的分解结果。

数据要求:
    - 二维数组 [N, M]（逐列独立，分数取平均）；序列长度需大于 2 个季节周期；
    - 必须指定周期长度（season，如 24）。

如何使用:
    python tools/run_traditional.py --task anomaly --method StlResidual_SingleVar_Seasonal \
        --data PSM --anomaly_ratio 1 --season 24
"""
import numpy as np
from statsmodels.tsa.seasonal import STL


class StlResidual_SingleVar_Seasonal:
    def __init__(self, season=24):
        self.season = int(season)
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        res = []
        for m in range(X.shape[1]):
            r = STL(X[:, m], period=self.season, robust=True).fit().resid
            res.append(r)
        self.res_train = np.stack(res, axis=1)
        self.std_ = self.res_train.std(axis=0) + 1e-9
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        res = []
        for m in range(X.shape[1]):
            r = STL(X[:, m], period=self.season, robust=True).fit().resid
            res.append(r)
        res = np.stack(res, axis=1)
        return (np.abs(res) / self.std_).mean(axis=1)
