# -*- coding: utf-8 -*-
"""
FeatureRf_MultiVar_ChannelFusion —— 多通道特征拼接 + 随机森林（多变量，中样本）

适用条件:
    - **多变量**分类，样本量中等；各通道特征共同决定类别；
    - 需要可解释、训练快；变量间复杂非线性交互交给树模型；
    - 通道数较多时尤为适合（特征拼接后随机森林自动选重要特征）。

数据要求:
    - 每个样本二维数组 [T_i, M]；标签整数数组。

如何使用:
    python tools/run_traditional.py --task classification --method FeatureRf_MultiVar_ChannelFusion \
        --data UEA --dataset Heartbeat
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class FeatureRf_MultiVar_ChannelFusion:
    def __init__(self, n_estimators=200, max_len=100, seed=0):
        self.n_estimators = int(n_estimators)
        self.max_len = int(max_len)
        self.seed = int(seed)
        self.clf = None

    @staticmethod
    def _resample(x, length):
        x = np.asarray(x, dtype=np.float64)
        if len(x) == length:
            return x
        idx = np.linspace(0, len(x) - 1, length).astype(int)
        return x[idx]

    @staticmethod
    def _features_1d(x):
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        f = [float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)),
             float(np.median(x)), float(np.sum(np.abs(np.diff(x)))),
             float(np.sum(x ** 2))]
        if n >= 8:
            f += list(np.abs(np.fft.rfft(x))[:4])
        else:
            f += [0.0, 0.0, 0.0, 0.0]
        return f

    def _feat(self, x):
        x = self._resample(x, self.max_len)
        out = []
        for m in range(x.shape[1]):
            out += self._features_1d(x[:, m])
        return out

    def fit(self, Xs, y):
        F = [self._feat(x) for x in Xs]
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1)
        self.clf.fit(np.array(F), np.asarray(y))
        return self

    def predict(self, Xs):
        F = [self._feat(x) for x in Xs]
        return self.clf.predict(np.array(F))
