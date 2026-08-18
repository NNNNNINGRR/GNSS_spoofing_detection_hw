# -*- coding: utf-8 -*-
"""
FeatureRf_SingleVar_MediumData —— 统计特征 + 随机森林（单变量，中样本）

适用条件:
    - **单变量**分类，样本量中等（几百~几千），序列等长或经重采样对齐；
    - 需要可解释（特征重要性）且对噪声稳健；
    - 比 DTW 快，适合批量实验。

数据要求:
    - 每个样本一维数组（内部会自动补/截断到统一长度 100）；
    - 标签整数数组。

如何使用:
    python tools/run_traditional.py --task classification --method FeatureRf_SingleVar_MediumData \
        --data UEA --dataset UWaveGestureLibrary
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def _features_1d(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    f = [
        float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)),
        float(np.median(x)), float(np.sum(np.abs(np.diff(x)))),
        float(np.count_nonzero(np.diff(np.sign(x))) / max(n - 1, 1)),
        float(np.sum(x ** 2)),
    ]
    if n >= 8:
        spec = np.abs(np.fft.rfft(x))[:4]
        f += list(spec)
    else:
        f += [0.0, 0.0, 0.0, 0.0]
    return f


class FeatureRf_SingleVar_MediumData:
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

    def fit(self, Xs, y):
        F = [self._feat(x) for x in Xs]
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1)
        self.clf.fit(np.array(F), np.asarray(y))
        return self

    def _feat(self, x):
        return _features_1d(self._resample(x, self.max_len))

    def predict(self, Xs):
        F = [self._feat(x) for x in Xs]
        return self.clf.predict(np.array(F))
