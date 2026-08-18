# -*- coding: utf-8 -*-
"""
ExponentialSmoothing_SingleVar_LevelTrend —— 指数平滑（水平/趋势，无季节）

适用条件:
    - 单变量序列，近似平稳或带缓慢趋势，**没有明显周期**；
    - 数据量中等到大；希望给近期观测更高权重；
    - 可解释、训练极快（statsmodels 实现）。

数据要求:
    - 一维浮点序列 [T,]；无需时间戳；不允许缺失值。

如何使用:
    python tools/run_traditional.py --task forecast --method ExponentialSmoothing_SingleVar_LevelTrend \
        --data ETTh1 --target OT --seq_len 96 --pred_len 96
"""
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ExponentialSmoothing_SingleVar_LevelTrend:
    def __init__(self, trend="add", seasonal=None):
        self.trend = trend
        self.seasonal = seasonal
        self.model = None

    def fit(self, y):
        self.model = ExponentialSmoothing(
            np.asarray(y, dtype=np.float64),
            trend=self.trend, seasonal=self.seasonal,
            initialization_method="estimated").fit(optimized=True)
        return self

    def forecast(self, history, steps):
        # 每个窗口在 history 上重新拟合（简单稳健），再向前预测
        m = ExponentialSmoothing(
            np.asarray(history, dtype=np.float64),
            trend=self.trend, seasonal=self.seasonal,
            initialization_method="estimated").fit(optimized=True)
        return m.forecast(steps)
