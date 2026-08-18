# -*- coding: utf-8 -*-
"""
HoltWinters_SingleVar_Seasonal —— 三次指数平滑（水平 + 趋势 + 季节）

适用条件:
    - 单变量序列，同时存在**趋势与稳定季节周期**（如日/周/年周期）；
    - 数据量中等以上（至少 2 个完整周期），要求周期长度已知；
    - 是“有趋势有季节”场景下最常用的传统基线之一。

数据要求:
    - 一维浮点序列 [T,]；需要指定季节周期长度（season）。

如何使用:
    python tools/run_traditional.py --task forecast --method HoltWinters_SingleVar_Seasonal \
        --data ETTh1 --target OT --seq_len 96 --pred_len 96 --season 24
"""
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltWinters_SingleVar_Seasonal:
    def __init__(self, season=24, seasonal="add"):
        self.season = int(season)
        self.seasonal = seasonal
        self.model = None

    def fit(self, y):
        self.model = ExponentialSmoothing(
            np.asarray(y, dtype=np.float64),
            trend="add", seasonal=self.seasonal,
            seasonal_periods=self.season,
            initialization_method="estimated").fit(optimized=True)
        return self

    def forecast(self, history, steps):
        # 每个窗口在 history 上重新拟合（简单稳健），再向前预测
        m = ExponentialSmoothing(
            np.asarray(history, dtype=np.float64),
            trend="add", seasonal=self.seasonal,
            seasonal_periods=self.season,
            initialization_method="estimated").fit(optimized=True)
        return m.forecast(steps)
