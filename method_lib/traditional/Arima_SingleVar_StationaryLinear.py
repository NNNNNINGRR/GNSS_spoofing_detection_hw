# -*- coding: utf-8 -*-
"""
Arima_SingleVar_StationaryLinear —— ARIMA（自回归 + 差分 + 移动平均）

适用条件:
    - 单变量序列，经差分后近似平稳、关系以**线性**为主；
    - 无强季节（有季节请用 Sarima_SingleVar_Seasonal）；
    - 样本量几百以上；需要可解释的系数与置信区间。

数据要求:
    - 一维浮点序列 [T,]；不允许缺失值；无需时间戳（默认为等间隔）。

如何使用:
    python tools/run_traditional.py --task forecast --method Arima_SingleVar_StationaryLinear \
        --data ETTh1 --target OT --seq_len 96 --pred_len 96 --arima_order 2,1,2
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class Arima_SingleVar_StationaryLinear:
    def __init__(self, order=(2, 1, 2)):
        if isinstance(order, str):
            order = tuple(int(x) for x in order.split(","))
        self.order = tuple(order)
        self.model = None

    def fit(self, y):
        self.model = ARIMA(np.asarray(y, dtype=np.float64), order=self.order).fit()
        return self

    def forecast(self, history, steps):
        fitted = self.model.apply(np.asarray(history, dtype=np.float64))
        return fitted.forecast(steps)
