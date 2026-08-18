# -*- coding: utf-8 -*-
"""
Sarima_SingleVar_Seasonal —— SARIMA（带季节项的 ARIMA）

适用条件:
    - 单变量序列，**同时存在平稳线性依赖与稳定季节周期**；
    - 样本量需要覆盖至少 2~3 个季节周期；
    - 比 Holt-Winters 更灵活（可显式建模季节自回归/移动平均）。

数据要求:
    - 一维浮点序列 [T,]；指定季节周期长度（season，如 24/7/12）。

如何使用:
    python tools/run_traditional.py --task forecast --method Sarima_SingleVar_Seasonal \
        --data ETTh1 --target OT --seq_len 96 --pred_len 96 --season 24
"""
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Sarima_SingleVar_Seasonal:
    def __init__(self, season=24, order=(1, 1, 1), seasonal_order=(1, 1, 1, 0)):
        self.season = int(season)
        self.order = tuple(order) if not isinstance(order, str) else tuple(int(x) for x in order.split(","))
        if isinstance(seasonal_order, str):
            seasonal_order = tuple(int(x) for x in seasonal_order.split(","))
        so = tuple(seasonal_order)
        if len(so) == 3:
            so = so + (self.season,)
        elif len(so) == 4 and so[3] == 0:
            so = so[:3] + (self.season,)     # 季节周期为 0 时用 season 补全
        self.seasonal_order = so
        self.model = None

    def fit(self, y):
        self.model = SARIMAX(
            np.asarray(y, dtype=np.float64),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False).fit(disp=False)
        return self

    def forecast(self, history, steps):
        fitted = self.model.apply(np.asarray(history, dtype=np.float64))
        return fitted.forecast(steps)
