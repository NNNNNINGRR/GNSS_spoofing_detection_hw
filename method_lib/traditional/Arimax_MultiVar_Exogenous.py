# -*- coding: utf-8 -*-
"""
Arimax_MultiVar_Exogenous —— ARIMAX（目标序列 + 外生解释变量）

适用条件:
    - “单目标 + 多解释变量”的多特征场景：目标序列的线性时序依赖 + 外生变量联合建模；
    - 外生变量未来值可合理假设（用最近值近似）或由其他模型给出；
    - 变量间关系以线性为主。

数据要求:
    - 二维浮点数组 [T, M]：**约定最后一列是目标变量，其余列为外生变量**；
    - 不允许缺失值。

如何使用:
    python tools/run_traditional.py --task forecast --method Arimax_MultiVar_Exogenous \
        --data ETTh1 --seq_len 96 --pred_len 96
"""
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


class Arimax_MultiVar_Exogenous:
    def __init__(self, order=(1, 1, 1)):
        self.order = tuple(order) if not isinstance(order, str) else tuple(int(x) for x in order.split(","))
        self.model = None

    def fit(self, Y):
        Y = np.asarray(Y, dtype=np.float64)
        target = Y[:, -1]
        exog = Y[:, :-1]
        self.model = SARIMAX(target, exog=exog, order=self.order,
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False)
        return self

    def forecast(self, history, steps):
        H = np.asarray(history, dtype=np.float64)
        exog_future = np.tile(H[-1, :-1], (steps, 1))   # 用最近外生值近似未来
        fitted = self.model.apply(H[:, -1], exog=H[:, :-1])
        return fitted.forecast(steps, exog=exog_future)
