# -*- coding: utf-8 -*-
"""
Var_MultiVar_LinearInteractions —— 向量自回归 VAR（多变量）

适用条件:
    - **多变量**序列，变量之间存在线性交互（一个变量的滞后影响另一个）；
    - 变量数少（一般 <10，否则参数爆炸）；序列平稳或经差分后平稳；
    - 需要解释“谁影响谁”（Granger 因果/脉冲响应）的场景。

数据要求:
    - 二维浮点数组 [T, M]（T 个时刻、M 个变量），不含时间戳列；
    - 不允许缺失值。

如何使用:
    python tools/run_traditional.py --task forecast --method Var_MultiVar_LinearInteractions \
        --data ETTh1 --seq_len 96 --pred_len 96 --var_maxlags 2
"""
import numpy as np
from statsmodels.tsa.api import VAR


class Var_MultiVar_LinearInteractions:
    def __init__(self, maxlags=2):
        self.maxlags = int(maxlags)
        self.model = None

    def fit(self, Y):
        self.model = VAR(np.asarray(Y, dtype=np.float64)).fit(maxlags=self.maxlags)
        return self

    def forecast(self, history, steps):
        Y = np.asarray(history, dtype=np.float64)
        return self.model.forecast(Y, steps=steps)
