# -*- coding: utf-8 -*-
"""
KalmanLocalLevel_SingleVar_Online —— 卡尔曼滤波（局部水平模型）

适用条件:
    - 单变量序列，近似“随机游走 + 噪声”（无强趋势/季节也可用阻尼趋势扩展）；
    - 强调**在线/实时**更新：每来一个新观测即可递推更新状态；
    - 需要预测区间（置信带）的场景。

数据要求:
    - 一维浮点序列 [T,]；等间隔；缺失值请先处理（卡尔曼假设规则采样）。

如何使用:
    python tools/run_traditional.py --task forecast --method KalmanLocalLevel_SingleVar_Online \
        --data ETTh1 --target OT --seq_len 96 --pred_len 96
"""
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents


class KalmanLocalLevel_SingleVar_Online:
    def __init__(self, level="local level"):
        self.level = level
        self.model = None

    def fit(self, y):
        self.model = UnobservedComponents(
            np.asarray(y, dtype=np.float64), level=self.level).fit(disp=False)
        return self

    def forecast(self, history, steps):
        fitted = self.model.apply(np.asarray(history, dtype=np.float64))
        return fitted.forecast(steps)
