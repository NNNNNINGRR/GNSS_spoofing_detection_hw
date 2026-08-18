# -*- coding: utf-8 -*-
"""
NaiveSeasonal_SingleVar_Periodic —— 季节朴素法（取上一周期同时刻的值）

适用条件:
    - 单变量序列，存在**已知且稳定的周期**（如 24 小时、7 天）；
    - 作为“有周期数据”的最低成本基线，常用于与 SARIMA/Holt-Winters 对照；
    - 周期长度必须由用户给出（season 参数），序列无需平稳。

数据要求:
    - 一维浮点序列 [T,]；必须告诉周期长度（如 hourly 数据 season=24）。

如何使用:
    python tools/run_traditional.py --task forecast --method NaiveSeasonal_SingleVar_Periodic \
        --data ETTh1 --target OT --seq_len 96 --pred_len 96 --season 24
"""
import numpy as np


class NaiveSeasonal_SingleVar_Periodic:
    def __init__(self, season=24):
        self.season = int(season)

    def fit(self, y):
        return self

    def forecast(self, history, steps):
        """把最近一个周期的值复制到未来。"""
        s = min(self.season, len(history))
        pattern = history[-s:]
        reps = int(np.ceil(steps / s))
        pred = np.tile(pattern, reps)[:steps]
        return np.asarray(pred, dtype=np.float64)
