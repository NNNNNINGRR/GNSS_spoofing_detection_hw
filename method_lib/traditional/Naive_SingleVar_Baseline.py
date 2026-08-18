# -*- coding: utf-8 -*-
"""
Naive_SingleVar_Baseline —— 朴素法（上一时刻值）

适用条件:
    - 单变量（一元）时间序列，仅作为最基础的下界基线；
    - 序列无明显趋势/季节，或只想快速获得一个“不可能更差”的对照；
    - 对精度要求低、计算预算几乎为零的场景。

数据要求:
    - 一维浮点序列（numpy 数组 [T,]），按时间升序排列；
    - 不需要时间戳、不需要标准化。

如何使用:
    python tools/run_traditional.py --task forecast --method Naive_SingleVar_Baseline \
        --data ETTh1 --target OT --seq_len 96 --pred_len 96 --num_windows 10
"""
import numpy as np


class Naive_SingleVar_Baseline:
    """预测 = 上一个已知值（随机游走假设）。"""

    def __init__(self):
        self.last = None

    def fit(self, y):
        """y: [T,] 训练序列，只需记住最后一个值。"""
        self.last = float(y[-1])
        return self

    def forecast(self, history, steps):
        """history: [T,] 窗口内历史值；返回 [steps,] 全为最后一个值的预测。"""
        base = float(history[-1])
        return np.full(steps, base, dtype=np.float64)
