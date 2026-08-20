# -*- coding: utf-8 -*-
"""全量预训练数据准备：ETTh1 / ETTm1 -> PSM 格式（Train 70% / Test 30%，标签全 0）。
7 通道，与 TEXBAT 特征数一致，保证三模型权重可迁移。
"""
import os
import pandas as pd

src = "/root/autodl-tmp/时间序列方法库/dataset/ETT-small"
dst = "/root/autodl-tmp/exp_gnss/data"
for name in ["ETTh1", "ETTm1"]:
    df = pd.read_csv(f"{src}/{name}.csv")
    n_train = int(len(df) * 0.7)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    lab = pd.DataFrame({"date": test["date"], "label": 0})
    out = f"{dst}/{name}_pre"
    os.makedirs(out, exist_ok=True)
    train.to_csv(f"{out}/Train.csv", index=False)
    test.to_csv(f"{out}/Test.csv", index=False)
    lab.to_csv(f"{out}/Test_label.csv", index=False)
    print(f"{name}: train={len(train)}, test={len(test)}, channels={len(df.columns)-1}")
