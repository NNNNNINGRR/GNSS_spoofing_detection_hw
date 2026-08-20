# -*- coding: utf-8 -*-
"""冒烟数据准备：
1) SQM_ds1：Train=前 3000 帧；Test/Test_label=围绕 onset 取 2000 帧（两类都有）
2) ETTh1：前 2000 帧转 PSM 格式（Train/Test/Test_label），7 通道与 TEXBAT 一致
"""
import sys
import pandas as pd

data_root, smoke = sys.argv[1], sys.argv[2]
onset_ds1 = 125.0
n = 2000

# 1) SQM_ds1
tr = pd.read_csv(f"{data_root}/SQM_ds1/Train.csv")
te = pd.read_csv(f"{data_root}/SQM_ds1/Test.csv")
lb = pd.read_csv(f"{data_root}/SQM_ds1/Test_label.csv")
tr.iloc[:3000].to_csv(f"{smoke}/SQM_ds1/Train.csv", index=False)
y = lb.iloc[:, 1].values
first1 = int((y == 1).argmax()) if (y == 1).any() else len(y) // 2
start = max(0, first1 - n // 2)
end = min(len(te), start + n)
te.iloc[start:end].to_csv(f"{smoke}/SQM_ds1/Test.csv", index=False)
lb.iloc[start:end].to_csv(f"{smoke}/SQM_ds1/Test_label.csv", index=False)
print(f"SQM_ds1 smoke: rows {start}:{end}, spoof_ratio={float(y[start:end].mean()):.3f}")

# 2) ETTh1 -> PSM 格式（7 通道）
ett = pd.read_csv("/root/autodl-tmp/时间序列方法库/dataset/ETT-small/ETTh1.csv")
train = ett.iloc[:2000].copy()
test = ett.iloc[2000:4000].copy()
lab_ett = pd.DataFrame({"date": test["date"], "label": 0})
train.to_csv(f"{smoke}/ETTh1/Train.csv", index=False)
test.to_csv(f"{smoke}/ETTh1/Test.csv", index=False)
lab_ett.to_csv(f"{smoke}/ETTh1/Test_label.csv", index=False)
print(f"ETTh1 smoke: train={len(train)}, test={len(test)}, channels={len(ett.columns)-1}")
