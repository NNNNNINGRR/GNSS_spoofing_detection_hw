# -*- coding: utf-8 -*-
"""冒烟数据子集：围绕欺骗起始帧取 n 帧（保证正常/欺骗两类都在）。
用法: python subset_smoke.py <Test.csv> <Test_label.csv> <n_frames> <out_test> <out_label>
"""
import sys
import pandas as pd

test = pd.read_csv(sys.argv[1])
lab = pd.read_csv(sys.argv[2])
n = int(sys.argv[3])
y = lab.iloc[:, 1].values
first1 = int((y == 1).argmax()) if (y == 1).any() else len(y) // 2
start = max(0, first1 - n // 2)
end = min(len(test), start + n)
test.iloc[start:end].to_csv(sys.argv[4], index=False)
lab.iloc[start:end].to_csv(sys.argv[5], index=False)
print(f"subset: rows {start}:{end} ({end - start}), first_spoof_idx={first1}, "
      f"spoof_ratio={float(y[start:end].mean()):.3f}")
