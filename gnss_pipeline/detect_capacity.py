# -*- coding: utf-8 -*-
"""用已有 metrics 分析各模型/版本对全部欺骗的检测能力（frame 级 TPR@FPR）。"""
import os
import glob
import numpy as np
import pandas as pd

DIR = r"D:\文献复现\SQM数据集制作\exp_gnss\results\full"
SCENS = [f"ds{i}" for i in range(1, 9)]

rows = []
for f in glob.glob(os.path.join(DIR, "*_metrics.csv")):
    base = os.path.basename(f)
    parts = base[: -len("_metrics.csv")].split("_")
    ver, scen = parts[0], parts[-1]
    model = "_".join(parts[1:-1])
    r = pd.read_csv(f).iloc[0]
    rows.append({
        "ver": ver, "model": model, "scen": scen,
        "tpr1": float(r["tpr@fpr1"]), "tpr5": float(r["tpr@fpr5"]),
        "auc": float(r["roc_auc"]),
    })
df = pd.DataFrame(rows)

print("== 各 模型×版本：8 个场景中 frame 级 TPR 达标数 ==")
print(f"{'模型':<12}{'版本':<4}{'TPR5>=0.99':<11}{'TPR5>=0.9':<11}{'TPR1>=0.9':<11}{'平均AUC'}")
for model in sorted(df["model"].unique()):
    for ver in ["v1", "v2"]:
        g = df[(df.model == model) & (df.ver == ver)]
        if len(g) == 0:
            continue
        print(f"{model:<12}{ver:<4}{int((g.tpr5 >= 0.99).sum()):<11}"
              f"{int((g.tpr5 >= 0.9).sum()):<11}{int((g.tpr1 >= 0.9).sum()):<11}"
              f"{g.auc.mean():.3f}")

print("\n== 各场景：TPR@FPR5% 最高的 模型×版本 ==")
for s in SCENS:
    g = df[df.scen == s].sort_values("tpr5", ascending=False)
    best = g.iloc[0]
    full = "全检(>=0.99)" if best.tpr5 >= 0.99 else (f"高({best.tpr5:.2f})" if best.tpr5 >= 0.9 else f"{best.tpr5:.2f}")
    print(f"{s}: {best.model}-{best.ver}  TPR@FPR5%={best.tpr5:.4f} [{full}]")
