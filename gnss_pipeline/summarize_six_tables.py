# -*- coding: utf-8 -*-
"""按 方法×版本 生成 6 张全场景结果表（markdown）。"""
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

DIR = r"D:\文献复现\SQM数据集制作\exp_gnss\results\full"
MODELS = ["Bi_FI", "LightTS", "DLinear"]
SCENS = [f"ds{i}" for i in range(1, 9)]
VER = {"v1": "V1（从头训练）", "v2": "V2（预训练+微调）"}
METRICS = ["roc_auc", "pr_auc", "tpr@fpr1", "tpr@fpr5", "f1", "mcc", "bal_acc"]

rows = {}
for f in glob.glob(os.path.join(DIR, "*_metrics.csv")):
    base = os.path.basename(f)
    parts = base[: -len("_metrics.csv")].split("_")
    ver, scen = parts[0], parts[-1]
    model = "_".join(parts[1:-1])
    rows[(ver, model, scen)] = pd.read_csv(f).iloc[0]

lines = [
    "# GNSS 欺骗检测全量结果（六表：方法 × 版本）",
    "",
    "- 评估口径：绝对时间对齐；阈值=正常训练分数 99 分位；ADD=连续 3 帧告警首报延迟（inf=未命中）",
    "- F1/MCC/BalAcc 为固定阈值口径；TPR@FPR 为固定虚警率协议（更公平的模型对比）",
    "",
]
for ver in ["v1", "v2"]:
    for model in MODELS:
        lines.append(f"## {model} {VER[ver]}")
        lines.append("")
        lines.append("| 场景 | ROC-AUC | PR-AUC | TPR@FPR1% | TPR@FPR5% | F1 | MCC | BalAcc | ADD(s) | 命中 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        acc = {m: [] for m in METRICS}
        adds, hits = [], []
        for s in SCENS:
            r = rows[(ver, model, s)]
            add = r["add_s"]
            adds.append(np.nan if add == "inf" else float(add))
            hits.append(int(r["hit"]))
            for m in METRICS:
                acc[m].append(float(r[m]))
            lines.append(
                f"| {s} | {float(r['roc_auc']):.4f} | {float(r['pr_auc']):.4f} | "
                f"{float(r['tpr@fpr1']):.4f} | {float(r['tpr@fpr5']):.4f} | "
                f"{float(r['f1']):.4f} | {float(r['mcc']):.4f} | {float(r['bal_acc']):.4f} | "
                f"{add} | {int(r['hit'])} |")
        add_mean = np.nanmean(adds) if adds else np.nan
        add_s = "inf" if np.isnan(add_mean) else f"{add_mean:.1f}"
        lines.append(
            f"| **宏平均** | {np.mean(acc['roc_auc']):.4f} | {np.mean(acc['pr_auc']):.4f} | "
            f"{np.mean(acc['tpr@fpr1']):.4f} | {np.mean(acc['tpr@fpr5']):.4f} | "
            f"{np.mean(acc['f1']):.4f} | {np.mean(acc['mcc']):.4f} | {np.mean(acc['bal_acc']):.4f} | "
            f"{add_s} | {sum(hits)}/8 |")
        lines.append("")

# 输出固定在 DIR（代码内常量）目录内
out = Path(DIR) / "results_six_tables.md"
if out.resolve().parent != Path(DIR).resolve():
    raise ValueError("output path escapes results dir")
with out.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
