# -*- coding: utf-8 -*-
"""传统方法测试集检测结果可视化：逐场景分数 + 欺骗真值段 + 清洁阈值 + 告警标记。

每方法一张图（8 场景子图）：
  灰线  = 异常分数（逐帧，50 Hz）
  绿区  = 欺骗真值（onset 之后）
  红点  = 实际告警帧（分数 ≥ 清洁标定 FPR1% 阈值）
  虚线  = 三档清洁标定阈值（0.1%/1%/5%）
  标注  = 首报延迟 ADD（连续 3 帧告警）与场景内实际 FPR / TPR

All text in English (cloud servers may lack CJK fonts).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCEN_DESC = {
    "ds1": "simple, static (high power)", "ds2": "intermediate, static (matched power)",
    "ds3": "intermediate, static (position push 0.4dB)", "ds4": "simple, dynamic",
    "ds5": "intermediate, dynamic", "ds6": "sophisticated, static",
    "ds7": "time-adjust, static", "ds8": "time-adjust, static",
}


def first_alarm(t, alarms, onset, k=3):
    run = 0
    for i, h in enumerate(alarms):
        if t[i] < onset:
            continue
        run = run + 1 if h else 0
        if run >= k:
            return t[i] - onset
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--method_dir", required=True, help="含 score/thresholds/label npy 的目录")
    ap.add_argument("--name", required=True, help="图标题/文件名用方法名")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    score = np.load(os.path.join(args.method_dir, "score.npy")).ravel()
    lab = np.load(os.path.join(args.method_dir, "label.npy")).ravel().astype(int)
    thr = np.load(os.path.join(args.method_dir, "thresholds.npy")).ravel()
    with open(os.path.join(args.data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    date = pd.to_datetime(pd.read_csv(os.path.join(args.data_dir, "Test.csv"))["date"])
    epoch = (date - pd.Timestamp("1970-01-01")).dt.total_seconds().values

    n = min(len(score), len(lab), len(epoch))
    score, lab, epoch = score[:n], lab[:n], epoch[:n]
    alarm = score >= thr[0]          # 清洁标定 FPR1% 工作点（thresholds.npy 顺序：1%/5%/10%）
    t_lo = np.percentile(score, 0.5)
    t_hi = min(float(thr[0]) * 1.3, np.percentile(score, 99.9) * 1.1)  # 限幅防长尾压扁曲线

    blocks = [b for b in manifest["test"] if b["start"] + b["rows"] <= n]
    ncol = 2
    nrow = (len(blocks) + 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.7 * nrow), squeeze=False)
    for i, b in enumerate(blocks):
        ax = axes[i // ncol][i % ncol]
        s, e = b["start"], b["start"] + b["rows"]
        t = epoch[s:e] - epoch[s]
        sc = np.minimum(score[s:e], t_hi)
        onset = b["onset_s"]
        ax.fill_between(t, t_lo, t_hi, where=(lab[s:e] == 1), color="green", alpha=0.15,
                        label="spoof ground truth")
        ax.plot(t, sc, color="0.45", lw=0.4, zorder=1)
        m = alarm[s:e] & (t >= onset - 5)
        ax.scatter(t[m], sc[m], s=1.2, color="red", zorder=3, label="alarm frames (@FPR1% thr)")
        for lvl, th, c in zip(["1%", "5%", "10%"], thr, ["k", "crimson", "orange"]):
            y = min(float(th), t_hi)
            ax.axhline(y, ls="--", lw=0.8, color=c, label=f"thr@{lvl}={th:.3g}")
        ax.axvline(onset, color="green", ls="-", lw=1.0)
        add = first_alarm(t, alarm[s:e], onset)
        rfpr = alarm[s:e][t < onset].mean() if (t < onset).any() else np.nan
        tpr = alarm[s:e][t >= onset].mean()
        addtxt = f"ADD={add:.2f}s" if np.isfinite(add) else "ADD=miss"
        ax.set_title(f"{b['scenario']} ({SCEN_DESC.get(b['scenario'], '')})  "
                     f"TPR={tpr:.3f}  real-FPR={rfpr:.3f}  {addtxt}", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("anomaly score", fontsize=8)
        ax.set_ylim(t_lo, t_hi)
        if i == 0:
            ax.legend(fontsize=6, loc="upper left", ncol=2)
    for j in range(len(blocks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{args.name}: per-scenario detection on test set "
                 f"(trained on cs+cd clean only, per-frame)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print("saved", args.out)


if __name__ == "__main__":
    main()
