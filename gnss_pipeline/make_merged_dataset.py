# -*- coding: utf-8 -*-
"""构建 v3.1/v3.0 两版特征并排列齐的 cs+cd 数据集（融合实验用）。

行对齐：每场景取 v3.1 指标的最高 CN0 通道，同一通道的 v3.0 指标按 time_s 内连接
（两次解算的历元几乎一致，交集 >99%）。v3.0 的 received_power/CN0 与 v3.1 冗余，不重复保留。

输出 anomaly_cscd_merged/：date + 13 列
  v31 组（8）：m_ratio, m_delta, m_elp, m_symdiff, m_manfredini, m_dd, received_power, CN0
  v30 组（5）：m_ratio_e025, m_delta_e025, m_elp_e025, m_symdiff_e025, m_manfredini_e025
+ Train/Test/Test_label/manifest.json（与 make_datasets cs+cd 口径一致）。
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from common import SCENARIOS, CLEAN_SCENARIOS, SPOOF_SCENARIOS  # noqa: E402
from make_datasets import ATTACK_ONSET_S, DYNAMIC_SCENARIOS  # noqa: E402

V31_COLS = ["m_ratio", "m_delta", "m_elp", "m_symdiff", "m_manfredini", "m_dd", "received_power", "CN0"]
V30_COLS = ["m_ratio", "m_delta", "m_elp", "m_symdiff", "m_manfredini", "received_power", "CN0"]
V30_RENAMED = [c + "_e025" for c in V30_COLS]


def best_channel(metrics_dir, s):
    best_ch, best = None, -1.0
    for ch in range(8):
        p = os.path.join(metrics_dir, f"{s}_ch{ch}.csv")
        if not os.path.exists(p):
            continue
        cn0 = pd.read_csv(p, usecols=["CN0"])["CN0"].mean()
        if cn0 > best:
            best, best_ch = cn0, ch
    return best_ch


def scenario_frame(m31, m30, s):
    """v3.1/v3.0 各取本版最高 CN0 通道（可为不同卫星），按最近邻时间对齐行。

    融合的是检测器分数而非原始特征，跨通道融合物理上成立；
    各版本保留自己的 received_power/CN0（共 15 列）。"""
    c31, c30 = best_channel(m31, s), best_channel(m30, s)
    a = pd.read_csv(os.path.join(m31, f"{s}_ch{c31}.csv"))[["time_s"] + V31_COLS]
    b = pd.read_csv(os.path.join(m30, f"{s}_ch{c30}.csv"))[["time_s"] + V30_COLS]
    b.columns = ["time_s"] + V30_RENAMED
    # time_s 网格对齐（20 ms，取整到 ms 防浮点抖动）
    # 两次解算的起始采样偏移有毫秒级差异，按最近邻对齐（±15 ms，同一 20 ms 历元）
    a = a.sort_values("time_s").reset_index(drop=True)
    b = b.sort_values("time_s").reset_index(drop=True)
    m = pd.merge_asof(a, b[["time_s"] + V30_RENAMED], on="time_s",
                      direction="nearest", tolerance=0.015).dropna(subset=V30_RENAMED)
    time_s = m["time_s"].values
    df = m[V31_COLS + V30_RENAMED].astype(np.float64)
    df.insert(0, "date", pd.to_datetime(time_s, unit="s", origin="1970-01-01"))
    return df, time_s, (int(c31), int(c30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m31", required=True, help="v3.1 metrics 目录")
    ap.add_argument("--m30", required=True, help="v3.0 metrics 目录")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # 输出路径规范化：仅允许在 --out 目录下写固定文件名（防路径穿越）
    OUT = Path(args.out).resolve()

    def out_path(name):
        if name != os.path.basename(name) or ".." in name:
            raise ValueError("invalid output filename")
        return OUT / name

    os.makedirs(OUT, exist_ok=True)
    manifest = {"features": V31_COLS + V30_RENAMED, "train": {}, "test": []}

    tr_parts = []
    for clean in ["cs", "cd"]:
        df, _, ch = scenario_frame(args.m31, args.m30, clean)
        tr_parts.append(df)
        manifest["train"][clean] = {"rows": len(df), "channel": ch}
    tr = pd.concat(tr_parts, ignore_index=True)

    te_parts, lab_parts = [], []
    start = 0
    for s in SPOOF_SCENARIOS:
        df, time_s, ch = scenario_frame(args.m31, args.m30, s)
        onset = ATTACK_ONSET_S[s]
        lab = (time_s >= onset).astype(int)
        te_parts.append(df)
        lab_parts.append(pd.DataFrame({"date": df["date"], "label": lab}))
        manifest["test"].append({"scenario": s, "start": start, "rows": len(df),
                                 "onset_s": onset,
                                 "type": "dynamic" if s in DYNAMIC_SCENARIOS else "static",
                                 "channel": ch})
        start += len(df)
    te = pd.concat(te_parts, ignore_index=True)
    lb = pd.concat(lab_parts, ignore_index=True)

    tr.to_csv(out_path("Train.csv"), index=False)
    te.to_csv(out_path("Test.csv"), index=False)
    lb.to_csv(out_path("Test_label.csv"), index=False)
    with out_path("manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"merged: Train={len(tr)} Test={len(te)} 15 列; "
          f"通道: train={manifest['train']} ; 欺骗帧={int(lb.label.sum())} ({lb.label.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
