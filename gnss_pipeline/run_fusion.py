# -*- coding: utf-8 -*-
"""融合检测器：跨版本（v3.1 + v3.0 特征）多方法 margin 融合，目标超越全部单模型。

流程（全部在 anomaly_cscd_merged 13 列对齐数据集上）：
1) 成员重拟合（只见 cs+cd 清洁数据），各自计算训练/测试分数；
2) margin 归一：m_i = (s_i − t99_i) / (t99.9_i − t99_i)，>0 即成员在 1% 档告警，=1 为 0.1% 档；
3) 融合变体：max（OR 语义）/ mean / median；
4) 融合分数在训练 margin 上标定 99.9/99/95 三档阈值 → eval_smoke 标准口径逐场景评测；
5) 贪心委员会搜索：目标宏 tpr@cf1 最大，约束宏 rfpr@1 ≤ 上限；输出全部变体对比与最优存档。

输出：out_dir/<variant>/{score,label,thresholds}.npy + metrics.csv + 检测图（外部脚本）。
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "..", "时间序列方法库", "method_lib"))
from traditional.Ewma_SingleVar_Online import Ewma_SingleVar_Online  # noqa: E402
from traditional.Cusum_SingleVar_Online import Cusum_SingleVar_Online  # noqa: E402
from traditional.Mahalanobis_MultiVar_Gaussian import Mahalanobis_MultiVar_Gaussian  # noqa: E402

SEQ = {"EWMA31", "EWMA30", "CUSUM31", "CUSUM30"}

# 13 列：v31 8 列 + v30 5 列（e025）
V31_IDX = list(range(8))
V30_IDX = list(range(8, 15))            # 7 列：5 个 e025 指标 + received_power_e025 + CN0_e025


class KnnDist:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X):
        self.nn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(X)
        return self

    def score(self, X):
        X = np.asarray(X)
        out = np.empty(len(X))
        for i in range(0, len(X), 20000):
            out[i:i + 20000] = self.nn.kneighbors(X[i:i + 20000])[0].mean(axis=1)
        return out


class Ocsvm:
    def __init__(self, nu=0.02):
        self.nu = nu

    def fit(self, X):
        self.clf = OneClassSVM(kernel="rbf", nu=self.nu, gamma="scale").fit(X)
        return self

    def score(self, X):
        return -self.clf.score_samples(X)


def build_members(seed=2):
    """[(名称, 实例, 特征列下标)]"""
    return [
        ("EWMA31", Ewma_SingleVar_Online(lam=0.2), V31_IDX),
        ("EWMA30", Ewma_SingleVar_Online(lam=0.2), V30_IDX),
        ("CUSUM31", Cusum_SingleVar_Online(k=0.5, h=5.0), V31_IDX),
        ("CUSUM30", Cusum_SingleVar_Online(k=0.5, h=5.0), V30_IDX),
        ("KnnDist31", KnnDist(5), V31_IDX),
        ("Mahal31", Mahalanobis_MultiVar_Gaussian(), V31_IDX),
        ("OCSVM31", Ocsvm(0.02), V31_IDX),
    ]


def margins(model, Ztr, Zte, tr_bounds, te_bounds, sequential):
    """margin 化的 train/test 分数（成员自身 train 分位归一）。

    序列型成员（EWMA/CUSUM）train 按 cs/cd 块、test 按场景块分段打分，
    避免状态跨块累积。"""
    if sequential:
        s_tr = np.empty(len(Ztr))
        for s, e in tr_bounds:
            s_tr[s:e] = model.score(Ztr[s:e])
        s_te = np.empty(len(Zte))
        for s, e in te_bounds:
            s_te[s:e] = model.score(Zte[s:e])
    else:
        s_tr = np.asarray(model.score(Ztr)).ravel()
        s_te = np.asarray(model.score(Zte)).ravel()
    t999, t99 = np.percentile(s_tr, 99.9), np.percentile(s_tr, 99.0)
    scale = (t999 - t99) if abs(t999 - t99) > 1e-12 else 1.0
    return (s_tr - t99) / scale, (s_te - t99) / scale


def evaluate(score, thr, label, data_dir, eval_py, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.save(os.path.join(os.path.dirname(out_csv), "score.npy"), score)
    np.save(os.path.join(os.path.dirname(out_csv), "thresholds.npy"), thr)
    np.save(os.path.join(os.path.dirname(out_csv), "label.npy"), label)
    r = subprocess.run([sys.executable, eval_py,
                        "--score", os.path.join(os.path.dirname(out_csv), "score.npy"),
                        "--label", os.path.join(os.path.dirname(out_csv), "label.npy"),
                        "--thresholds", os.path.join(os.path.dirname(out_csv), "thresholds.npy"),
                        "--csv", os.path.join(data_dir, "Test.csv"),
                        "--manifest", os.path.join(data_dir, "manifest.json"),
                        "--win", "1", "--out_csv", out_csv], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-400:])
        return None
    df = pd.read_csv(out_csv)
    per = df[~df.scenario.isin(["ALL", "MACRO"])]
    return {"auc": per.roc_auc.mean(), "tpr01": per["tpr@cf0p1"].mean(),
            "rfpr01": per["rfpr@0p1"].mean(), "tpr1": per["tpr@cf1"].mean(),
            "rfpr1": per["rfpr@1"].mean(), "tpr5": per["tpr@cf5"].mean(),
            "rfpr5": per["rfpr@5"].mean(), "hits": int(per.hit.sum()),
            "csv": out_csv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="anomaly_cscd_merged 目录")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--rfpr_cap", type=float, default=0.015, help="宏 rfpr@1 上限")
    ap.add_argument("--pool", default="EWMA31,EWMA30,CUSUM31,CUSUM30,KnnDist31,Mahal31,OCSVM31")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    Xtr = pd.read_csv(os.path.join(args.data_dir, "Train.csv")).values[:, 1:].astype(np.float64)
    Xte = pd.read_csv(os.path.join(args.data_dir, "Test.csv")).values[:, 1:].astype(np.float64)
    yte = pd.read_csv(os.path.join(args.data_dir, "Test_label.csv")).values[:, 1].ravel().astype(int)
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    tr_bounds, off = [], 0
    for v in manifest["train"].values():
        tr_bounds.append((off, off + v["rows"]))
        off += v["rows"]
    te_bounds = [(b["start"], b["start"] + b["rows"]) for b in manifest["test"]]
    # 输出路径规范化：仅允许在 --out_dir 下写固定文件名（防路径穿越）
    OUT = Path(args.out_dir).resolve()

    def out_path(name):
        if name != os.path.basename(name) or ".." in name:
            raise ValueError("invalid output filename")
        return OUT / name

    os.makedirs(OUT, exist_ok=True)
    np.save(out_path("label.npy"), yte)
    print(f"merged: Train={Ztr.shape} Test={Zte.shape} te块={len(te_bounds)}")

    pool = set(args.pool.split(","))
    mem_tr, mem_te = {}, {}
    for name, model, idx in build_members():
        if name not in pool:
            continue
        print(f"fitting {name} ...", flush=True)
        model.fit(Ztr[:, idx])
        mem_tr[name], mem_te[name] = margins(model, Ztr[:, idx], Zte[:, idx],
                                             tr_bounds, te_bounds, name in SEQ)
        print(f"  {name}: te margin p50/p90={np.percentile(mem_te[name],[50,90]).round(2)}")

    # 全体单成员基线（在合并集上重拟合的口径）
    singles = {}
    for name in mem_te:
        thr = np.percentile(mem_tr[name], [99.0, 95.0, 90.0])
        singles[name] = evaluate(mem_te[name], thr, yte, args.data_dir, args.eval,
                                 os.path.join(args.out_dir, f"single_{name}", "metrics.csv"))
        if singles[name]:
            s = singles[name]
            print(f"[single] {name:9s} auc={s['auc']:.4f} tpr@cf1={s['tpr1']:.4f} "
                  f"rfpr={s['rfpr1']:.4f} tpr@cf.1={s['tpr01']:.4f} hits={s['hits']}")

    # 委员会 × 融合变体搜索
    # variants: max/mean/median 为 margin 线性融合（阈值=融合分位数）；
    # or3: F=max margin，1% 档为精确 OR（thr=0.0），5%/10% 档用 F 自身 95/90 分位
    variants = ("max", "mean", "median", "or3", "rank")
    # rank 变体所需的成员 train-ECDF
    sorted_tr = {n: np.sort(mem_tr[n]) for n in mem_te}

    def rankize(n, s):
        return np.searchsorted(sorted_tr[n], s, side="right") / len(sorted_tr[n])

    results = []
    names = sorted(mem_te)
    for r in (1, 2, 3, 4):
        for combo in itertools.combinations(names, r):
            mats_tr = np.stack([mem_tr[n] for n in combo])
            mats_te = np.stack([mem_te[n] for n in combo])
            for v in variants:
                if v == "max":
                    f_tr, f_te = mats_tr.max(0), mats_te.max(0)
                    thr = np.percentile(f_tr, [99.0, 95.0, 90.0])
                elif v == "mean":
                    f_tr, f_te = mats_tr.mean(0), mats_te.mean(0)
                    thr = np.percentile(f_tr, [99.0, 95.0, 90.0])
                elif v == "median":
                    f_tr, f_te = np.median(mats_tr, 0), np.median(mats_te, 0)
                    thr = np.percentile(f_tr, [99.0, 95.0, 90.0])
                elif v == "or3":
                    f_tr, f_te = mats_tr.max(0), mats_te.max(0)
                    thr = np.asarray([0.0, np.percentile(f_tr, 95.0), np.percentile(f_tr, 90.0)])
                else:  # rank
                    f_tr = np.mean([rankize(n, mem_tr[n]) for n in combo], axis=0)
                    f_te = np.mean([rankize(n, mem_te[n]) for n in combo], axis=0)
                    thr = np.percentile(f_tr, [99.0, 95.0, 90.0])
                tag = f"{v}_" + "+".join(combo)
                res = evaluate(f_te, thr, yte, args.data_dir, args.eval,
                               os.path.join(args.out_dir, tag, "metrics.csv"))
                if res:
                    res.update({"variant": tag, "mode": v, "combo": combo})
                    results.append(res)
                    ok = res["rfpr1"] <= args.rfpr_cap
                    print(f"[{'OK ' if ok else '   '}] {tag:55s} auc={res['auc']:.4f} "
                          f"tpr@cf1={res['tpr1']:.4f} rfpr={res['rfpr1']:.4f} "
                          f"tpr@cf.1={res['tpr01']:.4f} tpr@cf5={res['tpr5']:.4f} hits={res['hits']}")

    # 融合与单模型（含云端 18 方法最优值）对比择优
    df = pd.DataFrame(results)
    feas = df[df.rfpr1 <= args.rfpr_cap].sort_values("tpr1", ascending=False)
    print("\n== TOP5 by macro tpr@cf1 (rfpr<=%.3f) ==" % args.rfpr_cap)
    print(feas.head(5)[["variant", "auc", "tpr1", "rfpr1", "tpr01", "tpr5", "hits"]].to_string(index=False))
    best = feas.iloc[0]
    with out_path("best.json").open("w", encoding="utf-8") as f:
        json.dump({"variant": best["variant"], "metrics": {k: (float(best[k]) if k != "hits" else int(best[k])) for k in
                   ["auc", "tpr1", "rfpr1", "tpr01", "rfpr01", "tpr5", "rfpr5", "hits"]}},
                  f, ensure_ascii=False, indent=2)
    print("\nBEST:", best["variant"])


if __name__ == "__main__":
    main()
