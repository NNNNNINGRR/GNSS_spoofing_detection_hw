# -*- coding: utf-8 -*-
"""单一数据集（v3.1，8 维特征）上的模型融合 + 有依据的算法改进，1%/5%/10% 三档虚警口径。

算法改进（均有文献依据，见各类 docstring）：
  - MEWMA：多元 EWMA（Lowry, Woodall, Champ, Rigdon 1992, Technometrics 34:46）。
  - MCUSUM：逐特征双边 CUSUM 取最大（Woodall & Ncube 1990 多元 CUSUM 思路）。
基线（method_lib，v3.1 上重拟合，同口径三档）：
  EWMA / CUSUM / KnnDist / Mahalanobis / OCSVM(nu=.02) / LOF / IF / StatThreshold / HotellingT2
融合（成员 margin 以各自训练分位 p99 归一，margin>0 ⟺ 1% 档告警）：
  max / mean / median / or3（1% 档精确 OR，5%/10% 用融合分自身 p95/p90）/ rank（训练 ECDF 秩均值）
输出：single_<name>/ 与 <variant>_/ 目录（score/label/thresholds npy + metrics.csv），
best.json 记录最优。序列型成员 train 按 cs/cd 块、test 按场景块分段打分。
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

ML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                  "时间序列方法库", "method_lib")
sys.path.insert(0, ML)
from traditional.Ewma_SingleVar_Online import Ewma_SingleVar_Online  # noqa: E402
from traditional.Cusum_SingleVar_Online import Cusum_SingleVar_Online  # noqa: E402
from traditional.Mahalanobis_MultiVar_Gaussian import Mahalanobis_MultiVar_Gaussian  # noqa: E402
from traditional.Lof_MultiVar_LocalDensity import Lof_MultiVar_LocalDensity  # noqa: E402
from traditional.IsolationForest_MultiVar_NonGaussian import IsolationForest_MultiVar_NonGaussian  # noqa: E402
from traditional.StatisticalThreshold_SingleVar_Gaussian import StatisticalThreshold_SingleVar_Gaussian  # noqa: E402
from traditional.HotellingT2_MultiVar_ProcessMonitoring import HotellingT2_MultiVar_ProcessMonitoring  # noqa: E402


class MEWMA:
    """多元 EWMA（Lowry et al. 1992）。E_t = λZ_t + (1−λ)E_{t−1}，
    统计量 Q_t = E_t' Σ_E⁻¹ E_t，稳态 Σ_E = λ/(2−λ)·Σ_Z。

    依据：库版 EWMA 对 8 维 |z| 取均值——特征反向偏移相互抵消且忽略协方差；
    SQM 指标强相关（delta/ratio/elp 互补），二次型保留方向与相关性信息。
    """

    def __init__(self, lam=0.2, reg=1e-6):
        self.lam, self.reg = float(lam), float(reg)

    def _run(self, Z):
        E = np.zeros(Z.shape[1])
        q = np.empty(len(Z))
        for i in range(len(Z)):
            E = self.lam * Z[i] + (1 - self.lam) * E
            q[i] = float(E @ self.Sigma_inv @ E)
        return q

    def fit(self, Z):
        cov = np.cov(Z, rowvar=False) * self.lam / (2 - self.lam)
        self.Sigma_inv = np.linalg.pinv(cov + self.reg * np.eye(Z.shape[1]))
        return self

    def score(self, Z):
        return self._run(np.asarray(Z, dtype=np.float64))


class MCUSUM:
    """逐特征双边 CUSUM 取最大（Woodall & Ncube 1990 思路）。
    S_f = max(C⁺_f, C⁻_f)，score_t = max_f S_f / h。

    依据：库版 CUSUM 先对 8 维 z 取均值再累积——功率类抬升与比值类下降符号相反
    直接抵消；逐特征累积+取最大保留最强单特征漂移（欺骗签名是特征特异的）。
    """

    def __init__(self, k=0.5, h=5.0):
        self.k, self.h = float(k), float(h)

    def _run(self, Z):
        cp = np.zeros(Z.shape[1])
        cm = np.zeros(Z.shape[1])
        s = np.empty(len(Z))
        for i in range(len(Z)):
            cp = np.maximum(0.0, cp + Z[i] - self.k)
            cm = np.maximum(0.0, cm - Z[i] - self.k)
            s[i] = np.maximum(cp, cm).max() / self.h
        return s

    def fit(self, Z):
        return self

    def score(self, Z):
        return self._run(np.asarray(Z, dtype=np.float64))


class KnnDist:
    def __init__(self, k=5):
        self.k = int(k)

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
        self.nu = float(nu)

    def fit(self, X):
        self.clf = OneClassSVM(kernel="rbf", nu=self.nu, gamma="scale").fit(X)
        return self

    def score(self, X):
        return -self.clf.score_samples(X)


SEQ = {"EWMA", "CUSUM", "MEWMA", "MCUSUM"}
TIERS_PCT = [99.0, 95.0, 90.0]          # FPR 1% / 5% / 10%


def build_all(seed=2):
    """全部单模型 [(名称, 实例)]；前 7 个进融合池。"""
    return [
        ("EWMA", Ewma_SingleVar_Online(lam=0.2)),
        ("MEWMA", MEWMA(lam=0.2)),
        ("MCUSUM", MCUSUM(k=0.5, h=5.0)),
        ("CUSUM", Cusum_SingleVar_Online(k=0.5, h=5.0)),
        ("KnnDist", KnnDist(5)),
        ("Mahalanobis", Mahalanobis_MultiVar_Gaussian()),
        ("OCSVM", Ocsvm(0.02)),
        ("LOF", Lof_MultiVar_LocalDensity()),
        ("IF", IsolationForest_MultiVar_NonGaussian(seed=seed)),
        ("StatThreshold", StatisticalThreshold_SingleVar_Gaussian()),
        ("HotellingT2", HotellingT2_MultiVar_ProcessMonitoring()),
    ]


def member_scores(model, Ztr, Zte, tr_b, te_b, sequential):
    if sequential:
        s_tr = np.empty(len(Ztr))
        for s, e in tr_b:
            s_tr[s:e] = model.score(Ztr[s:e])
        s_te = np.empty(len(Zte))
        for s, e in te_b:
            s_te[s:e] = model.score(Zte[s:e])
    else:
        s_tr = np.asarray(model.score(Ztr)).ravel()
        s_te = np.asarray(model.score(Zte)).ravel()
    t99, t999 = np.percentile(s_tr, 99.0), np.percentile(s_tr, 99.9)
    scale = (t999 - t99) if abs(t999 - t99) > 1e-12 else 1.0
    return (s_tr - t99) / scale, (s_te - t99) / scale


def evaluate(score, thr, label, data_dir, eval_py, out_csv):
    d = os.path.dirname(out_csv)
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/score.npy", score)
    np.save(f"{d}/thresholds.npy", thr)
    np.save(f"{d}/label.npy", label)
    r = subprocess.run([sys.executable, eval_py,
                        "--score", f"{d}/score.npy", "--label", f"{d}/label.npy",
                        "--thresholds", f"{d}/thresholds.npy",
                        "--csv", os.path.join(data_dir, "Test.csv"),
                        "--manifest", os.path.join(data_dir, "manifest.json"),
                        "--win", "1", "--out_csv", out_csv], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-300:])
        return None
    per = pd.read_csv(out_csv)
    per = per[~per.scenario.isin(["ALL", "MACRO"])]
    return {"auc": float(per.roc_auc.mean()),
            "tpr1": float(per["tpr@cf1"].mean()), "rfpr1": float(per["rfpr@1"].mean()),
            "tpr5": float(per["tpr@cf5"].mean()), "rfpr5": float(per["rfpr@5"].mean()),
            "tpr10": float(per["tpr@cf10"].mean()), "rfpr10": float(per["rfpr@10"].mean()),
            "hits": int(per.hit.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="anomaly_cscd（v3.1）目录")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--rfpr_cap", type=float, default=0.02, help="宏 rfpr@1% 上限")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    Xtr = pd.read_csv(os.path.join(args.data_dir, "Train.csv")).values[:, 1:].astype(np.float64)
    Xte = pd.read_csv(os.path.join(args.data_dir, "Test.csv")).values[:, 1:].astype(np.float64)
    yte = pd.read_csv(os.path.join(args.data_dir, "Test_label.csv")).values[:, 1].ravel().astype(int)
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    tr_b, off = [], 0
    for v in manifest["train"].values():
        tr_b.append((off, off + v["rows"]))
        off += v["rows"]
    te_b = [(b["start"], b["start"] + b["rows"]) for b in manifest["test"]]
    print(f"v3.1: Train={Ztr.shape} Test={Zte.shape}", flush=True)

    mem_tr, mem_te = {}, {}
    for name, model in build_all():
        print(f"fit {name} ...", flush=True)
        model.fit(Ztr)
        mem_tr[name], mem_te[name] = member_scores(model, Ztr, Zte, tr_b, te_b, name in SEQ)

    print("\n== 单模型基线（1%/5%/10% 三档）==")
    base = {}
    for name in mem_te:
        thr = np.percentile(mem_tr[name], TIERS_PCT)
        m = evaluate(mem_te[name], thr, yte, args.data_dir, args.eval,
                     os.path.join(args.out_dir, f"single_{name}", "metrics.csv"))
        if m:
            base[name] = m
            print(f"{name:13s} auc={m['auc']:.4f} | 1%: tpr={m['tpr1']:.4f} rfpr={m['rfpr1']:.4f} | "
                  f"5%: tpr={m['tpr5']:.4f} rfpr={m['rfpr5']:.4f} | 10%: tpr={m['tpr10']:.4f} "
                  f"rfpr={m['rfpr10']:.4f} | hits={m['hits']}")

    pool = ["EWMA", "MEWMA", "MCUSUM", "CUSUM", "KnnDist", "Mahalanobis", "OCSVM"]
    sorted_tr = {n: np.sort(mem_tr[n]) for n in pool}

    def rankize(n, s):
        return np.searchsorted(sorted_tr[n], s, side="right") / len(sorted_tr[n])

    print("\n== 融合搜索 ==")
    results = []
    for r in (2, 3, 4):
        for combo in itertools.combinations(pool, r):
            mats_tr = np.stack([mem_tr[n] for n in combo])
            mats_te = np.stack([mem_te[n] for n in combo])
            cands = {
                "max": (mats_tr.max(0), mats_te.max(0), np.percentile(mats_tr.max(0), TIERS_PCT)),
                "mean": (mats_tr.mean(0), mats_te.mean(0), np.percentile(mats_tr.mean(0), TIERS_PCT)),
                "median": (np.median(mats_tr, 0), np.median(mats_te, 0),
                           np.percentile(np.median(mats_tr, 0), TIERS_PCT)),
                "or3": (mats_tr.max(0), mats_te.max(0),
                        np.asarray([0.0, np.percentile(mats_tr.max(0), 95.0),
                                    np.percentile(mats_tr.max(0), 90.0)])),
                "rank": (np.mean([rankize(n, mem_tr[n]) for n in combo], axis=0),
                         np.mean([rankize(n, mem_te[n]) for n in combo], axis=0), None),
            }
            cands["rank"] = (cands["rank"][0], cands["rank"][1],
                             np.percentile(cands["rank"][0], TIERS_PCT))
            for v, (f_tr, f_te, thr) in cands.items():
                tag = f"{v}_" + "+".join(combo)
                m = evaluate(f_te, thr, yte, args.data_dir, args.eval,
                             os.path.join(args.out_dir, tag, "metrics.csv"))
                if not m:
                    continue
                m.update({"variant": tag})
                results.append(m)
                ok = m["rfpr1"] <= args.rfpr_cap
                print(f"[{'OK' if ok else '  '}] {tag:60s} auc={m['auc']:.4f} "
                      f"1%: {m['tpr1']:.4f}/{m['rfpr1']:.4f} "
                      f"5%: {m['tpr5']:.4f}/{m['rfpr5']:.4f} "
                      f"10%: {m['tpr10']:.4f} hits={m['hits']}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.out_dir, "all_variants.csv"), index=False)
    feas = df[df.rfpr1 <= args.rfpr_cap].sort_values("tpr5", ascending=False)
    print("\n== TOP5 by macro tpr@cf5%（rfpr@1%<=%.3f）==" % args.rfpr_cap)
    print(feas.head(5)[["variant", "auc", "tpr1", "rfpr1", "tpr5", "rfpr5", "tpr10", "hits"]]
          .to_string(index=False))
    if len(feas):
        b = feas.iloc[0]
        with open(os.path.join(args.out_dir, "best.json"), "w", encoding="utf-8") as f:
            json.dump({"variant": b["variant"],
                       "metrics": {k: (float(b[k]) if k != "hits" else int(b[k]))
                                   for k in ["auc", "tpr1", "rfpr1", "tpr5", "rfpr5",
                                             "tpr10", "rfpr10", "hits"]}}, f, indent=2)
        print("BEST:", b["variant"])


if __name__ == "__main__":
    main()
