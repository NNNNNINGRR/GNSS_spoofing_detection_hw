# -*- coding: utf-8 -*-
"""补齐传统家族 5 个方法在 v3.1、1%/5%/10% 三档口径下的评测（与其余 11 方法对齐）。

方法：PcaReconstruction / MahalanobisMCD / StlResidual(season=50) / StatZ / StatZCN0。
输出与 run_fusion_v31.py 的 single_* 完全一致：<out>/single_<name>/{score,label,thresholds}.npy + metrics.csv。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

ML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                  "时间序列方法库", "method_lib")
sys.path.insert(0, ML)
from traditional.PcaReconstruction_MultiVar_Correlated import PcaReconstruction_MultiVar_Correlated  # noqa: E402
from traditional.StlResidual_SingleVar_Seasonal import StlResidual_SingleVar_Seasonal  # noqa: E402


class MahalanobisMCD:
    def __init__(self, reg=1e-6, seed=2):
        self.reg, self.seed = float(reg), int(seed)

    def fit(self, X):
        mcd = MinCovDet(random_state=self.seed, support_fraction=None).fit(X)
        self.mean_ = mcd.location_
        self.cov_inv_ = np.linalg.pinv(mcd.covariance_ + self.reg * np.eye(X.shape[1]))
        return self

    def score(self, X):
        d = np.asarray(X) - self.mean_
        return np.einsum("ni,ij,nj->n", d, self.cov_inv_, d)


class StatZ:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.med_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.med_), axis=0) * 1.4826
        self.mad_ = np.where(mad > 1e-12, mad, 1e-12)
        return self

    def score(self, X):
        return np.max(np.abs(np.asarray(X) - self.med_) / self.mad_, axis=1)


class StatZCN0:
    def __init__(self, cn0_idx=-1, bin_db=2.0):
        self.ci, self.bin_db = int(cn0_idx), bin_db

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        cn0 = X[:, self.ci]
        lo = np.floor(cn0.min() / self.bin_db) * self.bin_db
        self.edges_ = np.arange(lo, cn0.max() + self.bin_db, self.bin_db)
        self.med_ = np.full((len(self.edges_), X.shape[1]), np.nan)
        self.mad_ = np.full_like(self.med_, np.nan)
        for i, b in enumerate(self.edges_):
            m = (cn0 >= b) & (cn0 < b + self.bin_db)
            if m.sum() < 200:
                continue
            seg = X[m]
            self.med_[i] = np.median(seg, axis=0)
            mad = np.median(np.abs(seg - self.med_[i]), axis=0) * 1.4826
            self.mad_[i] = np.where(mad > 1e-12, mad, 1e-12)
        gm = np.where(np.isnan(np.nanmedian(self.med_, axis=0)), np.median(X, axis=0), np.nanmedian(self.med_, axis=0))
        gd = np.where(np.isnan(np.nanmedian(self.mad_, axis=0)), 1.0, np.nanmedian(self.mad_, axis=0))
        for i in range(len(self.edges_)):
            self.med_[i] = np.where(np.isnan(self.med_[i]), gm, self.med_[i])
            self.mad_[i] = np.where(np.isnan(self.mad_[i]), gd, self.mad_[i])
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        idx = np.clip(np.searchsorted(self.edges_, X[:, self.ci]) - 1, 0, len(self.edges_) - 1)
        z = np.abs(X - self.med_[idx]) / self.mad_[idx]
        g = len(self.edges_) // 2
        z[:, self.ci] = np.abs(X[:, self.ci] - self.med_[g, self.ci]) / self.mad_[g, self.ci]
        return z.max(axis=1)


def blockwise(model, Z, bounds, sequential):
    if not sequential:
        return np.asarray(model.score(Z)).ravel()
    out = np.empty(len(Z))
    for s, e in bounds:
        out[s:e] = model.score(Z[s:e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval", required=True)
    args = ap.parse_args()
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

    methods = [("PcaReconstruction", PcaReconstruction_MultiVar_Correlated(), False),
               ("MahalanobisMCD", MahalanobisMCD(), False),
               ("StlResidual", StlResidual_SingleVar_Seasonal(season=50), True),
               ("StatZ", StatZ(), False),
               ("StatZCN0", StatZCN0(), False)]
    for name, model, seq in methods:
        print(f"fit {name} ...", flush=True)
        model.fit(Ztr)
        s_tr = blockwise(model, Ztr, tr_b, seq)
        s_te = blockwise(model, Zte, te_b, seq)
        d = os.path.join(args.out_dir, f"single_{name}")
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/score.npy", s_te)
        np.save(f"{d}/label.npy", yte)
        np.save(f"{d}/thresholds.npy", np.percentile(s_tr, [99.0, 95.0, 90.0]))
        r = subprocess.run([sys.executable, args.eval,
                            "--score", f"{d}/score.npy", "--label", f"{d}/label.npy",
                            "--thresholds", f"{d}/thresholds.npy",
                            "--csv", os.path.join(args.data_dir, "Test.csv"),
                            "--manifest", os.path.join(args.data_dir, "manifest.json"),
                            "--win", "1", "--out_csv", f"{d}/metrics.csv"],
                           capture_output=True, text=True)
        df = pd.read_csv(f"{d}/metrics.csv")
        per = df[~df.scenario.isin(["ALL", "MACRO"])]
        print(f"{name:18s} auc={per.roc_auc.mean():.4f} 1%:{per['tpr@cf1'].mean():.4f}/"
              f"{per['rfpr@1'].mean():.4f} 5%:{per['tpr@cf5'].mean():.4f} "
              f"10%:{per['tpr@cf10'].mean():.4f} hits={int(per.hit.sum())}"
              + ("" if r.returncode == 0 else " EVAL-ERR"))


if __name__ == "__main__":
    main()
