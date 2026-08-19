# -*- coding: utf-8 -*-
"""传统方法 v2：调参与条件化改进（冻结 v1 结果，本脚本独立输出到 v2 目录）。

新增方法（均 fit 只见清洁数据，score 越大越异常）：
  - MahalanobisMCD    稳健协方差（MinCovDet），抗清洁训练集尾部污染
  - KnnDist           到清洁训练集 k 近邻的平均欧氏距离
  - StatZCN0          C/N0 条件化统计基线：噪声尺度随 CN0 分档标定
                      （Pirsiavash 2017 式(13)：SQM 方差 ∝ C/(N0·T) 的工程简化）
  - Ocsvm_nu*_g*      OCSVM 网格（nu × gamma）

评测与 v1 完全同口径：thresholds = 训练分数 99.9/99/95 分位，
eval_smoke.py --win 1 + manifest 逐场景。
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

CN0_IDX = 7      # FEATURE_COLS 最后一列
CN0_BIN = 2.0    # dB


class MahalanobisMCD:
    def __init__(self, reg=1e-6, seed=0):
        self.reg, self.seed = float(reg), int(seed)

    def fit(self, X):
        mcd = MinCovDet(random_state=self.seed, support_fraction=None).fit(X)
        self.mean_ = mcd.location_
        self.cov_inv_ = np.linalg.pinv(mcd.covariance_ + self.reg * np.eye(X.shape[1]))
        return self

    def score(self, X):
        d = np.asarray(X) - self.mean_
        return np.einsum("ni,ij,nj->n", d, self.cov_inv_, d)


class KnnDist:
    def __init__(self, k=5):
        self.k = int(k)

    def fit(self, X):
        self.nn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1).fit(X)
        return self

    def score(self, X):
        X = np.asarray(X)
        out = np.empty(len(X))
        for i in range(0, len(X), 20000):   # 分块防内存
            d, _ = self.nn.kneighbors(X[i:i + 20000])
            out[i:i + 20000] = d.mean(axis=1)
        return out


class StatZCN0:
    """逐特征 z 分数，med/MAD 按 CN0 分档标定；CN0 自身用全局档。
    score = max_f |x_f - med_f(cn0)| / mad_f(cn0)"""

    def __init__(self, cn0_idx=CN0_IDX, bin_db=CN0_BIN):
        self.ci, self.bin_db = cn0_idx, bin_db

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        cn0 = X[:, self.ci]
        lo = np.floor(cn0.min() / self.bin_db) * self.bin_db
        hi = cn0.max()
        self.edges_ = np.arange(lo, hi + self.bin_db, self.bin_db)
        self.med_ = np.full((len(self.edges_), X.shape[1]), np.nan)
        self.mad_ = np.full_like(self.med_, np.nan)
        for i, b in enumerate(self.edges_):
            m = (cn0 >= b) & (cn0 < b + self.bin_db)
            if m.sum() < 200:      # 样本不足的档位留空，回退全局
                continue
            seg = X[m]
            self.med_[i] = np.median(seg, axis=0)
            mad = np.median(np.abs(seg - self.med_[i]), axis=0) * 1.4826
            self.mad_[i] = np.where(mad > 1e-12, mad, 1e-12)
        glob_med = np.nanmedian(self.med_, axis=0)
        glob_med = np.where(np.isnan(glob_med), np.median(X, axis=0), glob_med)
        glob_mad = np.nanmedian(self.mad_, axis=0)
        glob_mad = np.where(np.isnan(glob_mad), 1.0, glob_mad)
        for i in range(len(self.edges_)):
            self.med_[i] = np.where(np.isnan(self.med_[i]), glob_med, self.med_[i])
            self.mad_[i] = np.where(np.isnan(self.mad_[i]), glob_mad, self.mad_[i])
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        cn0 = X[:, self.ci]
        idx = np.clip(np.searchsorted(self.edges_, cn0) - 1, 0, len(self.edges_) - 1)
        z = np.abs(X - self.med_[idx]) / self.mad_[idx]
        # CN0 档内 z 无意义（自己和自己比），改用全局档的 z
        g = len(self.edges_) // 2
        z[:, self.ci] = np.abs(X[:, self.ci] - self.med_[g, self.ci]) / self.mad_[g, self.ci]
        return z.max(axis=1)


class OcsvmTuned:
    def __init__(self, nu=0.01, gamma="scale"):
        self.nu, self.gamma = float(nu), gamma

    def fit(self, X):
        self.clf = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma).fit(X)
        return self

    def score(self, X):
        return -self.clf.score_samples(X)


METHODS = {
    "MahalanobisMCD": lambda seed: MahalanobisMCD(seed=seed),
    "KnnDist": lambda seed: KnnDist(),
    "StatZCN0": lambda seed: StatZCN0(),
}
OCSVM_GRID = [(0.01, "scale"), (0.01, 0.5), (0.02, "scale"), (0.02, 0.5)]


def run_one(name, model, Ztr, Zte, yte, mdir, data_dir, eval_py):
    os.makedirs(mdir, exist_ok=True)
    model.fit(Ztr)
    s_tr = np.asarray(model.score(Ztr), dtype=np.float64).ravel()
    s_te = np.asarray(model.score(Zte), dtype=np.float64).ravel()
    np.save(os.path.join(mdir, "score.npy"), s_te)
    np.save(os.path.join(mdir, "label.npy"), yte)
    np.save(os.path.join(mdir, "thresholds.npy"), np.percentile(s_tr, [99.9, 99.0, 95.0]))
    cmd = [sys.executable, eval_py,
           "--score", os.path.join(mdir, "score.npy"),
           "--label", os.path.join(mdir, "label.npy"),
           "--thresholds", os.path.join(mdir, "thresholds.npy"),
           "--csv", os.path.join(data_dir, "Test.csv"),
           "--manifest", os.path.join(data_dir, "manifest.json"),
           "--win", "1", "--out_csv", os.path.join(mdir, "metrics.csv")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + ("" if ok else ": " + r.stderr[-300:]))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--only", default="", help="逗号分隔：只跑这些（MahalanobisMCD,KnnDist,StatZCN0,Ocsvm）")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Xtr = pd.read_csv(os.path.join(args.data_dir, "Train.csv")).values[:, 1:].astype(np.float64)
    Xte = pd.read_csv(os.path.join(args.data_dir, "Test.csv")).values[:, 1:].astype(np.float64)
    yte = pd.read_csv(os.path.join(args.data_dir, "Test_label.csv")).values[:, 1].ravel().astype(int)
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    print(f"Train={Ztr.shape} Test={Zte.shape}")

    only = set(x for x in args.only.split(",") if x)
    for name, mk in METHODS.items():
        if only and name not in only:
            continue
        run_one(name, mk(args.seed), Ztr, Zte, yte,
                os.path.join(args.out_dir, name), args.data_dir, args.eval)
    if not only or "Ocsvm" in only:
        for nu, g in OCSVM_GRID:
            name = f"Ocsvm_nu{nu}_g{g}"
            run_one(name, OcsvmTuned(nu, g), Ztr, Zte, yte,
                    os.path.join(args.out_dir, name), args.data_dir, args.eval)


if __name__ == "__main__":
    main()
