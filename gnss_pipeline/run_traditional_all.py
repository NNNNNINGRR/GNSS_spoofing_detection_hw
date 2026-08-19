# -*- coding: utf-8 -*-
"""全部传统方法 × 两版数据集（v3.0 七特征 / v3.1 八特征）统一验证驱动。

方法清单（fit 只见清洁数据 cs+cd；score 越大越异常）：
  库方法 10 个：IsolationForest / OneClassSvm / Mahalanobis / Lof / PcaReconstruction /
               HotellingT2 / StatisticalThreshold / Cusum / Ewma / StlResidual
  自定义 4 个：StatZ / StatZCN0 / KnnDist(k=5) / MahalanobisMCD
  OCSVM 网格 4 个：nu{0.01,0.02} × gamma{scale,0.5}

序列累积型方法（Cusum/Ewma/StlResidual）按场景块分段打分（train 按 cs/cd 块、
test 按 manifest 场景块），避免跨场景累积污染；无状态方法整段打分（等价）。

每方法输出：<out_dir>/<method>/{score,label,thresholds}.npy + metrics.csv +
<method>_detection.png（便于后续重绘）。支持 --resume 断点续跑。
"""
import argparse
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

CN0_IDX = -1     # CN0 恒为最后一列（两版特征集均如此）
SEQUENTIAL = {"Cusum_SingleVar_Online", "Ewma_SingleVar_Online", "StlResidual_SingleVar_Seasonal"}


# ---------- 自定义方法 ----------
class StatZ:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.med_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.med_), axis=0) * 1.4826
        self.mad_ = np.where(mad > 1e-12, mad, 1e-12)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.max(np.abs(X - self.med_) / self.mad_, axis=1)


class StatZCN0:
    def __init__(self, cn0_idx=CN0_IDX, bin_db=2.0):
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
        gm = np.nanmedian(self.med_, axis=0)
        gm = np.where(np.isnan(gm), np.median(X, axis=0), gm)
        gd = np.nanmedian(self.mad_, axis=0)
        gd = np.where(np.isnan(gd), 1.0, gd)
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
            d, _ = self.nn.kneighbors(X[i:i + 20000])
            out[i:i + 20000] = d.mean(axis=1)
        return out


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


class OcsvmTuned:
    def __init__(self, nu=0.01, gamma="scale"):
        self.nu, self.gamma = float(nu), gamma

    def fit(self, X):
        self.clf = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma).fit(X)
        return self

    def score(self, X):
        return -self.clf.score_samples(X)


def make_methods(ml_path, seed):
    """返回 [(名称, 实例化函数)]。"""
    sys.path.insert(0, ml_path)
    methods = []
    lib = ["IsolationForest_MultiVar_NonGaussian", "OneClassSvm_MultiVar_Boundary",
           "Mahalanobis_MultiVar_Gaussian", "Lof_MultiVar_LocalDensity",
           "PcaReconstruction_MultiVar_Correlated", "HotellingT2_MultiVar_ProcessMonitoring",
           "StatisticalThreshold_SingleVar_Gaussian", "Cusum_SingleVar_Online",
           "Ewma_SingleVar_Online", "StlResidual_SingleVar_Seasonal"]
    for name in lib:
        mod = importlib.import_module(f"traditional.{name}")
        cls = getattr(mod, name)
        if name == "StlResidual_SingleVar_Seasonal":
            methods.append((name, lambda c=cls: c(season=50)))   # 50 Hz → 1 s 周期
        else:
            methods.append((name, lambda c=cls: c()))
    methods += [("StatZ", StatZ), ("StatZCN0", StatZCN0),
                ("KnnDist", KnnDist), ("MahalanobisMCD", MahalanobisMCD)]
    for nu in (0.01, 0.02):
        for g in ("scale", 0.5):
            methods.append((f"Ocsvm_nu{nu}_g{g}", lambda n=nu, gg=g: OcsvmTuned(n, gg)))
    return methods


def score_blocks(model, X, bounds):
    """按 [ (start,end), ... ] 块分段打分后拼接（序列型方法需要）。"""
    out = np.empty(len(X))
    for s, e in bounds:
        out[s:e] = model.score(X[s:e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--method_lib", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eval", required=True, help="eval_smoke.py")
    ap.add_argument("--plot", required=True, help="plot_detection.py")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default="", help="逗号分隔只跑这些方法")
    args = ap.parse_args()

    with open(os.path.join(args.data_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    n_feat = len(manifest["features"])
    Xtr = pd.read_csv(os.path.join(args.data_dir, "Train.csv")).values[:, 1:].astype(np.float64)
    Xte = pd.read_csv(os.path.join(args.data_dir, "Test.csv")).values[:, 1:].astype(np.float64)
    yte = pd.read_csv(os.path.join(args.data_dir, "Test_label.csv")).values[:, 1].ravel().astype(int)
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    # 分段边界：train 按 cs/cd（manifest train rows），test 按场景块
    tr_b = []
    off = 0
    for v in manifest["train"].values():
        tr_b.append((off, off + v["rows"]))
        off += v["rows"]
    te_b = [(b["start"], b["start"] + b["rows"]) for b in manifest["test"]]
    print(f"特征 {n_feat} 维 | Train={Ztr.shape} Test={Zte.shape} | train块={len(tr_b)} test块={len(te_b)}")

    only = set(x for x in args.only.split(",") if x)
    for name, mk in make_methods(args.method_lib, args.seed):
        if only and name not in only:
            continue
        mdir = os.path.join(args.out_dir, name)
        if args.resume and os.path.exists(os.path.join(mdir, "metrics.csv")):
            print(f"[SKIP] {name}")
            continue
        os.makedirs(mdir, exist_ok=True)
        try:
            model = mk()
            model.fit(Ztr)
            s_tr = score_blocks(model, Ztr, tr_b) if name in SEQUENTIAL else np.asarray(model.score(Ztr)).ravel()
            s_te = score_blocks(model, Zte, te_b) if name in SEQUENTIAL else np.asarray(model.score(Zte)).ravel()
            s_tr = np.asarray(s_tr, dtype=np.float64).ravel()
            s_te = np.asarray(s_te, dtype=np.float64).ravel()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}")
            continue
        np.save(os.path.join(mdir, "score.npy"), s_te)
        np.save(os.path.join(mdir, "label.npy"), yte)
        np.save(os.path.join(mdir, "thresholds.npy"), np.percentile(s_tr, [99.9, 99.0, 95.0]))
        r = subprocess.run([sys.executable, args.eval,
                            "--score", f"{mdir}/score.npy", "--label", f"{mdir}/label.npy",
                            "--thresholds", f"{mdir}/thresholds.npy",
                            "--csv", os.path.join(args.data_dir, "Test.csv"),
                            "--manifest", os.path.join(args.data_dir, "manifest.json"),
                            "--win", "1", "--out_csv", f"{mdir}/metrics.csv"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL-eval] {name}: {r.stderr[-300:]}")
            continue
        subprocess.run([sys.executable, args.plot,
                        "--data_dir", args.data_dir, "--method_dir", mdir,
                        "--name", name, "--out", os.path.join(mdir, f"{name}_detection.png")],
                       capture_output=True, text=True)
        print(f"[OK] {name}")


if __name__ == "__main__":
    main()
