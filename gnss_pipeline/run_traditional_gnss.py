# -*- coding: utf-8
"""传统单类方法 + 统计基线，在 SQM_cscd（cs+cd 清洁训练）上做逐帧 GNSS 欺骗检测。

方法（fit 只见清洁数据，score 越大越异常）：
  - method_lib/traditional 的 5 个多变量异常方法（IF/OCSVM/Mahalanobis/LOF/PCA）
  - StatZ：统计 SQM 基线（逐特征对清洁标定的稳健 z 分数取最大，Paper A 对照的
    假设检验类检测器的简化版）

输出：每方法 out_dir/<m>/{score,label,thresholds}.npy，
阈值 = 训练（清洁）分数 99.9/99/95 分位（FPR 0.1/1/5% 口径），
随后用 eval_smoke.py --win 1（逐帧）+ manifest 逐场景评测。

用法：
  python run_traditional_gnss.py --data_dir <anomaly_cscd 目录> \
      --method_lib <method_lib 路径> --out_dir <输出目录> [--methods m1,m2] [--eval eval_smoke.py]
"""
import argparse
import importlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

TRADITIONAL = ["IsolationForest_MultiVar_NonGaussian",
               "OneClassSvm_MultiVar_Boundary",
               "Mahalanobis_MultiVar_Gaussian",
               "Lof_MultiVar_LocalDensity",
               "PcaReconstruction_MultiVar_Correlated"]


class StatZ:
    """统计基线：逐特征稳健 z（中位数/MAD），分数 = max_f |z_f|。"""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.med_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.med_), axis=0) * 1.4826
        self.mad_ = np.where(mad > 1e-12, mad, 1e-12)
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.max(np.abs(X - self.med_) / self.mad_, axis=1)


def load_method(name, ml_path):
    if name == "StatZ":
        return StatZ
    sys.path.insert(0, ml_path)
    mod = importlib.import_module(f"traditional.{name}")
    return getattr(mod, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="含 Train/Test/Test_label/manifest.json")
    ap.add_argument("--method_lib", default=None, help="method_lib 路径")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--methods", default=",".join(TRADITIONAL + ["StatZ"]))
    ap.add_argument("--eval", default=None, help="eval_smoke.py 路径（缺省不评测）")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Xtr = pd.read_csv(os.path.join(args.data_dir, "Train.csv")).values[:, 1:].astype(np.float64)
    te = pd.read_csv(os.path.join(args.data_dir, "Test.csv"))
    Xte = te.values[:, 1:].astype(np.float64)
    yte = pd.read_csv(os.path.join(args.data_dir, "Test_label.csv")).values[:, 1].ravel().astype(int)
    Xtr, Xte = np.nan_to_num(Xtr), np.nan_to_num(Xte)

    # 统一标准化（统计量只来自清洁训练集）：距离/协方差类方法必需，树类不受影响
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    print(f"Train={Ztr.shape} Test={Zte.shape} 欺骗帧率={yte.mean():.3f}")

    for name in args.methods.split(","):
        cls = load_method(name.strip(), args.method_lib or "")
        mdir = os.path.join(args.out_dir, name.strip())
        os.makedirs(mdir, exist_ok=True)
        try:
            takes_seed = getattr(getattr(cls.__init__, "__code__", None), "co_varnames", ())
            model = cls(seed=args.seed) if "seed" in takes_seed else cls()
            model.fit(Ztr)
            s_tr = np.asarray(model.score(Ztr), dtype=np.float64).ravel()
            s_te = np.asarray(model.score(Zte), dtype=np.float64).ravel()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {e}")
            continue
        np.save(os.path.join(mdir, "score.npy"), s_te)
        np.save(os.path.join(mdir, "label.npy"), yte)
        np.save(os.path.join(mdir, "thresholds.npy"),
                np.percentile(s_tr, [99.9, 99.0, 95.0]))
        print(f"[OK] {name}: score_tr[p50,p99,p999]="
              f"{np.percentile(s_tr,[50,99,99.9]).round(3)}, "
              f"score_te[p50,p99,p999]={np.percentile(s_te,[50,99,99.9]).round(3)}")

        if args.eval:
            cmd = [sys.executable, args.eval,
                   "--score", os.path.join(mdir, "score.npy"),
                   "--label", os.path.join(mdir, "label.npy"),
                   "--thresholds", os.path.join(mdir, "thresholds.npy"),
                   "--csv", os.path.join(args.data_dir, "Test.csv"),
                   "--manifest", os.path.join(args.data_dir, "manifest.json"),
                   "--win", "1",
                   "--out_csv", os.path.join(mdir, "metrics.csv")]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stderr[-500:])


if __name__ == "__main__":
    main()
