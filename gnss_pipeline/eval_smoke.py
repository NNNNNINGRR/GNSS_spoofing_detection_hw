# -*- coding: utf-8 -*-
"""统一评估脚本（Paper A 口径）。

两种模式：
1) manifest 模式（推荐）：--manifest 指向 make_datasets.py 生成的
   anomaly_cscd/manifest.json。一次训练、拼接测试集，按场景行界切分，
   逐场景输出：阈值无关指标 + 三档清洁标定阈值（FPR 0.1/1/5%，在训练
   即清洁数据分数上标定，Paper A 式工作点）下的 TPR/实际 FPR + ADD。
2) 旧单场景模式：--onset 指定攻击起始秒，行为与历史版本兼容。

窗口对齐：score/label npy 为窗口主序展平（PSMSegLoader step=1），
reshape(n_win, win)[:, -1] 取连续窗口的末帧，与 Test.csv 行号
i + (win-1) 一一对应；行号再经 manifest 映射到场景。
"""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             balanced_accuracy_score, roc_curve)

FPR_LEVELS = [0.1, 1.0, 5.0]  # %；对应 thresholds.npy 的三档


def tpr_at_fpr(y, s, alpha):
    """事后 ROC 口径：FPR<=alpha 约束下可达的最大 TPR（模型排序能力比较用）。"""
    order = np.argsort(-s)
    ys, tp, fp = y[order], np.cumsum(y[order]), np.cumsum(1 - y[order])
    n_pos = max(int(tp[-1]), 1)
    n_neg = max(int(fp[-1]), 1)
    tpr, fpr = tp / n_pos, fp / n_neg
    ok = fpr <= alpha
    return float(tpr[ok].max()) if ok.any() else 0.0


def first_alarm_delay(t, alarms, onset, k=3):
    run = 0
    for i, h in enumerate(alarms):
        if t[i] < onset:
            continue
        run = run + 1 if h else 0
        if run >= k:
            return float(t[i] - onset)
    return np.inf


def _frame_metrics(lab, score, thr_dict):
    """单场景（或全局）帧级指标。thr_dict: {fpr%: threshold}。"""
    res = {}
    if lab.min() == lab.max():
        res["roc_auc"] = res["pr_auc"] = float("nan")
        res["tpr@fpr1"] = res["tpr@fpr5"] = float("nan")
    else:
        res["roc_auc"] = round(roc_auc_score(lab, score), 4)
        res["pr_auc"] = round(average_precision_score(lab, score), 4)
        res["tpr@fpr1"] = round(tpr_at_fpr(lab, score, 0.01), 4)
        res["tpr@fpr5"] = round(tpr_at_fpr(lab, score, 0.05), 4)
    for fpr_lvl, thr in thr_dict.items():
        yhat = (score >= thr).astype(int)
        name = ("%g" % fpr_lvl).replace(".", "p")
        pos = lab == 1
        neg = lab == 0
        res[f"tpr@cf{name}"] = round(float(yhat[pos].mean()) if pos.any() else np.nan, 4)
        res[f"rfpr@{name}"] = round(float(yhat[neg].mean()) if neg.any() else np.nan, 4)
    return res


def _fixed_point_report(lab, score, thr):
    """固定阈值（FPR1% 工作点）下的混淆矩阵指标。"""
    yhat = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(lab, yhat, labels=[0, 1]).ravel()
    return {
        "f1": round(f1_score(lab, yhat, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(lab, yhat), 4),
        "bal_acc": round(balanced_accuracy_score(lab, yhat), 4),
        "tpr": round(tp / (tp + fn) if tp + fn else 0.0, 4),
        "fpr": round(fp / (fp + tn) if fp + tn else 0.0, 4),
    }


def load_frames(args):
    """读 npy 并按窗口末帧采样，返回 (score, lab, 行号数组)。"""
    score = np.load(args.score).ravel()
    lab = np.load(args.label).ravel()
    win = args.win
    n_win = score.size // win
    score = score[: n_win * win].reshape(n_win, win)[:, -1]
    lab = lab[: n_win * win].reshape(n_win, win)[:, -1]
    rows = np.arange(win - 1, win - 1 + n_win)  # 窗口末帧对应的 Test.csv 行号
    n = min(len(score), len(lab), len(rows))
    return score[:n], lab[:n], rows[:n]


def run_manifest(args):
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    thr_arr = np.load(args.thresholds).ravel()
    thr_dict = {lvl: float(t) for lvl, t in zip(FPR_LEVELS, thr_arr)}

    score, lab, rows = load_frames(args)
    te = pd.read_csv(args.csv)
    n_rows = len(te)
    if rows.max() >= n_rows:
        rows = rows[rows < n_rows]
        score, lab = score[: len(rows)], lab[: len(rows)]
    date = pd.to_datetime(te["date"])
    epoch = (date - pd.Timestamp("1970-01-01")).dt.total_seconds().values

    out_rows = []
    scen_frames = []  # (name, t_rel, score, lab, onset) 供画图
    for blk in manifest["test"]:
        s = blk["scenario"]
        sel = (rows >= blk["start"]) & (rows < blk["start"] + blk["rows"])
        sc, lb, rw = score[sel], lab[sel], rows[sel]
        if len(rw) == 0:
            print(f"警告：{s} 无采样帧，跳过")
            continue
        t0 = epoch[blk["start"]]
        t_rel = epoch[rw] - t0
        onset = blk["onset_s"]

        rec = {"scenario": s, "type": blk["type"], "n_frames": int(len(rw)),
               "onset_s": onset}
        rec.update(_frame_metrics(lb, sc, thr_dict))
        rec.update(_fixed_point_report(lb, sc, thr_dict[1.0]))
        add = first_alarm_delay(t_rel, (sc >= thr_dict[1.0]).astype(int), onset, args.k)
        rec["add_s"] = round(add, 2) if np.isfinite(add) else "inf"
        rec["hit"] = int(np.isfinite(add))
        rec["threshold@1%"] = round(thr_dict[1.0], 4)
        out_rows.append(rec)
        scen_frames.append((s, t_rel, sc, lb, onset))

    # 全局（拼接全部场景）
    t_all = np.concatenate([f[1] for f in scen_frames])
    s_all = np.concatenate([f[2] for f in scen_frames])
    l_all = np.concatenate([f[3] for f in scen_frames])
    overall = {"scenario": "ALL", "type": "-", "n_frames": int(len(s_all)), "onset_s": np.nan}
    overall.update(_frame_metrics(l_all, s_all, thr_dict))
    overall.update(_fixed_point_report(l_all, s_all, thr_dict[1.0]))
    overall["add_s"] = np.nan
    overall["hit"] = int(pd.DataFrame(out_rows)["hit"].sum()) if out_rows else 0
    df = pd.DataFrame(out_rows + [overall])

    # Paper A 主指标宏平均（逐场景 TPR 均值）
    per = pd.DataFrame(out_rows)
    macro = {"scenario": "MACRO", "type": "-"}
    for c in ["tpr@cf0p1", "tpr@cf1", "tpr@cf5", "roc_auc", "tpr@fpr1", "tpr@fpr5"]:
        if c in per.columns:
            macro[c] = round(per[c].mean(skipna=True), 4)
    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))

    if args.out_fig and scen_frames:
        n = len(scen_frames)
        ncol = 2
        nrow = (n + 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3 * nrow), squeeze=False)
        for i, (s, t, sc, lb, onset) in enumerate(scen_frames):
            ax = axes[i // ncol][i % ncol]
            ax.plot(t, sc, lw=0.6)
            for lvl, thr in thr_dict.items():
                ax.axhline(thr, ls=":", lw=0.8,
                           label=f"thr@{lvl}%={thr:.3g}")
            ax.axvline(onset, color="green", ls="--", lw=1.2)
            ax.fill_between(t, sc.min(), sc.max(), where=(lb == 1),
                            color="green", alpha=0.15)
            ax.set_title(s, fontsize=10)
            ax.set_xlabel("Time (s, scenario-relative)")
            if i == 0:
                ax.legend(fontsize=7)
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Anomaly score per scenario (clean-calibrated thresholds)", fontsize=11)
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=120)
        plt.close(fig)


def run_single(args):
    """旧模式：单场景，onset 由命令行给定。"""
    thr = float(np.load(args.threshold).ravel()[0])
    thr_arr = np.load(args.thresholds).ravel() if args.thresholds else np.asarray([np.nan] * 3)
    thr_dict = {lvl: float(t) for lvl, t in zip(FPR_LEVELS, thr_arr)}

    score, lab, rows = load_frames(args)
    te = pd.read_csv(args.csv)
    if rows.max() >= len(te):
        rows = rows[rows < len(te)]
        score, lab = score[: len(rows)], lab[: len(rows)]
    t = (pd.to_datetime(te["date"]) - pd.Timestamp("1970-01-01")).dt.total_seconds().values
    t = t[rows] - t[rows[0]]

    rec = {"scenario": "single", "n_frames": int(len(score))}
    rec.update(_frame_metrics(lab, score, thr_dict))
    rec.update(_fixed_point_report(lab, score, thr))
    add = first_alarm_delay(t, (score >= thr).astype(int), args.onset, args.k)
    rec["add_s"] = round(add, 2) if np.isfinite(add) else "inf"
    rec["hit"] = int(np.isfinite(add))
    rec["threshold@1%"] = round(thr, 4)
    pd.DataFrame([rec]).to_csv(args.out_csv, index=False)

    if args.out_fig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(t, score, lw=0.8, label="score")
        axes[0].axhline(thr, color="red", ls="--", lw=1.2, label=f"thr={thr:.3g}")
        axes[0].axvline(args.onset, color="green", ls="--", lw=1.5, label="onset")
        axes[0].fill_between(t, score.min(), score.max(), where=(lab == 1),
                             color="green", alpha=0.2, label="spoof GT")
        axes[0].set_title(f"Anomaly score (ADD={rec['add_s']}s)")
        axes[0].set_xlabel("Time (s)")
        axes[0].legend(fontsize=8)
        fpr, tpr, _ = roc_curve(lab, score)
        axes[1].plot(fpr, tpr, lw=1.5, label=f"AUC={rec['roc_auc']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k:", lw=0.8)
        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=120)
        plt.close(fig)
    print(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threshold", default=None, help="单阈值 npy（旧模式必填）")
    ap.add_argument("--thresholds", default=None,
                    help="三档清洁标定阈值 npy（manifest 模式必填）")
    ap.add_argument("--csv", required=True, help="Test.csv（含 date 列）")
    ap.add_argument("--win", type=int, default=96)
    ap.add_argument("--manifest", default=None,
                    help="anomaly_cscd/manifest.json → 逐场景模式")
    ap.add_argument("--onset", type=float, default=None, help="单场景模式攻击起始秒")
    ap.add_argument("--out_fig", default=None)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    if args.manifest:
        if not args.thresholds:
            ap.error("manifest 模式需要 --thresholds")
        run_manifest(args)
    else:
        if args.threshold is None or args.onset is None:
            ap.error("单场景模式需要 --threshold 与 --onset")
        run_single(args)


if __name__ == "__main__":
    main()
