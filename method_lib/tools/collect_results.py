# -*- coding: utf-8 -*-
"""collect_results.py —— 汇总 method_lib 中各任务的实验结果。

用法:
  python tools/collect_results.py [--out results_summary.json]
"""
import argparse, glob, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = ROOT


def parse_forecast(path):
    out = {}
    if not os.path.exists(path):
        return out
    txt = open(path, encoding="utf-8", errors="replace").read()
    for blk in re.split(r"\n\s*\n", txt):
        lines = [l.strip() for l in blk.splitlines() if l.strip()]
        if not lines:
            continue
        m = re.search(r"mse:([0-9.eE+-]+), mae:([0-9.eE+-]+)", "\n".join(lines))
        if m:
            toks = lines[0].split("_")
            key = "_".join(toks[3:5])          # e.g. ETTh1_96
            model = toks[5] if toks[5] != 'Bi' else 'Bi_FI'   # e.g. Bi_FI / DLinear
            out.setdefault(model, {})[key] = [round(float(m.group(1)), 4), round(float(m.group(2)), 4)]
    return out


def parse_anomaly(path):
    out = {}
    if not os.path.exists(path):
        return out
    txt = open(path, encoding="utf-8", errors="replace").read()
    for blk in re.split(r"\n\s*\n", txt):
        lines = [l.strip() for l in blk.splitlines() if l.strip()]
        if not lines:
            continue
        m = re.search(r"F-score : ([0-9.]+)", "\n".join(lines))
        if m:
            ds = lines[0].split("_")[2]
            model = lines[0].split("_")[3]
            out.setdefault(model, {}).setdefault(ds, []).append(round(float(m.group(1)), 4))
    return out


def parse_classification():
    out = {}
    for p in glob.glob(os.path.join(B, "results", "*", "result_classification.txt")):
        txt = open(p, encoding="utf-8", errors="replace").read()
        setting = p.split(os.sep)[-2]
        toks = setting.split("_")
        ds, model = toks[1], toks[2]
        m = re.search(r"accuracy:([0-9.]+)", txt)
        if m:
            out.setdefault(model, {})[ds] = round(float(m.group(1)), 4)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results_summary.json"))
    args = ap.parse_args()
    rep = {
        "forecast": parse_forecast(os.path.join(B, "result_long_term_forecast.txt")),
        "anomaly": parse_anomaly(os.path.join(B, "result_anomaly_detection.txt")),
        "classification": parse_classification(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))