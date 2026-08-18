# -*- coding: utf-8 -*-
"""compare_paper.py —— 把 method_lib 结果与 Bi-FI 论文 Table D.7/D.8/D.9 对照打印。

用法: python tools/compare_paper.py [results_summary.json]
"""
import argparse, json, os, sys

# 论文 Bi-FI 数值（Applied Soft Computing 167 (2024) 112383）
PAPER_FORECAST = {
    "ETTh1": (0.442, 0.438), "ETTh2": (0.376, 0.399), "ETTm1": (0.385, 0.395),
    "ETTm2": (0.278, 0.322), "ECL": (0.181, 0.274), "Exchange": (0.363, 0.408),
    "Weather": (0.253, 0.271), "Solar": (0.229, 0.268),
}
PAPER_ANOMALY = {"MSL": 82.20, "SWaT": 92.76, "SMD": 84.30, "SMAP": 68.84, "PSM": 96.01}
PAPER_CLS = {
    "EthanolConcentration": 0.297, "FaceDetection": 0.667, "Handwriting": 0.268,
    "Heartbeat": 0.776, "JapaneseVowels": 0.976, "PEMS-SF": 0.827,
    "SelfRegulationSCP1": 0.928, "SelfRegulationSCP2": 0.567,
    "SpokenArabicDigits": 0.989, "UWaveGestureLibrary": 0.875,
}

DS2ID = {"electricity": "ECL", "exchange_rate": "Exchange", "weather": "Weather", "Solar": "Solar"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", nargs="?", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_summary.json"))
    args = ap.parse_args()
    rep = json.load(open(args.summary, encoding="utf-8"))

    print("== 长序列预测（8 数据集 4 长度平均 MSE/MAE；仅 Bi-FI 与论文对照）==")
    fc = rep.get("forecast", {})
    bifi = fc.get("Bi_FI", {})
    for ds, (pm, pa) in PAPER_FORECAST.items():
        # 用数据集前缀聚合（model_id 可能是 ETTh1_96 等）
        keys = [k for k in bifi if k.split("_")[0] == ds]
        vals = [bifi[k] for k in keys]
        if vals:
            mse = round(sum(v[0] for v in vals) / len(vals), 4)
            mae = round(sum(v[1] for v in vals) / len(vals), 4)
            print(f"{ds:10s} 论文 {pm:.3f}/{pa:.3f}   复现 {mse:.4f}/{mae:.4f}")

    print("\n== 异常检测 F1（%）==")
    ad = rep.get("anomaly", {}).get("Bi_FI", {})
    for ds, pv in PAPER_ANOMALY.items():
        got = ad.get(ds, [])
        best = max(got) if got else None
        print(f"{ds:6s} 论文 {pv:.2f}   复现 {best if best is not None else 'N/A'}")

    print("\n== 分类 Accuracy ==")
    cl = rep.get("classification", {}).get("Bi_FI", {})
    for ds, pv in PAPER_CLS.items():
        got = cl.get(ds)
        print(f"{ds:24s} 论文 {pv:.3f}   复现 {got if got is not None else 'N/A'}")


if __name__ == "__main__":
    main()