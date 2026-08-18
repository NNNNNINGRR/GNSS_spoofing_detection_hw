# -*- coding: utf-8 -*-
"""run_full_experiments.py —— 方法库全量验证调度器（预测/异常/分类）。

用法: python tools/run_full_experiments.py [--stage forecast|anomaly|classification|all]
特点: 阶段内并行、断点续跑（已完成组自动跳过）、失败记录。
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/root/miniconda3/bin/python"
DS = "/root/autodl-tmp/bifi_repro_20260811/时间序列方法库/dataset"
LOG_DIR = os.path.join(ML, "logs", "full")
os.makedirs(LOG_DIR, exist_ok=True)

MODELS = ["Bi_FI", "DLinear", "iTransformer", "PatchTST", "LightTS",
          "FEDformer", "Reformer", "Informer", "Autoformer"]

FORECAST = [
    # ds, data, root, file, freq, enc_in
    ("ETTh1", "ETTh1", "ETT-small/", "ETTh1.csv", "h", 7),
    ("ETTh2", "ETTh2", "ETT-small/", "ETTh2.csv", "h", 7),
    ("ETTm1", "ETTm1", "ETT-small/", "ETTm1.csv", "t", 7),
    ("ETTm2", "ETTm2", "ETT-small/", "ETTm2.csv", "t", 7),
    ("ECL", "custom", "electricity/", "electricity.csv", "h", 321),
    ("Exchange", "custom", "exchange_rate/", "exchange_rate.csv", "d", 8),
    ("Weather", "custom", "weather/", "weather.csv", "h", 21),
    ("Solar", "Solar", "Solar/", "solar_AL.txt", "h", 137),
]
PRED_LENS = [96, 192, 336, 720]
ANOMALY = [("MSL", 55), ("SMAP", 25), ("SMD", 38), ("SWAT", 51), ("PSM", 25)]
CLASSIFY = ["EthanolConcentration", "FaceDetection", "Handwriting", "Heartbeat",
            "JapaneseVowels", "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2",
            "SpokenArabicDigits", "UWaveGestureLibrary"]
CLS_DM = {"EthanolConcentration": 2048, "Handwriting": 256, "Heartbeat": 256}

def forecast_cmd(m, ds, data, root, fname, freq, enc_in, pl):
    mid = f"{ds}_{pl}"
    return [PY, "-u", "run.py", "--task_name", "long_term_forecast", "--is_training", "1",
            "--model", m, "--model_id", mid, "--data", data,
            "--root_path", os.path.join(DS, root), "--data_path", fname,
            "--features", "M", "--freq", freq, "--seq_len", "96", "--label_len", "48",
            "--pred_len", str(pl), "--enc_in", str(enc_in), "--dec_in", str(enc_in),
            "--c_out", str(enc_in), "--d_model", "512", "--n_heads", "8",
            "--e_layers", "2", "--d_layers", "1", "--d_ff", "512",
            "--num_workers", "4", "--batch_size", "16" if (m == "PatchTST" and ds == "ECL") else "32", "--learning_rate", "0.0001",
            "--train_epochs", "10", "--patience", "3", "--des", "Full", "--itr", "1",
            "--no_save_npy", "--no_visual"]

def anomaly_cmd(m, ds, enc_in):
    return [PY, "-u", "run.py", "--task_name", "anomaly_detection", "--is_training", "1",
            "--model", m, "--model_id", ds, "--data", ds,
            "--root_path", os.path.join(DS, ds), "--seq_len", "100",
            "--enc_in", str(enc_in), "--dec_in", str(enc_in), "--c_out", str(enc_in),
            "--d_model", "512", "--n_heads", "8", "--e_layers", "2", "--d_layers", "1",
            "--d_ff", "512", "--num_workers", "4", "--batch_size", "128",
            "--learning_rate", "0.0001", "--train_epochs", "10", "--patience", "3",
            "--anomaly_ratio", "1", "--des", "Full", "--itr", "1"]

def classify_cmd(m, ds):
    dm = CLS_DM.get(ds, 128)
    return [PY, "-u", "run.py", "--task_name", "classification", "--is_training", "1",
            "--model", m, "--model_id", ds, "--data", "UEA",
            "--root_path", os.path.join(DS, ds), "--e_layers", "3", "--batch_size", "16",
            "--d_model", str(dm), "--d_ff", "256", "--top_k", "3",
            "--learning_rate", "0.001", "--train_epochs", "100", "--patience", "10",
            "--num_workers", "4", "--enc_in", "3", "--des", "Full", "--itr", "1"]

def done_key(prefix, key):
    """检查 result 文件/目录是否已包含该 setting（断点续跑）。"""
    if prefix == "long_term_forecast":
        p = os.path.join(ML, "result_long_term_forecast.txt")
        if os.path.exists(p):
            return any(l.startswith(f"long_term_forecast_{key}") for l in open(p, encoding="utf-8", errors="replace"))
    elif prefix == "anomaly_detection":
        p = os.path.join(ML, "result_anomaly_detection.txt")
        if os.path.exists(p):
            return any(l.startswith(f"anomaly_detection_{key}") for l in open(p, encoding="utf-8", errors="replace"))
    elif prefix == "classification":
        import glob
        return bool(glob.glob(os.path.join(ML, "results", f"classification_{key}_*", "result_classification.txt")))
    return False

def build_tasks(stage, datasets=None):
    tasks = []
    ds_set = set(datasets or [])
    if stage in ("forecast", "all"):
        for m in MODELS:
            for ds, data, root, fname, freq, enc_in in FORECAST:
                if ds_set and ds not in ds_set:
                    continue
                for pl in PRED_LENS:
                    key = f"{ds}_{pl}_{m}"
                    if not done_key("long_term_forecast", key):
                        tasks.append(("forecast", key, forecast_cmd(m, ds, data, root, fname, freq, enc_in, pl)))
    if stage in ("anomaly", "all"):
        for m in MODELS:
            for ds, enc_in in ANOMALY:
                if ds_set and ds not in ds_set:
                    continue
                key = f"{ds}_{m}"
                if not done_key("anomaly_detection", key):
                    tasks.append(("anomaly", key, anomaly_cmd(m, ds, enc_in)))
    if stage in ("classification", "all"):
        for m in MODELS:
            for ds in CLASSIFY:
                if ds_set and ds not in ds_set:
                    continue
                key = f"{ds}_{m}"
                if not done_key("classification", key):
                    tasks.append(("classification", key, classify_cmd(m, ds)))
    return tasks

def run_one(t):
    stage, key, cmd = t
    logf = os.path.join(LOG_DIR, f"{key}.log")
    with open(logf, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=ML, stdout=f, stderr=subprocess.STDOUT).returncode
    status = "OK" if rc == 0 else "FAIL"
    with open(os.path.join(LOG_DIR, "status.tsv"), "a", encoding="utf-8") as f:
        f.write(f"{status}\t{stage}\t{key}\t{rc}\n")
    return key, rc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["forecast", "anomaly", "classification", "all"])
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--datasets", default="", help="逗号分隔的数据集子集，例如 ETTh1,ECL,MSL,UWaveGestureLibrary")
    args = ap.parse_args()

    tasks = build_tasks(args.stage, [d.strip() for d in args.datasets.split(",") if d.strip()])
    total = len(tasks)
    print(f"[full] stage={args.stage} 待运行 {total} 组")
    if total == 0:
        print("[full] 全部已完成，无需运行")
        return
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for key, rc in ex.map(run_one, tasks):
            done += 1
            el = (time.time() - t0) / 60
            print(f"[full] {done}/{total} {key} rc={rc} elapsed={el:.1f}min", flush=True)
    print("[full] 阶段完成")

if __name__ == "__main__":
    main()