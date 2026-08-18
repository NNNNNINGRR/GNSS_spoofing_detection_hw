# -*- coding: utf-8 -*-
"""run_traditional.py —— 传统方法统一入口（预测 / 异常检测 / 分类）

不修改现有 run.py / exp / data_provider 接口，独立运行传统方法库：
    traditional/<方法名>.py，命名中体现了适用条件（如 Arima_SingleVar_StationaryLinear）。

用法示例:
    # 预测（单变量）
    python tools/run_traditional.py --task forecast --method Arima_SingleVar_StationaryLinear \
        --data ETTh1 --target OT --seq_len 96 --pred_len 96 --num_windows 10
    # 预测（多变量）
    python tools/run_traditional.py --task forecast --method Var_MultiVar_LinearInteractions \
        --data ETTh1 --seq_len 96 --pred_len 96 --num_windows 10
    # 异常检测
    python tools/run_traditional.py --task anomaly --method IsolationForest_MultiVar_NonGaussian \
        --data MSL --anomaly_ratio 1
    # 分类
    python tools/run_traditional.py --task classification --method KnnDtw_SingleVar_SmallData \
        --data UEA --dataset UWaveGestureLibrary

结果输出: 终端打印 + result_traditional_<task>.txt（与深度模型结果文件并列）。
"""
import argparse, importlib, inspect, os, sys
import numpy as np

ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(os.path.dirname(ML), "dataset")
sys.path.insert(0, ML)


def load_method(name):
    mod = importlib.import_module(f"traditional.{name}")
    return getattr(mod, name)


def build_kwargs(method, args):
    """只把方法 __init__ 接受的参数传进去。"""
    params = inspect.signature(method.__init__).parameters
    mapping = {
        "season": getattr(args, "season", None),
        "order": getattr(args, "arima_order", None),
        "seasonal_order": getattr(args, "seasonal_order", None),
        "maxlags": getattr(args, "var_maxlags", None),
        "components": getattr(args, "pca_components", None),
        "n_estimators": getattr(args, "n_estimators", None),
        "n_neighbors": getattr(args, "n_neighbors", None),
        "contamination": getattr(args, "contamination", None),
        "seed": getattr(args, "seed", None),
    }
    kw = {k: v for k, v in mapping.items() if k in params and v is not None}
    return kw


# ---------------- 数据加载 ----------------

def load_forecast(args):
    """读 CSV（date + 变量列 + OT），按 70/20/10 切分，返回原始值。"""
    import pandas as pd
    path = os.path.join(args.root_path, args.data_path)
    df = pd.read_csv(path)
    cols = list(df.columns)
    cols.remove(args.target)
    cols.remove("date")
    df = df[["date"] + cols + [args.target]]
    values = df[df.columns[1:]].values.astype(np.float64)
    n = len(df)
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    train = values[:num_train]
    val = values[num_train:num_train + (n - num_train - num_test)]
    test = values[n - num_test:]
    return train, val, test


def load_anomaly(args):
    """按 data 类型读取训练/测试/标签（与 data_provider 相同的文件约定）。"""
    if args.data == "PSM":
        import pandas as pd
        Xtr = pd.read_csv(os.path.join(args.root_path, "Train.csv")).values[:, 1:]
        Xte = pd.read_csv(os.path.join(args.root_path, "Test.csv")).values[:, 1:]
        yte = pd.read_csv(os.path.join(args.root_path, "Test_label.csv")).values[:, 1:].ravel()
    elif args.data in ("MSL", "SMAP", "SMD"):
        Xtr = np.load(os.path.join(args.root_path, f"{args.data}_train.npy"))
        Xte = np.load(os.path.join(args.root_path, f"{args.data}_test.npy"))
        yte = np.load(os.path.join(args.root_path, f"{args.data}_test_label.npy")).ravel()
    elif args.data == "SWAT":
        import pandas as pd
        tr = pd.read_csv(os.path.join(args.root_path, "swat_train2.csv")).values
        te = pd.read_csv(os.path.join(args.root_path, "swat2.csv")).values
        Xtr, Xte, yte = tr[:, :-1], te[:, :-1], te[:, -1].ravel()
    else:
        raise ValueError(f"不支持的异常数据集: {args.data}")
    Xtr = np.nan_to_num(np.asarray(Xtr, dtype=np.float64))
    Xte = np.nan_to_num(np.asarray(Xte, dtype=np.float64))
    return Xtr, Xte, yte.astype(int)


def load_uea(ds_name, flag):
    from data_provider.uea import _parse_uea_ts
    path = os.path.join(DS, ds_name, f"{ds_name}_{flag}.ts")
    samples, labels = _parse_uea_ts(path)
    classes = sorted(set(labels))
    code = {c: i for i, c in enumerate(classes)}
    y = np.array([code[l] for l in labels], dtype=np.int64)
    return samples, y


# ---------------- 任务执行 ----------------

def run_forecast(method, args, train, val, test):
    if args.max_train:
        train = train[:args.max_train]
    if args.max_test:
        test = test[:args.max_test]
    target_col = -1 if args.target else None
    single = "SingleVar" in args.method
    target_only = "Exogenous" in args.method   # Arimax：多特征拟合、单目标评估
    if target_only:
        ytr, yte = train, test[:, target_col] if test.ndim == 2 else test
    elif single:
        ytr = train[:, target_col] if train.ndim == 2 else train
        yte = test[:, target_col] if test.ndim == 2 else test
    else:
        ytr, yte = train, test

    model = method(**build_kwargs(method, args))
    model.fit(ytr)

    # 在测试段均匀取 num_windows 个窗口做滚动评估
    seq_len, pred_len = args.seq_len, args.pred_len
    starts = np.linspace(0, max(len(yte) - pred_len - 1, 0), args.num_windows).astype(int)
    mse_list, mae_list = [], []
    for s in starts:
        if s == 0:
            history = ytr[-seq_len:] if len(ytr) >= seq_len else ytr
        elif target_only:
            history = test[s - seq_len:s]           # Arimax 需要完整多变量窗口
        else:
            history = (test[:, target_col] if single else test)[s - seq_len:s]
        if len(history) < 1:
            continue
        pred = np.asarray(model.forecast(history, pred_len), dtype=np.float64)
        true = yte[s:s + pred_len]
        if pred.shape[0] == 0:
            continue
        L = min(pred.shape[0], len(true))
        mse_list.append(float(np.mean((pred[:L] - true[:L]) ** 2)))
        mae_list.append(float(np.mean(np.abs(pred[:L] - true[:L]))))
    mse = float(np.mean(mse_list)) if mse_list else float("nan")
    mae = float(np.mean(mae_list)) if mae_list else float("nan")
    return {"mse": mse, "mae": mae}


def run_anomaly(method, args, Xtr, Xte, yte):
    if args.max_train:
        Xtr = Xtr[:args.max_train]
    if args.max_test:
        Xte = Xte[:args.max_test]
        yte = yte[:args.max_test]
    model = method(**build_kwargs(method, args))
    model.fit(Xtr)
    score_tr = np.asarray(model.score(Xtr)).ravel()
    score_te = np.asarray(model.score(Xte)).ravel()
    combined = np.concatenate([score_tr, score_te])
    th = np.percentile(combined, 100 - args.anomaly_ratio)
    pred = (score_te > th).astype(int)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(yte, pred)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f), "threshold": float(th)}


def run_classification(method, args, Xs_tr, y_tr, Xs_te, y_te):
    single = "SingleVar" in args.method
    if single:
        # 单变量方法：若样本是多通道 [T, M]，取第一通道；单通道则压成一维
        Xs_tr = [x[:, 0] if x.ndim == 2 and x.shape[1] > 1 else x for x in Xs_tr]
        Xs_te = [x[:, 0] if x.ndim == 2 and x.shape[1] > 1 else x for x in Xs_te]
    model = method(**build_kwargs(method, args))
    model.fit(Xs_tr, y_tr)
    pred = model.predict(Xs_te)
    acc = float(np.mean(pred == y_te))
    return {"accuracy": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["forecast", "anomaly", "classification"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--data", default="ETTh1")
    ap.add_argument("--root_path", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--dataset", default=None, help="分类数据集名（文件夹名）")
    ap.add_argument("--target", default="OT")
    ap.add_argument("--freq", default="h")
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--label_len", type=int, default=48)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--num_windows", type=int, default=10, help="预测评估窗口数")
    ap.add_argument("--anomaly_ratio", type=float, default=1.0)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--arima_order", default=None)
    ap.add_argument("--seasonal_order", default=None)
    ap.add_argument("--var_maxlags", type=int, default=None)
    ap.add_argument("--pca_components", default=None)
    ap.add_argument("--n_estimators", type=int, default=None)
    ap.add_argument("--n_neighbors", type=int, default=None)
    ap.add_argument("--contamination", type=float, default=None)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    method = load_method(args.method)
    if args.root_path is None:
        args.root_path = os.path.join(DS, args.data if args.data != "UEA" else args.dataset)
    if args.data_path is None:
        args.data_path = f"{args.data}.csv"

    if args.task == "forecast":
        train, val, test = load_forecast(args)
        res = run_forecast(method, args, train, val, test)
        line = f"traditional_forecast_{args.method}_{args.data}_pl{args.pred_len}"
        print(f"{line}\nmse:{res['mse']:.6f}, mae:{res['mae']:.6f}")
        with open(os.path.join(ML, "result_traditional_forecast.txt"), "a", encoding="utf-8") as f:
            f.write(f"{line}  \n")
            f.write(f"mse:{res['mse']:.6f}, mae:{res['mae']:.6f}\n\n")
    elif args.task == "anomaly":
        Xtr, Xte, yte = load_anomaly(args)
        res = run_anomaly(method, args, Xtr, Xte, yte)
        line = f"traditional_anomaly_{args.method}_{args.data}_ratio{args.anomaly_ratio}"
        print(f"{line}\nAccuracy : {res['accuracy']:.4f}, Precision : {res['precision']:.4f}, "
              f"Recall : {res['recall']:.4f}, F-score : {res['f1']:.4f}")
        with open(os.path.join(ML, "result_traditional_anomaly.txt"), "a", encoding="utf-8") as f:
            f.write(f"{line}  \n")
            f.write(f"Accuracy : {res['accuracy']:.4f}, Precision : {res['precision']:.4f}, "
                    f"Recall : {res['recall']:.4f}, F-score : {res['f1']:.4f}\n\n")
    else:
        ds = args.dataset or os.path.basename(os.path.normpath(args.root_path))
        Xs_tr, y_tr = load_uea(ds, "TRAIN")
        Xs_te, y_te = load_uea(ds, "TEST")
        res = run_classification(method, args, Xs_tr, y_tr, Xs_te, y_te)
        line = f"traditional_classification_{args.method}_{ds}"
        print(f"{line}\naccuracy:{res['accuracy']:.6f}")
        with open(os.path.join(ML, "result_traditional_classification.txt"), "a", encoding="utf-8") as f:
            f.write(f"{line}  \n")
            f.write(f"accuracy:{res['accuracy']:.6f}\n\n")


if __name__ == "__main__":
    main()
