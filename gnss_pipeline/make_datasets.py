"""由指标 CSV 生成三数据集：UEA 分类 / PSM 异常检测 / custom 传统模型。

Paper A 对齐（Iqbal et al. 2024）：
- 特征集 = 7 特征（0.5 chip 口径）+ 双 delta = 8 维；
- 异常检测训练集 = cs + cd 两个清洁场景拼接（论文用 DS-0+DS-1 训练单一模型），
  测试 = 全部欺骗场景拼接，附 manifest（逐场景行号/起始时刻）供逐场景评测。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import CLEAN_SCENARIOS, SCENARIOS, SPOOF_SCENARIOS

# 主特征：Paper A 7 特征（基础 4 指标为 ±0.5 chip 口径）+ 双 delta
METRIC_COLS = ["m_ratio", "m_delta", "m_elp", "m_symdiff", "m_manfredini", "m_dd", "received_power"]
FEATURE_COLS = METRIC_COLS + ["CN0"]

# 各欺骗场景攻击信号起始时刻（秒，相对本场景 .bin 文件起点）
# 以 TEXBAT 原始文档与论文为准（本地数据仅交叉验证，详见 docs/06_TEXBAT_攻击起始时间_来源与交叉验证.md）：
#   - ds1: Lemmenes et al., ION GNSS+ 2016 Table 2（spoofing signal onset=125；原始 2012 论文仅定性 ~90-100s，已被修订值取代）
#   - ds2: Lemmenes 2016（110.1）+ Iqbal/Asif (NUS) Table I（110）
#   - ds3: Lemmenes 2016（118.9）+ NUS（120）
#   - ds4: Lemmenes 2016（113.8）+ NUS（114）
#   - ds5: NUS Table I（102）
#   - ds6: NUS Table I（105）；MDPI Sensors 2017（约 110-150s 注入，一致）
#   - ds7/ds8: Humphreys, TEXBAT Data Sets 7 and 8 (2016) 技术报告（0-110s 无欺骗，110s 起注入）
# 注意编号映射：本地 dsN = TEXBAT 官方 ds(N+1)；Paper A 的 DS-7（sophisticated）= 本地 ds6。
# 本地交叉验证（v3.0 解算，CN0 最高通道）：ds1-6 功率跳变点与上表 ±1s 吻合；ds7 按设计无功率跳变（合成相量幅度恒定），CN0/Doppler 变化佐证。
ATTACK_ONSET_S = {
    "ds1": 125.0,
    "ds2": 110.0,
    "ds3": 120.0,
    "ds4": 114.0,
    "ds5": 102.0,
    "ds6": 105.0,
    "ds7": 110.0,
    "ds8": 110.0,
}

# 场景分组（TEXBAT 官方定义，经 docs/ 原始文档复核）：
#   - Humphreys et al. 2012 ION GNSS（tbION_for_distribution.pdf）Table I：
#     场景 1-4 = Static、场景 5-6 = Dynamic（基于 cleanDynamic(cd)）
#   - Lemmenes et al. 2016（LemmenesGNSSpaper.pdf）："six using a static antenna and
#     two using a moving antenna"；其 Table 1（static scenario offsets）含 1/2/3/4/7/8
# 静态：ds1~ds4、ds7、ds8（基于 cleanStatic cs）；动态：ds5、ds6（基于 cleanDynamic cd）
STATIC_SCENARIOS = ["ds1", "ds2", "ds3", "ds4", "ds7", "ds8"]
DYNAMIC_SCENARIOS = ["ds5", "ds6"]
SCENARIO_TRAIN = {s: "cs" for s in STATIC_SCENARIOS}
SCENARIO_TRAIN.update({s: "cd" for s in DYNAMIC_SCENARIOS})


def available_scenarios(metrics_dir):
    """只返回存在指标文件的场景。"""
    return [s for s in SCENARIOS if any(
        os.path.exists(os.path.join(metrics_dir, f"{s}_ch{ch}.csv")) for ch in range(8))]


def load_scenario_metrics(metrics_dir, scenario):
    """合并某场景全部通道的指标 CSV，返回 (df, best_channel)。"""
    frames = []
    best_ch, best_cn0 = None, -1.0
    for ch in range(8):
        p = os.path.join(metrics_dir, f"{scenario}_ch{ch}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        cn0 = df["CN0"].mean()
        if cn0 > best_cn0:
            best_cn0, best_ch = cn0, ch
        frames.append((ch, df))
    if not frames:
        raise FileNotFoundError(f"{scenario} 无指标文件")
    return frames, best_ch


def check_features(df, scenario):
    """v3.0 指标缺少 m_dd（且基础指标为 0.25 chip 口径），拒绝构建。"""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{scenario}: 指标缺少列 {missing}。当前特征集需要 v3.1 解算 + "
            "新版 build_metrics.py（0.5 chip 口径 + m_dd）。")


def build_classification(metrics_dir, out_dir, window=128, stride=64, binary=False):
    """UEA .ts：每通道滑窗，8 维指标（含 CN0），标签=场景（或 clean/spoof）。

    切分按场景整块 70/30（时间序列滑窗高度相关，随机切分会把重叠窗口同时
    放进训练/测试造成泄漏——Paper A IV-B1 明确警示过这一点）。
    """
    cases, labels, scen_of = [], [], []
    for s in available_scenarios(metrics_dir):
        frames, _ = load_scenario_metrics(metrics_dir, s)
        label = "spoof" if (binary and s in SPOOF_SCENARIOS) else ("clean" if binary else s)
        for ch, df in frames:
            check_features(df, s)
            vals = df[FEATURE_COLS].values.astype(np.float64)
            if len(vals) < window:
                continue
            for i in range(0, len(vals) - window + 1, stride):
                cases.append(vals[i : i + window])
                labels.append(label)
                scen_of.append(s)
    if not cases:
        raise RuntimeError("没有足够数据生成分类数据集（窗口大于序列长度？）")
    cases = np.asarray(cases)
    labels = np.asarray(labels)
    scen_of = np.asarray(scen_of)

    # 按场景分块：每个场景的窗口随机打乱后，场景内前 70% 进训练（块间不混）
    rng = np.random.default_rng(42)
    tr_idx, te_idx = [], []
    for s in sorted(set(scen_of)):
        idx = np.where(scen_of == s)[0]
        rng.shuffle(idx)
        k = int(len(idx) * 0.7)
        tr_idx.extend(idx[:k])
        te_idx.extend(idx[k:])
    tr_idx, te_idx = np.asarray(sorted(tr_idx)), np.asarray(sorted(te_idx))
    write_uea(cases[tr_idx], labels[tr_idx], os.path.join(out_dir, "classification", "SQM_TEXBAT_TRAIN.ts"))
    write_uea(cases[te_idx], labels[te_idx], os.path.join(out_dir, "classification", "SQM_TEXBAT_TEST.ts"))
    print(f"分类: {len(cases)} cases (train {len(tr_idx)} / test {len(te_idx)}), "
          f"标签={sorted(set(labels))}, 场景块切分（无重叠泄漏）")


def write_uea(cases, labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    class_names = sorted(set(labels))
    with open(path, "w", encoding="utf-8") as f:
        f.write("@problemName SQM_TEXBAT\n")
        f.write("@timeStamps false\n@missing false\n@univariate false\n")
        f.write(f"@dimensions {cases.shape[-1]}\n@equalLength true\n")
        f.write(f"@classLabel true {len(class_names)} {' '.join(class_names)}\n@data\n")
        for c, lab in zip(cases, labels):
            line = ":".join(",".join(f"{v:.6f}" for v in dim) for dim in c.T)
            f.write(f"{line}:{lab}\n")


def _scenario_frame(metrics_dir, scenario, best_ch):
    """取某场景 best_ch 通道的指标，转 PSM 格式（date + 特征列）。"""
    df = pd.read_csv(os.path.join(metrics_dir, f"{scenario}_ch{best_ch}.csv"))
    check_features(df, scenario)
    time_s = df["time_s"].values  # 相对本场景 .bin 起点
    df = df[FEATURE_COLS].copy()
    df.insert(0, "date", pd.to_datetime(time_s, unit="s", origin="1970-01-01"))
    return df, time_s


def build_anomaly(metrics_dir, out_dir, train="cs", test=None, use_all_spoof=True):
    """PSM 风格（旧口径）：Train=单一清洁场景，Test=欺骗场景，标签 0/1。"""
    if test is None:
        test = SPOOF_SCENARIOS if use_all_spoof else ["ds3", "ds7"]
    _, best_ch = load_scenario_metrics(metrics_dir, train)
    tr_df, _ = _scenario_frame(metrics_dir, train, best_ch)

    te_parts, lab_parts = [], []
    for s in test:
        _, best_ch = load_scenario_metrics(metrics_dir, s)
        df, te_time_s = _scenario_frame(metrics_dir, s, best_ch)
        te_parts.append(df)
        onset = ATTACK_ONSET_S.get(s, 0.0)
        lab_parts.append(pd.DataFrame({"date": df["date"],
                                       "label": (te_time_s >= onset).astype(int)}))
    te_df = pd.concat(te_parts, ignore_index=True)
    lab_df = pd.concat(lab_parts, ignore_index=True)

    out = os.path.join(out_dir, "anomaly")
    os.makedirs(out, exist_ok=True)
    tr_df.to_csv(os.path.join(out, "Train.csv"), index=False)
    te_df.to_csv(os.path.join(out, "Test.csv"), index=False)
    lab_df.to_csv(os.path.join(out, "Test_label.csv"), index=False)
    print(f"异常检测(旧口径): Train={len(tr_df)} ({train}), Test={len(te_df)} (场景 {test})")


def build_anomaly_cscd(metrics_dir, out_dir, test=None):
    """Paper A 口径：单一模型，Train = cs + cd 拼接（全部清洁数据），
    Test = 全部欺骗场景拼接；附 manifest 记录逐场景行界与攻击起始时刻。

    manifest 结构：
      {"train": {"cs": {"rows": n, "channel": ch}, "cd": {...}},
       "test": [{"scenario": "ds1", "start": 0, "rows": n, "onset_s": 125.0,
                 "type": "static"}, ...],
       "features": FEATURE_COLS}
    """
    if test is None:
        test = SPOOF_SCENARIOS
    out = os.path.join(out_dir, "anomaly_cscd")
    os.makedirs(out, exist_ok=True)

    manifest = {"features": FEATURE_COLS, "train": {}, "test": []}

    tr_parts = []
    for clean in ["cs", "cd"]:
        if clean not in available_scenarios(metrics_dir):
            raise SystemExit(f"缺少清洁场景 {clean} 的指标，无法构建 cs+cd 训练集")
        _, best_ch = load_scenario_metrics(metrics_dir, clean)
        df, _ = _scenario_frame(metrics_dir, clean, best_ch)
        tr_parts.append(df)
        manifest["train"][clean] = {"rows": len(df), "channel": int(best_ch)}
    tr_df = pd.concat(tr_parts, ignore_index=True)

    te_parts, lab_parts = [], []
    start = 0
    for s in test:
        if s not in available_scenarios(metrics_dir):
            print(f"警告：场景 {s} 无指标，跳过")
            continue
        _, best_ch = load_scenario_metrics(metrics_dir, s)
        df, te_time_s = _scenario_frame(metrics_dir, s, best_ch)
        onset = ATTACK_ONSET_S.get(s, 0.0)
        lab = (te_time_s >= onset).astype(int)
        te_parts.append(df)
        lab_parts.append(pd.DataFrame({"date": df["date"], "label": lab}))
        manifest["test"].append({
            "scenario": s,
            "start": start,
            "rows": len(df),
            "onset_s": onset,
            "type": "dynamic" if s in DYNAMIC_SCENARIOS else "static",
            "channel": int(best_ch),
        })
        start += len(df)
    te_df = pd.concat(te_parts, ignore_index=True)
    lab_df = pd.concat(lab_parts, ignore_index=True)

    tr_df.to_csv(os.path.join(out, "Train.csv"), index=False)
    te_df.to_csv(os.path.join(out, "Test.csv"), index=False)
    lab_df.to_csv(os.path.join(out, "Test_label.csv"), index=False)
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    n_spoof = int(lab_df["label"].sum())
    print(f"异常检测(cs+cd): Train={len(tr_df)} "
          f"(cs ch{manifest['train']['cs']['channel']} + cd ch{manifest['train']['cd']['channel']}), "
          f"Test={len(te_df)} 帧跨 {len(manifest['test'])} 场景, 欺骗帧={n_spoof} "
          f"({n_spoof/len(te_df)*100:.1f}%), manifest 已写入")


def build_traditional(metrics_dir, out_dir, target="CN0"):
    """custom CSV：date + 8 指标（含 CN0 与 m_dd）+ OT（目标）。"""
    out = os.path.join(out_dir, "traditional")
    os.makedirs(out, exist_ok=True)
    for s in available_scenarios(metrics_dir):
        frames, _ = load_scenario_metrics(metrics_dir, s)
        df = pd.concat([f for _, f in frames], ignore_index=True)
        check_features(df, s)
        df["date"] = pd.to_datetime(df["time_s"], unit="s", origin="1970-01-01")
        cols = ["date"] + FEATURE_COLS + ["OT"]
        df["OT"] = df[target]
        df[cols].to_csv(os.path.join(out, f"{s}.csv"), index=False)
        print(f"传统: {s}.csv {len(df)} 行")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios_dir", required=True, help="metrics 目录（含 <场景>_ch<k>.csv）")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--binary", action="store_true", help="分类用 clean/spoof 二类")
    ap.add_argument("--anomaly_test", default="ds3,ds7", help="异常测试场景，逗号分隔；all=全部欺骗")
    ap.add_argument("--anomaly_train", default=None, choices=["cs", "cd", "cs+cd"],
                    help="训练清洁场景；cs+cd=Paper A 口径（单模型，cs+cd 拼接训练，全部欺骗测试，附 manifest）")
    ap.add_argument("--anomaly_only", action="store_true",
                    help="只生成异常数据集（跳过分类/传统，用于逐场景快速制作）")
    args = ap.parse_args()

    if args.anomaly_train == "cs+cd":
        test = SPOOF_SCENARIOS if args.anomaly_test == "all" else args.anomaly_test.split(",")
        avail = available_scenarios(args.scenarios_dir)
        test = [t for t in test if t in avail]
        if not test:
            raise SystemExit("没有可用的欺骗场景指标")
        build_anomaly_cscd(args.scenarios_dir, args.output_dir, test=test)
        return

    if not args.anomaly_only:
        build_classification(args.scenarios_dir, args.output_dir, args.window, args.stride, args.binary)
    test = SPOOF_SCENARIOS if args.anomaly_test == "all" else args.anomaly_test.split(",")
    avail = available_scenarios(args.scenarios_dir)
    test = [t for t in test if t in avail]
    if not test:
        print("警告：没有可用的欺骗场景指标，跳过异常数据集")
    else:
        if args.anomaly_train:
            train = args.anomaly_train
        else:
            trains = {SCENARIO_TRAIN.get(s, "cs") for s in test}
            if len(trains) > 1:
                raise SystemExit(
                    f"测试场景同时含静态与动态（{sorted(trains)}），无法用单一清洁场景训练，"
                    "请按场景分别生成（--anomaly_test 单场景）或显式指定 --anomaly_train")
            train = trains.pop()
        build_anomaly(args.scenarios_dir, args.output_dir, train=train, test=test)
    if not args.anomaly_only:
        build_traditional(args.scenarios_dir, args.output_dir)


if __name__ == "__main__":
    main()
