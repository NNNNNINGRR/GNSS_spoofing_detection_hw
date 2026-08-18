"""计算 SQM 指标（50 Hz），输出每场景每通道 CSV。

对齐说明（Paper A: Iqbal et al., IEEE TMLCN 2024）：
- 基础 4 指标（ratio/delta/elp/symdiff）论文用 d=0.5 chip 相关器；
  本接收机 E/L=±0.25 chip（GNSS-SDR 默认 early_late_space_chips=0.25），
  VE/VL=±0.5 chip（very_early_late_space_chips=0.5），
  因此主口径用 VE/VL 计算（需要 v3.1 及以后 dump：v3.0 的 VE/VL 恒为 0）。
  0.25 chip 的 E/L 版本保留为 *_e025 列，仅追溯用，不进数据集特征。
- 双 delta（double delta，Pirsiavash/Broumandan/Lachapelle, ITSNT 2017, eq.(5)）：
  m_dd = [(I_{-0.5} − I_{+0.5}) − (I_{-0.1} − I_{+0.1})] / I_0，
  监测对（宽间距）减跟踪对（窄间距）之差，除以 prompt（同时消除导航数据位符号）；
  对称相关器下无攻击名义均值为 0（实测 +0.19 恒定偏置来自带限 ACF 曲率与间距差，
  训练/测试同分布，不影响异常检测）。本文间距：监测对 ±0.5 chip（VE/VL）、
  跟踪对 ±0.25 chip（E/L，DLL 实际间距，Pirsiavash 用 0.2 chip）。
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import CN0_MIN, FS, SCENARIOS
from parse_tracking import is_20ms_record, read_dat


def calibrate_sigma_N0(scenarios_dir, calib="cs"):
    """sigma_N0 = cs（无欺骗）场景全部通道 20 ms 记录 |psi_VE| 的标准差。

    与 symdiff 指标同间距（±0.5 chip，Paper A 的 d=0.5）。
    """
    vals = []
    for ch in range(8):
        p = os.path.join(scenarios_dir, calib, f"trk_ch_{ch}.dat")
        if not os.path.exists(p):
            continue
        rec = read_dat(p)
        m = is_20ms_record(rec) & (rec["CN0_SNV_dB_Hz"] >= CN0_MIN)
        vals.append(np.abs(rec["VeryEarly_I"][m] + 1j * rec["VeryEarly_Q"][m]))
    if not vals:
        raise FileNotFoundError(f"找不到标定场景 {calib} 的 dump")
    sigma = float(np.std(np.concatenate(vals)))
    if sigma == 0.0:
        raise ValueError(
            "sigma_N0=0：dump 的 VE/VL 全为零（v3.0 旧数据）。"
            "基础指标需 ±0.5 chip 相关器，请用 v3.1 及以后的解算结果。")
    return sigma


def check_veml_valid(rec, scenario, ch):
    """v3.0 dump 的 VE/VL 恒为 0，无法计算 0.5 chip 指标，直接报错。"""
    if np.all(rec["VeryEarly_I"] == 0.0) and np.all(rec["VeryLate_I"] == 0.0):
        raise ValueError(
            f"{scenario}_ch{ch}: VE/VL 全为零（v3.0 数据）。"
            "0.5 chip 指标与双 delta 需要 v3.1 解算结果（results/原始解算/v3.1）。")


def prompt_power_dB(I_P, Q_P, n_baseline=100):
    """逐通道相对信号功率（prompt 功率代理，dB）。

    |P|^2 与该卫星解扩后载波功率成正比（未标定、逐通道）。
    基线取该通道前 n_baseline 个有效 20 ms 记录（约 2 s，环路已稳定、
    欺骗尚未注入）的 |P|^2 中位数；记录不足时回退到全序列中位数。
    当前阶段用其填充 received_power（论文 eq.(6) 的逐通道代理），
    待离线 received_power.py 完成后可合并覆盖为接收机级真值。
    """
    p2 = I_P.astype(np.float64) ** 2 + Q_P.astype(np.float64) ** 2
    n = min(n_baseline, len(p2))
    base = float(np.median(p2[:n])) if n > 0 else 0.0
    if base <= 0.0:
        base = float(np.median(p2)) if len(p2) else 1.0
    if base <= 0.0:
        base = 1.0
    return 10.0 * np.log10(np.maximum(p2, 1e-30) / base)


def compute_metrics(rec, sigma_N0):
    """对 20 ms 记录向量化计算指标（0.5 chip 主口径 + 双 delta + 0.25 chip 追溯列）。"""
    # ±0.5 chip 对（Paper A 的 d=0.5）
    I_VE, Q_VE = rec["VeryEarly_I"], rec["VeryEarly_Q"]
    I_VL, Q_VL = rec["VeryLate_I"], rec["VeryLate_Q"]
    # ±0.25 chip 对（DLL 跟踪对，双 delta 的第二对）
    I_E, Q_E = rec["Early_I"], rec["Early_Q"]
    I_L, Q_L = rec["Late_I"], rec["Late_Q"]
    I_P, Q_P = rec["Prompt_I"], rec["Prompt_Q"]
    P = I_P + 1j * Q_P

    m_ratio = (I_VE + I_VL) / I_P
    m_delta = (I_VE - I_VL) / I_P
    m_elp = np.arctan2(Q_VE, I_VE) - np.arctan2(Q_VL, I_VL)
    m_symdiff = np.abs((I_VE - I_VL) + 1j * (Q_VE - Q_VL)) / sigma_N0

    Ex = sum(rec[f"MF_I_{i}"] + 1j * rec[f"MF_Q_{i}"] for i in range(4))      # d<0 早侧
    Lx = sum(rec[f"MF_I_{i}"] + 1j * rec[f"MF_Q_{i}"] for i in range(5, 9))  # d>0 晚侧
    m_manfredini = np.abs(Ex - Lx) / np.abs(P)

    # 双 delta：监测对(±0.5) E−L 差 与 跟踪对(±0.25) E−L 差 之差，按 Delta 惯例除以 I_P
    m_dd = ((I_VE - I_VL) - (I_E - I_L)) / I_P

    # 0.25 chip 旧口径（追溯列，不进数据集特征）
    m_ratio_e025 = (I_E + I_L) / I_P
    m_delta_e025 = (I_E - I_L) / I_P
    m_elp_e025 = np.arctan2(Q_E, I_E) - np.arctan2(Q_L, I_L)

    pp_db = prompt_power_dB(I_P, Q_P)

    return pd.DataFrame(
        {
            "time_s": rec["PRN_start_sample_count"].astype(np.float64) / FS,
            "m_ratio": m_ratio,
            "m_delta": m_delta,
            "m_elp": m_elp,
            "m_symdiff": m_symdiff,
            "m_manfredini": m_manfredini,
            "m_dd": m_dd,
            "received_power": pp_db,     # 当前 = prompt 功率代理（逐通道）
            "prompt_power_dB": pp_db,    # 逐通道相对信号功率（独立列，可追溯）
            "m_ratio_e025": m_ratio_e025,
            "m_delta_e025": m_delta_e025,
            "m_elp_e025": m_elp_e025,
            "CN0": rec["CN0_SNV_dB_Hz"],
            "carrier_doppler_hz": rec["carrier_doppler_hz"],
            "PRN": rec["PRN"],
        }
    )


def merge_received_power(df, received_power_dir, scenario):
    """按 20 ms 网格对齐接收机级 received_power（离线 eq.(6)），覆盖代理值。"""
    p = os.path.join(received_power_dir, f"received_power_{scenario}.csv")
    if not os.path.exists(p):
        return df
    rp = pd.read_csv(p)
    t = np.round(df["time_s"].values * 50) / 50.0
    rt = np.round(rp["time_s"].values * 50) / 50.0
    idx = np.searchsorted(rt, t)
    idx = np.clip(idx, 0, len(rt) - 1)
    df["received_power"] = rp["received_power_dB"].values[idx]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios_dir", required=True, help="含 <场景>/trk_ch_*.dat 的目录")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--calib_scenario", default="cs")
    ap.add_argument("--received_power_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sigma_N0 = calibrate_sigma_N0(args.scenarios_dir, args.calib_scenario)
    with open(os.path.join(args.output_dir, "sigma_N0.txt"), "w") as f:
        f.write(f"sigma_N0 = {sigma_N0:.6f} (标定场景: {args.calib_scenario}, 相关器: |psi_VE| @0.5 chip)\n")
    print(f"sigma_N0 = {sigma_N0:.6f}")

    for s in SCENARIOS:
        for ch in range(8):
            p = os.path.join(args.scenarios_dir, s, f"trk_ch_{ch}.dat")
            if not os.path.exists(p):
                continue
            rec = read_dat(p)
            m = is_20ms_record(rec) & (rec["CN0_SNV_dB_Hz"] >= CN0_MIN) & (rec["carrier_lock_test"] > 0)
            if m.sum() == 0:
                continue
            check_veml_valid(rec[m], s, ch)
            df = compute_metrics(rec[m], sigma_N0)
            if args.received_power_dir:
                df = merge_received_power(df, args.received_power_dir, s)
            out = os.path.join(args.output_dir, f"{s}_ch{ch}.csv")
            df.to_csv(out, index=False)
            print(f"{os.path.basename(out)}: {len(df)} 行, PRN={int(df['PRN'].iloc[0])}, "
                  f"CN0均值={df['CN0'].mean():.1f} dB-Hz, "
                  f"m_dd均值={df['m_dd'].mean():+.4f}, "
                  f"received_power(代理) 均值={df['received_power'].mean():.2f} dB")


if __name__ == "__main__":
    main()
