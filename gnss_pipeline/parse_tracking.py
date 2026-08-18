"""解析 212 字节 tracking dump：读取 -> 选择 20 ms 记录 -> 清洗。"""
import argparse
import os

import numpy as np

from common import CN0_MIN, FS, RECORD_BYTES, TRACKING_DTYPE


def read_dat(path):
    size = os.path.getsize(path)
    if size % RECORD_BYTES != 0:
        raise ValueError(f"{path}: size {size} 不是 {RECORD_BYTES} 的整数倍（版本不匹配或被中断）")
    return np.fromfile(path, dtype=TRACKING_DTYPE)


def is_20ms_record(rec):
    """位同步后、导航位边界的 20 ms 累积记录：MF 累加器非零。"""
    return (rec["MF_I_0"] != 0.0) | (rec["MF_Q_0"] != 0.0)


def clean(rec, cn0_min=CN0_MIN):
    """只保留 20 ms 记录，并剔除 CN0 过低或失锁历元。返回 (rec, time_s)。"""
    rec = rec[is_20ms_record(rec)]
    ok = (rec["CN0_SNV_dB_Hz"] >= cn0_min) & (rec["carrier_lock_test"] > 0.0)
    rec = rec[ok]
    time_s = rec["PRN_start_sample_count"].astype(np.float64) / FS
    return rec, time_s


def main():
    ap = argparse.ArgumentParser(description="解析单通道 tracking dump")
    ap.add_argument("--dat", required=True, help="trk_ch_<k>.dat 路径")
    ap.add_argument("--out", required=True, help="输出 .npz 路径")
    ap.add_argument("--cn0_min", type=float, default=CN0_MIN)
    args = ap.parse_args()
    rec = read_dat(args.dat)
    rec, time_s = clean(rec, args.cn0_min)
    np.savez_compressed(args.out, rec=rec, time_s=time_s)
    n20 = int(is_20ms_record(read_dat(args.dat)).sum())
    print(f"{os.path.basename(args.dat)}: 总记录={len(read_dat(args.dat))}, "
          f"20ms 有效={n20}, 清洗后={len(rec)} -> {args.out}")


if __name__ == "__main__":
    main()
