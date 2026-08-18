"""共享常量与 212 字节 tracking dump 的 dtype 定义。"""
import numpy as np

FS = 25_000_000.0  # TEXBAT 采样率
SCENARIOS = ["cs", "cd", "ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7", "ds8"]
CLEAN_SCENARIOS = ["cs", "cd"]
SPOOF_SCENARIOS = ["ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7", "ds8"]
CN0_MIN = 30.0  # 有效历元的最低 CN0 (dB-Hz)

# 212 字节/记录，字段顺序与 dll_pll_veml_tracking::log_data() 写入顺序一致（小端）
_MF_IQ = [(f"MF_I_{i}", "<f4") for i in range(9)] + [(f"MF_Q_{i}", "<f4") for i in range(9)]
TRACKING_DTYPE = np.dtype(
    [
        ("abs_VE", "<f4"), ("abs_E", "<f4"), ("abs_P", "<f4"), ("abs_L", "<f4"), ("abs_VL", "<f4"),
        ("VeryEarly_I", "<f4"), ("VeryEarly_Q", "<f4"),
        ("Early_I", "<f4"), ("Early_Q", "<f4"),
        ("Late_I", "<f4"), ("Late_Q", "<f4"),
        ("VeryLate_I", "<f4"), ("VeryLate_Q", "<f4"),
        ("Prompt_I", "<f4"), ("Prompt_Q", "<f4"),
        ("PRN_start_sample_count", "<u8"),
        ("acc_carrier_phase_rad", "<f4"), ("carrier_doppler_hz", "<f4"),
        ("carrier_doppler_rate_hz", "<f4"), ("code_freq_chips", "<f4"),
        ("code_freq_rate_chips", "<f4"), ("carr_error_hz", "<f4"),
        ("carr_error_filt_hz", "<f4"), ("code_error_chips", "<f4"),
        ("code_error_filt_chips", "<f4"), ("CN0_SNV_dB_Hz", "<f4"),
        ("carrier_lock_test", "<f4"), ("aux1", "<f4"), ("aux2", "<f8"),
        ("PRN", "<u4"), ("TOW_ms", "<u8"), ("WN", "<i4"),
    ]
    + _MF_IQ
)

RECORD_BYTES = TRACKING_DTYPE.itemsize  # 应为 212


def psi(rec, name):
    """由 I/Q 字段构造复数相关器输出。"""
    return rec[name + "_I"] + 1j * rec[name + "_Q"]
