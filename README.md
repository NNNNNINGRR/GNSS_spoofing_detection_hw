# GNSS_spoofing_detection_hw

基于时间序列方法库（method_lib）的 GNSS 欺骗检测实验代码。

## 包含模型
- Bi_FI、LightTS、DLinear（异常检测：窗口重建误差）

## 代码适配（相对原始方法库）
- `exp/exp_anomaly_detection.py`
  - 阈值改为 train-only（`percentile(train_energy, 100-anomaly_ratio)`），消除测试泄漏；
  - 测试时保存 `score.npy / label.npy / pred_raw.npy / threshold.npy` 供统一评估；
  - 支持 `--init_checkpoint` 从预训练权重初始化（V2 预训练+微调）。
- `run.py`：新增 `--init_checkpoint` 参数。

## 实验协议
- 数据：TEXBAT 异常检测集（Train=cs/cd 正常，Test=ds1-8，50 Hz，7 特征）
- 窗口：seq_len=96（≈1.92 s）
- 评估：ROC-AUC / PR-AUC / TPR@FPR / F1 / MCC / 检测延迟（ADD）等
