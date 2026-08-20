# 归档：深度模型 GNSS 欺骗检测——工作结果与问题（冻结）

**日期**：2026-08-18　**状态**：深度模型线冻结，转向传统方法　**数据**：TEXBAT v3.1（GNSS-SDR 解算）

---

## 0. 一句话结论

在"训练只见清洁数据（cs+cd）、清洁标定阈值、部署即用"的口径下，
**三个深度模型（Bi_FI / LightTS / DLinear）全部无法检测出欺骗**：
清洁标定阈值在任何测试场景都从未被触发（tpr@cf ≈ 0、实际 FPR ≈ 0、ADD 全部 inf），
即判决器输出恒为"无攻击"。模型有排序能力（DLinear 宏平均 AUC 0.958），
但**工作点完全失效**，不构成可部署的检测器。

## 1. 实验设置（冻结时的最终协议）

- **特征**（8 维，50 Hz 逐帧，`build_metrics.py`）：
  m_ratio / m_delta / m_elp / m_symdiff（±0.5 chip，VE/VL 口径，对齐 Iqbal 2024 的 d=0.5）、
  m_manfredini（MF×9，±0.1016 chip）、m_dd（双 delta，Pirsiavash ITSNT 2017 式(5)）、
  received_power（prompt 功率代理）、CN0
- **数据**：v3.1 解算（VE/VL 有效）；每场景取 CN0 均值最高通道
- **训练**：Train = cs + cd 拼接（45476 帧，仅清洁）；Test = ds1–ds8 拼接（176799 帧，欺骗帧占 76.8%）
- **模型**：Bi_FI / LightTS / DLinear，seq_len=96（1.92 s 窗口），重建误差（L1 均值）为异常分数
- **阈值**：训练（清洁）分数按 FPR 0.1% / 1% / 5% 三档分位标定（thresholds.npy）
- **评测**：`eval_smoke.py --manifest` 逐场景；事后 TPR@FPR（排序能力）+ 工作点 tpr@cf / rfpr + ADD（连续 3 帧告警）
- 运行：`run_full_gnss.sh`（云端，seed=2 单次）

## 2. 结果汇总（完整逐场景表见 `cscd_{Bi_FI,LightTS,DLinear}_s2_metrics.csv`）

| 模型 | 宏平均 ROC-AUC | 宏平均 TPR@FPR1% | TPR@FPR5% | 工作点 tpr@cf1（8 场景） | ADD 命中 |
|---|---|---|---|---|---|
| DLinear | 0.958 | 0.573 | 0.889 | **全部 ≈ 0** | 0/8 |
| LightTS | 0.741 | 0.132 | 0.322 | ≈ 0（仅 ds2/ds5 恰好触发，ADD 60/77 s） | 2/8 |
| Bi_FI | 0.549 | 0.033 | 0.112 | 全部 ≈ 0 | 0/8 |

场景难度（三模型一致）：ds1≈ds4≈ds5（易，AUC>0.99）> ds2≈ds8 > ds6 > ds7 > **ds3（最难：位置推送、0.4 dB 功率优势，AUC 0.827）**。

## 3. 问题清单

| # | 问题 | 证据 | 严重度 |
|---|---|---|---|
| P0 | **清洁标定阈值跨场景迁移失效**：训练（cs ch0 + cd ch4）与测试（各 ds 最高 CN0 通道）的重建误差分布整体漂移，阈值恒高于测试分数 | tpr@cf1≈0 且 rfpr@1≈0（同时为零=从未触发，不是漏检而是不判决）；DLinear threshold@1%=0.4495 对 8 个场景无一适用 | 致命（=本次冻结的直接原因） |
| P1 | 低畸变攻击特征区分度不足：ds3（位置推送 0.4 dB）与 ds6（sophisticated 载波对齐）在严虚警档（FPR1%）检出率骤降 | DLinear ds3 tpr@fpr1=0.117、ds6=0.052（对应 tpr@fpr5=0.535/0.815） | 高 |
| P2 | Bi_FI 系统性反向：攻击期特征更平稳 → 重建误差更低 → 分数反向 | ds5/ds6 AUC 0.37/0.40（<0.5） | 高（该模型不适用） |
| P3 | 单种子单次运行，无方差估计；模型间差异显著性未知 | seed=2 only | 中 |
| P4 | F1/MCC/BalAcc 列因 P0 退化为"全判无攻击"特征值（BalAcc=0.5、MCC≈0），已无信息量 | 全部 CSV | 中 |
| P5 | 窗口跨场景边界的 96 个混合窗按末帧归属场景（影响 <0.5% 帧） | eval 对齐逻辑 | 低 |

## 4. 根因分析

1. **分数尺度依赖通道/场景**：重建误差与各通道噪声水平（CN0）、特征尺度强相关。
   cs（ch0，CN0≈52）/ cd（ch4）训练的分数分布 ≠ 各 ds 通道的分数分布。
   Paper A 是逐点模型 + 逐点归一化特征（[0,1] 缩放 + 静态/动态条件标签），
   我们用窗口重建误差替代后丢掉了这层尺度不变性。
2. **窗口重建范式与 SQM 特征性质不匹配**：欺骗接管完成后跟踪环重新"干净"地锁在欺骗信号上，
   窗口内特征自相关性反而更好 → 重建误差回落（P2 的直接原因）。
   检测依赖的应是"与清洁总体分布的偏离"（点级密度/距离），而非"时序可预测性"。
3. **ds3 类攻击本就不在 SQM 特征的检测能力射程内**（Paper B 用码-载波一致性物理量解决此类）。

## 5. 已验证有效、保留复用的资产

- v3.1 数据管线（build_metrics：0.5 chip 口径 + 双 delta）、make_datasets cs+cd 数据集 + manifest
- eval_smoke.py manifest 逐场景评测（三档清洁阈值 + 事后 ROC + ADD）
- 场景语义表（本地 dsN = TEXBAT ds(N+1)；动态 = ds4/ds5；sophisticated = ds6）
- 云端复现：`SEEDS=.. bash run_full_gnss.sh`；GitHub `GNSS_spoofing_detection_hw@8d6ffb9`

## 6. 去向：传统方法线（下一步）

深度模型的问题 1、2 都指向**逐帧点式判决**的传统单类方法——与 Paper A 同粒度：
- method_lib/traditional 8 个异常方法：IsolationForest / OneClassSvm / Mahalanobis /
  Lof / PcaReconstruction / KnnEd / KnnDtw×2（fit 清洁 / score 越大越异常）
- 另加统计 SQM 基线（逐特征对清洁标定的 z 检验，Paper A 对照的 Wesson/Manfredini/Sun 一类）
- 统一用 SQM_cscd + manifest 评测（`--win 1` 逐帧口径），清洁标定三档阈值

## 7. 复现索引

- 数据集：`results/datasets/v3.1/anomaly_cscd/`（本地）＝ `/root/autodl-tmp/exp_gnss/data/SQM_cscd`（云端）
- 结果：`exp_gnss/results/cscd/`（本地）＝ `experiment_results/cscd/`（GitHub）
- 训练日志：`full_cscd.log`；图：`figures/cscd/cscd_{模型}_s2.png`
