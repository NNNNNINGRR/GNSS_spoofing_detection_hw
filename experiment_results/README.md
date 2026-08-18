# GNSS 欺骗检测全量实验结果（Bi-FI / LightTS / DLinear）

- 版本一（V1）：TEXBAT 正常数据从头训练（10 epochs, lr=1e-4）
- 版本二（V2）：ETTh1 -> ETTm1 预训练（各 10 epochs）-> TEXBAT 微调（5 epochs, lr=5e-5）
- 数据：TEXBAT 异常检测集（Train=cs/cd 正常，Test=ds1-8，50 Hz，7 特征，seq_len=96）
- 评估：ROC-AUC / PR-AUC / TPR@FPR(1%,5%) / F1 / MCC / BalancedAcc / ADD(K=3) / 命中
- 目录：metrics/ 逐场景指标，figures/ 每场景异常分数+ROC 图，summary_full.csv 汇总
