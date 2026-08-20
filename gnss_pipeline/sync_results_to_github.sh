#!/bin/bash
# 将全量实验结果同步到 GitHub 私有仓库 GNSS_spoofing_detection_hw
set -e
REPO=/root/autodl-tmp/GNSS_spoofing_detection_hw
SRC=/root/autodl-tmp/exp_gnss
cd "$REPO"

mkdir -p experiment_results/metrics experiment_results/figures
cp $SRC/results/full/*_metrics.csv experiment_results/metrics/
cp $SRC/results/full/summary_full.csv experiment_results/
cp $SRC/results/full/full_run.log experiment_results/
cp $SRC/figures/full_v*.png experiment_results/figures/

cat > experiment_results/README.md <<'EOF'
# GNSS 欺骗检测全量实验结果（Bi-FI / LightTS / DLinear）

- 版本一（V1）：TEXBAT 正常数据从头训练（10 epochs, lr=1e-4）
- 版本二（V2）：ETTh1 -> ETTm1 预训练（各 10 epochs）-> TEXBAT 微调（5 epochs, lr=5e-5）
- 数据：TEXBAT 异常检测集（Train=cs/cd 正常，Test=ds1-8，50 Hz，7 特征，seq_len=96）
- 评估：ROC-AUC / PR-AUC / TPR@FPR(1%,5%) / F1 / MCC / BalancedAcc / ADD(K=3) / 命中
- 目录：metrics/ 逐场景指标，figures/ 每场景异常分数+ROC 图，summary_full.csv 汇总
EOF

grep -q 'experiment_results' .gitignore || printf '\n# 实验结果显示（仅结果，不含数据/权重）\n!experiment_results/\n!experiment_results/**\n' >> .gitignore

git add -A
git add -f experiment_results
git -c user.name="codex-cloud" -c user.email="codex-cloud@users.noreply.github.com" \
  commit -m "Add full GNSS spoofing detection experiment results (V1/V2 x 8 scenarios x Bi_FI/LightTS/DLinear)"
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519_gnss_cloud -o StrictHostKeyChecking=accept-new"
git push origin main
echo "--- verify ---"
git ls-remote origin main
