#!/bin/bash
# 将适配后的方法库代码同步到 GitHub 私有仓库 GNSS_spoofing_detection_hw
set -e
REPO=git@github.com:NNNNNINGRR/GNSS_spoofing_detection_hw.git
KEY=/root/.ssh/id_ed25519_gnss_cloud
export GIT_SSH_COMMAND="ssh -i $KEY -o StrictHostKeyChecking=accept-new"

cd /root/autodl-tmp
rm -rf GNSS_spoofing_detection_hw
git clone "$REPO" GNSS_spoofing_detection_hw
cd GNSS_spoofing_detection_hw

# 放入适配后的方法库代码（仅代码）
cp -r /root/autodl-tmp/时间序列方法库/method_lib ./method_lib
find ./method_lib -name '__pycache__' -type d -prune -exec rm -rf {} +
find ./method_lib -name '*.pyc' -delete

cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc

# 运行产物
checkpoints/
results/
test_results/
logs/

# 数据/大文件/敏感文件
dataset/
*.npy
*.csv
*.tar.gz
*.pdf
.DS_Store
*.key
*_key
EOF

cat > README.md <<'EOF'
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
EOF

git add -A
git -c user.name="codex-cloud" -c user.email="codex-cloud@users.noreply.github.com" \
  commit -m "Add GNSS spoofing detection adaptation (Bi-FI/LightTS/DLinear): train-only threshold, save scores, init_checkpoint for fine-tuning"
git push -u origin main
echo "--- verify ---"
git ls-remote origin main
