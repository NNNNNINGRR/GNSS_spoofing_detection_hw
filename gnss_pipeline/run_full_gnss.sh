#!/bin/bash
# GNSS 欺骗检测全量实验（Paper A 口径）：模型 × 单一训练（cs+cd 清洁数据）× 8 场景拼接测试
#
# 协议（对齐 Iqbal et al. 2024 与用户设定）：
#   - 训练只用清洁场景 cs+cd（Train.csv），测试为 8 个欺骗场景拼接（Test.csv）；
#   - 特征 8 维：m_ratio/m_delta/m_elp/m_symdiff(±0.5chip) + m_manfredini + m_dd(双delta)
#     + received_power + CN0（来自 v3.1 解算 + 新版 build_metrics.py）；
#   - 阈值在训练（清洁）分数上按 FPR 0.1/1/5% 三档标定（thresholds.npy）；
#   - 逐场景报 TPR@三档 + 实际FPR + ROC/PR-AUC + ADD（连续3帧告警）。
# 旧 V1/V2（逐场景训练、ETT 预训练微调）分支已删除。
set -e
export PATH=/root/miniconda3/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

ML=/root/autodl-tmp/GNSS_spoofing_detection_hw/method_lib
DATA=/root/autodl-tmp/exp_gnss/data/SQM_cscd
W=/root/autodl-tmp/exp_gnss/weights
RES=/root/autodl-tmp/exp_gnss/results/cscd
FIG=/root/autodl-tmp/exp_gnss/figures
EVAL=/root/autodl-tmp/exp_gnss/code/eval_smoke.py
mkdir -p $W $RES $FIG
cd $ML

MODELS="${MODELS:-Bi_FI LightTS DLinear}"
SEEDS="${SEEDS:-2}"
COMMON="--task_name anomaly_detection --is_training 1 --data PSM --seq_len 96 \
  --enc_in 8 --dec_in 8 --c_out 8 --d_model 64 --n_heads 8 --e_layers 2 --d_ff 128 \
  --dropout 0.1 --batch_size 32 --num_workers 4 --checkpoints $W --anomaly_ratio 1"

for SEED in $SEEDS; do
  for M in $MODELS; do
    ID=cscd_${M}_s${SEED}
    echo "===== train $M (seed=$SEED, cs+cd) ====="
    rm -rf test_results
    python -u run.py $COMMON --model $M --model_id $ID --seed $SEED \
      --root_path $DATA --learning_rate 0.0001 --train_epochs 10 --patience 3 \
      --des Cscd > $RES/${ID}.log 2>&1
    T=$(dirname $(find test_results -name score.npy | head -1))
    echo "===== eval $M (seed=$SEED) -> $T ====="
    python $EVAL --score $T/score.npy --label $T/label.npy --thresholds $T/thresholds.npy \
      --csv $DATA/Test.csv --manifest $DATA/manifest.json --win 96 \
      --out_fig $FIG/${ID}.png --out_csv $RES/${ID}_metrics.csv
  done
done

echo "########## SUMMARY ##########"
head -1 $RES/cscd_Bi_FI_s2_metrics.csv > $RES/summary_cscd.csv 2>/dev/null || true
cat $RES/cscd_*_metrics.csv >> $RES/summary_cscd.csv 2>/dev/null || true
echo CSCD_DONE
