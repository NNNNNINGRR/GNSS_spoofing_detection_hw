#!/bin/bash
# GNSS 欺骗检测冒烟测试：3 模型 × (V1 从头 / V2 预训练+微调)，1 epoch，SQM_ds1 前 2000 帧
set -e
export PATH=/root/miniconda3/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

ML=/root/autodl-tmp/时间序列方法库/method_lib
DATA=/root/autodl-tmp/exp_gnss/data
SMOKE=$DATA/smoke
W=/root/autodl-tmp/exp_gnss/weights
RES=/root/autodl-tmp/exp_gnss/results/smoke
FIG=/root/autodl-tmp/exp_gnss/figures
EVAL=/root/autodl-tmp/exp_gnss/code/eval_smoke.py
PREP=/root/autodl-tmp/exp_gnss/code/prep_smoke.py
MODELS="${MODELS:-Bi_FI LightTS DLinear}"
mkdir -p $SMOKE/PSM $SMOKE/SQM_ds1 $SMOKE/ETTh1 $W $RES $FIG
rm -f $RES/*.csv

echo "===== [1/3] prepare smoke data ====="
head -2000 /root/autodl-tmp/时间序列方法库/dataset/PSM/Train.csv      > $SMOKE/PSM/Train.csv
head -2000 /root/autodl-tmp/时间序列方法库/dataset/PSM/Test.csv       > $SMOKE/PSM/Test.csv
head -2000 /root/autodl-tmp/时间序列方法库/dataset/PSM/Test_label.csv > $SMOKE/PSM/Test_label.csv
python $PREP $DATA $SMOKE

cd $ML
COMMON="--task_name anomaly_detection --is_training 1 --data PSM --seq_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 --d_model 64 --n_heads 8 --e_layers 2 --d_ff 128 \
  --dropout 0.1 --batch_size 32 --learning_rate 0.0001 --train_epochs 1 --patience 3 \
  --anomaly_ratio 1 --num_workers 4 --seed 2 --checkpoints $W"

echo "===== [2/3] V1: from scratch on SQM_ds1 ====="
for M in $MODELS; do
  echo "----- V1 $M -----"
  rm -rf test_results
  python -u run.py $COMMON --model $M --model_id smoke_v1_$M --root_path $SMOKE/SQM_ds1 \
    --des SmokeV1 > $RES/v1_$M.log 2>&1
  T=$(dirname $(find test_results -name score.npy | head -1))
  python $EVAL --score $T/score.npy --label $T/label.npy --threshold $T/threshold.npy \
    --csv $SMOKE/SQM_ds1/Test.csv --win 96 --onset 125 \
    --out_fig $FIG/smoke_v1_${M}.png --out_csv $RES/v1_${M}_metrics.csv
done

echo "===== [3/3] V2: pretrain(ETTh1) + finetune(SQM_ds1) ====="
for M in $MODELS; do
  echo "----- V2 pretrain $M -----"
  rm -rf test_results
  python -u run.py $COMMON --model $M --model_id smoke_pre_$M --root_path $SMOKE/ETTh1 \
    --des SmokePre > $RES/pre_$M.log 2>&1
  CK=$(find $W -path "*smoke_pre_${M}*" -name checkpoint.pth | tail -1)
  echo "pretrain ckpt: $CK"
  echo "----- V2 finetune $M -----"
  rm -rf test_results
  python -u run.py $COMMON --model $M --model_id smoke_v2_$M --root_path $SMOKE/SQM_ds1 \
    --des SmokeV2 --init_checkpoint $CK > $RES/v2_$M.log 2>&1
  T=$(dirname $(find test_results -name score.npy | head -1))
  python $EVAL --score $T/score.npy --label $T/label.npy --threshold $T/threshold.npy \
    --csv $SMOKE/SQM_ds1/Test.csv --win 96 --onset 125 \
    --out_fig $FIG/smoke_v2_${M}.png --out_csv $RES/v2_${M}_metrics.csv
done

echo "===== SMOKE SUMMARY ====="
cat $RES/v1_*_metrics.csv $RES/v2_*_metrics.csv
echo SMOKE_DONE
