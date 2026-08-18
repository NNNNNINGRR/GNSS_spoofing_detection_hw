#!/usr/bin/env bash
# 冒烟测试：9 个模型在 ETTh1 pred_len=96 上各训练 1 个 epoch，验证方法库端到端可用。
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0
cd /root/autodl-tmp/bifi_repro_20260811/method_lib
LOG=logs/smoke.log
for M in Bi_FI DLinear iTransformer PatchTST LightTS FEDformer Reformer Informer Autoformer; do
  echo "===== $M =====" | tee -a $LOG
  /root/miniconda3/bin/python -u run.py --task_name long_term_forecast --is_training 1 \
    --model_id ETTh1_smoke --model $M --data ETTh1 \
    --root_path ../dataset/ETT-small/ --data_path ETTh1.csv \
    --features M --freq h --seq_len 96 --label_len 48 --pred_len 96 \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 32 --n_heads 8 \
    --e_layers 2 --d_layers 1 --d_ff 256 --num_workers 4 --batch_size 32 \
    --learning_rate 0.0001 --train_epochs 1 --patience 3 --des Smoke --itr 1 2>&1 | tee -a $LOG
  echo "rc=$?" | tee -a $LOG
done
echo SMOKE_ALL_DONE