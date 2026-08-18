#!/usr/bin/env bash
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0
cd /root/autodl-tmp/bifi_repro_20260811/时间序列方法库/method_lib
LOG=logs/full_validation.log
mkdir -p logs
MODELS="Bi_FI DLinear iTransformer PatchTST LightTS FEDformer Reformer Informer Autoformer"

for M in $MODELS; do
  echo "########## $M forecast ##########" | tee -a $LOG
  /root/miniconda3/bin/python -u run.py --task_name long_term_forecast --is_training 1 --model $M \
    --model_id ETTh1_v --data ETTh1 --root_path ../dataset/ETT-small/ --data_path ETTh1.csv \
    --features M --freq h --seq_len 96 --label_len 48 --pred_len 96 \
    --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 512 \
    --num_workers 4 --batch_size 32 --learning_rate 0.0001 --train_epochs 1 --patience 3 --des Val --itr 1 2>&1 | tee -a $LOG
  echo "rc=$?" | tee -a $LOG
done

for M in $MODELS; do
  echo "########## $M anomaly ##########" | tee -a $LOG
  /root/miniconda3/bin/python -u run.py --task_name anomaly_detection --is_training 1 --model $M \
    --model_id MSL_v --data MSL --root_path ../dataset/MSL/ \
    --seq_len 100 --enc_in 55 --dec_in 55 --c_out 55 --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 512 \
    --num_workers 4 --batch_size 128 --learning_rate 0.0001 --train_epochs 1 --patience 3 --anomaly_ratio 1 --des Val --itr 1 2>&1 | tee -a $LOG
  echo "rc=$?" | tee -a $LOG
done

for M in $MODELS; do
  echo "########## $M classification ##########" | tee -a $LOG
  /root/miniconda3/bin/python -u run.py --task_name classification --is_training 1 --model $M \
    --model_id UWave_v --data UEA --root_path ../dataset/UWaveGestureLibrary/ \
    --e_layers 3 --batch_size 16 --d_model 128 --d_ff 256 --top_k 3 \
    --learning_rate 0.001 --train_epochs 1 --patience 10 --num_workers 4 --enc_in 3 --des Val --itr 1 2>&1 | tee -a $LOG
  echo "rc=$?" | tee -a $LOG
done
echo FULL_VALIDATION_DONE