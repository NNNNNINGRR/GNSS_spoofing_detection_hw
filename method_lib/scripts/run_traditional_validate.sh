#!/usr/bin/env bash
# 传统方法并行验证：预测/异常/分类各跑代表性数据集与方法。
# 用法: bash scripts/run_traditional_validate.sh [--quick]
export PATH=/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

cd /root/autodl-tmp/bifi_repro_20260811/时间序列方法库/method_lib
mkdir -p logs/traditional
LOG=logs/traditional/validate.log

rm -f result_traditional_forecast.txt result_traditional_anomaly.txt result_traditional_classification.txt

cat > /tmp/trad_jobs.txt <<'EOF'
forecast Naive_SingleVar_Baseline ETTh1 2000 500 0
forecast NaiveSeasonal_SingleVar_Periodic ETTh1 2000 500 24
forecast ExponentialSmoothing_SingleVar_LevelTrend ETTh1 2000 500 0
forecast HoltWinters_SingleVar_Seasonal ETTh1 2000 500 24
forecast Arima_SingleVar_StationaryLinear ETTh1 2000 500 0
forecast Sarima_SingleVar_Seasonal ETTh1 2000 500 24
forecast KalmanLocalLevel_SingleVar_Online ETTh1 2000 500 0
forecast Var_MultiVar_LinearInteractions ETTh1 2000 500 0
forecast Arimax_MultiVar_Exogenous ETTh1 2000 500 0
anomaly StatisticalThreshold_SingleVar_Gaussian PSM 3000 2000 0
anomaly Cusum_SingleVar_Online PSM 3000 2000 0
anomaly Ewma_SingleVar_Online PSM 3000 2000 0
anomaly StlResidual_SingleVar_Seasonal PSM 3000 2000 24
anomaly Mahalanobis_MultiVar_Gaussian MSL 3000 2000 0
anomaly PcaReconstruction_MultiVar_Correlated MSL 3000 2000 0
anomaly HotellingT2_MultiVar_ProcessMonitoring MSL 3000 2000 0
anomaly IsolationForest_MultiVar_NonGaussian MSL 3000 2000 0
anomaly Lof_MultiVar_LocalDensity MSL 3000 2000 0
anomaly OneClassSvm_MultiVar_Boundary MSL 3000 2000 0
classification KnnDtw_SingleVar_SmallData UWaveGestureLibrary 0 0 0
classification KnnEd_SingleVar_Aligned UWaveGestureLibrary 0 0 0
classification FeatureRf_SingleVar_MediumData UWaveGestureLibrary 0 0 0
classification KnnDtw_MultiVar_ChannelFusion Heartbeat 0 0 0
classification FeatureRf_MultiVar_ChannelFusion Heartbeat 0 0 0
EOF

run_one() {
  task=$1; method=$2; data=$3; maxtr=$4; maxte=$5; season=$6
  args="--task $task --method $method"
  if [ "$task" = "forecast" ]; then
    if [[ "$data" == ETTh* || "$data" == ETTm* ]]; then
      rp="../dataset/ETT-small/"
    else
      rp="../dataset/$data/"
    fi
    args="$args --data $data --root_path $rp --data_path $data.csv --target OT --seq_len 96 --pred_len 96 --num_windows 5 --max_train $maxtr --max_test $maxte"
  elif [ "$task" = "anomaly" ]; then
    args="$args --data $data --root_path ../dataset/$data/ --anomaly_ratio 1 --max_train $maxtr --max_test $maxte"
  else
    args="$args --data UEA --dataset $data"
  fi
  if [ -n "$season" ] && [ "$season" != "0" ]; then args="$args --season $season"; fi
  echo "===== $task $method $data =====" >> "$LOG"
  /root/miniconda3/bin/python tools/run_traditional.py $args >> "$LOG" 2>&1
  echo "rc=$?" >> "$LOG"
}
export -f run_one
export LOG

cat /tmp/trad_jobs.txt | xargs -P 4 -n 6 bash -c 'run_one "$@"' _
echo TRADITIONAL_VALIDATE_DONE | tee -a "$LOG"
