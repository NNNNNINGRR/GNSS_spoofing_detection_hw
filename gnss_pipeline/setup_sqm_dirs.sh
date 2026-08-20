#!/bin/bash
# 在云端创建 SQM 逐场景异常检测数据目录（Train=cs/cd, Test=dsX, Test_label）
set -e
cd /root/autodl-tmp/exp_gnss/data
rm -rf SQM_ "SQM_;" cp
for s in ds1 ds2 ds3 ds4 ds5 ds6 ds7 ds8; do
  mkdir -p "SQM_$s"
  cp "anomaly_per_scenario/$s/anomaly/Train.csv"      "SQM_$s/"
  cp "anomaly_per_scenario/$s/anomaly/Test.csv"       "SQM_$s/"
  cp "anomaly_per_scenario/$s/anomaly/Test_label.csv" "SQM_$s/"
done
echo "--- verify ---"
ls SQM_ds1 SQM_ds7
