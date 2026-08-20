#!/bin/bash
# 全量实验监控：当前任务、已完成指标数、最近日志
R=/root/autodl-tmp/exp_gnss/results/full
echo "时间: $(date '+%H:%M:%S')"
echo "当前任务: $(grep -hE '=====|##########' $R/full_run.log $R/full_run_rest.log 2>/dev/null | tail -1)"
echo "已完成 metrics: $(ls $R/*_metrics.csv 2>/dev/null | wc -l)"
LATEST=$(ls -t $R/*.log 2>/dev/null | grep -v full_run | head -1)
if [ -n "$LATEST" ]; then
  echo "最近日志: $LATEST"
  tail -3 "$LATEST"
fi
if grep -q FULL_DONE $R/full_run_rest.log 2>/dev/null; then
  echo "STATUS: FULL_DONE"
else
  echo "STATUS: RUNNING"
fi
