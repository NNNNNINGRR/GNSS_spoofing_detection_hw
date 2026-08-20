#!/bin/bash
cd /root/autodl-tmp/时间序列方法库/method_lib || exit 1
/root/miniconda3/bin/python - <<'PY'
import ast
for f in ["exp/exp_anomaly_detection.py", "run.py"]:
    ast.parse(open(f, encoding="utf-8").read())
    print(f, "syntax OK")
PY
/root/miniconda3/bin/python run.py --help 2>&1 | grep -A1 init_checkpoint
