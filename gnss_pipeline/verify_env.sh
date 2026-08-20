#!/bin/bash
# 云端环境验证：run.py 参数 + PSM 数据加载
cd /root/autodl-tmp/时间序列方法库/method_lib || exit 1
/root/miniconda3/bin/python run.py --help 2>&1 | head -8
echo '--- loader test ---'
/root/miniconda3/bin/python - <<'PY'
import sys
sys.path.insert(0, '.')
from data_provider.data_loader import PSMSegLoader
d = PSMSegLoader('/root/autodl-tmp/exp_gnss/data/SQM_ds1', win_size=96)
print('train', d.train.shape, 'test', d.test.shape, 'labels', d.test_labels.shape)
PY
