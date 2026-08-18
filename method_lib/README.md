# Bi-FI 方法库与工具库（method_lib）

统一封装论文 *Bi-Branching Feature Interaction Representation Learning for Multivariate Time Series (Bi-FI)*
及其 8 个对比方法，支持长序列预测 / 异常检测 / 分类三种任务，共用同一套数据加载、训练、评测与结果收集工具。

## 模型清单（model/，均为 `Model(configs)` 统一接口）
| 模型 | 来源 |
|---|---|
| Bi_FI | 论文官方仓库 zeg-datamining/Bi-FI（含兼容性修复） |
| DLinear | Time-Series-Library（LTSF-Linear 论文实现） |
| iTransformer | Time-Series-Library |
| PatchTST | Time-Series-Library |
| LightTS | Time-Series-Library |
| FEDformer | Time-Series-Library |
| Reformer | Time-Series-Library |
| Informer | Time-Series-Library |
| Autoformer | Time-Series-Library |

对比方法官方 Bi-FI 仓库未提供源码，这里从 thuml/Time-Series-Library（MIT）移植统一版本；完整仓库保留在 vendor/Time-Series-Library 作为来源与参考。

## 传统方法库（traditional/，统一入口 tools/run_traditional.py）

新增 24 个**统计/经典机器学习方法**，命名格式为 `<方法>_<SingleVar|MultiVar>_<适用条件>.py`，文件头部均写明“适用条件 / 数据要求 / 如何使用”。统一入口不改动现有 run.py 接口：

```bash
# 预测（单/多变量）
python tools/run_traditional.py --task forecast --method Arima_SingleVar_StationaryLinear \
  --data ETTh1 --target OT --seq_len 96 --pred_len 96
# 异常检测
python tools/run_traditional.py --task anomaly --method IsolationForest_MultiVar_NonGaussian \
  --data MSL --anomaly_ratio 1
# 分类
python tools/run_traditional.py --task classification --method KnnDtw_SingleVar_SmallData \
  --data UEA --dataset UWaveGestureLibrary
```

方法清单：

- 预测（9）：Naive_SingleVar_Baseline、NaiveSeasonal_SingleVar_Periodic、ExponentialSmoothing_SingleVar_LevelTrend、HoltWinters_SingleVar_Seasonal、Arima_SingleVar_StationaryLinear、Sarima_SingleVar_Seasonal、KalmanLocalLevel_SingleVar_Online、Var_MultiVar_LinearInteractions、Arimax_MultiVar_Exogenous；
- 分类（5）：KnnDtw_SingleVar_SmallData、KnnEd_SingleVar_Aligned、FeatureRf_SingleVar_MediumData、KnnDtw_MultiVar_ChannelFusion、FeatureRf_MultiVar_ChannelFusion；
- 异常检测（10）：StatisticalThreshold_SingleVar_Gaussian、Cusum_SingleVar_Online、Ewma_SingleVar_Online、StlResidual_SingleVar_Seasonal、Mahalanobis_MultiVar_Gaussian、PcaReconstruction_MultiVar_Correlated、HotellingT2_MultiVar_ProcessMonitoring、IsolationForest_MultiVar_NonGaussian、Lof_MultiVar_LocalDensity、OneClassSvm_MultiVar_Boundary。

批量验证脚本：`scripts/run_traditional_validate.sh`（云端并行验证用）。

## 目录结构
- run.py / run_bifi.py  统一训练入口（TSL 风格，支持全部参数）；run_bifi.py 为 Bi-FI 官方原版入口
- exp/                  三任务实验类（long_term_forecast / anomaly_detection / classification）
- data_provider/        数据加载（本地文件；UEA .ts 轻量解析）
- model/                9 个模型
- layers/               共享网络层（TSL 版本，覆盖 AutoCorrelation/Fourier/MultiWavelet 等）
- utils/                指标、早停、时间特征、masking、参数打印
- tools/collect_results.py   结果汇总 → results_summary.json
- tools/compare_paper.py     与论文 Table D.7/D.8/D.9 对照打印
- scripts/run_smoke.sh       9 模型端到端冒烟
- vendor/Time-Series-Library 对比方法源码来源（完整克隆）

## 用法
```bash
cd /root/autodl-tmp/bifi_repro_20260811/method_lib

# 单个模型训练+测试（预测）
python run.py --task_name long_term_forecast --is_training 1 --model Bi_FI \
  --model_id ETTh1_96 --data ETTh1 --root_path ../dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --freq h --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --e_layers 2 --d_ff 512 \
  --batch_size 32 --learning_rate 0.0001 --train_epochs 10 --des Exp

# 换模型只需 --model DLinear / iTransformer / PatchTST / LightTS / FEDformer / Reformer / Informer / Autoformer
# 异常检测：--task_name anomaly_detection --data MSL/SMAP/SMD/SWAT/PSM --seq_len 100 --anomaly_ratio 1
# 分类：--task_name classification --data UEA --root_path ../dataset/<Name>/
```

## 环境要求
- Python 3.12 + PyTorch 2.8 + CUDA（云端 miniconda3 已就绪）
- `reformer_pytorch`（Reformer / Bi-FI 的 SelfAttention_Family 顶层导入需要）
- 已做 NumPy2 / pandas3 兼容修复（np.inf、频率别名 t→min、Solar 时间标记 4 维、UEA 分类加载器）

## 数据
数据集位于上级 `../dataset/`（ETT-small、electricity、exchange_rate、weather、Solar、MSL、SMAP、SMD、SWaT、PSM、10 个 UEA .ts），与 TSL 官方数据格式一致。
