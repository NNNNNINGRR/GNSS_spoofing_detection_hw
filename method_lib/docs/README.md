# 时间序列方法库教学文档（总览）

本目录为“时间序列方法库”中全部 9 个方法的配套教学文档，目标读者是**不熟悉神经网络与时间序列分析的科研人员**。每篇文档都遵循统一结构：

1. **前半部分——方法思想**：用段落化文字与公式解释“这个方法要解决什么问题、核心思想是什么、数学上怎么做”，并尽量给出直觉类比。
2. **后半部分——源码讲解**：对照 `method_lib/model/<模型名>.py` 的代码逐段讲解，代码中出现的每一步都对应到前面的公式，并注明张量形状的变化。

## 文档清单

| 文档 | 方法 | 一句话概括 |
|---|---|---|
| [00_基础概念与公共模块.md](00_基础概念与公共模块.md) | 公共基础 | 时间序列、神经网络、注意力、嵌入、损失与公共层源码 |
| [01_Bi_FI.md](01_Bi_FI.md) | Bi-FI | 双分支：频域卷积捕获变量间关系 + 变量令牌 Transformer 捕获变量内时序 |
| [02_DLinear.md](02_DLinear.md) | DLinear | 趋势/季节分解后各用一个线性层直接回归 |
| [03_iTransformer.md](03_iTransformer.md) | iTransformer | 把“变量”当令牌做注意力，时间维用线性层压缩 |
| [04_PatchTST.md](04_PatchTST.md) | PatchTST | 先把每个变量切成小块（patch），再对小块做 Transformer |
| [05_LightTS.md](05_LightTS.md) | LightTS | 连续/间隔双采样 + 极简 MLP，不用注意力 |
| [06_FEDformer.md](06_FEDformer.md) | FEDformer | 趋势/季节分解 + 在频域做注意力 |
| [07_Reformer.md](07_Reformer.md) | Reformer | 用局部敏感哈希把注意力复杂度从 O(L²) 降到 O(L log L) |
| [08_Informer.md](08_Informer.md) | Informer | 稀疏概率注意力 + 蒸馏 + 生成式解码器 |
| [09_Autoformer.md](09_Autoformer.md) | Autoformer | 自相关机制按周期聚合信息，替代注意力 |
| [10_数据入口与私有数据集制作指南.md](10_数据入口与私有数据集制作指南.md) | 数据入口 | 三种任务的数据格式、放置方式、运行参数与自检 |
| [11_数据预处理全流程与代码解析.md](11_数据预处理全流程与代码解析.md) | 数据预处理 | 读取→切分→标准化→时间特征→滑窗→打包全流程源码解析，私有数据集端到端使用 |
| [12_传统方法库使用与私有数据集指南.md](12_传统方法库使用与私有数据集指南.md) | 传统方法 | 24 个统计/经典方法的使用、私有数据集接入、每个方法的数据输入要求 |
| [合稿_时间序列方法库教学全集.md](合稿_时间序列方法库教学全集.md) | 合稿 | 全部文档合并版，便于连续阅读 |

## 方法分类速览

论文 Bi-FI 把 8 个对比方法分成两类：

- **变量内方法（intra-variable）**：只关注“同一个变量随时间怎么变”，如 DLinear、PatchTST、LightTS、iTransformer；
- **变量间方法（inter-variable）**：关注“同一时刻不同变量之间的关系”，如 Informer、Autoformer、Reformer、FEDformer；
- **Bi-FI**：两条分支同时学两类信息，最后相加融合。

## 阅读顺序建议

1. 先读 [00_基础概念与公共模块.md](00_基础概念与公共模块.md)，掌握记号（$X \in \mathbb{R}^{T\times M}$、$T$=时间长度、$M$=变量数、$d_{\text{model}}$=隐层维数）和公共层（嵌入、注意力、编码器层）的实现；
2. 读 [02_DLinear.md](02_DLinear.md)（最简单）建立“输入→模型→损失→训练”的完整印象；
3. 再按兴趣读 Transformer 系（03、04、06、07、08、09）；
4. 最后读 [01_Bi_FI.md](01_Bi_FI.md)，因为它是这些思想的“合体”。

## 本库支持的三种任务

- **长序列预测（long-term forecasting）**：给过去 $T$ 个时刻的 $M$ 个变量，预测未来 $H$ 个时刻的值；
- **异常检测（anomaly detection）**：模型学会“重建”正常序列，重建误差大的时间点判为异常；
- **分类（classification）**：给一整条多元序列，输出它属于哪个类别。

所有模型都共用同一套训练框架（`run.py` + `exp/` + `data_provider/`），切换模型只需改 `--model` 参数。

## 符号约定

- $B$：批大小（batch size），一次送入模型的样本数；
- $T$ 或 $L$：输入序列长度（代码中 `seq_len`，常用 96）；
- $H$ 或 $P$：预测长度（代码中 `pred_len`，常用 96/192/336/720）；
- $M$ 或 $N$：变量数（代码中 `enc_in`，如 ECL 数据集 321 个变量）；
- $d_{\text{model}}$：隐层维数（代码中 `d_model`）；
- $d_{ff}$：前馈网络隐层维数（代码中 `d_ff`）；
- $h$：注意力头数（代码中 `n_heads`）；
- 张量形状：PyTorch 中一般写作 `[B, T, M]`（批、时间、变量），代码注释沿用这一写法。

## 如何对照源码

所有源码位于 `method_lib/model/`。每个文件都定义了一个名为 `Model` 的类，统一提供：

```python
class Model(nn.Module):
    def __init__(self, configs): ...      # 搭建网络结构
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec): ...  # 前向计算
```

`configs` 是运行参数集合（`run.py` 解析的命令行参数），包含 `seq_len`、`pred_len`、`enc_in`、`d_model` 等。文档中讲解的代码片段均来自本仓库实际文件，可与原文直接对照。
