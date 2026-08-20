# 04 Mahalanobis-MCD（最小协方差行列式稳健估计）——方法详解与逐场景结果

> 系列文档之四。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_MahalanobisMCD/metrics.csv` 一致。**本方法为反面案例**：稳健化改造在本数据上失效，其失败机理与正方法同样有论文价值。

## 一、方法原理与数学定义

MCD（Minimum Covariance Determinant, Rousseeuw 1984）是稳健统计中最经典的 location–scatter 联合估计：在 $n$ 个样本中寻找基数为 $h=\lceil(n+p+1)/2\rceil$（默认支撑比例 50%）的子集，使其协方差行列式最小——直觉是"最紧凑的一半样本"代表未受污染的核心分布：

$$(\hat{\boldsymbol{\mu}}_{\mathrm{MCD}},\hat{\Sigma}_{\mathrm{MCD}})=\arg\min_{|H|=h}\ \det\Big(\mathrm{Cov}(X_H)\Big),$$

检测统计量仍是马氏距离的二次型，只是参数换成 MCD 估计：

$$s_t=(\boldsymbol{z}_t-\hat{\boldsymbol{\mu}}_{\mathrm{MCD}})^{\top}(\hat{\Sigma}_{\mathrm{MCD}}+\varepsilon I)^{-1}(\boldsymbol{z}_t-\hat{\boldsymbol{\mu}}_{\mathrm{MCD}}).$$

MCD 具有约 $(1-h/n)/2=25\%$ 的崩溃点（理论上可容忍训练集四分之一的任意污染），渐近效率可通过重加权步骤提升。**动机**：cs+cd"清洁"数据并非无菌——多径、周跳恢复期、位同步瞬态等"脏正常"帧混在训练集中，若经典样本协方差被这些尾巴拉宽，椭球被吹胀、阈值变松、检出下降；MCD 应聚焦核心分布、收缩椭球、提高对边缘欺骗的检出。

**失效预告**：MCD 收缩协方差的同时也改变了清洁分数分布的形状（尾部变重、尺度变小），而阈值标定仍按"全训练集分数的分位数"进行——统计量与标定样本不一致，属于**方法论层面的口径错配**，实测全面失效。

## 二、源码解析

实现为本项目自建类（`run_extras_v31.py` 中 `MahalanobisMCD`，基于 sklearn 的 `MinCovDet`）：

```python
class MahalanobisMCD:
    def __init__(self, reg=1e-6, seed=2):
        self.reg, self.seed = float(reg), int(seed)

    def fit(self, X):
        mcd = MinCovDet(random_state=self.seed,
                        support_fraction=None).fit(X)   # None → h=(n+p+1)/2，50% 支撑
        self.mean_ = mcd.location_                      # 稳健位置 μ_MCD
        self.cov_inv_ = np.linalg.pinv(
            mcd.covariance_ + self.reg * np.eye(X.shape[1]))  # 稳健散布 Σ_MCD + 正则伪逆
        return self

    def score(self, X):
        d = np.asarray(X) - self.mean_
        return np.einsum("ni,ij,nj->n", d, self.cov_inv_, d)  # 逐帧二次型
```

要点解析：(1) `support_fraction=None` 即默认 $h=(n+p+1)/2$，45476 样本下子集约 22742 帧——**cs/cd 两簇各占一半的训练集里，50% 支撑大概率整体丢掉其中一簇**（如只保留 cs 静态簇），这是比"尾部污染"严重得多的问题：估计的不是"核心清洁分布"而是"半个训练集"；(2) sklearn 默认做重加权（reweighting）一步，但无法挽回选错簇的后果；(3) `score` 与经典马氏完全同构，唯一差别在参数来源；(4) 驱动层阈值仍取**全训练集**（含被 MCD 视为离群的另一半）分数的 p99/p95/p90——被丢弃那一半的分数极高，把分位阈值推得极高，测试帧几乎无法越限。这一条是失效的直接技术原因。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MahalanobisMCD/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.802 |
| TPR@1%（实际FPR） | 0.188（0.011） |
| TPR@5%（实际FPR） | 0.188（0.062） |
| TPR@10%（实际FPR） | 0.188（0.121） |
| ADD / hit | 0.62 s / 1 |

**分析**：三档 TPR 完全相同（0.188）——分数分布中出现"平台"，即三档阈值全部落入同一个分数密集区（阈值过高且集中）。AUC 0.802 说明排序能力尚存（攻击初期距离仍高），但工作点被标定毁掉。对照经典马氏同场景 0.998：稳健化损失了 0.81 的检出。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MahalanobisMCD/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.789 |
| TPR@1%（实际FPR） | 0.134（0.011） |
| TPR@5%（实际FPR） | 0.144（0.062） |
| TPR@10%（实际FPR） | 0.150（0.121） |
| ADD / hit | 6.18 s / 1 |

**分析**：同样的三档平台形态（0.13–0.15）。经典马氏在该场景为满分——MCD 让它跌到 0.13，是全方法中"改造致损"最极端的对照。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MahalanobisMCD/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.677 |
| TPR@1%（实际FPR） | 0.026（0.010） |
| TPR@5%（实际FPR） | 0.059（0.061） |
| TPR@10%（实际FPR） | 0.089（0.121） |
| ADD / hit | 10.86 s / 1 |

**分析**：本就是最难场景，叠加标定失配后三档全在 0.1 以下。值得注意的是实际 FPR：1% 档 0.010 恰好等于名义值——不是纪律好，而是"阈值高到清洁段也只有 1% 越限"，与检出无关。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MahalanobisMCD/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.758 |
| TPR@1%（实际FPR） | 0.015（0.012） |
| TPR@5%（实际FPR） | 0.033（0.067） |
| TPR@10%（实际FPR） | 0.051（0.124） |
| ADD / hit | 100.40 s / 1 |

**分析**：全场最差表现（1% 档 0.015，ADD 100 s）。若 MCD 的 50% 支撑恰好保留了 cs 静态簇而丢弃 cd 动态簇，则动态测试帧（ds4/ds5）本身就成为"离群"，分数分布整体错位——动态场景受害最重支持这一诊断。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MahalanobisMCD/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.786 |
| TPR@1%（实际FPR） | 0.141（0.008） |
| TPR@5%（实际FPR） | 0.145（0.052） |
| TPR@10%（实际FPR） | 0.148（0.101） |
| ADD / hit | 3.18 s / 1 |

**分析**：同为动态，ds5 表现（0.141）反而好于 ds4（0.015）——匹配功率场景的畸变特征抬升幅度大，部分帧仍越过被推高的阈值；而 ds4 依赖的功率方向证据被错位椭球进一步压低。这种"动态两场景间的巨大反差"（差 9 倍）本身就是标定错配的证据。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MahalanobisMCD/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.732 |
| TPR@1%（实际FPR） | 0.039（0.007） |
| TPR@5%（实际FPR） | 0.071（0.052） |
| TPR@10%（实际FPR） | 0.098（0.099） |
| ADD / hit | 6.28 s / 1 |

**分析**：三档 <0.1。经典马氏在该场景 1% 档 0.410，MCD 0.039——损失一个数量级。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MahalanobisMCD/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.743 |
| TPR@1%（实际FPR） | 0.044（0.011） |
| TPR@5%（实际FPR） | 0.072（0.066） |
| TPR@10%（实际FPR） | 0.101（0.134） |
| ADD / hit | 115.78 s / 1 |

**分析**：与 ds8 几乎相同的低检出；ADD 115.78 s。无特殊场景效应，纯粹是标定失配的底色。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MahalanobisMCD/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.784 |
| TPR@1%（实际FPR） | 0.061（0.011） |
| TPR@5%（实际FPR） | 0.089（0.066） |
| TPR@10%（实际FPR） | 0.116（0.134） |
| ADD / hit | 2.04 s / 1 |

**分析**：篡改幅度大，ADD 2.04 s 尚可，但 TPR 仍 <0.12。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.7588 |
| TPR@1% / 实际 FPR | 0.0811 / 0.0103 |
| TPR@5% / 实际 FPR | 0.1002 / 0.0611 |
| TPR@10% / 实际 FPR | 0.1177 / 0.1012 |
| ADD 命中 | 8/8 |

**综合分析（失败机理，论文素材）**：(1) 与经典马氏（AUC 0.948、1% 档 0.744）相比全面坍塌，唯一差别是参数估计方式——**证明"更稳健的估计"不会自动带来更好的检测器**；(2) 失效的两层机理：表层是**标定口径错配**（阈值按全训练集分位标定，而 MCD 把一半训练集当作离群，其高分把 p99 推到测试分布之外）；深层是**多簇结构下的支撑选择歧义**（cs/cd 两簇各半时，50% 支撑等价于随机选簇，估计的不是核心分布而是"半个世界"）；(3) 可修复方向（供论文讨论）：(i) 阈值改按 MCD 支撑子集内的分数分位标定，(ii) 支撑比例提到 0.75–0.9，(iii) 对 cs/cd 先分簇再各自估计（条件化马氏）——但任何修复都意味着丢弃"单一稳健估计"的原始卖点；(4) 方法论教训一句话：**单类检测中，估计器与阈值标定必须共享同一"什么是正常"的定义**。
