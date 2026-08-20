# 05 PCA 重构残差——方法详解与逐场景结果

> 系列文档之五。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_PcaReconstruction/metrics.csv` 一致。

## 一、方法原理与数学定义

PCA 重构残差是 Hotelling T² 的"半个版本"：只用残差空间（SPE），不看得分空间。对标准化特征做主成分分解，保留累计方差贡献 90% 的前 $k$ 个主成分载荷 $P_k$，将样本向该子空间投影再重构，以**重构误差范数**为异常分数：

$$\hat{\boldsymbol{z}}_t=P_kP_k^{\top}\boldsymbol{z}_t,\qquad
s_t=\big\|\boldsymbol{z}_t-\hat{\boldsymbol{z}}_t\big\|_2 .$$

直觉模型：清洁数据的全部变化由少数"清洁模式"（前 $k$ 个主成分方向）线性组合生成，任何不能被这些模式重构的成分都是异常。它与 T² 互补：T² 问"样本在子空间内的位置是否偏离清洁质心"，SPE 问"样本是否有子空间之外的能量"。

**适用前提与弱点**：SPE 的有效性依赖两点——(i) 清洁子空间在测试条件下不变；(ii) 欺骗确实产生子空间外能量。SQM 特征的现实是：各通道 CN0/噪声水平不同，使**小方差主成分方向的尺度随通道漂移**，重构误差被这种尺度漂移支配而非欺骗支配；且畸变类欺骗若被 90% 子空间容纳（见 Hotelling T² 文档对 ds2 的分析），残差几乎无信号。8 维小特征集上截断的收益（抗噪）远小于代价（丢方向）。

## 二、源码解析

实现位于 `method_lib/traditional/PcaReconstruction_MultiVar_Correlated.py`：

```python
def fit(self, X):
    X = np.asarray(X, dtype=np.float64)
    self.mean_ = X.mean(axis=0)
    self.std_ = X.std(axis=0) + 1e-9
    Z = (X - self.mean_) / self.std_
    self.pca = PCA(n_components=self.components)   # components=0.9
    self.pca.fit(Z)
    return self

def score(self, X):
    X = np.asarray(X, dtype=np.float64)
    Z = (X - self.mean_) / self.std_
    T = self.pca.transform(Z)                      # [N,k]
    rec = self.pca.inverse_transform(T)            # 子空间重构 [N,8]
    return np.linalg.norm(Z - rec, axis=1)         # 逐帧残差范数 SPE
```

要点解析：(1) 与 HotellingT2 共用同一 `fit`（PCA(n_components=0.9)），`score` 只保留 `norm(Z-rec)` 一项——两方法的全部差异就是统计量公式那一行，构成严格的受控对照；(2) `inverse_transform(transform(Z))` 数学上即 $P_kP_k^{\top}\boldsymbol{z}$（sklearn 内部已去均值再回加）；(3) 残差维数为 $8-k$（本数据 $k\approx3\text{–}4$），SPE 是 4–5 维残差向量的联合范数——单一标量承载"所有次要方向"的信息，任一次要方向的通道尺度漂移都会污染它；(4) 无状态逐帧方法。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/PcaReconstruction/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.587 |
| TPR@1%（实际FPR） | 0.018（0.002） |
| TPR@5%（实际FPR） | 0.082（0.022） |
| TPR@10%（实际FPR） | 0.150（0.061） |
| ADD / hit | inf / 0 |

**分析**：高功率阶跃主要体现为子空间内的大幅偏移（T² 的领地），SPE 只看到功率组合略微偏离清洁线性关系的一点残余——AUC 0.587、1% 档 0.018、ADD 未命中。与 Hotelling T²（同场景 0.885/0.998）对照可直接读出"残差空间单打独斗"的代价。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/PcaReconstruction/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.431 |
| TPR@1%（实际FPR） | 0.005（0.002） |
| TPR@5%（实际FPR） | 0.018（0.019） |
| TPR@10%（实际FPR） | 0.037（0.060） |
| ADD / hit | inf / 0 |

**分析**：AUC 0.431 低于 0.5——分数在该场景**反向**：匹配功率下畸变特征的变化被主成分容纳（子空间内），而残差方向主要记录的是清洁噪声的通道差异，欺骗帧的 SPE 反而略低于清洁均值附近。反向排序是"签名完全落在保留子空间内"的标志性行为。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/PcaReconstruction/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.558 |
| TPR@1%（实际FPR） | 0.053（0.002） |
| TPR@5%（实际FPR） | 0.107（0.023） |
| TPR@10%（实际FPR） | 0.156（0.057） |
| ADD / hit | 58.44 s / 1 |

**分析**：微漂移落在子空间内的码相位族方向，残差无信号；ADD 58 s 来自漂移后期偶发越限。三档 <0.16。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/PcaReconstruction/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.730 |
| TPR@1%（实际FPR） | 0.182（0.002） |
| TPR@5%（实际FPR） | 0.325（0.022） |
| TPR@10%（实际FPR） | 0.408（0.058） |
| ADD / hit | 142.70 s / 1 |

**分析**：本方法最好场景之一也仅 0.182@1%。动态运动把大量能量放进次要方向（残差被运动噪声抬高基线），功率阶跃在残差上的相对对比进一步被稀释；ADD 142.70 s 为全方法最长之一。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/PcaReconstruction/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.321 |
| TPR@1%（实际FPR） | 0.001（0.024） |
| TPR@5%（实际FPR） | 0.009（0.092） |
| TPR@10%（实际FPR） | 0.023（0.163） |
| ADD / hit | inf / 0 |

**分析**：AUC 0.321，全方法全场景最低——强烈的反向排序。动态清洁段（cd）贡献训练残差的长尾，而欺骗期特征组合反而更接近清洁子空间（载波/码锁定良好时残差小），SPE 在该场景完全失效。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/PcaReconstruction/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.465 |
| TPR@1%（实际FPR） | 0.041（0.020） |
| TPR@5%（实际FPR） | 0.090（0.087） |
| TPR@10%（实际FPR） | 0.141（0.157） |
| ADD / hit | 65.88 s / 1 |

**分析**：AUC 0.465，又是反向。机理同 ds2/ds5：载波对齐的欺骗信号在清洁子空间内"更干净"，残差低于清洁。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/PcaReconstruction/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.859 |
| TPR@1%（实际FPR） | 0.577（0.002） |
| TPR@5%（实际FPR） | 0.675（0.019） |
| TPR@10%（实际FPR） | 0.711（0.055） |
| ADD / hit | 102.94 s / 1 |

**分析**：与 Hotelling T² 一致的反转模式：时间调整类攻击在本方法下反而是最好场景（1% 档 0.577@0.002）。时间篡改逐步改变特征组合关系，缓慢地把能量漏进残差方向；ADD 102.94 s 反映积累。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/PcaReconstruction/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.857 |
| TPR@1%（实际FPR） | 0.553（0.002） |
| TPR@5%（实际FPR） | 0.664（0.020） |
| TPR@10%（实际FPR） | 0.704（0.055） |
| ADD / hit | 75.42 s / 1 |

**分析**：与 ds7 几乎复制（0.553/0.664/0.704），ADD 75 s。虚警纪律好，再次确认"残差空间对时间篡改敏感、对几何类攻击失明"的互补格局。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.6010 |
| TPR@1% / 实际 FPR | 0.1788 / 0.0070 |
| TPR@5% / 实际 FPR | 0.2463 / 0.0380 |
| TPR@10% / 实际 FPR | 0.3002 / 0.0675 |
| ADD 命中 | 5/8 |

**综合分析**：(1) 全方法倒数第二（仅好于 STL 与 StatZ 族），且**三个场景 AUC < 0.5（反向）**——欺骗样本重构得比清洁还好的现象本身是重要证据：欺骗接管后跟踪环锁定质量不低于清洁段，几何类攻击的主要签名在"清洁模式"的方向上而不在其正交补上；(2) 与 Hotelling T² 的受控对照（同一 PCA、同一 fit，只差统计量一项）给出干净的分解：T² 项贡献了绝大部分检测能力（0.827 vs 0.601），SPE 项单独反而引入反向风险；合并版（T²+SPE）优于 SPE 单独版；(3) 低维教训与 Hotelling 文档一致：8 维特征集上"截断子空间+残差"的建模范式不如全协方差；(4) 论文定位：作为"子空间残差范式在低维 SQM 特征上失效"的反例，与 02/03 文档共同构成 PCA 族的三点对照实验。
