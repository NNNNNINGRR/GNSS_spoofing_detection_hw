# 03 Mahalanobis 距离——方法详解与逐场景结果

> 系列文档之三。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_Mahalanobis/metrics.csv` 一致。

## 一、方法原理与数学定义

马氏距离（Mahalanobis, 1936）度量样本到分布的"考虑形状后"的距离。把清洁训练集建模为多元高斯 $\mathcal{N}(\boldsymbol{\mu},\Sigma)$，用样本均值与样本协方差（加小正则 $\varepsilon=10^{-6}$ 保证可逆）替代参数，检测统计量为二次型：

$$s_t=(\boldsymbol{z}_t-\boldsymbol{\mu})^{\top}(\Sigma+\varepsilon I)^{-1}(\boldsymbol{z}_t-\boldsymbol{\mu}),\qquad \boldsymbol{z}_t\in\mathbb{R}^{8} .$$

在 $\Sigma=\sigma^2 I$ 的特例下退化为欧氏距离平方；一般情形下它完成三件事：

1. **方差加权**：沿清洁数据方差大的方向（CN0 慢波动、received_power 漂移）的偏离被降权，沿方差小的方向（m_symdiff、m_manfredini 等高信噪比 SQM 指标）的微小偏移被放大；
2. **去相关**：$\Sigma^{-1}$ 自动扣除特征间相关性——SQM 指标强相关（ratio 与 delta 近似负相关、功率与 CN0 正相关），欧氏/均值型度量会把同一份证据重复计分，马氏距离不会；
3. **椭圆等值线**：清洁分布的等密度线是椭球，$s_t$ 恰为椭球半径平方，若高斯假设成立则 $s_t\sim\chi^2_8$，阈值可由卡方分布直接给出（本文采用更稳健的清洁分位数标定，不依赖分布假设）。

**适用条件与弱点**：要求清洁分布近似高斯且协方差可被 45k 样本充分估计（8 维下无压力）；对多簇清洁数据（cs 静态簇 + cd 动态簇）单一高斯是"平均化"建模，簇间过渡区会被高估为正常；对欺骗期"自成一体"的密集簇，距离只衡量到清洁质心的加权距离，若欺骗簇恰好落在清洁椭球延长方向内则漏检。

## 二、源码解析

实现位于 `method_lib/traditional/Mahalanobis_MultiVar_Gaussian.py`，是全库最简洁的方法：

```python
class Mahalanobis_MultiVar_Gaussian:
    def __init__(self, reg=1e-6):
        self.reg = float(reg)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)                     # μ：清洁均值
        cov = np.cov(X, rowvar=False)                   # Σ：8×8 样本协方差
        self.cov_inv_ = np.linalg.pinv(cov + self.reg * np.eye(cov.shape[0]))
        return self                                      # 伪逆 + 正则：数值稳健

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        d = X - self.mean_
        return np.einsum("ni,ij,nj->n", d, self.cov_inv_, d)
        # 逐帧二次型 d'Σ⁻¹d；einsum 一次完成批量双线性型，比循环快两个量级
```

要点解析：(1) `fit` 仅两步——估计 $\boldsymbol{\mu},\Sigma$ 并预算 $\Sigma^{-1}$，训练复杂度 $O(N\cdot8^2)$，毫秒级；(2) `pinv + reg*I` 处理潜在病态（特征近线性相关时协方差接近奇异），保证数值稳定；(3) `score` 用 `einsum("ni,ij,nj->n")` 批量计算 176k 帧的二次型，单次调用约 0.1 s；(4) 无状态逐帧方法、无非线性超参——**它是后续一切"分布偏离型"方法（MEWMA、MCD、kNN）的线性基准**：MEWMA 相当于把马氏距离作用在 EWMA 状态上，kNN 相当于非参数版，MCD 相当于稳健版。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/Mahalanobis/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.999 |
| TPR@1%（实际FPR） | 0.998（0.000） |
| TPR@5%（实际FPR） | 0.998（0.004） |
| TPR@10%（实际FPR） | 0.998（0.029） |
| ADD / hit | 0.04 s / 1 |

**分析**：功率阶跃把样本推出清洁椭球极远（马氏距离可达清洁分位的数百倍），三档全部 0.998、ADD 0.04 s。1% 与 5% 档实际 FPR 均接近零——功率方向虽是清洁大方差方向（被降权），但 10 dB 阶跃的偏移量远超降权后的容限。教科书式的"强签名场景"。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/Mahalanobis/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 1.000（0.000） |
| TPR@5%（实际FPR） | 1.000（0.005） |
| TPR@10%（实际FPR） | 1.000（0.029） |
| ADD / hit | 0.04 s / 1 |

**分析**：全方法中少有的"完美场景"——AUC、三档 TPR 全部 1.000，ADD 0.04 s。机理：匹配功率下功率特征不动，但畸变特征（symdiff/manfredini/dd）方差小、是清洁椭球的"窄方向"，其抬升被 $\Sigma^{-1}$ 强烈放大。与 Hotelling T²（同为二次型但 90% 截断+大方差加权）在本场景的崩溃（0.006）对比，验证了**全协方差 vs 子空间加权**的差别正是"窄方向证据是否被保留"。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/Mahalanobis/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.787 |
| TPR@1%（实际FPR） | 0.286（0.002） |
| TPR@5%（实际FPR） | 0.362（0.009） |
| TPR@10%（实际FPR） | 0.434（0.034） |
| ADD / hit | 0.20 s / 1 |

**分析**：微漂移使样本持续略偏于椭球边缘，单帧距离不够远——1% 档 0.286。但 ADD 仅 0.20 s：攻击瞬间拉偏过渡期有一段距离显著抬升（首批越限即触发），随后样本在新的"欺骗椭球"内稳定、距离回落，导致 TPR 停在 0.29–0.43 区间。**这是"逐帧分布偏离型"方法对慢漂移攻击的典型形态：抓得住开头、守不住全程**——全程覆盖需要序贯累积（MCUSUM，见系列 16）。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/Mahalanobis/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.982 |
| TPR@1%（实际FPR） | 0.709（0.000） |
| TPR@5%（实际FPR） | 0.864（0.006） |
| TPR@10%（实际FPR） | 0.916（0.032） |
| ADD / hit | 0.40 s / 1 |

**分析**：动态场景的清洁协方差由 cd 段贡献了运动方向的大方差（椭球在运动方向被拉长），功率阶跃仍在多数帧上越出椭球，1% 档 0.709、5% 档 0.864。虚警纪律极好（1% 档实际 0.000）。运动导致部分欺骗帧落回"运动方向延长的椭球"内是检出未达 0.99 的原因。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/Mahalanobis/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 0.999（0.019） |
| TPR@5%（实际FPR） | 0.999（0.098） |
| TPR@10%（实际FPR） | 0.999（0.204） |
| ADD / hit | 0.06 s / 1 |

**分析**：动态+匹配功率下依然近满分（AUC 1.000，三档 0.999）——本场景拉偏过程的畸变特征抬升幅度大，加上动态清洁方差使椭球适配良好。1% 档实际 FPR 0.019（动态清洁长尾），10% 档升至 0.204：放宽阈值对该场景没有检出收益（已饱和），只有虚警代价。

### ds6（DS-7，sophisticated·载波对齐+匹配功率）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/Mahalanobis/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.914 |
| TPR@1%（实际FPR） | 0.410（0.022） |
| TPR@5%（实际FPR） | 0.754（0.101） |
| TPR@10%（实际FPR） | 0.875（0.205） |
| ADD / hit | 0.06 s / 1 |

**分析**：拉偏瞬态（攻击初期）距离显著抬升使 ADD 仅 0.06 s；但载波对齐使稳态欺骗样本贴近清洁椭球，中后期距离回落，1% 档只守到 0.410。5%/10% 档放宽到 0.754/0.875，但实际 FPR 同步涨到 0.10/0.21——分数分布重叠明显。与 ds3 同属"抓头不守尾"形态，且更严重。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/Mahalanobis/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.909 |
| TPR@1%（实际FPR） | 0.698（0.001） |
| TPR@5%（实际FPR） | 0.746（0.016） |
| TPR@10%（实际FPR） | 0.775（0.045） |
| ADD / hit | 100.08 s / 1 |

**分析**：时间调整的缓慢联合偏移最终把样本稳定推出椭球——1% 档 0.698 为全方法该场景前列；但 ADD 100 s：偏移需要约 100 s 积累才首次越限（连续 3 帧）。虚警纪律好（0.001）。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/Mahalanobis/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.991 |
| TPR@1%（实际FPR） | 0.848（0.000） |
| TPR@5%（实际FPR） | 0.966（0.014） |
| TPR@10%（实际FPR） | 0.979（0.042） |
| ADD / hit | 0.40 s / 1 |

**分析**：篡改幅度大、特征偏移快，1% 档 0.848、ADD 0.40 s。与 ds7 的对比再次呈现"篡改速率决定可见时间"规律（100 s → 0.4 s）。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.9477 |
| TPR@1% / 实际 FPR | 0.7436 / 0.0054 |
| TPR@5% / 实际 FPR | 0.8362 / 0.0316 |
| TPR@10% / 实际 FPR | 0.8719 / 0.0774 |
| ADD 命中 | 8/8 |

**综合分析**：(1) **阈值迁移最守纪律的距离型方法**：1% 档实际 FPR 0.0054 与名义 1% 几乎一致（全方法第二好，仅次于 CUSUM/MCUSUM 的零），说明清洁椭球在测试通道上高度稳定——论文中可作为"工作点可直接部署"的代表；(2) 检出结构呈"瞬态强、稳态弱"：六个场景 ADD ≤0.4 s，但 ds3/ds6 的稳态 TPR 被欺骗自稳态压制；(3) 纵向对比链条清晰：全协方差（本方法 0.948）> 90% 子空间（Hotelling 0.827）> 稳健 MCD（0.759）> 非参数 kNN 检出更高但虚警差（0.952/0.040）——**协方差建模方式每退一步，性能换一个方向的损失**；(4) 作为融合候选：其纪律性使它在 455 变体搜索中多次入选低虚警组合（如 rank_MCUSUM+KnnDist 之外的多成员组合）。
