# 08 局部离群因子 LOF（k=20）——方法详解与逐场景结果

> 系列文档之八。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_LOF/metrics.csv` 一致。

## 一、方法原理与数学定义

局部离群因子（Local Outlier Factor, Breunig, Kriegel, Ng & Sander, SIGMOD 2000）把"离群"从全局距离升级为**局部密度比**。对样本 $p$，先定义其 $k$ 距离 $d_k(p)$（第 $k$ 近邻距离）、可达距离 $\mathrm{reach}_k(p,q)=\max\{d_k(q),\,d(p,q)\}$ 与局部可达密度：

$$\mathrm{lrd}(p)=\left(\frac{1}{|N_k(p)|}\sum_{q\in N_k(p)}\mathrm{reach}_k(p,q)\right)^{-1},$$

即"到邻域的平均可达距离的倒数"。检测统计量为邻域密度比的均值：

$$s_t=\mathrm{LOF}(p)=\frac{1}{|N_k(p)|}\sum_{q\in N_k(p)}\frac{\mathrm{lrd}(q)}{\mathrm{lrd}(p)},\qquad k=20 .$$

$\mathrm{LOF}\approx 1$：与邻域密度一致（正常）；$\mathrm{LOF}\gg 1$：自身密度显著低于邻居（局部离群）。LOF 的卖点是对**不等密度簇**的处理：全局距离法在"密簇的小偏移"与"疏簇的大偏移"之间无法统一尺度，LOF 通过除以邻居密度实现局部归一——cs 静态簇密、cd 动态簇疏的理想适配场景。

**已知弱点**：(1) **集群式盲区**——若异常样本彼此邻近（形成自己的小簇），其 lrd 与邻居相当，LOF→1，完全漏检；欺骗攻击的样本恰恰是连续成串的（一个场景 1.5 万帧欺骗数据互相为邻）；(2) 密度比的数值病态（分母趋零时爆炸）；(3) 与 kNN 同样的"清洁库密度 = 尺度"的迁移税。

## 二、源码解析

实现位于 `method_lib/traditional/Lof_MultiVar_LocalDensity.py`：

```python
def fit(self, X):
    X = np.asarray(X, dtype=np.float64)
    self.mean_ = X.mean(axis=0)
    self.std_ = X.std(axis=0) + 1e-9
    Z = (X - self.mean_) / self.std_
    self.clf = LocalOutlierFactor(
        n_neighbors=self.n_neighbors,      # k=20
        contamination=self.contamination,  # 0.01（仅影响内部 offset，不影响 score_samples）
        novelty=True)                      # 关键：允许对新样本打分
    self.clf.fit(Z)
    return self

def score(self, X):
    Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
    return -self.clf.decision_function(Z)  # decision_function 越大越正常 → 取负
```

要点解析：(1) `novelty=True` 使 LOF 进入"novelty detection"模式，可对训练集之外样本计算 $\mathrm{LOF}$——否则只能对训练集内部互评（`fit_predict`），无法用于测试帧；(2) `decision_function = -LOF 归一化`（sklearn 内部再做缩放平移），取负后"越大越异常"统一到全库口径；(3) `contamination=0.01` 只影响内部 `offset_`（用于二值 predict），对连续分数无影响，阈值标定完全交给驱动层的清洁分位；(4) $k=20$ 是 sklearn 惯例值，作用是平滑密度估计（比 kNN 检测的 $k=5$ 大，因为密度估计需要更多邻居）；(5) 无状态逐帧方法。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/LOF/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.998 |
| TPR@1%（实际FPR） | 0.999（0.135） |
| TPR@5%（实际FPR） | 0.999（0.279） |
| TPR@10%（实际FPR） | 0.999（0.391） |
| ADD / hit | 0.04 s / 1 |

**分析**：功率阶跃帧的邻居仍在清洁簇内、自身远离——LOF 巨大，检出 0.999、ADD 0.04 s。但**实际 FPR 三档 0.135/0.279/0.391**，名义 1% 的阈值在清洁段实际放行 13.5%：LOF 分数（密度比）的清洁分布在测试通道上整体上移，阈值迁移税比 kNN（0.019）更重一个量级——密度比值对参照库密度变化极其敏感。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/LOF/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 1.000（0.136） |
| TPR@5%（实际FPR） | 1.000（0.286） |
| TPR@10%（实际FPR） | 1.000（0.391） |
| ADD / hit | 0.04 s / 1 |

**分析**：检出完美（三档 1.000），FPR 结构与 ds1 完全一致（0.136/0.286/0.391）——LOF 的清洁分数分布在静态场景间高度一致，说明迁移税是系统性的（分数定义所致）而非数据噪声。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/LOF/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.683 |
| TPR@1%（实际FPR） | 0.406（0.134） |
| TPR@5%（实际FPR） | 0.526（0.279） |
| TPR@10%（实际FPR） | 0.610（0.389） |
| ADD / hit | 0.04 s / 1 |

**分析**：微漂移样本沿清洁流形边缘缓慢滑出，其局部密度只有轻微下降（邻居尚在），LOF 信号弱——AUC 0.683 为该场景全方法较低档。ADD 0.04 s 再次来自拉偏瞬态。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/LOF/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.931 |
| TPR@1%（实际FPR） | 0.834（0.135） |
| TPR@5%（实际FPR） | 0.914（0.278） |
| TPR@10%（实际FPR） | 0.943（0.397） |
| ADD / hit | 0.18 s / 1 |

**分析**：功率阶跃在动态簇上同样产生强局部稀疏，1% 档 0.834。注意这是"名义 1%、实际 13.5% 虚警"下的数字。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/LOF/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 1.000（0.165） |
| TPR@5%（实际FPR） | 1.000（0.357） |
| TPR@10%（实际FPR） | 1.000（0.489） |
| ADD / hit | 0.06 s / 1 |

**分析**：检出满分；FPR 略高于静态（0.165/0.357/0.489），动态簇密度低使清洁帧的 LOF 基线抬高。**1% 名义阈值实际虚警 16.5%**——在严格虚警预算的部署里该工作点不可用。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/LOF/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.928 |
| TPR@1%（实际FPR） | 0.868（0.165） |
| TPR@5%（实际FPR） | 0.943（0.360） |
| TPR@10%（实际FPR） | 0.966（0.493） |
| ADD / hit | 0.06 s / 1 |

**分析**：名义档下检出 0.868（表观全方法前列），但实际 FPR 0.165——**其"高检出"部分是虚警放水换来的**。若把阈值收紧到实际虚警 1%，其 TPR 会大幅缩水（从其分数分布与 kNN/EWMA 的对比推断约 0.4–0.6 区间）。这组数字是"名义 FPR vs 实际 FPR 必须成对报告"的最有力论据。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/LOF/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.924 |
| TPR@1%（实际FPR） | 0.714（0.005） |
| TPR@5%（实际FPR） | 0.756（0.032） |
| TPR@10%（实际FPR） | 0.791（0.069） |
| ADD / hit | 62.68 s / 1 |

**分析**：时间篡改的缓移最终使样本滑入低密度区，1% 档 0.714、FPR 低位（0.005，静态场景且远离簇边界时 LOF 迁移良好）。ADD 62.68 s。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/LOF/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.962 |
| TPR@1%（实际FPR） | 0.743（0.006） |
| TPR@5%（实际FPR） | 0.825（0.041） |
| TPR@10%（实际FPR） | 0.874（0.094） |
| ADD / hit | 2.42 s / 1 |

**分析**：大幅篡改下 0.743@0.006，ADD 2.42 s——LOF 在静态弱签名场景的可用性与 kNN 相当。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.9283 |
| TPR@1% / 实际 FPR | 0.8203 / 0.1100 |
| TPR@5% / 实际 FPR | 0.8701 / 0.2391 |
| TPR@10% / 实际 FPR | 0.8979 / 0.3391 |
| ADD 命中 | 8/8 |

**综合分析**：(1) 排序能力优秀（AUC 0.928，六个场景 ≥0.92），但**阈值迁移为全部可用方法最差**——1% 档实际 FPR 0.110（名义的 11 倍），5%/10% 档 0.24/0.34。根源是密度比统计量对"参照库密度 = 尺度"的绝对依赖：测试通道的清洁密度与 cs/cd 不一致时分数整体平移；(2) 有趣的对照：**kNN（绝对距离）迁移税 0.040，LOF（相对密度）迁移税 0.110**——把距离归一到邻居密度反而放大了迁移敏感度，因为分子分母同时受库密度影响；(3) 论文写作建议：以本方法为"名义 FPR 具有误导性"的核心例证（ds6 表观 0.868 vs 实际虚警 16.5%），并给出"密度型方法在跨通道部署前必须做通道内再标定"的工程结论；(4) 未进入最终融合委员会（455 变体搜索中含 LOF 的组合全部因 rfpr 超标被剪枝），与上述分析自洽。
