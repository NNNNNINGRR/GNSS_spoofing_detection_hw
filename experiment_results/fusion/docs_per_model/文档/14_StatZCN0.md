# 14 StatZ-CN0（C/N0 条件化统计基线）——方法详解与逐场景结果

> 系列文档之十四。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_StatZCN0/metrics.csv` 一致。**本方法为 StatZ 的物理条件化改造，
> 动机正确但实现粒度不当，实测不敌无条件版**——其教训直接指向后续工作方向。

## 一、方法原理与数学定义

SQM 指标的噪声方差不是常数，而随载噪比变化：Pirsiavash, Broumandan & Lachapelle (2017) 式(13)给出 SQM 统计量方差与 $C/N_0$、相干积分时间 $T$ 的关系（$\mathrm{Var}\{\mathrm{SQM}\}\propto \tfrac{1}{C/N_0\cdot T}$ 的量级结构）。物理直觉：弱信号（低 CN0）时相关器输出噪声大、SQM 指标基线波动大；强信号时波动小。若用**同一组** med/MAD 标定全部帧，低 CN0 通道/时段的帧会被系统性判为异常（虚警源），高 CN0 帧的阈值又偏松（漏检源）。

StatZ-CN0 的条件化：把每特征的中位数与 MAD 按 CN0 以 2 dB 分档分别估计（训练时每档样本 >200 帧才启用，不足回退全局），测试帧按其 CN0 取对应档标定：

$$s_t=\max_{f}\ \frac{\left|x_{f,t}-\mathrm{med}_f(\mathrm{CN0}_t)\right|}{1.4826\cdot\mathrm{MAD}_f(\mathrm{CN0}_t)},$$

其中 $\mathrm{CN0}_t$ 为该帧的载噪比（8 维特征的第 8 维），$\mathrm{med}_f(\cdot),\mathrm{MAD}_f(\cdot)$ 为按 CN0 档位查表的分段常数。CN0 自身不参与 max（自己与自己比较无意义），改用全局档的 z。

**预期收益与风险**：收益——跨通道/跨场景（各 ds 最优通道 CN0 不同）的尺度归一，理论上修复"动态清洁段长尾"这类虚警；风险——(i) 分档使每档样本量缩至数百～数千，med/MAD 估计方差增大；(ii) CN0 档内的特征边际分布对比度可能反而下降（同一 CN0 下帧间更同质，max-z 缺少区分对象）；(iii) 2 dB 硬分档在档边界产生不连续。

## 二、源码解析

实现为 `run_extras_v31.py` 中的自建类：

```python
class StatZCN0:
    def __init__(self, cn0_idx=-1, bin_db=2.0):
        self.ci, self.bin_db = int(cn0_idx), bin_db   # CN0 列=最后一维；2 dB 分档

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        cn0 = X[:, self.ci]
        lo = np.floor(cn0.min() / self.bin_db) * self.bin_db
        self.edges_ = np.arange(lo, cn0.max() + self.bin_db, self.bin_db)  # 档边界
        self.med_ = np.full((len(self.edges_), X.shape[1]), np.nan)
        self.mad_ = np.full_like(self.med_, np.nan)
        for i, b in enumerate(self.edges_):
            m = (cn0 >= b) & (cn0 < b + self.bin_db)
            if m.sum() < 200:        # 样本不足的档留空 → 回退全局
                continue
            seg = X[m]
            self.med_[i] = np.median(seg, axis=0)
            mad = np.median(np.abs(seg - self.med_[i]), axis=0) * 1.4826
            self.mad_[i] = np.where(mad > 1e-12, mad, 1e-12)
        # 全局档（nan 回退）：各档 nan 位置填全体的 med/MAD
        ...
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        # 按当前帧 CN0 查档（searchsorted），逐特征 z 后取 max；CN0 维用全局档
        idx = np.clip(np.searchsorted(self.edges_, X[:, self.ci]) - 1, 0, len(self.edges_) - 1)
        z = np.abs(X - self.med_[idx]) / self.mad_[idx]
        g = len(self.edges_) // 2
        z[:, self.ci] = np.abs(X[:, self.ci] - self.med_[g, self.ci]) / self.mad_[g, self.ci]
        return z.max(axis=1)
```

要点解析：(1) `edges_` 由训练 CN0 范围生成（约 44–54 dB-Hz → 5–6 档），每档数千帧满足 >200 阈；(2) `searchsorted` 查档向量化，176k 帧毫秒级；(3) CN0 列特殊处理（全局档 z），避免"用条件预测条件"的逻辑循环；(4) 该实现保留了 StatZ 的 max-MAD 聚合骨架——**条件化与聚合方式两个变量同时在场**，实验结论需归因谨慎（见下文分析）；(5) 无状态逐帧方法。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZCN0/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.790 |
| TPR@1%（实际FPR） | 0.081（0.026） |
| TPR@5%（实际FPR） | 0.128（0.113） |
| TPR@10%（实际FPR） | 0.191（0.213） |
| ADD / hit | 8.70 s / 1 |

**分析**：条件化把 1% 档从 StatZ 的 0.015 提到 0.081、ADD 从未命中变 8.70 s——CN0 档内功率阶跃帧的相对偏离更突出（同档内比较更公平）。但绝对水平仍不可用，且 5%/10% 档实际 FPR（0.113/0.213）**超过名义值**——档内估计方差开始放血。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZCN0/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.766 |
| TPR@1%（实际FPR） | 0.011（0.024） |
| TPR@5%（实际FPR） | 0.049（0.113） |
| TPR@10%（实际FPR） | 0.093（0.209） |
| ADD / hit | inf / 0 |

**分析**：匹配功率下无强特征证据，条件化独木难支，ADD 未命中。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZCN0/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.447 |
| TPR@1%（实际FPR） | 0.003（0.023） |
| TPR@5%（实际FPR） | 0.015（0.109） |
| TPR@10%（实际FPR） | 0.032（0.205） |
| ADD / hit | inf / 0 |

**分析**：AUC 0.447——**反向**。机理：位置推送的漂移特征（码相位族）与 CN0 相关（拖偏过程中 CN0 轻微下降），条件化后"漂移被 CN0 解释掉"——把与攻击伴生的 CN0 变化当作条件变量扣除，等于删除了攻击线索的一部分。这是条件化方法的经典陷阱：**条件变量本身被攻击污染时，条件化会吸收信号**。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZCN0/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.687 |
| TPR@1%（实际FPR） | 0.001（0.025） |
| TPR@5%（实际FPR） | 0.004（0.116） |
| TPR@10%（实际FPR） | 0.011（0.214） |
| ADD / hit | inf / 0 |

**分析**：动态场景 CN0 波动大，帧在档间跳变，硬分档的不连续 + 档内估计噪声使分数近乎随机（0.001@1%）。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZCN0/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.775 |
| TPR@1%（实际FPR） | 0.007（0.004） |
| TPR@5%（实际FPR） | 0.034（0.011） |
| TPR@10%（实际FPR） | 0.063（0.025） |
| ADD / hit | inf / 0 |

**分析**：本场景 CN0 稳定、帧集中一两档内，条件化退化为局部 StatZ——1% 档 0.007 仍弱。注意实际 FPR（0.004/0.011/0.025）低于名义：cd 训练档的 MAD 偏大、动态帧 z 偏小。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZCN0/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.763 |
| TPR@1%（实际FPR） | 0.010（0.003） |
| TPR@5%（实际FPR） | 0.049（0.011） |
| TPR@10%（实际FPR） | 0.094（0.024） |
| ADD / hit | 17.88 s / 1 |

**分析**：三档 ≤0.10，ADD 17.88 s。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZCN0/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.634 |
| TPR@1%（实际FPR） | 0.001（0.022） |
| TPR@5%（实际FPR） | 0.006（0.109） |
| TPR@10%（实际FPR） | 0.014（0.208） |
| ADD / hit | inf / 0 |

**分析**：弱签名 + 条件化噪声，近乎全灭（0.001@1%）。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZCN0/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.754 |
| TPR@1%（实际FPR） | 0.000（0.023） |
| TPR@5%（实际FPR） | 0.003（0.113） |
| TPR@10%（实际FPR） | 0.009（0.214） |
| ADD / hit | inf / 0 |

**分析**：即使大幅篡改也几乎无帧越限。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.7021 |
| TPR@1% / 实际 FPR | 0.0141 / 0.0189 |
| TPR@5% / 实际 FPR | 0.0359 / 0.0868 |
| TPR@10% / 实际 FPR | 0.0633 / 0.1261 |
| ADD 命中 | 2/8 |

**综合分析（条件化的经验教训，论文素材）**：(1) **动机正确、实现不当**：噪声方差随 CN0 变化的物理（Pirsiavash 式 13）无争议，但"2 dB 硬分档 + med/MAD"的实现把每档样本切薄、估计方差放大，且档内同质性反而压缩 max-z 的对比度——条件化没有修复 StatZ 的饱和，只是换了饱和的位置；(2) **更深层的陷阱：条件变量被攻击污染**——ds3 的 AUC 反向（0.447）证明位置推送的漂移与 CN0 变化伴生，按 CN0 条件化等于把一部分攻击签名当"正常条件变化"扣除。推论：**条件变量必须选择不受攻击影响或影响可建模的量**（如接收机自知的运动状态、天线类型），而非攻击可直接扰动的信号质量指标；(3) 两个负面结果（StatZ 与 StatZCN0）合起来的方法论贡献：给出"单帧统计检验族在本特征集上的两条红线"——尺度估计须抗重尾（MAD 不行，宜分位数/秩），条件变量须外生于攻击；(4) 正确的后续形态：连续方差模型 $\sigma_f(\mathrm{CN0})$（如对 $\log\mathrm{Var}$ 做稳健回归）+ 秩变换特征 + 序贯累积（与 MCUSUM 结合），这是论文 future work 的具体入口。
