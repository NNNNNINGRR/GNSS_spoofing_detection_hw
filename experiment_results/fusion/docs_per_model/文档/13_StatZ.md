# 13 StatZ（稳健 max-z 统计基线）——方法详解与逐场景结果

> 系列文档之十三。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_StatZ/metrics.csv` 一致。**本方法为自建非参数单帧检验基线，实测因
> z 分数饱和而失效**，其失败机理与 MCD/STL 同属论文的"负面结果"素材。

## 一、方法原理与数学定义

StatZ 是"逐特征阈值检验"的最简非参数形式：用**中位数**与 **MAD**（中位绝对偏差）替代均值/标准差，以获得对训练集"脏正常"帧（多径、周跳残余）的稳健性，再取跨特征最大值作为单帧检验统计量：

$$\mathrm{med}_f=\mathrm{median}\big(x_{f,\mathcal{D}}\big),\qquad
\mathrm{MAD}_f=\mathrm{median}\big(\left|x_{f,\mathcal{D}}-\mathrm{med}_f\right|\big),$$

$$s_t=\max_{f\in\{1,\dots,8\}}\ \frac{\left|x_{f,t}-\mathrm{med}_f\right|}{1.4826\cdot \mathrm{MAD}_f} .$$

系数 1.4826 使 MAD 在高斯分布下与标准差一致（$E[\mathrm{MAD}]=\sigma/\Phi^{-1}(0.75)\approx\sigma/1.349$）。**设计动机**：(1) max 聚合保留"最强单特征证据"（对照 StatThreshold 的均值稀释）；(2) 中位数/MAD 有 50% 崩溃点，理论上抗脏训练数据。**预期失效模式**（设计时已知的风险）：MAD 对**有界/重尾分布**极端保守——若某特征分布有硬边界（dB 域的 received_power 相对值），大部分样本集中在边界一侧，MAD 极小、z 分数在尾部"饱和"（上限被压到远小于高斯预期的水平），清洁分数的 p99 与 p99.9 几乎重合，分位阈值失去分辨力。

## 二、源码解析

实现为 `run_extras_v31.py` 中的自建类：

```python
class StatZ:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.med_ = np.median(X, axis=0)                    # 逐特征中位数
        mad = np.median(np.abs(X - self.med_), axis=0) * 1.4826
        self.mad_ = np.where(mad > 1e-12, mad, 1e-12)       # 防零下限
        return self

    def score(self, X):
        return np.max(np.abs(np.asarray(X) - self.med_) / self.mad_, axis=1)
        # 逐特征稳健 z，取行内最大 = "最强单特征证据"
```

要点解析：(1) `fit` 只有两组统计量（中位数、MAD），训练 O(N log N)；(2) `np.where(mad>1e-12,...)` 防止退化特征除零；(3) `score` 的 `max(axis=1)` 是与 StatThreshold（mean）的唯一聚合差异——两方法构成"聚合方式 × 尺度估计"的 2×2 实验一角（mean+std=StatThreshold，max+MAD=StatZ；另两角 max+std 与 mean+MAD 未单独跑，趋势可由二者插值推断）；(4) 无状态逐帧方法。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZ/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.783 |
| TPR@1%（实际FPR） | 0.015（0.017） |
| TPR@5%（实际FPR） | 0.054（0.074） |
| TPR@10%（实际FPR） | 0.106（0.140） |
| ADD / hit | inf / 0 |

**分析**：AUC 0.783 说明功率阶跃确实产生最大单特征偏离，但 z 分数饱和把欺骗与清洁的分数都压进同一窄带——1% 档 0.015、ADD 未命中。实际 FPR 0.017 恰为名义值：不是纪律好，而是清洁分数已被压平、阈值无论设在哪都是这个放行率。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZ/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.770 |
| TPR@1%（实际FPR） | 0.009（0.016） |
| TPR@5%（实际FPR） | 0.041（0.075） |
| TPR@10%（实际FPR） | 0.082（0.142） |
| ADD / hit | inf / 0 |

**分析**：同样的饱和形态，三档 ≤0.08。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZ/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.682 |
| TPR@1%（实际FPR） | 0.007（0.016） |
| TPR@5%（实际FPR） | 0.040（0.070） |
| TPR@10%（实际FPR） | 0.080（0.140） |
| ADD / hit | inf / 0 |

**分析**：微漂移 + 饱和双重不利，三档 ≤0.08。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZ/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.750 |
| TPR@1%（实际FPR） | 0.004（0.017） |
| TPR@5%（实际FPR） | 0.017（0.076） |
| TPR@10%（实际FPR） | 0.037（0.145） |
| ADD / hit | inf / 0 |

**分析**：动态长尾进一步压缩清洁/欺骗分数间距，全部场景最弱（1% 档 0.004）。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZ/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.770 |
| TPR@1%（实际FPR） | 0.009（0.011） |
| TPR@5%（实际FPR） | 0.044（0.055） |
| TPR@10%（实际FPR） | 0.085（0.106） |
| ADD / hit | inf / 0 |

**分析**：同形态。注意其实际 FPR（0.011/0.055/0.106）反而略低于名义——cd 段的 MAD 较大使动态帧 z 偏低，进一步压平。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZ/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.746 |
| TPR@1%（实际FPR） | 0.008（0.012） |
| TPR@5%（实际FPR） | 0.041（0.056） |
| TPR@10%（实际FPR） | 0.079（0.104） |
| ADD / hit | inf / 0 |

**分析**：三档 ≤0.08。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZ/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.732 |
| TPR@1%（实际FPR） | 0.005（0.015） |
| TPR@5%（实际FPR） | 0.024（0.071） |
| TPR@10%（实际FPR） | 0.053（0.138） |
| ADD / hit | inf / 0 |

**分析**：弱签名 + 饱和，1% 档 0.005。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StatZ/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.770 |
| TPR@1%（实际FPR） | 0.004（0.015） |
| TPR@5%（实际FPR） | 0.024（0.071） |
| TPR@10%（实际FPR） | 0.053（0.139） |
| ADD / hit | inf / 0 |

**分析**：即使大幅篡改也无法穿透饱和带——全部 8 场景 ADD 均未命中，是全方法中唯一"零命中"。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.7504 |
| TPR@1% / 实际 FPR | 0.0076 / 0.0148 |
| TPR@5% / 实际 FPR | 0.0358 / 0.0685 |
| TPR@10% / 实际 FPR | 0.0717 / 0.1280 |
| ADD 命中 | 0/8 |

**综合分析（失效解剖，论文素材）**：(1) 失效主因是**z 饱和**：训练分数诊断显示 p99≈p99.9（同一量级，如 29.97 vs 30.15）——MAD 极小的特征（dB 域 received_power 的分布半边贴界）主导 max，其余特征的证据被这个特征的低动态范围封顶；分位阈值在饱和带内失去分辨力，名义 FPR 与实际 FPR "完美一致"（0.015）恰是分数被压平的数学后果；(2) 对照实验链条：StatThreshold（mean+std，AUC 0.940）≫ StatZ（max+MAD，0.750）——**在本特征集上"稳健尺度估计"的代价远大于"max 聚合"的收益**；修复方向明确：(i) 尺度改用分位数差（IQR 型）或按特征分别选尺度，(ii) 对 dB 域特征先做分布变换（logit/秩）再算 z；(3) 方法论教训：**分位数阈值标定隐含"分数分布连续且不饱和"的前提**，统计量设计阶段就应检查训练分数的分位谱（p50/p90/p99/p99.9 的间隔），本文所有可用方法均通过该检查、三个失效方法（MCD/STL/StatZ 族）均未通过——该检查本身可作为论文的"方法可用性预检"贡献。
