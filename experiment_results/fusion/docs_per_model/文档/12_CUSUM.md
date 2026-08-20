# 12 CUSUM（累积和，k=0.5，h=5）——方法详解与逐场景结果

> 系列文档之十二。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_CUSUM/metrics.csv` 一致。本方法以"零实际虚警"著称，
> 但库版维度聚合存在方向抵消缺陷（改进② MCUSUM 的靶点）。

## 一、方法原理与数学定义

CUSUM（Cumulative Sum, Page 1954）是序贯变化检测的奠基方法。对标准化观测的通道均值 $\bar z_t=\frac{1}{F}\sum_f z_{f,t}$ 维护双边累积量：

$$C^{+}_t=\max\!\big(0,\ C^{+}_{t-1}+\bar z_t-k\big),\qquad
C^{-}_t=\max\!\big(0,\ C^{-}_{t-1}-\bar z_t-k\big),$$

$$s_t=\frac{\max\big(C^{+}_t,\ C^{-}_t\big)}{h},\qquad k=0.5,\ h=5 .$$

参数 $k$（参考值/容许带）为"每帧允许的噪声漂移量"，$h$（决策区间）为告警门限。**统计最优性**：对幅度已知的确定性漂移，CUSUM 在给定虚警约束下使平均检测延迟（ARL1）达到下界（Lorden, 1971 的 CUSUM 最优性）；对幅度 $\delta$ 的持续漂移，$C^{\pm}$ 以每帧 $(\delta-k)$ 的速度线性累积——**指数式信噪比放大**：$n$ 帧后累积偏移 $n\delta$ 对噪声 $\sqrt{n}\sigma$，信噪比按 $\sqrt{n}$ 增长。同时 max(0,·) 的"重置"使单帧噪声无法累积——**对短时波动完全免疫**（虚警纪律的来源）。

**库版聚合缺陷**（改进②的靶点）：累积前对 8 维取均值——欺骗引发的各特征偏移**方向异号**（received_power/CN0 上升、m_ratio/m_delta 下降），部分抵消；且"漂移仅发生于个别特征族"是常态。逐特征改进见第 16 篇。

## 二、源码解析

实现位于 `method_lib/traditional/Cusum_SingleVar_Online.py`：

```python
def fit(self, X):
    X = np.asarray(X, dtype=np.float64)
    self.mean_ = X.mean(axis=0)
    self.std_ = X.std(axis=0) + 1e-9
    return self

def score(self, X):
    z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
    s_plus = np.zeros(z.shape[0] + 1)
    s_minus = np.zeros(z.shape[0] + 1)
    for i in range(z.shape[0]):
        s_plus[i + 1] = max(0.0, s_plus[i] + z[i].mean() - self.k)   # 通道均值→累积
        s_minus[i + 1] = max(0.0, s_minus[i] - z[i].mean() - self.k)
    return np.maximum(s_plus[1:], s_minus[1:]) / self.h
```

要点解析：(1) `z[i].mean()`——**先均值后累积**，方向抵消发生处；MCUSUM 把 `.mean()` 移出递推、对每维各自累积后取 max；(2) 边界数组 `s[0]=0` 表示每段起始状态归零——驱动层按块调用（cs/cd、各场景）；(3) 递推循环纯 Python（176k 帧约 0.5 s），可向量化但无必要；(4) $k=0.5,h=5$ 为 SPC 惯例（检测约 1σ 漂移、ARL0≈465），未调参；(5) score 除以 $h$ 只是量纲归一，阈值标定交给清洁分位。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/CUSUM/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.999 |
| TPR@1%（实际FPR） | 0.997（0.000） |
| TPR@5%（实际FPR） | 0.998（0.013） |
| TPR@10%（实际FPR） | 0.998（0.034） |
| ADD / hit | 0.04 s / 1 |

**分析**：功率阶跃使 $\bar z$ 每帧偏移数 σ，$C^+$ 三帧内冲破 $h$——0.997@零虚警、ADD 0.04 s。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/CUSUM/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 0.997（0.000） |
| TPR@5%（实际FPR） | 1.000（0.005） |
| TPR@10%（实际FPR） | 1.000（0.026） |
| ADD / hit | 0.04 s / 1 |

**分析**：匹配功率下功率特征不动，但畸变族特征同向抬升，均值未严重抵消——近满分、三档实际 FPR ≤0.026，纪律极佳。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/CUSUM/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.757 |
| TPR@1%（实际FPR） | 0.284（0.000） |
| TPR@5%（实际FPR） | 0.421（0.020） |
| TPR@10%（实际FPR） | 0.484（0.046） |
| ADD / hit | 25.74 s / 1 |

**分析**：位置推送的漂移集中在码相位族（2–3 个特征），均值后幅度再除以 8——每帧累积速度 $(\bar\delta-k)$ 几乎贴着容许带 $k=0.5$，累积极慢：ADD 25.74 s、1% 档 0.284。**同数据同参数下 MCUSUM（逐特征）0.919@0**——方向抵消损失了约 0.64 的检出，是"聚合方式"单变量对照的最强证据。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/CUSUM/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.910 |
| TPR@1%（实际FPR） | 0.734（0.000） |
| TPR@5%（实际FPR） | 0.791（0.006） |
| TPR@10%（实际FPR） | 0.809（0.028） |
| ADD / hit | 85.16 s / 1 |

**分析**：运动噪声使 $\bar z$ 频繁越过 $\pm k$ 的反向，$C^{\pm}$ 被反复重置——功率阶跃的累积被运动打断，ADD 85.16 s（MCUSUM 同场景 96.3 s，同样受累）。"序贯累积 vs 快变噪声背景"的固有张力。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/CUSUM/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 0.996（0.000） |
| TPR@5%（实际FPR） | 0.999（0.229） |
| TPR@10%（实际FPR） | 0.999（0.326） |
| ADD / hit | 0.06 s / 1 |

**分析**：该场景拉偏幅度大、畸变同向，均值未抵消——0.996@零虚警、ADD 0.06 s。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/CUSUM/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.745 |
| TPR@1%（实际FPR） | 0.583（0.000） |
| TPR@5%（实际FPR） | 0.614（0.205） |
| TPR@10%（实际FPR） | 0.633（0.300） |
| ADD / hit | 0.06 s / 1 |

**分析**：载波对齐使漂移微弱且分散在少数特征，均值稀释 + 5%/10% 档虚警快速上升（0.21/0.30，动态长尾与漂移段分数重叠）——1% 档 0.583@0 尚可，宽档不可用。MCUSUM 同场景 0.939@0。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/CUSUM/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.866 |
| TPR@1%（实际FPR） | 0.662（0.000） |
| TPR@5%（实际FPR） | 0.711（0.030） |
| TPR@10%（实际FPR） | 0.738（0.071） |
| ADD / hit | 115.10 s / 1 |

**分析**：慢速时间篡改的累积速度贴着容许带，ADD 115.10 s——全部方法该场景最慢之一；1% 档 0.662@零虚警（篡改后期终于累积越限）。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/CUSUM/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.937 |
| TPR@1%（实际FPR） | 0.696（0.000） |
| TPR@5%（实际FPR） | 0.849（0.040） |
| TPR@10%（实际FPR） | 0.876（0.096） |
| ADD / hit | 0.04 s / 1 |

**分析**：大幅篡改累积快，ADD 0.04 s、0.696@0。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.9017 |
| TPR@1% / 实际 FPR | 0.7437 / 0.0000 |
| TPR@5% / 实际 FPR | 0.7977 / 0.0685 |
| TPR@10% / 实际 FPR | 0.8172 / 0.1158 |
| ADD 命中 | 8/8 |

**综合分析**：(1) **虚警纪律全场唯一满分**：1% 档实际 FPR 逐场景全 0.0000（约 4.4 万清洁帧零误报）——max(0,·) 重置机制对短时波动的免疫在真实数据上完全兑现，这一性质使 CUSUM 族成为任何融合中"零虚警贡献者"；(2) 检出受均值聚合拖累（宏 0.744），结构性弱在 ds3（0.284）/ds6（0.583）；(3) 5%/10% 档虚警快速上升（0.069/0.116）的原因值得注意：宽档阈值下，**清洁段自身的慢漂移**（接收机时钟、温度）也能短程累积越限——CUSUM 的虚警不是白噪声型而是"清洁慢变化型"，宽档需要差分或去趋势预处理；(4) 库版→MCUSUM 的单变量对照（仅改聚合）：宏 1% 档 0.744→0.875、AUC 0.902→0.992、ds3 0.284→0.919——本系列最重要的改进证据链；(5) 融合角色：库版未入委员会（被 MCUSUM 全面替代），但作为"MCUSUM 的消融基线"必须出现在论文。
