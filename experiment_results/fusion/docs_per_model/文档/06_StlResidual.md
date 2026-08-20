# 06 STL 残差（季节-趋势分解）——方法详解与逐场景结果

> 系列文档之六。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_StlResidual/metrics.csv` 一致。**本方法为时序分解族的适用性边界反例**。

## 一、方法原理与数学定义

STL（Seasonal-Trend decomposition using Loess, Cleveland et al. 1990）把序列分解为三个成分：

$$x_{f,t}=\tau_{f,t}+s_{f,t}+r_{f,t},$$

$\tau$ 为缓慢变化的趋势（loess 平滑）、$s$ 为周期成分（周期 $p$，鲁棒迭代加权）、$r$ 为残差。本实现取周期 $p=50$ 帧（1 s），`robust=True`。检测的思想：清洁序列的残差 $r$ 是平稳噪声，其尺度 $\hat{\sigma}_{r,f}$ 由清洁训练集估计；异常帧表现为残差放大：

$$s_t=\frac{1}{F}\sum_{f=1}^{F}\frac{\left|r_{f,t}\right|}{\hat{\sigma}^{\text{train}}_{r,f}},\qquad F=8 .$$

**为什么原理性不匹配**：STL 的设计对象是"趋势 + 显著周期 + 平稳余项"的慢时序（小时级电力负荷、日周期气象）。GNSS 跟踪环特征是快变噪声驱动序列——不存在稳定的 1 s 周期（把周期参数设为 50 只是让算法可运行），逐帧波动本身就是主体。更致命的是**趋势项会吸收慢漂移**：位置推送/时间调整类攻击的签名恰恰是慢漂移，趋势平滑器把它当作"正常趋势变化"吸收进 $\tau$，残差 $r$ 里剩下的只有噪声——**分解过程精确地删除了要检测的信号**。序列型方法按场景块分段打分（训练 cs/cd 块、测试场景块），每块独立分解。

## 二、源码解析

实现位于 `method_lib/traditional/StlResidual_SingleVar_Seasonal.py`：

```python
def fit(self, X):
    X = np.asarray(X, dtype=np.float64)
    res = []
    for m in range(X.shape[1]):
        r = STL(X[:, m], period=self.season, robust=True).fit().resid  # 逐列 STL，取残差
        res.append(r)
    self.res_train = np.stack(res, axis=1)
    self.std_ = self.res_train.std(axis=0) + 1e-9                     # 清洁残差尺度 σ_r,f
    return self

def score(self, X):
    X = np.asarray(X, dtype=np.float64)
    res = []
    for m in range(X.shape[1]):
        r = STL(X[:, m], period=self.season, robust=True).fit().resid  # 测试块同样分解
        res.append(r)
    res = np.stack(res, axis=1)
    return (np.abs(res) / self.std_).mean(axis=1)                      # 平均标准化残差
```

要点解析：(1) `fit` 对 8 列各做一次鲁棒 STL，只用残差的 std 做尺度——趋势与周期被整体丢弃；(2) `score` 对**测试块**再次分解：块内首尾的分解边界效应（loess 无法外推的端点）会给首尾帧带来系统性残差偏差，分块打分意味着每个场景块首尾各有一小段分数不可信；(3) `robust=True` 的迭代加权进一步压低大残差权重——对真异常同样"鲁棒"掉，等于双重削弱；(4) 计算量：STL 为 $O(n)$ 每次迭代、鲁棒版多次迭代，45k×8 列训练 + 8 块×22k×8 列测试约需数分钟，是全部方法中最慢的之一。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StlResidual/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.540 |
| TPR@1%（实际FPR） | 0.003（0.000） |
| TPR@5%（实际FPR） | 0.029（0.010） |
| TPR@10%（实际FPR） | 0.070（0.032） |
| ADD / hit | inf / 0 |

**分析**：功率阶跃是阶跃型信号，loess 趋势在阶跃后的几帧内即平滑跟上，只有阶跃瞬间 2–3 帧产生大残差——AUC 0.540（接近随机），工作点全废，ADD 未命中。攻击对 STL 而言"太慢也太快"：慢到被趋势吸收，快到只在瞬间闪现。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StlResidual/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.535 |
| TPR@1%（实际FPR） | 0.023（0.001） |
| TPR@5%（实际FPR） | 0.073（0.015） |
| TPR@10%（实际FPR） | 0.116（0.040） |
| ADD / hit | 61.74 s / 1 |

**分析**：峰畸变的抬升同样被趋势吸收，AUC 0.535。ADD 61.74 s 来自拉偏过渡期的瞬时残差。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StlResidual/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.551 |
| TPR@1%（实际FPR） | 0.006（0.001） |
| TPR@5%（实际FPR） | 0.036（0.015） |
| TPR@10%（实际FPR） | 0.077（0.038） |
| ADD / hit | 59.92 s / 1 |

**分析**：位置推送的慢漂移是趋势项的"本职工作"，残差几乎不含签名——AUC 0.551。这是"分解删除信号"机理最纯粹的体现。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StlResidual/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.633 |
| TPR@1%（实际FPR） | 0.015（0.000） |
| TPR@5%（实际FPR） | 0.075（0.011） |
| TPR@10%（实际FPR） | 0.139（0.035） |
| ADD / hit | 158.44 s / 1 |

**分析**：动态场景运动多普勒使特征自带大波动，残差基线被抬高，信噪比进一步恶化；ADD 158.44 s 为全方法最长。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StlResidual/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.410 |
| TPR@1%（实际FPR） | 0.017（0.011） |
| TPR@5%（实际FPR） | 0.066（0.069） |
| TPR@10%（实际FPR） | 0.108（0.126） |
| ADD / hit | 84.72 s / 1 |

**分析**：AUC 0.410——反向。清洁动态段的残差（运动噪声）大于欺骗段（锁定良好），排序倒挂。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StlResidual/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.430 |
| TPR@1%（实际FPR） | 0.011（0.009） |
| TPR@5%（实际FPR） | 0.044（0.064） |
| TPR@10%（实际FPR） | 0.083（0.124） |
| ADD / hit | 65.80 s / 1 |

**分析**：AUC 0.430，反向。接管完成后欺骗信号比清洁更"平稳"，残差更小，机理与 ds5 相同。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StlResidual/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.621 |
| TPR@1%（实际FPR） | 0.027（0.003） |
| TPR@5%（实际FPR） | 0.112（0.023） |
| TPR@10%（实际FPR） | 0.186（0.055） |
| ADD / hit | inf / 0 |

**分析**：慢速时间篡改被趋势吸收得最彻底（这正是趋势模型眼中的"正常变化"），ADD 未命中。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/StlResidual/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.693 |
| TPR@1%（实际FPR） | 0.043（0.001） |
| TPR@5%（实际FPR） | 0.163（0.018） |
| TPR@10%（实际FPR） | 0.257（0.046） |
| ADD / hit | 48.26 s / 1 |

**分析**：篡改幅度大、速度快，部分逃过趋势平滑进入残差——AUC 0.693 为该方法唯一超过 0.6 的场景，仍远不可用。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.5516 |
| TPR@1% / 实际 FPR | 0.0180 / 0.0031 |
| TPR@5% / 实际 FPR | 0.0747 / 0.0281 |
| TPR@10% / 实际 FPR | 0.1296 / 0.0480 |
| ADD 命中 | 6/8 |

**综合分析**：(1) 全方法垫底（AUC 0.552，六场景 <0.7、两场景反向）。失败是**结构性**的而非调参性的：趋势项吸收慢漂移签名 + 鲁棒加权压制瞬态残差 + 1 s 周期假设无物理对应，三重不利叠加；(2) 与序贯方法（EWMA/CUSUM 族）的对比最能说明问题——两者都利用时间结构，但方向相反：EWMA/CUSUM 用历史累积**放大**持续偏离，STL 用平滑**消除**持续偏离。同一"时间维度"，一个是检测器一个是滤波器；(3) 论文定位：作为"时序分解范式不适用于快变跟踪环特征"的边界证据，并给出周期参数选择的物理要求（若坚持用分解类方法，周期应取导航数据位周期 20 ms 的整数倍并结合差分而非原始特征）。
