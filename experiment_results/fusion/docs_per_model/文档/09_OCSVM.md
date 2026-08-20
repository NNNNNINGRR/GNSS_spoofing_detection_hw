# 09 One-Class SVM（RBF 核，ν=0.02）——方法详解与逐场景结果

> 系列文档之九。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_OCSVM/metrics.csv` 一致。

## 一、方法原理与数学定义

单类支持向量机（One-Class SVM, Schölkopf et al., Neural Computation 2001）在核空间中构造把清洁数据与原点分开的最大间隔超平面。通过核映射 $\phi:\mathbb{R}^8\to\mathcal{H}$（本文用 RBF 核 $K(\boldsymbol{x},\boldsymbol{x}')=\exp(-\gamma\|\boldsymbol{x}-\boldsymbol{x}'\|^2)$），求解：

$$\min_{\boldsymbol{w},\rho,\boldsymbol{\xi}}\ \frac{1}{2}\|\boldsymbol{w}\|^2+\frac{1}{\nu n}\sum_{i=1}^{n}\xi_i-\rho,$$
$$\text{s.t.}\quad \boldsymbol{w}^{\top}\phi(\boldsymbol{x}_i)\ge\rho-\xi_i,\quad \xi_i\ge0 ,$$

对偶解为稀疏的支持向量展开 $\boldsymbol{w}=\sum_i\alpha_i\phi(\boldsymbol{x}_i)$（$\alpha_i\ge0$，至多 $\nu n$ 个非零），检测统计量取决策值的负：

$$s_t=\rho-\sum_{i\in SV}\alpha_i K(\boldsymbol{x}_i,\boldsymbol{z}_t)\qquad(\text{越大越异常}).$$

参数含义：$\nu\in(0,1]$ 同时是训练误差率上界与支持向量比例下界（$\nu=0.02$ → 约 2% 清洁帧落在边界外、约 900 个支持向量刻画清洁域边界）；$\gamma=\text{scale}=1/(8\cdot\mathrm{Var})$ 控制 RBF 宽度。**几何直觉**：RBF 核下等价于在清洁样本周围"包络"出一个非凸的密度等高域，域内（核相似度高）判正常、域外异常——可视为"核平滑版的马氏椭球"，能表达任意形状的清洁域。

**已知行为**：$\gamma$ 偏大时边界紧贴训练样本（过拟合训练噪声，测试清洁段大量出界→虚警偏高）；$\nu$ 越小边界越紧。本文网格搜索确认 gamma=0.5 会让判决饱和（三档同一阈值附近），故取 scale。

## 二、源码解析

实现为 `run_fusion_v31.py` 中的包装类（库版同名方法与之等价）：

```python
class Ocsvm:
    def __init__(self, nu=0.02):
        self.nu = float(nu)

    def fit(self, X):
        self.clf = OneClassSVM(kernel="rbf", nu=self.nu, gamma="scale").fit(X)
        return self     # 训练 = 二次规划求 α、ρ；45k 样本约 1–3 分钟

    def score(self, X):
        return -self.clf.score_samples(X)
        # score_samples = 到边界的符号距离（sklearn 缩放），取负统一"越大越异常"
```

要点解析：(1) `gamma="scale"` 即 $1/(d\cdot\mathrm{Var}(X))$，对已标准化的输入约 $1/8$——RBF 宽度约为特征标准差的 $\sqrt{8}\approx2.8$ 倍，边界平滑不贴点；(2) `score_samples` 返回 $\rho-\boldsymbol{w}^{\top}\phi$ 的缩放版，负值在边界外（异常侧）；(3) 支持向量约 $\nu n\approx900$ 个，测试帧打分是与这 900 个 SV 的核求和——推理复杂度与训练集大小脱钩，适合部署；(4) 训练复杂度 $O(n^2\sim n^3)$，45k 样本是单机舒适上限；(5) 无状态逐帧方法。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/OCSVM/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.999 |
| TPR@1%（实际FPR） | 0.998（0.013） |
| TPR@5%（实际FPR） | 0.999（0.174） |
| TPR@10%（实际FPR） | 0.999（0.339） |
| ADD / hit | 0.04 s / 1 |

**分析**：功率阶跃样本远离 RBF 域（与所有 SV 的核相似度趋零），决策值负得多——0.998 检出、ADD 0.04 s。1% 档实际 FPR 0.013 接近名义；5%/10% 档快速涨到 0.174/0.339，反映 RBF 域边界附近清洁分数的厚过渡带。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/OCSVM/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 1.000（0.012） |
| TPR@5%（实际FPR） | 1.000（0.174） |
| TPR@10%（实际FPR） | 1.000（0.334） |
| ADD / hit | 0.04 s / 1 |

**分析**：畸变特征组合离开 RBF 包络域，AUC 满分、三档 1.000、ADD 0.04 s。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/OCSVM/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.642 |
| TPR@1%（实际FPR） | 0.352（0.017） |
| TPR@5%（实际FPR） | 0.465（0.174） |
| TPR@10%（实际FPR） | 0.555（0.332） |
| ADD / hit | 0.04 s / 1 |

**分析**：微漂移样本贴着 RBF 域边缘滑动，核相似度仅微降——AUC 0.642，为该场景中等偏弱。ADD 0.04 s 依旧是瞬态贡献。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/OCSVM/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.932 |
| TPR@1%（实际FPR） | 0.805（0.013） |
| TPR@5%（实际FPR） | 0.886（0.173） |
| TPR@10%（实际FPR） | 0.919（0.341） |
| ADD / hit | 0.18 s / 1 |

**分析**：动态清洁域由 cd 簇的 SV 刻画，功率阶跃样本出域明显——0.805@0.013，表现稳健。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/OCSVM/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 0.999（0.059） |
| TPR@5%（实际FPR） | 0.999（0.219） |
| TPR@10%（实际FPR） | 1.000（0.352） |
| ADD / hit | 0.06 s / 1 |

**分析**：检出近满分；动态清洁段实际 FPR 0.059（1% 档）——cd 簇边界刻画略松。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/OCSVM/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.949 |
| TPR@1%（实际FPR） | 0.824（0.066） |
| TPR@5%（实际FPR） | 0.942（0.231） |
| TPR@10%（实际FPR） | 0.967（0.361） |
| ADD / hit | 0.06 s / 1 |

**分析**：**OCSVM 的招牌场景**——sophisticated 攻击下拉偏瞬态与稳态样本都处于 RBF 域外稀疏区（核相似度对"位置微移"敏感），1% 档 0.824@0.066、5% 档 0.942，全方法该场景第一梯队（与 kNN 0.785、EWMA 0.953@0.036 各有胜负）。RBF 边界的局部自适应能力在此兑现。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/OCSVM/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.845 |
| TPR@1%（实际FPR） | 0.706（0.003） |
| TPR@5%（实际FPR） | 0.734（0.034） |
| TPR@10%（实际FPR） | 0.754（0.100） |
| ADD / hit | 83.06 s / 1 |

**分析**：时间篡改缓移逐步出域，1% 档 0.706@0.003，ADD 83 s。虚警纪律在静态弱签名场景良好。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/OCSVM/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.965 |
| TPR@1%（实际FPR） | 0.797（0.003） |
| TPR@5%（实际FPR） | 0.879（0.042） |
| TPR@10%（实际FPR） | 0.916（0.115） |
| ADD / hit | 0.40 s / 1 |

**分析**：大幅篡改快速出域：0.797@0.003、ADD 0.40 s。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.9164 |
| TPR@1% / 实际 FPR | 0.8100 / 0.0232 |
| TPR@5% / 实际 FPR | 0.8630 / 0.1525 |
| TPR@10% / 实际 FPR | 0.8888 / 0.2842 |
| ADD 命中 | 8/8 |

**综合分析**：(1) **ds6（sophisticated）最佳单方法之一**（0.824@1% 档），证明核边界对"域外稀疏样本"的判别力——这也是它进入融合候选池的原因；(2) 虚警结构居中：1% 档 0.023（好于 LOF 的 0.110、差于马氏的 0.005），5% 档起涨速快（RBF 边界过渡带厚）；(3) 参数敏感性实验（记录于传统 v2 批次）：gamma=0.5 时判决饱和（三档 TPR/FPR 几乎相同、实际 FPR≈0.35，完全不可用），nu 在 0.01–0.02 间影响很小——**γ 是 OCSVM 在本任务上的唯一敏感参数，scale 是安全选择**；(4) 计算成本：训练 1–3 分钟、推理仅与约 900 支持向量求核，50 Hz 实时无碍；(5) 与最终融合的关系：进入 7 成员候选池，在多数高分组合中出现（如 or3 搜索的 top 组合 max_EWMA+MCUSUM+OCSVM 等），最终委员会为三成员精简配置故未入选，其 ds6 能力由 MCUSUM+EWMA 并集覆盖。
