# 15 MEWMA（多元 EWMA，改进①）——方法详解与逐场景结果

> 系列文档之十五。协议同前：v3.1 八维特征、cs+cd 清洁训练、清洁分位三档阈值（1%/5%/10%）、逐帧判决。
> 表格数值与 `results/fusion/single_MEWMA/metrics.csv` 一致。本方法为**改进①**（文献依据：
> Lowry, Woodall, Champ & Rigdon 1992, Technometrics 34:46–53），最终进入融合委员会。

## 一、方法原理与数学定义

MEWMA（Multivariate EWMA）把 EWMA 从标量监视推广到向量状态并用协方差感知的二次型作检验统计量。对标准化特征向量维护 EWMA 状态：

$$\boldsymbol{E}_t=\lambda\,\boldsymbol{z}_t+(1-\lambda)\,\boldsymbol{E}_{t-1},\qquad \boldsymbol{E}_0=\boldsymbol{0},\qquad \lambda=0.2,$$

检测统计量为二次型：

$$s_t=Q_t=\boldsymbol{E}_t^{\top}\ \Sigma_E^{-1}\ \boldsymbol{E}_t,\qquad
\Sigma_E=\frac{\lambda}{2-\lambda}\,\Sigma_z,$$

$\Sigma_z$ 为清洁特征的 8×8 协方差（加 $10^{-6}$ 正则后求逆）。$\Sigma_E$ 是零均值平稳输入下 EWMA 状态的稳态协方差（Lowry et al. 1992 的标准形式），使 $Q_t$ 在清洁时近似服从 $\chi^2_8$——本文阈值仍用清洁分位标定，不依赖该分布近似。

**改进依据（针对库版 EWMA 的两个缺陷）**：

1. **保留偏移方向**：库版统计量 $\frac{1}{F}\sum_f|E_{f,t}|$ 对每维取绝对值再平均——欺骗引发的各维偏移方向异号（功率/CN0 升、ratio/delta 降）时，$|E|$ 平均虽不抵消（取了绝对值），但把"方向组合"这一判别信息全部丢弃；二次型 $Q_t=\boldsymbol{E}^{\top}\Sigma^{-1}\boldsymbol{E}$ 完整保留向量，能区分"沿清洁相关方向的漂移"（正常波动常走的路）与"正交方向的漂移"（异常）。
2. **协方差加权**：SQM 特征强相关（ratio 与 delta 近似负相关），库版各维独立求和等价于把相关维度重复计分、方差高估、信噪比受损；$\Sigma^{-1}$ 权重恰与马氏距离同理——沿大方差/强相关方向的偏离降权，沿独立小方差方向升权。

**代价与边界**：$Q_t$ 对"幅度相同但方向不同"的漂移不再线性可比（椭球度量）；对协方差估计误差敏感（8 维、45k 样本下可忽略）；λ 同 EWMA 决定记忆深度（约 5 帧有效、稳态方差缩至 0.11 倍）。

## 二、源码解析

实现为 `run_fusion_v31.py` 中的自建类：

```python
class MEWMA:
    """多元 EWMA（Lowry et al. 1992）。"""
    def __init__(self, lam=0.2, reg=1e-6):
        self.lam, self.reg = float(lam), float(reg)

    def _run(self, Z):
        E = np.zeros(Z.shape[1])                       # 状态归零（分块起点）
        q = np.empty(len(Z))
        for i in range(len(Z)):
            E = self.lam * Z[i] + (1 - self.lam) * E   # 向量 EWMA 递推
            q[i] = float(E @ self.Sigma_inv @ E)       # 二次型 Q_t
        return q

    def fit(self, Z):
        cov = np.cov(Z, rowvar=False) * self.lam / (2 - self.lam)  # Σ_E = λ/(2-λ)·Σ_z
        self.Sigma_inv = np.linalg.pinv(cov + self.reg * np.eye(Z.shape[1]))
        return self

    def score(self, Z):
        return self._run(np.asarray(Z, dtype=np.float64))
```

要点解析：(1) `fit` 只构造 $\Sigma_E^{-1}$——一次 8×8 协方差估计与伪逆，训练毫秒级；(2) `_run` 每段从 $\boldsymbol{E}=\boldsymbol{0}$ 起递推（驱动层按 cs/cd、场景块分段调用，杜绝跨块状态）；(3) 递推内联二次型 `E @ Sigma_inv @ E`，176k 帧毫秒级；(4) 与库版 EWMA 的代码差异仅两处——状态从逐维绝对值平均变为向量保留、统计量从均值变为二次型——构成严格的消融对照；(5) λ=0.2 与库版对齐，未调参（Lowry 建议区间 0.1–0.4）。

## 三、逐场景检测结果

### ds1（DS-2，简单·静态·高功率 ~10 dB）

![ds1](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MEWMA/ds1.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.999 |
| TPR@1%（实际FPR） | 0.998（0.000） |
| TPR@5%（实际FPR） | 0.998（0.068） |
| TPR@10%（实际FPR） | 0.999（0.315） |
| ADD / hit | 0.04 s / 1 |

**分析**：功率阶跃使 $\boldsymbol{E}$ 沿功率方向大幅伸长，二次型立即放大——0.998@0.000、ADD 0.04 s。注意 5% 档实际 FPR 仅 0.068，显著低于库版 EWMA 同档的 0.126——**协方差加权压低虚警的第一个证据**（同 TPR 下纪律更好）。

### ds2（DS-3，中等·静态·匹配功率）

![ds2](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MEWMA/ds2.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 1.000（0.000） |
| TPR@5%（实际FPR） | 1.000（0.080） |
| TPR@10%（实际FPR） | 1.000（0.332） |
| ADD / hit | 0.04 s / 1 |

**分析**：满分三档、零虚警。与库版同 TPR，但 5% 档 FPR 减半（0.080 vs 0.142）。

### ds3（DS-4，位置推送·0.4 dB）

![ds3](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MEWMA/ds3.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.900 |
| TPR@1%（实际FPR） | 0.327（0.002） |
| TPR@5%（实际FPR） | 0.687（0.076） |
| TPR@10%（实际FPR） | 0.891（0.318） |
| ADD / hit | 0.04 s / 1 |

**分析**：慢漂移场景中 MEWMA 略逊于库版（0.327 vs 0.399@1% 档）——二次型对"贴着清洁相关方向缓慢爬行"的漂移（位置推送恰沿码相位族的相关方向）不如 $|E|$ 平均敏感：椭球在相关方向被拉长、降权。**方向信息是双刃剑**：对正交于清洁结构的攻击有利，对沿清洁结构漂移的攻击不利。最终由 MCUSUM 补位。

### ds4（DS-5，简单·动态·高功率）

![ds4](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MEWMA/ds4.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 0.854（0.000） |
| TPR@5%（实际FPR） | 1.000（0.071） |
| TPR@10%（实际FPR） | 1.000（0.308） |
| ADD / hit | 0.04 s / 1 |

**分析**：1% 档 0.854 略低于库版 0.979——功率阶跃方向在动态清洁协方差中与运动方向部分共线（被降权），但 5% 档即回到 1.000。

### ds5（DS-6，中等·动态·匹配功率）

![ds5](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MEWMA/ds5.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 1.000 |
| TPR@1%（实际FPR） | 0.999（0.017） |
| TPR@5%（实际FPR） | 0.999（0.247） |
| TPR@10%（实际FPR） | 1.000（0.484） |
| ADD / hit | 0.06 s / 1 |

**分析**：检出近满分；1% 档实际 FPR 0.017 优于库版 0.032（虚警减半），5%/10% 档 FPR 也全面低于库版（0.247/0.484 vs 0.328/0.518）——动态场景中协方差加权的纪律收益最明显。

### ds6（DS-7，sophisticated）

![ds6](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MEWMA/ds6.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.985 |
| TPR@1%（实际FPR） | 0.786（0.022） |
| TPR@5%（实际FPR） | 0.998（0.244） |
| TPR@10%（实际FPR） | 0.999（0.484） |
| ADD / hit | 0.06 s / 1 |

**分析**：1% 档 0.786 低于库版 0.953——同 ds3 的机理（残余漂移沿清洁相关方向），但 5% 档 0.998 追平。**与库版在 ds3/ds6 的互补方向相反**（库版强在相关方向漂移、MEWMA 强在虚警纪律），这是两通道并集融合价值的另一来源。

### ds7（DS-8，时间调整）

![ds7](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MEWMA/ds7.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.939 |
| TPR@1%（实际FPR） | 0.711（0.000） |
| TPR@5%（实际FPR） | 0.837（0.005） |
| TPR@10%（实际FPR） | 0.860（0.046） |
| ADD / hit | 69.46 s / 1 |

**分析**：与库版相当（0.711 vs 0.741@1%），5%/10% 档虚警极低（0.005/0.046，库版 0.019/0.057）。ADD 69.46 s。

### ds8（DS-9，时间调整·大幅）

![ds8](D:/文献复现/SQM数据集制作/exp_gnss/results/fusion/figs_scenes/MEWMA/ds8.png)

| 指标 | 值 |
|---|---|
| ROC-AUC | 0.991 |
| TPR@1%（实际FPR） | 0.971（0.000） |
| TPR@5%（实际FPR） | 0.983（0.007） |
| TPR@10%（实际FPR） | 0.985（0.050） |
| ADD / hit | 0.04 s / 1 |

**分析**：0.971@0.000，5% 档 FPR 0.007（库版 0.027）——纪律收益再次确认。

## 四、宏平均汇总与综合分析

| 指标 | 宏平均（8 场景） |
|---|---|
| ROC-AUC | 0.9766 |
| TPR@1% / 实际 FPR | 0.8307 / 0.0051 |
| TPR@5% / 实际 FPR | 0.9376 / 0.0998 |
| TPR@10% / 实际 FPR | 0.9667 / 0.2920 |
| ADD 命中 | 8/8 |

**综合分析**：(1) **改进①的净收益是"虚警纪律"而非"检出"**：1% 档实际 FPR 从库版 0.0089 降至 0.0051（−43%），5%/10% 档同样全面下降（0.100/0.292 vs 0.153/0.303），而宏检出略降（0.831 vs 0.875@1% 档）——协方差加权消除了相关特征重复计分带来的虚警膨胀，同时沿清洁相关方向的慢漂移检测灵敏度让渡给了方向分辨；(2) **与库版形成方向互补**：库版强在 ds3/ds6（相关方向漂移），MEWMA 强在虚警结构与 ds1/ds2/ds8 的同检出零虚警——两通道告警在时间轴上错位互补，是 or3 融合选择它的核心理由（融合后 ds3 TPR=1.0 的并集证据之一）；(3) 学术定位：MEWMA 是 MSPC 的成熟方法（Lowry 1992 被引逾 4000 次），本文的增量不在方法本身，而在(a)首次将其用于 GNSS SQM 欺骗检测、(b)定量给出"方向/协方差信息在欺骗检测中的收益-代价结构"（虚警 −43%、相关方向漂移检出 −0.07）；(4) 计算成本与库版同量级（毫秒级），部署无碍。
