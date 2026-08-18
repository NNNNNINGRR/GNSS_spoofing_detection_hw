# 第 2 篇：DLinear —— 分解 + 线性回归

对应源码：`method_lib/model/DLinear.py`；论文：*Are Transformers Effective for Time Series Forecasting?*（AAAI 2023）。

---

## 第一部分：方法思想

### 1. 要解决的问题

2023 年前后的主流观点是“Transformer 很强，所以时间序列预测也应该用 Transformer”。但 DLinear 的作者提出了一个尖锐的质疑：**时间序列预测真的需要那么复杂的模型吗？**

他们发现，很多时间序列有明显的**趋势（trend）**和**季节（seasonal）**成分：

- 趋势：缓慢上升或下降的整体走向（例如负荷逐年增长）；
- 季节：周期性重复的模式（例如每天、每周、每年的周期）。

而 Transformer 之类的模型会把趋势和季节“搅在一起”学，反而学不好。DLinear 的想法非常朴素：

> **先把趋势和季节分开，然后对每一部分直接做线性回归（一个线性层），加起来就是预测。**

### 2. 核心公式

设输入 $X\in\mathbb{R}^{T\times M}$。第一步，用移动平均把序列分解（第 0 篇 5 节）：

$$
X = S + T_r
$$

其中 $S$ 是季节部分（残差），$T_r$ 是趋势部分（移动平均）。第二步，对每一部分、沿时间维做线性映射：

$$
\hat S = W_S\, S,\qquad \hat T_r = W_{T_r}\, T_r
$$

$W_S,W_{T_r}\in\mathbb{R}^{H\times T}$ 是共享（或每个变量独立）的权重矩阵。最后：

$$
\hat X = \hat S + \hat T_r \in \mathbb{R}^{H\times M}
$$

这里 $H$ 是预测长度。也就是说，整个模型只有两个线性层（甚至没有激活函数），却能在多个公开基准上打败复杂 Transformer。

### 3. 为什么“简单”反而有效

1. **归纳偏置正确**：趋势近似线性（短窗口内），季节有周期性——线性层天然适合这两种模式；把两者分开后，模型不用自己“猜”哪个部分是趋势；
2. **没有过拟合**：参数少，训练快，不容易把噪声学进去；
3. **对分布漂移不敏感**：与复杂模型相比，线性模型更稳定。

局限性也很明显：如果序列有很强的非线性或复杂的变量间交互，DLinear 表达能力不足。这也是 Bi-FI 等模型存在的意义。

### 4. 初始化的小技巧

源码把线性层的权重初始化为全 $1/T$：

$$
W_{ij}=\frac{1}{T}
$$

这意味着初始状态每个输出位置都是输入的“平均值”，相当于从一个非常保守的起点开始学习，有助于稳定收敛。

### 5. 三任务

- 预测/异常：直接输出线性回归结果；
- 分类：把输出压平后接一个线性层到类别数。

---

## 第二部分：源码讲解

### 1. 初始化

```python
class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection':
            self.pred_len = configs.seq_len      # 异常/分类输出与输入等长
        else:
            self.pred_len = configs.pred_len
        self.decompsition = series_decomp(configs.moving_avg)   # 移动平均分解
        self.individual = individual
        self.channels = configs.enc_in
```

`series_decomp(configs.moving_avg)` 就是第 0 篇的分解模块；`individual=False` 表示所有变量共享同一组线性层。

```python
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
```

两种模式：

- `individual=True`：每个变量一个独立的线性层（$M$ 组 $W_S,W_{T_r}$）；
- `individual=False`（默认）：所有变量共享同一组权重。

```python
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter(
            (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
```

对应 4 节的初始化：权重矩阵每个元素都是 $1/T$，即初始输出为输入均值。

### 2. 编码器（核心）

```python
    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)      # 分解
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        # 形状: [B, T, M] -> [B, M, T]
```

分解后把维度换成 `[B, M, T]`：时间维放在最后，方便对每个变量的时间序列做线性映射。

```python
        if self.individual:
            seasonal_output = torch.zeros([...])
            trend_output = torch.zeros([...])
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)
```

对应公式：

$$
\hat S = W_S S,\qquad \hat T_r = W_{T_r} T_r,\qquad \hat X = \hat S + \hat T_r
$$

最后 `permute` 把形状变回 `[B, H, M]`（预测长度在中间）。

### 3. 三个任务的入口

```python
    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)     # 压平
        output = self.projection(output)                   # -> num_class
        return output
```

预测和异常检测完全复用 `encoder`；分类多一个投影层：

```python
        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)
```

### 4. 统一入口

```python
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc)
```

注意：DLinear 完全忽略时间戳标记（`x_mark_enc`、`x_dec`），因为它不需要位置编码或时间编码——线性层本身就隐含了“每个时间位置对应一个权重”的信息。

## 小结

DLinear 用“趋势/季节分解 + 两个线性层”就完成了预测，是全库最简单的模型。理解它有助于建立“模型复杂度 ≠ 效果好”的判断，也是理解 Autoformer、FEDformer（都用了同一套分解）的基础。
