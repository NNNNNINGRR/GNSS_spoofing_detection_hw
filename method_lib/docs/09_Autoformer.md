# 第 9 篇：Autoformer —— 用自相关替代注意力

对应源码：`method_lib/model/Autoformer.py`；论文：*Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*（NeurIPS 2021）。

---

## 第一部分：方法思想

### 1. 要解决的问题

时间序列预测中最常见的两种模式：

- **趋势**：缓慢变化（用移动平均就能提取）；
- **季节/周期**：重复出现的模式（例如 24 小时、7 天）。

标准注意力是按“点对点相似度”聚合信息，但周期性信息是“**整段模式按时间滞后重复**”的。Autoformer 提出两个关键改动：

1. 把趋势/季节分解**内嵌到模型每一层**（而不是只在入口分解一次）；
2. 用**自相关机制（Auto-Correlation）**替代注意力：按周期滞后聚合信息。

### 2. 序列分解（Decomposition）

每一层都做：

$$
X = X_{\text{seasonal}} + X_{\text{trend}}
$$

趋势用移动平均提取，季节是残差。编码器/解码器只对季节部分做深度建模，趋势部分直接传递并逐步累积。

### 3. 自相关机制

#### 3.1 什么是自相关

自相关衡量序列与它自身“滞后 $\tau$ 步”的相似度：

$$
R_{XX}(\tau)=\lim_{L\to\infty}\frac1L\sum_{t=1}^{L}X_t X_{t-\tau}
$$

对离散序列，可以用 FFT 高效计算（自相关 = 序列与其翻转的卷积）：

$$
R_{XX} = \text{iFFT}\left(\text{FFT}(X)\cdot\overline{\text{FFT}(X)}\right)
$$

自相关峰值对应的 $\tau$ 就是序列的**主周期**。

#### 3.2 周期依赖发现（Period-based Dependencies）

对自相关序列取前 $k=\lfloor c\log L\rfloor$ 个峰值，得到候选周期 $\{\tau_1,\dots,\tau_k\}$：

$$
\tau_1,\dots,\tau_k=\arg\operatorname{Topk}_{\tau}\left(R_{XX}(\tau)\right)
$$

#### 3.3 时间延迟聚合（Time Delay Aggregation）

对每个候选周期，把值序列**滚动** $\tau_i$ 步，按自相关权重加权求和：

$$
\text{AutoCorrelation}(X)=\sum_{i=1}^{k}\text{softmax}\left(R_{XX}(\tau_i)\right)\cdot \text{Roll}(X,\tau_i)
$$

直觉：如果序列以 24 小时为周期，那么把序列整体滚动 24 步，和它自己几乎重合；自相关机制找到这些周期，把“一个周期前”的信息搬过来帮助预测。

复杂度同样是 $O(L\log L)$（因为用了 FFT），但语义比注意力更贴合周期数据。

### 4. 整体结构

- 编码器：多层（自相关 + 分解 + 前馈），每层输出季节表示；
- 解码器：输入 = 最近 `label_len` 步季节部分 + 补零；趋势部分用均值外推；经过自相关自注意力、交叉注意力、逐层分解；
- 输出 = 趋势累积 + 季节重建。

$$
\hat Y = \text{Trend}_{\text{dec}} + \text{Seasonal}_{\text{dec}}
$$

---

## 第二部分：源码讲解

### 1. 初始化

```python
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)          # 分解模块
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, ...)
        self.encoder = Encoder([
            EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor, ...),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                moving_avg=configs.moving_avg, ...)
            for l in range(configs.e_layers)
        ], norm_layer=my_Layernorm(configs.d_model))
```

注意：Autoformer 用 `DataEmbedding_wo_pos`（**没有位置编码**）——因为自相关机制本身基于周期，位置编码反而多余。

解码器：

```python
        self.decoder = Decoder([
            DecoderLayer(
                AutoCorrelationLayer(AutoCorrelation(True, ...), ...),   # 掩码自相关
                AutoCorrelationLayer(AutoCorrelation(False, ...), ...),  # 交叉自相关
                configs.d_model, configs.c_out, configs.d_ff,
                moving_avg=configs.moving_avg, ...)
            for l in range(configs.d_layers)
        ], norm_layer=my_Layernorm(configs.d_model),
           projection=nn.Linear(configs.d_model, configs.c_out, bias=True))
```

### 2. 自相关实现（`layers/AutoCorrelation.py`）

```python
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        ...
        queries = queries.view(B * H, L, -1)
        keys = keys.view(B * H, L, -1)
        values = values.view(B * H, L, -1)

        # 周期依赖发现：用 FFT 计算自相关
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        ...
        # 找 Top-k 周期并做时间延迟聚合
        V = self.time_delay_agg_training(values, corr)
        return V.contiguous(), None
```

对应公式：

$$
R_{XX}=\text{iFFT}\left(\text{FFT}(Q)\cdot\overline{\text{FFT}(K)}\right)
$$

`time_delay_agg_training`：

```python
    def time_delay_agg_training(self, values, corr):
        top_k = int(self.factor * math.log(length))          # k = c ln L
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)            # 权重
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)   # 滚动 τ_i 步
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].unsqueeze(...))
        return delays_agg
```

对应公式：

$$
\sum_{i=1}^{k}\text{softmax}(R(\tau_i))\cdot\text{Roll}(V,\tau_i)
$$

### 3. 预测前向

```python
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)       # 分解
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)    # 趋势外推
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)  # 季节补零

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        return trend_part + seasonal_part
```

与 FEDformer 相似：趋势用均值外推，季节补零进解码器，最后两者相加。

### 4. 异常与分类

```python
    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)
```

## 小结

Autoformer 把“注意力”替换成“自相关”：用 FFT 找周期、按滞后滚动聚合，复杂度 $O(L\log L)$ 且语义贴合周期数据；同时把趋势/季节分解内嵌到每一层。它与 FEDformer 同属“分解 + 频域/周期”路线，是理解时间序列专用 Transformer 的重要一环。
