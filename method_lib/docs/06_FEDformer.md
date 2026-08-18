# 第 6 篇：FEDformer —— 在频域做注意力

对应源码：`method_lib/model/FEDformer.py`；论文：*FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting*（ICML 2022）。

---

## 第一部分：方法思想

### 1. 要解决的问题

时间序列往往由少数几个“主频率”支配：比如电力负荷有 24 小时周期，气温有一年周期。标准 Transformer 的注意力在时域逐点计算，$T\times T$ 的矩阵里大部分是噪声，而且难以直接表达“周期性”。

FEDformer 的思路：

> **先做趋势/季节分解，再在频域（频率分量）上做注意力。** 频域注意力只关注少数最重要的频率，计算量小且天然适配周期性。

### 2. 趋势/季节分解

沿用第 0 篇的移动平均分解：

$$
X = X_{\text{seasonal}} + X_{\text{trend}}
$$

趋势部分几乎不需要复杂建模（用均值外推即可），主要建模对象是季节部分。

### 3. 频域增强（Frequency Enhanced Block）

对季节部分做快速傅里叶变换：

$$
\hat X(k)=\sum_{t=0}^{T-1}X_{\text{seasonal}}(t)\,e^{-2\pi j k t/T}
$$

然后只保留 **modes 个最重要的频率分量**（可选“随机选”或“选最低频”），对它们做可学习的线性变换（复数乘法），再逆傅里叶变换回去：

$$
Y = \text{iFFT}\left(W_{\text{freq}}\odot \hat X_{\text{selected}}\right)
$$

其中 $\odot$ 表示逐元素复数乘法。这个过程在代码里是 `FourierBlock`：

1. `torch.fft.rfft` 得到频域；
2. 随机/低频选择 `modes` 个频率索引；
3. 可学习权重（复数）逐元素相乘；
4. `torch.fft.irfft` 回时域。

### 4. 频域注意力（Frequency Enhanced Attention）

注意力同样搬到频域：查询 $Q$、键 $K$、值 $V$ 都先做 FFT，只保留 modes 个频率，在频域做交互，再 iFFT 回时域：

$$
\hat Q=\text{FFT}(Q),\ \hat K=\text{FFT}(K),\ \hat V=\text{FFT}(V)
$$

$$
\text{Attn}_{\text{freq}} = \text{iFFT}\left(\text{softmax}\left(\frac{\hat Q\hat K^\top}{\sqrt{d}}\right)\hat V\right)
$$

由于只保留 modes 个频率，复杂度降为 $O(N)$（$N$ 为序列长度），而不是 $O(T^2)$。

### 5. 编码器-解码器

- 编码器：多层“频域自注意力 + 分解 + 前馈”；
- 解码器：输入是“季节部分（取最后 label_len 步 + 补零）+ 趋势外推”，经过频域自注意力 + 频域交叉注意力，输出季节与趋势之和。

$$
\hat Y = \text{Trend}_{\text{dec}} + \text{Seasonal}_{\text{dec}}
$$

---

## 第二部分：源码讲解

### 1. 初始化：选择频域模块

```python
class Model(nn.Module):
    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        super().__init__()
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(...)
        self.dec_embedding = DataEmbedding(...)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=1, base='legendre')
            ...
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model,
                n_heads=configs.n_heads, seq_len=self.seq_len,
                modes=self.modes, mode_select_method=self.mode_select)
            decoder_self_att = FourierBlock(..., seq_len=self.seq_len // 2 + self.pred_len, ...)
            decoder_cross_att = FourierCrossAttention(...)
```

默认使用傅里叶版本（`version='fourier'`）；也可选小波版本。

### 2. FourierBlock：频域线性变换

```python
def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index
```

选择要保留的频率索引：随机抽 `modes` 个或取最低频 `modes` 个。

```python
class FourierBlock(nn.Module):
    def __init__(self, ...):
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.weights1 = nn.Parameter(self.scale * torch.rand(n_heads, in_channels // n_heads,
                                                             out_channels // n_heads, len(self.index)))
        self.weights2 = nn.Parameter(...)

    def forward(self, q, k, v, mask):
        x = q.permute(0, 2, 3, 1)
        x_ft = torch.fft.rfft(x, dim=-1)          # 频域
        ...
        out_ft = torch.zeros_like(x_ft)
        out_ft[..., self.index] = self.compl_mul1d("bcd,de->bce", x_ft[..., self.index], self.weights1)
        ...
        x = torch.fft.irfft(out_ft, n=x.size(-1)) # 回时域
        return x
```

对应公式：

$$
Y=\text{iFFT}\left(W\odot \hat X_{\text{selected}}\right)
$$

只对被选中的 `self.index` 频率做复数乘法，其余频率置零。

### 3. 预测前向

```python
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)          # 分解
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)  # 趋势外推
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))  # 季节补零

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        return trend_part + seasonal_part
```

对应 5 节公式：解码器同时接收季节部分（补零）和趋势部分（均值外推），输出两者之和。

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
        output = output * x_mark_enc.unsqueeze(-1)   # 用 padding mask 清零
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)
```

异常/分类只使用编码器；分类时用 `x_mark_enc`（padding 掩码）把填充位置清零。

## 小结

FEDformer 的核心是把注意力搬到频域：分解出季节，FFT 后只保留关键频率做可学习变换与注意力，再 iFFT 回来。它证明了频域信息对时间序列建模的价值——这也是 Bi-FI inter 分支（FFT + 卷积）的思想源头之一。
