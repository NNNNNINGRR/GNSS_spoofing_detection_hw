# 第 4 篇：PatchTST —— 时间序列的“分词”

对应源码：`method_lib/model/PatchTST.py`；论文：*A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*（ICLR 2023）。

---

## 第一部分：方法思想

### 1. 要解决的问题

自然语言处理（NLP）中，Transformer 处理的是“词”而不是“字”。把一句话拆成词，每个词有完整语义，注意力才能在词之间建立关系。时间序列中也有类似问题：

1. **单点没有语义**：单独一个时间点的数值（比如 0.372）几乎不携带信息，模式要由连续一段点共同体现；
2. **注意力太碎**：标准 Transformer 在 $T$ 个时间点上做注意力，$T\times T$ 的矩阵里大量是相邻点的“平庸关系”，计算浪费且难学到长周期模式。

PatchTST 的想法：

> **把时间序列切成小块（patch），每个小块当作一个“词”，再交给 Transformer。**

### 2. 什么是 Patch（分块）

设输入 $X\in\mathbb{R}^{T\times M}$。对**每个变量**单独处理：把它的 $T$ 个时间点按固定长度 $P$（patch 长度）和步长 $S$（stride）切成若干个小块：

$$
\text{patch 数量} = \left\lfloor\frac{T-P}{S}\right\rfloor + 2
$$

（代码里先在末尾补 $S$ 个点，所以是 $+2$。）例如 $T=96, P=16, S=8$，得到约 12 个 patch。

每个 patch 是一个 $P$ 维向量，代表一小段局部模式（如“一个上升沿”“一个波峰”）。

### 3. Patch 嵌入

把每个 patch 用线性层映射成 $d_{\text{model}}$ 维，并加上位置编码：

$$
u_i = W_{\text{patch}}\, p_i + \text{PE}(i),\qquad i=1,\dots,N_{\text{patch}}
$$

其中 $p_i\in\mathbb{R}^{P}$ 是第 $i$ 个 patch，$W_{\text{patch}}\in\mathbb{R}^{d_{\text{model}}\times P}$。

### 4. 通道独立的 Transformer

PatchTST 对 $M$ 个变量**分别独立**运行 Transformer（channel independence）：

- 把 $M$ 个变量合并成一批：形状 `[B*M, N_patch, d_model]`；
- 每个变量的 patch 序列自己过编码器：

$$
Z = \text{Encoder}(U),\qquad Z\in\mathbb{R}^{B\cdot M\times N_{\text{patch}}\times d_{\text{model}}}
$$

这样模型专注于单变量的时间模式，避免变量间噪声干扰；代价是放弃变量间关系（这正是 Bi-FI 想补回来的部分）。

### 5. 预测头

编码器输出每个 patch 的表示，把它们展平后用一个线性层映射到 $H$ 步预测：

$$
\hat Y_m = W_{\text{head}}\, \text{Flatten}(Z_m),\qquad \hat Y_m\in\mathbb{R}^{H}
$$

对所有变量拼起来得到 $[B,H,M]$。

### 6. 为什么有效

- patch 相当于“降采样 + 语义聚合”：输入长度从 $T$ 降到 $N_{\text{patch}}$，注意力矩阵从 $T^2$ 降到 $N_{\text{patch}}^2$，计算量大幅下降；
- 局部模式（趋势片段、尖峰）被完整保留在一个 patch 里，模型更容易识别；
- 通道独立让模型聚焦单变量规律，在预测任务上常常超过混合变量模型。

---

## 第二部分：源码讲解

### 1. Patch 嵌入（`layers/Embed.py` 的 `PatchEmbedding`）

```python
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]                     # 变量数 M
        x = self.padding_patch_layer(x)         # 末尾补 padding 个点
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # 滑窗切块
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
```

对应公式：

$$
u_i = W_{\text{patch}}\,p_i + \text{PE}(i)
$$

- `unfold` 在最后一维（时间）上滑窗，得到 `[B, M, N_patch, P]`；
- `reshape` 把变量合并进批维：`[B*M, N_patch, P]`；
- `value_embedding` 把每个 patch 从 $P$ 维映射到 $d_{\text{model}}$ 维；
- `position_embedding` 加上 patch 位置信息。

### 2. 模型初始化

```python
class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
```

默认 patch 长度 16、步长 8。

```python
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, ...),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, ...)
            for l in range(configs.e_layers)
        ], norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2)))
```

注意：归一化用的是 `BatchNorm1d`（配合转置），而不是 LayerNorm——这是 PatchTST 的一个实现细节，作用同样是稳定训练。

```python
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, ...)
        elif self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, ...)
        elif self.task_name == 'classification':
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)
```

`head_nf` 是展平后每个变量的特征数 = $d_{\text{model}}\times N_{\text{patch}}$。

### 3. `FlattenHead`

```python
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):     # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)   # -> [bs x nvars x d_model*patch_num]
        x = self.linear(x)    # -> [bs x nvars x target_window]
        return self.dropout(x)
```

对应公式：

$$
\hat Y_m = W_{\text{head}}\, \text{Flatten}(Z_m)
$$

### 4. 预测前向

```python
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1)              # [B, M, T]
        enc_out, n_vars = self.patch_embedding(x_enc)   # [B*M, N_patch, d_model]
        enc_out, attns = self.encoder(enc_out)          # 通道独立注意力
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)           # [B, M, d_model, N_patch]
        dec_out = self.head(enc_out)                    # [B, M, H]
        dec_out = dec_out.permute(0, 2, 1)              # [B, H, M]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
```

数据流：

$$
[B,T,M]\to[B,M,T]\to[B\cdot M,N_{\text{patch}},d]\to[B,M,N_{\text{patch}},d]\to[B,M,H]\to[B,H,M]
$$

前后是实例归一化与反归一化。

### 5. 分类与异常

```python
    def anomaly_detection(self, x_enc):
        # 与 forecast 相同，只是 head 输出 seq_len（重建）

    def classification(self, x_enc, x_mark_enc):
        # 归一化、patch、编码器后：
        output = self.flatten(enc_out)              # [B, M, d*N_patch]
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)            # [B, num_class]
```

## 小结

PatchTST 把“时间序列分词”思想带进 Transformer：每个变量切成 patch、独立建模，注意力在 patch 之间做。它用更少的计算量取得了很强的预测性能，是“变量内建模”路线的代表；Bi-FI 的 intra 分支思路与其互补。
