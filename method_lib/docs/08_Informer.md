# 第 8 篇：Informer —— 稀疏注意力 + 生成式解码

对应源码：`method_lib/model/Informer.py`；论文：*Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*（AAAI 2021）。

---

## 第一部分：方法思想

### 1. 要解决的问题

长序列预测要同时解决三个问题：

1. **注意力太贵**：$O(T^2)$ 的复杂度让长序列不可行；
2. **解码太慢**：传统 Transformer 逐点生成预测，$H$ 步要解码 $H$ 次；
3. **长距离依赖**：序列越长，模型越难捕捉远处信息。

Informer 的三大贡献：

- **ProbSparse 注意力**：只让少数“重要”查询参与完整注意力；
- **自注意力蒸馏**：逐层减半序列长度，压缩冗余；
- **生成式解码器**：一次前向输出全部 $H$ 步预测。

### 2. ProbSparse 注意力

并非所有查询都值得做完整注意力。Informer 用一个“稀疏性度量”选出重要查询。对第 $i$ 个查询，度量其注意力分布与均匀分布的差异（KL 散度近似）：

$$
M(q_i,K)=\max_j\frac{q_i k_j^\top}{\sqrt{d_k}}-\frac1{L_K}\sum_{j=1}^{L_K}\frac{q_i k_j^\top}{\sqrt{d_k}}
$$

直观理解：如果一个查询对所有键的注意力分数都差不多（均匀），说明它“谁都不特别关注”，信息量低；$M$ 越大，说明该查询越“专一”，越重要。

Informer 只对 $M$ 值最大的 $u=c\ln L_Q$ 个查询计算完整注意力，其余查询直接用键的均值作为上下文：

$$
\text{Attn}(Q,K,V)=\begin{cases}
\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V, & q\in Q_{\text{top}}\\
\bar V, & \text{其他}
\end{cases}
$$

复杂度降到 $O(L\ln L)$。

### 3. 自注意力蒸馏

每经过一层编码器，用一维卷积 + 最大池化把序列长度减半：

$$
X_{l+1}=\text{MaxPool}\left(\text{Conv1d}(\text{ReLU}(X_l))\right)
$$

这样高层只保留最重要的信息，序列长度逐层减半，进一步降低计算量。

### 4. 生成式解码器

解码器输入由两部分拼接：

- 已知的最近 `label_len` 步真实值（作为“起始令牌”）；
- 未来 `pred_len` 步占位（全零）。

$$
X_{\text{dec}}=[X_{T-\text{label\_len}+1:T};\ \underbrace{0,\dots,0}_{\text{pred\_len}}]
$$

解码器一次前向输出全部预测，无需逐点自回归，大幅提速。

### 5. 整体结构

$$
\hat Y=\text{Decoder}\left(\text{DecoderEmbedding}(X_{\text{dec}}),\ \text{Encoder}(X_{\text{enc}})\right)
$$

解码器内有掩码自注意力（防止看到未来）和交叉注意力（向编码器取信息）。

---

## 第二部分：源码讲解

### 1. 初始化

```python
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, ...)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, ...)

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(ProbAttention(False, configs.factor, ...),
                               configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, ...)
            for l in range(configs.e_layers)
        ], [
            ConvLayer(configs.d_model) for l in range(configs.e_layers - 1)
        ] if configs.distil and ('forecast' in configs.task_name) else None,
          norm_layer=torch.nn.LayerNorm(configs.d_model))
```

编码器用 `ProbAttention`，层间插入 `ConvLayer`（蒸馏）。

```python
        self.decoder = Decoder([
            DecoderLayer(
                AttentionLayer(ProbAttention(True, ...), ...),   # 掩码自注意力
                AttentionLayer(ProbAttention(False, ...), ...),  # 交叉注意力
                configs.d_model, configs.d_ff, ...)
            for l in range(configs.d_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model),
           projection=nn.Linear(configs.d_model, configs.c_out, bias=True))
```

### 2. ProbSparse 度量（`layers/SelfAttention_Family.py`）

```python
    def _prob_QK(self, Q, K, sample_k, n_top):
        # 随机采样部分键，估计每个查询的稀疏性
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)   # 稀疏性度量
        M_top = M.topk(n_top, sorted=False)[1]                            # 选 Top-u 查询
        Q_reduce = Q[...M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top
```

对应公式：

$$
M(q_i,K)=\max_j\frac{q_i k_j^\top}{\sqrt{d_k}}-\frac1{L_K}\sum_j\frac{q_i k_j^\top}{\sqrt{d_k}}
$$

只对采样到的键估计 $M$，选出最大的 $u=c\ln L_Q$ 个查询，对这些查询计算完整 $QK^\top$。

```python
    def _get_initial_context(self, V, L_Q):
        # 未选中的查询用 V 的均值做上下文
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
```

### 3. 预测前向

```python
    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)     # 起始令牌 + 零占位
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out
```

`x_dec` 由训练框架构造：前 `label_len` 步是真实值，后面 `pred_len` 步为零——对应生成式解码器的输入公式。

### 4. 蒸馏层 `ConvLayer`

`layers/Transformer_EncDec.py` 中：

```python
class ConvLayer(nn.Module):
    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))   # 一维卷积
        x = nn.functional.gelu(x)
        x = self.maxPool(x)                     # 最大池化，长度减半
        return x
```

### 5. 异常与分类

```python
    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)         # 重建

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

Informer 用 ProbSparse 度量挑选重要查询、用蒸馏减半序列、用生成式解码器一步输出全部预测，是长序列 Transformer 的高效代表。它的 ProbSparse 思想与 Reformer 的哈希分桶都是“让注意力变稀疏”的不同路线。
