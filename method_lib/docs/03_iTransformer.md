# 第 3 篇：iTransformer —— 倒置的 Transformer

对应源码：`method_lib/model/iTransformer.py`；论文：*iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*（ICLR 2024）。

---

## 第一部分：方法思想

### 1. 要解决的问题

标准 Transformer 把“同一时刻的所有变量”作为一个令牌，注意力在时间位置之间做。这带来两个问题：

1. **时间令牌的语义混乱**：不同时刻的向量代表“该时刻所有变量的快照”，变量顺序被打乱，模型很难学到“哪个变量和哪个变量相关”；
2. **注意力被时间维支配**：序列很长时，注意力矩阵 $T\times T$ 巨大且大部分是噪声。

iTransformer 提出一个“倒过来”的思路：

> **把注意力从“时间之间”搬到“变量之间”**——每个变量自己是一条完整的序列，把它当作一个令牌；时间维则用线性层直接压缩。

### 2. 核心思想

#### 2.1 变量令牌化

输入 $X\in\mathbb{R}^{T\times M}$。iTransformer 先转置：

$$
X^\top \in \mathbb{R}^{M\times T}
$$

然后用一个线性层（`DataEmbedding_inverted`）把每个变量的整条时间序列 $T$ 维压缩成 $d_{\text{model}}$ 维：

$$
h_m = W_{\text{emb}}\, X_{:,m} \in \mathbb{R}^{d_{\text{model}}},\quad m=1,\dots,M
$$

这样得到了 $M$ 个令牌 $\{h_1,\dots,h_M\}$，每个令牌浓缩了一个变量过去 $T$ 步的全部信息。

#### 2.2 变量间注意力

在这 $M$ 个令牌上做标准多头自注意力：

$$
Q=HW_Q,\ K=HW_K,\ V=HW_V
$$

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

注意力矩阵是 $M\times M$，$M$ 通常远小于 $T$（例如 7 个气象变量 vs 96 个时间步），所以计算量大幅下降，而且每个注意力分数有清晰的物理含义：“变量 $m_1$ 与变量 $m_2$ 的关系有多强”。

#### 2.3 时间维用线性层

注意力输出形状是 $[B, M, d_{\text{model}}]$。要做预测，需要把每个变量的表示“展开”回时间轴。iTransformer 用一个线性层：

$$
\hat Y = W_{\text{out}}\, Z^\top,\qquad Z\in\mathbb{R}^{M\times d_{\text{model}}},\ W_{\text{out}}\in\mathbb{R}^{H\times d_{\text{model}}}
$$

即对每个变量，从 $d_{\text{model}}$ 维表示直接映射到 $H$ 步预测。

### 3. 归一化的重要性

iTransformer 沿用了 Non-stationary Transformer 的实例归一化：

$$
\tilde X = \frac{X-\mu}{\sigma+\epsilon}
$$

预测后反归一化：

$$
\hat Y = \hat Y_{\text{norm}}\cdot\sigma+\mu
$$

这能显著提升对非平稳序列（均值和方差随时间漂移）的预测效果。

### 4. 与 Bi-FI 的 intra 分支的关系

Bi-FI 的 intra 分支用的正是 iTransformer 的“变量令牌”思路；区别是 iTransformer 只有这一个分支，而 Bi-FI 还叠加了频域分支。所以读懂了本篇，就懂了一半 Bi-FI。

---

## 第二部分：源码讲解

### 1. 初始化

```python
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, ...)
```

`DataEmbedding_inverted(c_in=seq_len, d_model=d_model)`：输入 `[B, T, M]`，内部转置成 `[B, M, T]`，线性层把 $T$ 压成 $d_{\text{model}}$，输出 `[B, M, d_model]`。

```python
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, ...),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, ...)
            for l in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))
```

标准编码器，但作用对象是 $M$ 个变量令牌。

```python
        if self.task_name == 'long_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
```

输出头与 Bi-FI 一致：预测输出 $H$ 步、异常输出 $T$ 步重建、分类输出类别数。

### 2. 预测前向

```python
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)      # [B, M, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
```

数据流：

$$
[B,T,M]\xrightarrow{\text{归一化}}[B,T,M]\xrightarrow{\text{倒置嵌入}}[B,M,d]\xrightarrow{\text{注意力}}[B,M,d]\xrightarrow{\text{投影+转置}}[B,H,M]\xrightarrow{\text{反归一化}}[B,H,M]
$$

`permute(0,2,1)` 把 `[B, M, H]` 变成 `[B, H, M]`；`[:, :, :N]` 截取前 $N$ 个变量（保证输出变量数与输入一致）。

### 3. 异常检测与分类

```python
    def anomaly_detection(self, x_enc):
        # 归一化 -> 嵌入 -> 编码器 -> 投影(seq_len) -> 反归一化
        ...

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)          # [B, M*d_model]
        output = self.projection(output)                      # [B, num_class]
        return output
```

分类把 $M\times d_{\text{model}}$ 全部压平再线性映射到类别数。

## 小结

iTransformer 的核心是“把时间令牌换成变量令牌”：注意力学变量间关系，线性层学时间变化。它计算量小、可解释性强（注意力权重=变量相关性），是理解 Bi-FI intra 分支和现代时间序列 Transformer 的关键模型。
