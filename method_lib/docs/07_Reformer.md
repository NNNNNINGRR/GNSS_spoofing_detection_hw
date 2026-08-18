# 第 7 篇：Reformer —— 用哈希加速注意力

对应源码：`method_lib/model/Reformer.py`；论文：*Reformer: The Efficient Transformer*（ICLR 2020）。

---

## 第一部分：方法思想

### 1. 要解决的问题

标准 Transformer 的注意力矩阵是 $T\times T$，计算和显存都是 $O(T^2)$。序列一长（比如几千步），普通显卡就装不下了。Reformer 要解决的核心问题是：

> **能不能只让“相似”的位置互相注意，而不必计算所有两两关系？**

### 2. 局部敏感哈希（LSH）注意力

思路来自局部敏感哈希（Locality-Sensitive Hashing）：把高维向量映射到少量“桶”（bucket）里，**相似的向量大概率进同一个桶**。

对注意力来说，查询 $q_i$ 和键 $k_j$ 的相似度由点积决定。Reformer 对 $Q$ 和 $K$ 做随机旋转后再分段哈希：

$$
h(q)=\arg\max_b \left(\text{round}\left(\frac{Rq}{\|Rq\|}\right)\right),\qquad R \text{ 是随机旋转矩阵}
$$

也就是把向量投影到单位球面上，按角度分桶。于是注意力只需在**同桶内**计算：

$$
\text{Attn}_{\text{LSH}}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,\quad \text{仅对同一桶内的 }(i,j)
$$

复杂度从 $O(T^2)$ 降到 $O(T\log T)$。

### 3. 多次哈希提高准确性

哈希有随机性，真正的近邻偶尔会被分到不同桶。Reformer 用 $n_{\text{hashes}}$ 次独立哈希，每次各自做注意力，最后合并结果：

$$
\text{Out}=\text{Concat}_{r=1}^{n_{\text{hashes}}}\left(\text{Attn}^{(r)}_{\text{LSH}}(Q,K,V)\right)
$$

### 4. 其他改进（本库使用部分）

- 可逆残差层（reversible layers）：节省显存（本库通过 `reformer_pytorch` 库提供）；
- 分块前馈：避免大 FFN 占用显存；
- 位置编码：Reformer 用可学习位置编码。

### 5. 本库中的结构

本库的 Reformer 是“编码器 + 线性输出”结构：

$$
\hat Y = W_{\text{out}}\,\text{Encoder}_{\text{LSH}}(X_{\text{embed}})
$$

没有复杂解码器，预测时把输入和“占位解码输入”拼在一起过编码器，取最后 $H$ 步。

---

## 第二部分：源码讲解

### 1. 初始化

```python
class Model(nn.Module):
    def __init__(self, configs, bucket_size=4, n_hashes=4):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, ...)
        self.encoder = Encoder([
            EncoderLayer(
                ReformerLayer(None, configs.d_model, configs.n_heads,
                              bucket_size=bucket_size, n_hashes=n_hashes),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
            for l in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))

        if self.task_name == 'classification':
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
        else:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
```

`ReformerLayer` 来自 `reformer_pytorch` 库（`layers/SelfAttention_Family.py` 中封装），`bucket_size` 是桶大小、`n_hashes` 是哈希次数。

### 2. 长序列预测

```python
    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)   # 拼接占位解码输入
        if x_mark_enc is not None:
            x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)   # [B, T+pred_len, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)                # [B, T+pred_len, c_out]
        return dec_out
```

数据流：

$$
[B,T,M]\xrightarrow{\text{拼接占位}}[B,T+H,M]\xrightarrow{\text{嵌入}}[B,T+H,d]\xrightarrow{\text{LSH编码器}}[B,T+H,d]\xrightarrow{\text{线性}}[B,T+H,M]
$$

`forward` 最后取 `dec_out[:, -self.pred_len:, :]`，即预测部分。

### 3. 异常检测与分类

```python
    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out)
        return self.projection(enc_out)                  # 重建输入

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)       # 清零 padding
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)
```

### 4. 关于 `ReformerLayer`

`layers/SelfAttention_Family.py` 中：

```python
class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None,
                 causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.attention = LSHSelfAttention(
            dim=d_model, heads=n_heads, bucket_size=bucket_size, n_hashes=n_hashes)
```

真正的哈希注意力由 `reformer_pytorch.LSHSelfAttention` 实现：把输入分桶、桶内注意力、多次哈希合并——对应第 2、3 节公式。

## 小结

Reformer 用“局部敏感哈希分桶”把注意力限制在相似位置上，复杂度从 $O(T^2)$ 降到 $O(T\log T)$。它保留了 Transformer 的表达能力，同时大幅降低长序列的计算与显存开销。
