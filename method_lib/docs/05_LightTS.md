# 第 5 篇：LightTS —— 极简双采样 MLP

对应源码：`method_lib/model/LightTS.py`；论文：*Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures*（2022）。

---

## 第一部分：方法思想

### 1. 要解决的问题

Transformer 虽然强大，但计算复杂、参数多。LightTS 的作者问：**能不能不用注意力，只用简单的 MLP（多层感知机）就达到接近的效果？**

他们观察到两个关键点：

1. 时间序列的相邻点高度冗余（每秒的数据和下一秒几乎一样），不需要逐个点建模；
2. 模式存在于不同尺度：有的规律在“连续的小段”里，有的规律在“每隔一段”里。

于是 LightTS 提出：

> **用两种互补的采样方式压缩序列，然后交给极简的 MLP 处理。**

### 2. 连续采样与间隔采样

把输入 $X\in\mathbb{R}^{T\times M}$ 重新组织成 $C$ 块、每块 $S$ 点（$T=C\times S$）：

- **连续采样（continuous sampling）**：保持原始顺序，按块切分：块 1 是 $X_{1:S}$，块 2 是 $X_{S+1:2S}$……
- **间隔采样（interval sampling）**：每隔 $C$ 个点取一个：第 1 块是 $X_1,X_{1+C},X_{1+2C},\dots$，即把“每块的第 $i$ 个点”聚在一起。

两种视角互补：

- 连续采样保留局部连贯性（波形形状）；
- 间隔采样把“同一相位”的点放在一起（例如每天同一时刻），更容易发现周期。

### 3. IEBlock：交互嵌入块

LightTS 的核心模块是 IEBlock，包含三个投影：

1. **空间投影**：对每个块内的 $S$ 个点做 MLP（压缩块内信息）；
2. **通道投影**：在块与块之间做线性混合（学块间关系），权重初始化为单位矩阵，表示“先不做混合，由训练决定”；
3. **输出投影**：把压缩后的特征映射到目标长度。

公式上，设 $x\in\mathbb{R}^{S\times C}$（$S$ 个点、$C$ 个块）：

$$
h = \text{LeakyReLU}(xW_1)W_2,\qquad h \leftarrow h + W_{\text{ch}}\,h,\qquad y = hW_3
$$

### 4. 高速公路连接（Highway）

LightTS 还保留一条“高速公路”：直接用线性层从原始输入回归预测：

$$
\hat Y_{\text{highway}} = W_{\text{ar}}\, X
$$

最终输出 = 双采样 MLP 的输出 + 高速公路输出：

$$
\hat Y = \hat Y_{\text{MLP}} + \hat Y_{\text{highway}}
$$

高速公路保证模型至少能学到线性趋势，MLP 负责补充非线性模式。

### 5. 整体流程

1. 输入 $[B,T,M]$；
2. 分别做连续采样和间隔采样，各过一层 IEBlock；
3. 把两路输出拼接，再过第三层 IEBlock；
4. 加上高速公路线性预测。

整个过程没有注意力，全是线性层 + LeakyReLU，因此计算极快，特别适合对速度敏感的场景。

---

## 第二部分：源码讲解

### 1. IEBlock

```python
class IEBlock(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_node):
        super().__init__()
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 4)
        )
        self.channel_proj = nn.Linear(self.num_node, self.num_node)
        torch.nn.init.eye_(self.channel_proj.weight)   # 单位矩阵初始化
        self.output_proj = nn.Linear(self.hid_dim // 4, self.output_dim)
```

对应 3 节公式：

- `spatial_proj`：$x\mapsto \text{LeakyReLU}(xW_1)W_2$；
- `channel_proj`：块间线性混合，单位初始化；
- `output_proj`：输出投影。

```python
    def forward(self, x):
        x = self.spatial_proj(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1) + self.channel_proj(x.permute(0, 2, 1))
        x = self.output_proj(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)
```

各种 `permute` 只是为了对齐维度：输入 `[B, S, C]`（块内点、块数），空间投影在点维做，通道投影在块维做。

### 2. 模型初始化

```python
class Model(nn.Module):
    def __init__(self, configs, chunk_size=24):
        super().__init__()
        self.seq_len = configs.seq_len
        if self.task_name == 'long_term_forecast':
            self.pred_len = configs.pred_len
        else:
            self.pred_len = configs.seq_len
        self.chunk_size = min(configs.pred_len, configs.seq_len, chunk_size)
        if self.seq_len % self.chunk_size != 0:
            self.seq_len += (self.chunk_size - self.seq_len % self.chunk_size)  # 补齐
        self.num_chunks = self.seq_len // self.chunk_size
```

`chunk_size` 是块大小 $S$，默认 24；如果 $T$ 不能被 $S$ 整除，就把 $T$ 补到能整除。$C=T/S$ 是块数。

```python
        self.layer_1 = IEBlock(input_dim=self.chunk_size, hid_dim=self.d_model // 4,
                               output_dim=self.d_model // 4, num_node=self.num_chunks)
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)
        self.layer_2 = IEBlock(input_dim=self.chunk_size, hid_dim=self.d_model // 4,
                               output_dim=self.d_model // 4, num_node=self.num_chunks)
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)
        self.layer_3 = IEBlock(input_dim=self.d_model // 2, hid_dim=self.d_model // 2,
                               output_dim=self.pred_len, num_node=self.enc_in)
        self.ar = nn.Linear(self.seq_len, self.pred_len)     # 高速公路
```

### 3. 编码器（核心）

```python
    def encoder(self, x):
        B, T, N = x.size()
        x = torch.cat([x, torch.zeros((B, self.seq_len - T, N)).to(x.device)], dim=1)  # 补齐

        highway = self.ar(x.permute(0, 2, 1))        # [B, N, pred_len]
        highway = highway.permute(0, 2, 1)

        # 连续采样
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)                  # [B, N, S, C]
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)  # [B*N, S, C]
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)   # 压缩块维 -> [B*N, S]

        # 间隔采样
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)

        x3 = torch.cat([x1, x2], dim=-1)             # [B*N, S*2? 实际是 d_model//2]
        x3 = x3.reshape(B, N, -1).permute(0, 2, 1)
        out = self.layer_3(x3)
        out = out + highway
        return out
```

关键点：

- `reshape(B, num_chunks, chunk_size, N)` 后取 `permute(0,3,2,1)` 得到 `[B, N, S, C]`：连续采样；
- `reshape(B, chunk_size, num_chunks, N)` 后取 `permute(0,3,1,2)` 得到 `[B, N, S, C]`：间隔采样（把同一相位的点聚到一块）；
- 两路都过 IEBlock 后用 `chunk_proj` 把块维压缩成 1；
- 拼接两路（连续+间隔）再过 `layer_3`；
- 最后 `out + highway` 对应 4 节的加法公式。

### 4. 三任务入口

```python
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output
```

## 小结

LightTS 证明了：不需要注意力，只要用“连续+间隔”双采样把序列压缩、再配合简单 MLP 和线性高速公路，就能又快又好地预测。它的双路互补思想与 Bi-FI 的“双分支互补”有异曲同工之处。
