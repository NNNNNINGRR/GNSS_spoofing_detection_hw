# 第 1 篇：Bi-FI —— 双分支特征交互表示学习

对应源码：`method_lib/model/Bi_FI.py`；论文：*Bi-Branching Feature Interaction Representation Learning for Multivariate Time Series*（Applied Soft Computing 2024）。

---

## 第一部分：方法思想

### 1. 要解决的问题

多元时间序列里同时存在两类信息：

1. **变量之间的关系（inter-variable）**：同一时刻，不同传感器/变量的取值常常互相影响。例如气象站里温度升高，气压和湿度往往随之变化；
2. **变量自身随时间的变化（intra-variable）**：同一个变量有趋势和周期性，例如温度按天、按年起伏。

过去的模型往往只抓一类：

- 把“同一时刻的所有变量”作为输入（如标准 Transformer），能学变量间关系，却容易丢失每个变量自身的时间顺序；
- 把“每个变量单独建模”（如 iTransformer、PatchTST），能学时间模式，却忽略了变量间的相互影响。

Bi-FI 的出发点很朴素：**两条路都走，然后把两条路的结果加在一起**。这就是“双分支（Bi-Branching）”的含义。

### 2. 总体结构

设输入多元序列为 $X\in\mathbb{R}^{T\times M}$（$T$ 个时间步、$M$ 个变量）。Bi-FI 定义：

$$
\hat Y = F_{\text{inter}}(X) + F_{\text{intra}}(X)
$$

其中：

- $F_{\text{inter}}$：变量间分支（inter-variable branch），在**频域**用卷积学习变量间关系；
- $F_{\text{intra}}$：变量内分支（intra-variable branch），把每个变量当作一个“令牌”，用 **Transformer 注意力**学习时间模式。

两个分支都输出 $\mathbb{R}^{T\times M}$ 形状的表示（或预测值），逐元素相加得到最终输出。

### 3. 变量间分支 $F_{\text{inter}}$：频域 + 二维卷积

#### 3.1 为什么用频域

变量之间可能存在“时间错位”（一个变量滞后于另一个变量）。在时域里这种滞后关系很难直接看到；但做傅里叶变换（FFT）后，滞后表现为**相位差**，幅度关系则体现在频谱大小上。所以频域更适合发现变量间的隐藏关联。

#### 3.2 计算过程

第一步：把输入 $X$ 做嵌入，映射到 $T\times H$（$H$ 为隐层维数）：

$$
x_{\text{inter}} = \text{Embedding}_{\text{inter}}(X) = \underbrace{\text{value}(X)}_{\text{值嵌入}} + \underbrace{\text{pos}(X)}_{\text{位置嵌入}} + \underbrace{\text{time}(X)}_{\text{时间嵌入}}
$$

第二步：沿时间维做快速傅里叶变换（rFFT，只保留正频率）：

$$
X(k)=\sum_{t=0}^{T-1}x_{\text{inter}}[t]\,e^{-2\pi j k t/T},\quad k=0,1,\dots,L-1
$$

其中 $j$ 是虚数单位，$L=\lfloor T/2\rfloor+1$。FFT 结果是复数，拆成实部 $\Re$ 和虚部 $\Im$，作为两个“通道”拼在一起：

$$
\tilde X\in\mathbb{R}^{H\times L\times 2},\qquad \tilde X_{h,k,0}=\Re X_h(k),\ \tilde X_{h,k,1}=\Im X_h(k)
$$

第三步：对这个 $H\times L\times 2$ 的张量做两层二维卷积：

$$
X'=\text{Conv2D}(\tilde X),\qquad X'=\text{ReLU}(\text{Conv2D}(X'))
$$

卷积核大小是 $(2,1)$，即每次在频率维上取相邻两个频率、在“实部/虚部”通道上取 1，把相邻频率的信息混合起来。

第四步：把卷积输出的两个通道重新拼成复数，做逆傅里叶变换（iFFT）回到时域：

$$
x_{\text{time}}=\text{iFFT}(X')
$$

第五步：用全连接层把长度压缩到需要的输出长度：

$$
\hat x_{\text{inter}}=\text{Dense}(x_{\text{time}})
$$

直观理解：FFT 把“波形”拆成不同频率的正弦波；卷积在频域里混合相邻频率；iFFT 再合成回时域。这样模型既能捕捉周期性，又能捕捉变量间的相位/幅度关系。

### 4. 变量内分支 $F_{\text{intra}}$：变量令牌注意力

#### 4.1 变量令牌化

把输入转置，让每个变量变成序列里的一个“位置”：

$$
X\in\mathbb{R}^{T\times M} \xrightarrow{\text{转置}} X^\top\in\mathbb{R}^{M\times T}
$$

再用 MLP 把每个变量的整条时间序列压缩成 $H$ 维向量：

$$
x_{\text{intra}}=\text{MLP}(X^\top)\in\mathbb{R}^{M\times H}
$$

于是 $M$ 个变量变成 $M$ 个“令牌”，每个令牌代表一个变量的浓缩时间信息。

#### 4.2 在变量令牌上做自注意力

$$
Q=x_{\text{intra}}W_Q,\quad K=x_{\text{intra}}W_K,\quad V=x_{\text{intra}}W_V
$$

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

这里注意力矩阵是 $M\times M$：第 $(i,j)$ 个元素表示“变量 $i$ 要参考变量 $j$ 多少”。注意这不同于传统 Transformer 的“时间位置注意力”，而是“**变量间注意力**”——但它的输入是每个变量完整的时间模式，所以本质上在融合变量间关系的同时保留了时间信息。

#### 4.3 输出投影

注意力输出经过前馈网络（FFN）后，用线性层映射回时间长度：

$$
\hat x_{\text{intra}}=\text{Linear}(\text{FFN}(\text{Attn}(x_{\text{intra}})))
$$

### 5. 归一化与融合

为了避免序列均值和方差漂移影响训练，Bi-FI 先对输入做“实例归一化”：

$$
\bar X = \frac{X-\mu}{\sigma+\epsilon},\qquad \mu=\frac1T\sum_t X_t,\ \sigma^2=\frac1T\sum_t (X_t-\mu)^2
$$

模型在归一化后的数据上训练，输出后再反归一化：

$$
\hat Y = \hat Y_{\text{norm}}\cdot\sigma + \mu
$$

最后两个分支相加：

$$
\hat Y = F_{\text{inter}}(X)+F_{\text{intra}}(X)
$$

### 6. 三种任务如何共用同一模型

- **预测**：输出未来 $H$ 步；
- **异常检测**：输出与输入等长的“重建”，重建误差大即异常；
- **分类**：两个分支的输出都压平后映射到类别数，相加后过 softmax。

论文中 Bi-FI 在三个任务上都取得了与最强基线相当或更好的结果，证明了“同时学变量间与变量内信息”的价值。

---

## 第二部分：源码讲解

### 1. 类的初始化（`__init__`）

```python
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.hidden_size = configs.d_model
```

`configs` 就是运行参数（`--seq_len 96` 等）。`d_model` 是隐层维数 $H$。

```python
        self.intra_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, ...)
        self.inter_embedding = DataEmbedding(configs.enc_in, configs.d_model, ...)
```

- `intra_embedding`：变量内分支的嵌入，即 4.1 节的“变量令牌化”（`DataEmbedding_inverted` 先转置再线性层）；
- `inter_embedding`：变量间分支的嵌入，即 3.2 节的“值+位置+时间嵌入”（`DataEmbedding`）。

```python
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.seq_len, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(self.seq_len, self.d_model, kernel_size=(2, 1), stride=(1, 1)),
            nn.ReLU()
        )
```

这就是 3.3 节的两层二维卷积：输入通道 $H$ → 输出通道 $T$（`seq_len`）→ 输出通道 $d_{\text{model}}$。卷积核 $(2,1)$ 在频率维上每次看两个相邻频率。

```python
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout,
                activation=configs.activation)
            for l in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))
```

变量内分支的标准多头自注意力编码器，对应 4.2 节的公式。

```python
        if self.task_name == 'long_term_forecast':
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projector = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.num_class = configs.num_class
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projector = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
```

不同任务用不同输出头：

- 预测：$d_{\text{model}}\to H$（预测长度）；
- 异常：$d_{\text{model}}\to T$（重建原长）；
- 分类：$d_{\text{model}}\times M\to$ 类别数。

### 2. 频域分支 `inter_frequency`

```python
    def inter_frequency(self, x):
        x_fft = torch.fft.rfft(x, dim=1)          # 沿时间维做 FFT
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag
        x_fft = torch.stack([x_fft_real, x_fft_imag], dim=-1).permute(0, 2, 1, 3)
```

对应公式：

$$
X(k)=\sum_{t}x[t]e^{-2\pi jkt/T},\quad \tilde X=[\Re X(k);\ \Im X(k)]
$$

`rfft` 只保留正频率，输出形状 `[B, L, H]`（复数）；`stack + permute` 把它变成 `[B, H, L, 2]`，最后两维是（实部，虚部）。

```python
        y = self.conv_layers(x_fft)               # 二维卷积
        y = torch.view_as_complex(y)              # 最后两维重新解释为复数
        x_time = torch.fft.irfft(y, dim=2)        # 逆 FFT 回时域
        B, C, T = x_time.shape
        output = nn.Linear(T, self.out_len, bias=True).to(x_time.device)(x_time)
        return output
```

对应公式：

$$
x_{\text{time}}=\text{iFFT}(\text{Conv2D}(\tilde X)),\qquad \hat x_{\text{inter}}=\text{Linear}(x_{\text{time}})
$$

### 3. 预测前向 `forecast`

```python
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, N = x_enc.shape

        intra_in = self.intra_embedding(x_enc, x_mark_enc)
        intra_out, attns = self.encoder(intra_in, attn_mask=None)
        intra_out = self.projector(intra_out).permute(0, 2, 1)[:, :, :N]

        inter_in = self.inter_embedding(x_enc, x_mark_enc)
        inter_out = self.inter_frequency(inter_in)
        inter_out = inter_out.permute(0, 2, 1)[:, :, :N]

        co_out = self.dropout(inter_out + intra_out)      # 双分支相加
        co_out = co_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        co_out = co_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return co_out
```

逐句对应：

1. 前四行做实例归一化（第 5 节公式）；
2. `intra_in` 形状 `[B, M, H]`（变量令牌），编码器输出后 `projector` 映射到 `[B, M, H]`，`permute` 成 `[B, H, M]` 并截取前 $N$ 个变量；
3. `inter_in` 形状 `[B, T, H]`，`inter_frequency` 返回 `[B, H, H]`（$H$ 个通道、预测长度），同样截取 $N$；
4. `inter_out + intra_out` 就是公式 $\hat Y=F_{\text{inter}}+F_{\text{intra}}$；
5. 最后两行反归一化，把均值方差加回去。

注意：inter 分支的输出通道数是 $d_{\text{model}}$，所以要求 $d_{\text{model}}\ge M$（本库复现时对 ECL 等大变量数据集使用 `d_model=512`），否则截取不到 $N$ 个变量。

### 4. 异常检测与分类

```python
    def anomaly_detection(self, x_enc, x_mark_enc):
        # 同样的归一化、双分支、相加、反归一化
        # 区别：输出长度 L = seq_len（重建输入）
        ...
        return co_out

    def classification(self, x_enc, x_mark_enc):
        intra_out = ...reshape(batch, -1)
        intra_out = self.projector(intra_out)               # (B, num_class)
        inter_out = ...reshape(batch, -1)
        inter_out = nn.Linear(L, self.num_class)(inter_out) # (B, num_class)
        co_out = self.dropout(inter_out + intra_out)        # 双分支相加
        return co_out
```

异常检测把“预测”换成“重建”：输出与输入等长，训练时用重建误差（L1）做损失，测试时重建误差大的时刻判为异常。分类则是把两个分支分别压平、各自映射到类别数，再相加。

### 5. 统一入口 `forward`

```python
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, x_mark_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
```

`forward` 根据 `task_name` 选择分支；预测时取最后 `pred_len` 个时间步作为输出。

## 小结

Bi-FI = 频域卷积分支（学变量间关系） + 变量令牌注意力分支（学变量内时间模式），两条分支输出相加。它的创新点不在某个惊艳的模块，而在“把两类信息显式拆开、并行学习、再融合”。
