# Tensor Parallelism 与 Sequence Parallelism

本文统一讲解 Megatron 风格 Tensor Parallelism（TP）与 Sequence Parallelism（SP）。

两者的关系可以概括为：

| 方案 | 切分对象 | 主要目标 | 主要通信 |
| --- | --- | --- | --- |
| TP | 线性层权重与 hidden/intermediate 维度 | 分摊权重与矩阵计算 | AllReduce，或 ReduceScatter + AllGather |
| SP | TP 算子之间的 token/sequence 激活 | 减少激活复制和逐元素重复计算 | AllGather + ReduceScatter |
| TP + SP | 权重维度与 token 维度交替分片 | 同时获得 TP 容量收益与 SP 激活收益 | 每个 Attention/MLP 子层一组 AllGather + ReduceScatter |

本文的通信量统一采用“每个 Rank 在一次 Collective 中发送到网络的字节数”这一口径。

## 导航

* [Tensor Parallelism：权重与计算切分](#tensor-parallelism权重与计算切分)
* [Sequence Parallelism：激活切分](#sequence-parallelism激活切分)
* [通信与通信量总结](#通信与通信量总结)
* [性能瓶颈与判断方法](#性能瓶颈与判断方法)

---

## Tensor Parallelism：权重与计算切分

### TP 在并行体系中的位置

TP（Tensor Parallelism，张量并行）是把一个算子内部的权重张量和计算任务切分到多张 GPU 上。

它属于层内模型并行，即多张 GPU 同时执行 Transformer 的同一层：

```text
同一个 Transformer Layer：

Rank 0 ─┐
Rank 1 ─┼─ 共同计算这一层
Rank 2 ─┼─
Rank 3 ─┘
```

TP 主要解决两个问题：

1. 单张 GPU 无法容纳完整模型权重；
2. 希望多张 GPU 同时计算一个请求，缩短大矩阵运算时间。

但 TP 也会把原本位于一张 GPU 内部的数据依赖，变成跨 GPU 通信。TP 的核心问题因此不是“能不能把矩阵切开”，而是：

> 如何选择切分方式，使连续两个线性层之间尽量少通信？

Megatron 风格 TP 的关键设计，就是将 Column Parallel Linear 与 Row Parallel Linear 成对使用，从而把两个线性层之间的中间激活保持为分片状态，只在必要的位置执行归约。

---

### 统一符号

为了避免 Batch、Sequence 和 Token 维度反复出现，先把前导维度展平。

设输入隐藏状态为：

$$X\in\mathbb{R}^{M\times H}$$

其中：

* $M$ 表示参与当前线性层计算的 token 数，通常有 $M=B\times S$；
* $B$ 是 batch size；
* $S$ 是序列长度；
* $H$ 是 hidden size。

线性层权重为：

$$W\in\mathbb{R}^{H\times I}$$

输出为：

$$Y=XW\in\mathbb{R}^{M\times I}$$

假设 TP Group 中有 $p$ 个 Rank。

如果张量元素占用 $b$ 字节，则完整输出 $Y$ 的数据量为：

$$N_Y=MIb$$

---

### 为什么线性层可以切分

矩阵乘法：

$$Y=XW$$

既可以沿输出维度切，也可以沿输入维度切。

沿输出维度切分时，不同 Rank 负责产生不同的输出特征；沿输入维度切分时，不同 Rank 产生同一个输出的局部贡献，最后需要把局部贡献相加。

这分别对应：

* Column Parallel Linear；
* Row Parallel Linear。

---

### Column Parallel Linear

#### 切分方式

将权重 $W$ 沿输出维度 $I$ 切成 $p$ 份：

$$W=[W_0,W_1,\ldots,W_{p-1}]$$

其中：

$$W_r\in\mathbb{R}^{H\times I/p}$$

每个 Rank 都持有完整输入 $X$，但只持有一部分权重：

```text
Rank 0：W0
Rank 1：W1
...
Rank p-1：Wp-1
```

第 $r$ 个 Rank 计算：

$$Y_r=XW_r$$

其中：

$$Y_r\in\mathbb{R}^{M\times I/p}$$

完整输出为：

$$Y=[Y_0,Y_1,\ldots,Y_{p-1}]$$

不同 Rank 计算的是不同输出列，因此称为 Column Parallel Linear。

---

#### 为什么 Column Parallel 后可以不通信

Column Parallel 的每个 Rank 已经得到完整输出的一部分。

如果下一步计算本身也能沿这一维度切分，就不需要立即把 $Y_0,\ldots,Y_{p-1}$ 收集起来。

例如 MLP（Multi-Layer Perceptron，多层感知机）的第一个线性层：

$$Z=XW_{\text{up}}$$

通常把 Intermediate Size 沿输出维度切分：

```text
Rank 0：Z0
Rank 1：Z1
...
Rank p-1：Zp-1
```

激活函数是逐元素操作：

$$A_r=\sigma(Z_r)$$

每个 Rank 可以直接在本地计算自己的分片，不需要知道其他 Rank 的 $Z$。

因此：

```text
Column Parallel Linear
        ↓
分片输出
        ↓
本地 Activation
```

中间没有通信。

---

#### 什么时候需要 AllGather

如果后续算子要求每个 Rank 都获得完整 $Y$，则需要执行 AllGather：

```text
初始：
Rank 0：Y0
Rank 1：Y1
...
Rank p-1：Yp-1

AllGather 后：
所有 Rank：[Y0,Y1,...,Yp-1]
```

完整输出大小为 $N_Y$，每个 Rank 初始持有 $N_Y/p$。

Ring AllGather 中，每 Rank 的发送量为：

$$V_{\text{AllGather}}=\frac{p-1}{p}N_Y$$

不过，经典 Transformer TP 会尽量避免在 Column Parallel 后立即 AllGather，而是让下一层直接消费分片结果。

---

### Row Parallel Linear

#### 切分方式

Row Parallel Linear 将输入 $X$ 沿 hidden dimension 切分：

$$X=[X_0,X_1,\ldots,X_{p-1}]$$

其中：

$$X_r\in\mathbb{R}^{M\times H/p}$$

权重 $W$ 沿输入维度切分：

$$W=\begin{bmatrix}W_0 \\ W_1 \\ \vdots \\ W_{p-1}\end{bmatrix}$$

其中：

$$W_r\in\mathbb{R}^{H/p\times I}$$

第 $r$ 个 Rank 计算：

$$Y^{(r)}=X_rW_r$$

需要注意，每个 $Y^{(r)}$ 都具有完整输出形状：

$$Y^{(r)}\in\mathbb{R}^{M\times I}$$

但是它只是完整输出的一部分数值贡献，而不是输出维度上的一个分片。

---

#### 为什么需要归约

根据分块矩阵乘法：

$$XW=[X_0,X_1,\ldots,X_{p-1}]\begin{bmatrix}W_0 \\ W_1 \\ \vdots \\ W_{p-1}\end{bmatrix}$$

展开后：

$$Y=\sum_{r=0}^{p-1}X_rW_r$$

也就是：

$$Y=\sum_{r=0}^{p-1}Y^{(r)}$$

因此，不同 Rank 的局部结果必须逐元素相加。

如果所有 Rank 都需要完整输出 $Y$，就执行 AllReduce。

---

#### Row Parallel 的 AllReduce 通信量

每个 Rank 的局部输出大小都是 $N_Y=MIb$。

Ring AllReduce 的每 Rank 发送量为：

$$V_{\text{AllReduce}}=2\frac{p-1}{p}N_Y$$

其中：

$$\text{AllReduce}=\text{ReduceScatter}+\text{AllGather}$$

因此：

$$V_{\text{AllReduce}}=\frac{p-1}{p}N_Y+\frac{p-1}{p}N_Y$$

这里需要特别注意：虽然每张 GPU 只保存了 $1/p$ 的 Row Parallel 权重，但其局部输出 $Y^{(r)}$ 仍然具有完整输出形状。

所以 Row Parallel 的通信量由输出激活大小决定，而不是由每 Rank 权重大小决定。

---

### 为什么要把 Column Parallel 和 Row Parallel 配对

考虑一个两层 MLP：

$$Y=\sigma(XW_1)W_2$$

设：

$$W_1\in\mathbb{R}^{H\times I}$$

$$W_2\in\mathbb{R}^{I\times H}$$

第一层使用 Column Parallel：

$$W_1=[W_{1,0},W_{1,1},\ldots,W_{1,p-1}]$$

每个 Rank 得到：

$$Z_r=XW_{1,r}$$

其中：

$$Z_r\in\mathbb{R}^{M\times I/p}$$

本地执行激活：

$$A_r=\sigma(Z_r)$$

第二层使用 Row Parallel，每个 Rank 保存：

$$W_{2,r}\in\mathbb{R}^{I/p\times H}$$

于是可以直接使用本地 $A_r$：

$$Y^{(r)}=A_rW_{2,r}$$

最后执行：

$$Y=\sum_{r=0}^{p-1}Y^{(r)}$$

整体过程为：

```text
完整输入 X
    │
    ▼
Column Parallel W1
    │
各 Rank 得到不同 Intermediate 分片
    │
    ▼
本地 Activation
    │
    ▼
Row Parallel W2
    │
各 Rank 得到完整输出的局部贡献
    │
    ▼
AllReduce
    │
    ▼
完整输出 Y
```

这种配对只在第二个线性层后通信一次。

如果第一层 Column Parallel 后立即 AllGather，再执行完整的第二层，就会产生额外通信和权重复制，失去 TP 的意义。

---

### Attention 中的 TP

#### QKV Projection

设输入为：

$$X\in\mathbb{R}^{B\times S\times H}$$

QKV Projection 通常使用 Column Parallel。

对于 MHA（Multi-Head Attention，多头注意力），设 Attention Head 数为 $N_h$，每个 Head 维度为 $D_h$：

$$H=N_hD_h$$

TP Degree 为 $p$ 时，每个 Rank 通常负责：

$$\frac{N_h}{p}$$

个 Attention Heads。

例如：

```text
N_h = 32
TP  = 4

Rank 0：Head 0～7
Rank 1：Head 8～15
Rank 2：Head 16～23
Rank 3：Head 24～31
```

Q、K、V 的本地形状大致为：

$$Q_r,K_r,V_r\in\mathbb{R}^{B\times S\times N_h/p\times D_h}$$

因为每个 Head 的 Attention 可以独立计算，所以 QKV Projection 后不需要立即通信。

---

#### 本地 Attention

每个 Rank 对自己持有的 Head 执行：

$$O_r=\operatorname{Attention}(Q_r,K_r,V_r)$$

其中：

$$O_r\in\mathbb{R}^{B\times S\times N_h/p\times D_h}$$

将 Head 维度展平后：

$$O_r\in\mathbb{R}^{B\times S\times H/p}$$

此时各 Rank 持有 Attention 输出的 hidden-dimension 分片。

---

#### Output Projection

Attention Output Projection 为：

$$Y=OW_O$$

输入 $O$ 已沿 hidden dimension 切分，因此 $W_O$ 自然使用 Row Parallel。

每个 Rank 计算：

$$Y^{(r)}=O_rW_{O,r}$$

最后：

$$Y=\sum_{r=0}^{p-1}Y^{(r)}$$

因此需要一次 AllReduce，或者在结合 SP 时使用 ReduceScatter。

---

### MLP 中的 TP

对于普通 Transformer MLP：

$$Z=\sigma(XW_{\text{up}})W_{\text{down}}$$

其中：

$$W_{\text{up}}\in\mathbb{R}^{H\times I}$$

$$W_{\text{down}}\in\mathbb{R}^{I\times H}$$

通常采用：

* Up Projection：Column Parallel；
* Down Projection：Row Parallel。

对于 SwiGLU（Swish-Gated Linear Unit，Swish 门控线性单元）等门控 MLP：

$$Z=\operatorname{SiLU}(XW_{\text{gate}})\odot XW_{\text{up}}$$

Gate Projection 和 Up Projection 都使用 Column Parallel，并沿相同 Intermediate Dimension 切分。

每个 Rank 本地计算：

$$Z_r=\operatorname{SiLU}(XW_{\text{gate},r})\odot XW_{\text{up},r}$$

之后通过 Row Parallel Down Projection 恢复 hidden size，并执行一次归约。

---

### 每个 Transformer Layer 中有多少次 TP 通信

经典 Megatron 风格 Transformer Layer 通常包含两组 Column–Row 配对：

```text
第一组：
QKV Column Parallel
→ Attention
→ Output Row Parallel
→ AllReduce

第二组：
Gate/Up Column Parallel
→ Activation
→ Down Row Parallel
→ AllReduce
```

因此，每层通常有两次主要 TP AllReduce。

对于 $L$ 层模型，一次完整 Forward 可能包含约：

$$2L$$

次 TP Collective。

在 Decode 阶段，每生成一个 token 都要经过全部 $L$ 层，因此可能每生成一个 token 就执行约 $2L$ 次跨 Rank 同步。

这正是 TP 对通信延迟非常敏感的原因。

---

### TP 通信量与张量形状

假设每个 Row Parallel 输出张量为：

$$Y\in\mathbb{R}^{B\times S\times H}$$

数据量为：

$$N_Y=BSHb$$

一次 Ring AllReduce 的每 Rank 发送量为：

$$V_{\text{AR}}=2\frac{p-1}{p}BSHb$$

如果每层有两次相同大小的 AllReduce，则每层每 Rank 发送量近似为：

$$V_{\text{TP, layer}}=4\frac{p-1}{p}BSHb$$

对于 $L$ 层：

$$V_{\text{TP, model}}=4L\frac{p-1}{p}BSHb$$

这只是简化估算。实际通信量还会受到以下因素影响：

* Attention 和 MLP 输出形状是否完全相同；
* 是否使用 SP；
* 是否融合 Residual；
* 是否使用量化通信；
* 是否使用异步 Collective；
* 是否存在额外的 Vocab Parallel 通信。

对于 Decode，通常 $S=1$，但 $B$ 是当前 Decode Batch 中的活跃请求数。

此时通信张量可能不大，却需要执行大量 Collective，因此更容易 latency-bound。

---

### TP 与 Prefill、Decode 的差异

#### Prefill

Prefill 中：

$$M=B\times S$$

通常较大。

每张 GPU 上的 GEMM 规模较大，TP 可以有效分摊计算。与此同时，通信张量也较大，更容易充分利用链路带宽。

因此 Prefill 中 TP 往往具有较好的扩展性。

#### Decode

Decode 中通常：

$$M=B$$

如果并发较低，$B$ 很小。

增加 TP Degree 后，每个 Rank 的 GEMM 输出维度继续缩小，但 AllReduce 次数不变。此时可能出现：

* GEMM 时间快速下降；
* Collective 固定延迟基本不变；
* 通信占总延迟的比例升高；
* GPU 数量增加但 TPOT 不再下降。

所以 Decode 的最优 TP Degree 往往小于“模型能够使用的最大 TP Degree”。

---

### TP 与 MHA、GQA、MQA

MHA 中通常有：

$$N_q=N_{kv}$$

因此可以自然地按 Head 切分。

GQA（Grouped-Query Attention，分组查询注意力）中：

$$N_q>N_{kv}$$

MQA（Multi-Query Attention，多查询注意力）中通常有：

$$N_{kv}=1$$

如果 TP Degree 为 $p$，理想情况要求 Query Head 和 KV Head 能够合理分配到各 Rank。

例如：

```text
Query Heads = 64
KV Heads    = 8
TP          = 8
```

每个 Rank 可以持有：

```text
8 个 Query Heads
1 个 KV Head
```

但如果：

```text
Query Heads = 64
KV Heads    = 8
TP          = 16
```

KV Head 数少于 TP Rank 数。此时可能采用：

* 在多个 Rank 上复制同一个 KV Head；
* 让若干 Rank 组成一个 KV Head Sharing Group；
* 对 Q 与 KV 使用不同切分方式；
* 限制 TP Degree；
* 将 KV Cache 改为 Context Parallel 分片。

因此，GQA 和 MQA 中的 TP Degree 不能只看 Query Head 数。

---

### Vocab Parallelism

Embedding 和 LM Head 的 Vocabulary Dimension 通常很大，也可以沿词表切分。

假设词表大小为 $V$，Embedding Dimension 为 $H$：

$$W_E\in\mathbb{R}^{V\times H}$$

使用 $p$ 个 Rank 时，每个 Rank 保存：

$$W_{E,r}\in\mathbb{R}^{V/p\times H}$$

#### 输入 Embedding

每个 Rank 只处理落在自己词表区间内的 token id，其他 token 的本地结果置零。

然后通过 AllReduce 得到完整 Embedding。

#### 输出 LM Head

Logits 为：

$$Z=XW_{\text{LM}}^\mathsf{T}$$

沿 Vocabulary Dimension 切分后，每个 Rank 得到：

$$Z_r\in\mathbb{R}^{M\times V/p}$$

如果只需要采样，不一定要 AllGather 完整 $V$ 维 Logits。

系统可以：

* 各 Rank 计算本地 Top-$k$；
* 收集各 Rank 的候选；
* 再执行全局 Top-$k$ 或 Sampling。

这样可以避免传输完整词表 Logits。

---

### TP 的硬件拓扑要求

TP 的 Collective 位于每一层的关键路径，因此通常应限制在高速互联域内：

```text
NVSwitch / NVLink
        优于
PCIe
        优于
跨节点 InfiniBand / RoCE
```

常见部署方式是：

```text
节点内：TP
节点间：PP 或 DP
```

例如两台 8 卡服务器部署 16 卡模型：

```text
Node 0：TP Group 0，负责 Pipeline Stage 0
Node 1：TP Group 1，负责 Pipeline Stage 1
```

即：

```text
TP = 8
PP = 2
```

这样高频 AllReduce 留在节点内，跨节点只传递 Pipeline Boundary 激活。

---

### TP 的性能判断

TP 是否有效，取决于：

$$T_{\text{single GPU compute}}-T_{\text{TP compute}}>T_{\text{communication}}+T_{\text{synchronization}}$$

观察 Nsight Systems（NVIDIA Nsight Systems，英伟达系统级性能分析器）时间线时，如果发现：

```text
GEMM 很短
NCCL Collective 很长
GPU 经常等待同步
```

通常意味着：

* TP Degree 过大；
* Decode Batch 太小；
* TP 跨越了低速链路；
* Collective 无法与计算重叠。

此时应考虑：

* 降低 TP Degree；
* 增大 Decode Batch；
* 使用 DP 扩展吞吐；
* 把 TP Group 限制在 NVLink 域内；
* 使用 SP 将 AllReduce 改写为 ReduceScatter 与 AllGather，并缩短激活复制时间。

---

## Sequence Parallelism：激活切分

### 术语范围

SP（Sequence Parallelism，序列并行）在不同论文和系统中可能指不同技术。

本节所说的 SP 特指 Megatron 风格 Sequence Parallelism：

> 在 TP Group 内，把 LayerNorm、Residual 和逐元素算子之间的激活沿 token 或 sequence 维度切分。

它不同于：

* UP（Ulysses Parallelism，尤利西斯并行）：通过 AllToAll 在序列分片和 Head 分片之间转换；
* CP（Context Parallelism，上下文并行）：对 Attention 上下文和 KV Cache 进行分布式切分；
* Ring Attention：让 KV Block 在设备间环形移动。

Megatron SP 本身通常不能独立存在，而是与 TP 配套使用。

---

### 为什么普通 TP 仍然存在激活复制

考虑普通 TP 中 Row Parallel Linear 的输出。

每个 Rank 计算局部贡献：

$$Y^{(r)}=X_rW_r$$

然后执行 AllReduce：

$$Y=\sum_{r=0}^{p-1}Y^{(r)}$$

AllReduce 后，每个 Rank 都拥有完整 $Y$：

```text
Rank 0：完整 Y
Rank 1：完整 Y
...
Rank p-1：完整 Y
```

接下来执行的操作可能包括：

* Residual Add；
* LayerNorm；
* RMSNorm；
* Dropout；
* 激活缩放；
* 其他逐元素算子。

这些操作并不需要所有 Rank 都保存完整 token 集合，因为不同 token 之间通常彼此独立。

但在普通 TP 中，每个 Rank 都会重复保存和计算完整激活，造成：

* 激活显存重复；
* LayerNorm 等操作重复计算；
* AllReduce 后完整结果复制到所有 Rank。

训练阶段需要保存大量前向激活，因此这种重复尤其昂贵。

---

### SP 的核心思想

SP 将激活沿 token 维度切分。

令：

$$T=B\times S$$

完整激活为：

$$X\in\mathbb{R}^{T\times H}$$

在 $p$ 个 Rank 上沿 token 维度切分：

$$X=[X_0;X_1;\ldots;X_{p-1}]$$

其中：

$$X_r\in\mathbb{R}^{T/p\times H}$$

执行 SP 后：

```text
Rank 0：前 1/p token
Rank 1：第 2/p token
...
Rank p-1：最后 1/p token
```

每个 Rank 对自己的 token 子集执行：

* LayerNorm；
* Residual；
* Dropout；
* 逐元素运算。

因为这些算子对每个 token 独立，所以不需要跨 Rank 通信。

---

### SP 如何与 TP 连接

SP 的关键不是单纯把序列切开，而是在两个张量布局之间切换：

#### Sequence-Parallel Layout

每个 Rank 持有：

$$X_r\in\mathbb{R}^{T/p\times H}$$

即：

```text
部分 token
完整 hidden dimension
```

#### Tensor-Parallel Linear 所需布局

Column Parallel Linear 通常要求每个 TP Rank 获得全部 token：

$$X\in\mathbb{R}^{T\times H}$$

每个 Rank 使用不同的权重分片，对全部 token 计算不同输出特征。

因此，在进入 Column Parallel Linear 前，需要把序列分片恢复为完整 token 集合。

这通过 AllGather 完成。

---

### SP 中的 AllGather

初始时每个 Rank 持有：

$$X_r\in\mathbb{R}^{T/p\times H}$$

完整激活大小为：

$$N_X=THb$$

每 Rank 本地分片大小为：

$$\frac{N_X}{p}$$

执行 AllGather 后，每个 Rank 得到：

$$X\in\mathbb{R}^{T\times H}$$

Ring AllGather 每 Rank 发送量为：

$$V_{\text{AG}}=\frac{p-1}{p}N_X$$

之后，各 Rank 可以执行 Column Parallel Linear：

$$Z_r=XW_r$$

得到 hidden 或 intermediate dimension 上的分片。

---

### SP 中的 ReduceScatter

经过一组 Column–Row Parallel 运算后，每个 Rank 得到完整输出形状的局部贡献：

$$Y^{(r)}\in\mathbb{R}^{T\times H}$$

普通 TP 会执行 AllReduce，让每个 Rank 都获得完整 $Y$。

SP 则执行 ReduceScatter。

先归约：

$$Y=\sum_{r=0}^{p-1}Y^{(r)}$$

再沿 token 维度切分：

$$Y=[Y_0;Y_1;\ldots;Y_{p-1}]$$

每个 Rank 最终得到：

$$Y_r\in\mathbb{R}^{T/p\times H}$$

这样输出立即回到 Sequence-Parallel Layout。

Ring ReduceScatter 的每 Rank 发送量为：

$$V_{\text{RS}}=\frac{p-1}{p}N_Y$$

其中：

$$N_Y=THb$$

---

### 一组 SP + TP 的完整过程

以 Attention 子层为例：

```text
Sequence-Parallel 输入
每 Rank：[T/p, H]
        │
        ▼
AllGather
        │
每 Rank：[T, H]
        │
        ▼
QKV Column Parallel
        │
每 Rank：全部 token，部分 Heads
        │
        ▼
Local Attention
        │
        ▼
Output Row Parallel
        │
每 Rank：[T, H] 的局部贡献
        │
        ▼
ReduceScatter
        │
每 Rank：[T/p, H]
        │
        ▼
本地 Residual / LayerNorm
```

MLP 子层类似：

```text
Sequence-Parallel 输入
        │
        ▼
AllGather
        │
        ▼
Gate/Up Column Parallel
        │
        ▼
本地 Activation
        │
        ▼
Down Row Parallel
        │
        ▼
ReduceScatter
        │
        ▼
Sequence-Parallel 输出
```

---

### 为什么 SP 不会增加理论总通信量

普通 TP 中，一组 Row Parallel 输出执行一次 AllReduce：

$$V_{\text{AllReduce}}=2\frac{p-1}{p}N$$

SP 将其拆成：

* 进入下一组 Column Parallel 前的 AllGather；
* 当前 Row Parallel 后的 ReduceScatter。

二者通信量分别为：

$$V_{\text{AllGather}}=\frac{p-1}{p}N$$

$$V_{\text{ReduceScatter}}=\frac{p-1}{p}N$$

合计：

$$V_{\text{SP pair}}=2\frac{p-1}{p}N$$

因此：

$$V_{\text{SP pair}}=V_{\text{AllReduce}}$$

从 Ring 算法的理想字节数看，SP 没有增加通信量。

它只是把一次 AllReduce 拆成：

$$\text{AllReduce}=\text{ReduceScatter}+\text{AllGather}$$

并让两个阶段分别位于不同算子边界。

---

### SP 的真正收益

SP 的主要收益不是减少理论通信字节数，而是减少激活复制。

普通 TP 中，每 Rank 在 LayerNorm 等位置持有：

$$T\times H$$

大小的完整激活。

SP 中，每 Rank 只持有：

$$\frac{T}{p}\times H$$

大小的序列分片。

理想情况下，这部分激活显存降低为原来的：

$$\frac{1}{p}$$

此外：

* LayerNorm 只处理本地 token；
* Residual Add 只处理本地 token；
* Dropout 只处理本地 token；
* 不再在每个 Rank 上重复执行完整 token 集合的逐元素操作。

---

### SP 为什么在训练中更重要

训练阶段需要为反向传播保存：

* LayerNorm 输入；
* Attention 输入和输出；
* MLP 激活；
* Dropout Mask；
* Residual 中间结果。

这些激活的生命周期跨越整个前向和反向过程。

因此，SP 能显著降低训练显存。

推理阶段没有反向传播，中间激活通常在一层执行完成后很快释放，所以 SP 的显存收益相对较小。

但 SP 在以下推理场景中仍可能有价值：

* 长 Prompt Prefill；
* 大 Batch Prefill；
* 大规模 DiT；
* 激活或 Workspace 占用较高；
* 需要与 TP 通信融合；
* 希望避免 TP Rank 上的逐元素重复计算。

---

### SP 与 CP 的区别

SP 主要处理的是：

```text
LayerNorm
Residual
Dropout
逐元素激活
TP 线性层之间的张量布局
```

CP 主要处理的是：

```text
Attention 的长上下文
Q、K、V 的分布
KV Cache 分片
分布式 Softmax
```

SP 中，进入 Attention 计算前通常仍会通过 AllGather 获得完整 token 集合。

CP 则试图让 Attention 本身在上下文分片状态下完成，不要求每张 GPU 都拥有完整上下文。

因此：

> SP 是 TP 周围的激活分片；CP 是 Attention 内部的上下文分片。

---

## 通信与通信量总结

令完整 hidden state 的数据量为：

$$N=BSHb$$

其中 $B$ 为 batch size，$S$ 为序列长度，$H$ 为 hidden size，$b$ 为每个元素的字节数，TP Degree 为 $p$。

普通 TP 在 Attention Output Projection 和 MLP Down Projection 后各执行一次 Ring AllReduce：

$$V_{\text{TP, layer}}=2\times2\frac{p-1}{p}N=4\frac{p-1}{p}N$$

TP + SP 将每次 AllReduce 拆到相邻算子边界，改为一次 ReduceScatter 和下一子层前的一次 AllGather。一个 Transformer Layer 包含 Attention 与 MLP 两组：

$$V_{\text{TP+SP, layer}}=2\left(\frac{p-1}{p}N+\frac{p-1}{p}N\right)=4\frac{p-1}{p}N$$

因此，在理想 Ring 字节数模型下：

$$V_{\text{TP+SP, layer}}=V_{\text{TP, layer}}$$

SP 省下的是完整激活的复制、保存和逐元素重复计算，而不是理论网络字节数。实际性能仍会受 Collective 启动次数、算子融合和通信调度影响。

---

## 性能瓶颈与判断方法

TP 与 TP + SP 的主要瓶颈具有明显的阶段差异：

| 场景 | 常见瓶颈 | 原因 |
| --- | --- | --- |
| Prefill | 通信带宽、显存容量 | $BSH$ 较大，Collective 消息大，但 GEMM 也较容易摊薄通信 |
| Decode | Collective 启动延迟、同步、小 GEMM | 每步 $S=1$，消息不大但每层仍有固定次数的通信 |
| 大 TP Degree | 小矩阵效率和同步开销 | 每 Rank 计算量继续减小，而 Collective 延迟不会同比下降 |
| 跨节点 TP | 网络带宽与尾延迟 | 高频 Collective 位于每层关键路径 |
| GQA/MQA | KV Head 切分与复制 | $N_{kv}$ 可能小于 TP Degree |
| 训练 | 激活显存 | SP 通常能显著降低 LayerNorm、Residual 等位置的激活复制 |

判断 TP Degree 是否过大时，应同时观察 GEMM 时间、NCCL 时间、通信前后的空闲区间以及每 Rank 的矩阵形状。若 GEMM 已很短而 Collective 主导时间线，继续增加 TP 通常不会改善延迟。

部署上通常优先把 TP Group 限制在 NVLink/NVSwitch 域内，再用 DP 或 PP 跨节点扩展。
