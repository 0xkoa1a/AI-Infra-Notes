---
title: "Tensor Parallelism 与 Sequence Parallelism"
order: 2
---

# Tensor Parallelism 与 Sequence Parallelism

## Tensor Parallelism

Megatron 风格 TP 的关键设计，就是将 Column Parallel Linear 与 Row Parallel Linear 成对使用，从而把两个线性层之间的中间激活保持为分片状态，只在必要的位置执行归约。

***

### 统一符号

把前导 batch x sequence 维度展平。

设输入隐藏状态为：

$$X\in\mathbb{R}^{M\times H}$$

其中：

* $M$ 表示参与当前线性层计算的 token 数， $M=B\times S$；
* $B$ 是 batch size；
* $S$ 是序列长度；
* $H$ 是 hidden size。

线性层权重为：

$$W\in\mathbb{R}^{H\times I}$$

输出为：

$$Y=XW\in\mathbb{R}^{M\times I}$$

假设有 $p$ 个 Rank。

***

### Column Parallel Linear

#### 切分方式

将权重 $W$ 沿输出维度 $I$ 切成 $p$ 份：

$$W=[W_0,W_1,\ldots,W_{p-1}]$$

其中：

$$W_r\in\mathbb{R}^{H\times (I/p)}$$

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

***

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

每个 $Y^{(r)}$ 都具有完整输出形状：

$$Y^{(r)}\in\mathbb{R}^{M\times I}$$

但是它只是完整输出的 partial sum。

***

### Column Parallel 和 Row Parallel 恰好可以配对

考虑一个两层 MLP：

$$Y=\sigma(XW_1)W_2$$

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
各 Rank 得到完整输出形状的 partial sum
    │
    ▼
AllReduce
    │
    ▼
完整输出 Y
```

Column Parallel Linear 的输出分片正好可以作为 Row Parallel Linear 的输入分片，不需要通信。

***

### Attention 中的 TP

Multi-Head Self Attention 可以分为 QKV Projection、Attention、Output Projection 三个阶段。其中 Attention 中每个 head 的计算是独立的，天然适合 Column Parallel。

将 Q、K、V 按照 head 维度切分，每个 rank 负责在若干个 head 上进行 QKV Projection 和 Attention 计算，得到局部输出 $O_r$（包含全部的序列和部分的 heads）。Output Projection 为：

$$Y=OW_O$$

输入 $O$ 已沿 head 维度切分，因此 $W_O$ 自然使用 Row Parallel。

最后将每个 rank 得到的 partial sum 归约，得到完整输出 $Y$。

***


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

每层有两次 TP AllReduce。


假设每个 Row Parallel 输出张量为：

$$Y\in\mathbb{R}^{B\times S\times H}$$

数据量为：

$$N_Y=BSHb$$

一次 Ring AllReduce 的每 Rank 发送量为：

$$V_{\text{AR}}=2\frac{p-1}{p}BSHb$$

> Ring AllReduce = ReduceScatter + AllGather
> 每个 rank 每轮将自己负责的分片（大小为 BSHb / p）发送给下一个 rank，同时接收上一个 rank 的分片。经过 p-1 轮后，每个 rank 得到完整结果。

两次相同大小的 AllReduce，则每层每 Rank 发送量近似为：

$$V_{\text{TP, layer}}=4\frac{p-1}{p}BSHb$$

***

## Sequence Parallelism

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
* 其他逐元素算子。

不同 token 之间通常彼此独立，但在普通 TP 中，每个 Rank 都会重复保存和计算完整激活。

***

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

***

### SP 如何与 TP 连接

Sequence-Parallel 时，每个 rank 持有：
- 部分 序列
- 完整 hidden dimension

进入 Tensor-Parallel 前，每个 rank 需要持有完整的 $X$。

因此 SP -> TP 的过程包括一个 **AllGather**。

经过一组 Column Parallel Linear + Row Parallel Linear 运算后，每个 Rank 得到完整输出形状的 partial sum：

$$Y^{(r)}\in\mathbb{R}^{T\times H}$$

普通 TP 会执行 AllReduce，让每个 Rank 都获得完整 $Y$。

SP 则执行 **ReduceScatter**。每个 Rank 最终得到：

$$Y_r\in\mathbb{R}^{T/p\times H}$$

这样输出立即回到部分序列 + 完整 hidden dimension 的状态。


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


***

### SP + TP 的通信量

普通 TP 中，一组 Row Parallel 输出执行一次 AllReduce：

$$V_{\text{AllReduce}}=2\frac{p-1}{p}N$$

SP 将其拆成：

* 进入下一组 Column Parallel 前的 AllGather；
* 当前 Row Parallel 后的 ReduceScatter。

**因此和单纯 TP 相比，SP + TP 的通信量不变**。

**但是大通信（AllReduce）被拆成了小通信（AllGather + ReduceScatter），更容易被计算掩盖**。

***

## 通信量总结

令完整 hidden state 的数据量为：

$$N=BSHb$$

普通 TP 在 Attention Output Projection 和 MLP Down Projection 后各执行一次 Ring AllReduce：

$$V_{\text{TP, layer}}=2\times2\frac{p-1}{p}N=4\frac{p-1}{p}N$$

TP + SP 将每次 AllReduce 拆到相邻算子边界，改为一次 ReduceScatter 和下一子层前的一次 AllGather。一个 Transformer Layer 包含 Attention 与 MLP 两组：

$$V_{\text{TP+SP, layer}}=2\left(\frac{p-1}{p}N+\frac{p-1}{p}N\right)=4\frac{p-1}{p}N$$

因此，在理想 Ring 字节数模型下：

$$V_{\text{TP+SP, layer}}=V_{\text{TP, layer}}$$

SP 省下的是完整激活的复制保存和逐元素重复计算。
