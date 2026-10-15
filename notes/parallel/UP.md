---
title: "Ulysses Parallelism"
order: 3
---

# Ulysses Parallelism

**Ulysses Parallelism（UP）沿序列分片激活，在 Attention 前后用 AllToAll 转换为 head 分片、再恢复序列分片。** 每卡因此可以在自己的 heads 上复用完整序列的单卡 Attention kernel。

纯 UP 本身不切分线性层权重。若未组合其他参数分片方案，每卡仍持有完整参数，并对本地 token 执行投影、LayerNorm 和 MLP。[Ulysses 原论文](https://arxiv.org/html/2309.14509v2#S3.SS1)

## 动机：本地序列还不足以完成 Attention

线性投影与 MLP 可以独立处理不同 token，因此把序列分给多张 GPU 后，它们仍能本地执行。Attention 则依赖跨 token 的上下文：本地 Query 需要访问远端、且由 mask 允许的 K、V，只持有本地序列的 QKV 并不足够。

UP 利用另一个独立性：不同 heads 可以分别计算。将任务重新按 head 分工，让每卡收齐自己负责的 heads 的全部 token，就能直接调用本地 Attention kernel。计算结束后，再按 token 将输出交回原卡。

## 方法：在序列分片与 head 分片之间转换

下文以标准多头注意力（MHA）为例，假设序列长度与 head 数都可被并行度均分。

| 符号 | 含义 |
|---|---|
| $B,S$ | batch size 与完整序列长度 |
| $N_h,D_h$ | head 数与每个 head 的维度 |
| $H=N_hD_h$ | 隐藏维度 |
| $p$ | UP 组中的 Rank 数；图中每个 Rank 对应一张 GPU |

每卡从输入 $X_r\in\mathbb{R}^{B\times S/p\times H}$ 开始，本地投影生成 QKV。

<UlyssesLayout />

> 图取 $B=1$、$S=4$、$N_h=4$、$p=2$。每格代表一个 token 在一个 head 上的向量，省略内部的 $D_h$ 维；Q、K、V 使用相同的分发方式。

**第一次 AllToAll 按 head 分发 QKV。** 源 Rank $r$ 把本地 heads 切为 $p$ 组，第 $j$ 组交给目标 Rank $j$。目标卡沿序列维度拼接各来源的这一组，得到可用于本地 Attention 的输入。每份发送块 $Q_{r\rightarrow j}$ 的形状为 $[B,S/p,N_h/p,D_h]$，K、V 同理。

**本地 Attention 处理完整序列中的部分 heads。** 第 $j$ 卡计算：

$$
O'_j=\operatorname{Attention}(Q'_j,K'_j,V'_j).
$$

这时所需上下文已经齐全，可以复用单卡 FlashAttention 等实现。

**第二次 AllToAll 恢复输出 O 的序列分片。** 每卡按 token 将本地输出分发给原卡，目标卡沿 head 维度拼接。两次通信都属于前向过程，具体形状如下：

| 数据位置 | 每卡形状 |
|---|---|
| 第一次 AllToAll 前的 Q、K、V | $[B,S/p,N_h,D_h]$ |
| 第一次 AllToAll 后的 Q′、K′、V′ | $[B,S,N_h/p,D_h]$ |
| 本地 Attention 输出 O′ | $[B,S,N_h/p,D_h]$ |
| 第二次 AllToAll 后的 O | $[B,S/p,N_h,D_h]$ |

AllToAll 做的是**重分发与拼接**。每卡在交换前后持有的单个张量元素数相同；恢复后的 O 已具有本地 token 的完整 head 特征，可继续执行 Output Projection、Residual、LayerNorm 和 MLP。[论文第 3.1 节](https://arxiv.org/html/2309.14509v2#S3.SS1)

## 通信量：只交换分片中的非本地部分

设每个元素占 $b$ 字节，一份完整隐藏状态大小为 $N=BSHb$。以下统计**一次前向、每 Rank 的理论发送字节数**，采用均匀 AllToAll，不含发给自己的部分，也不叠加接收量。

标准 MHA 的完整 Q、K、V、O 各约为 $N$ 字节。每卡只持有各自的分片，其中 $(p-1)/p$ 需要发给其他卡：

| 通信 | 每卡参与交换的数据大小 | 发给其他卡的字节数 |
|---|---|---|
| QKV AllToAll | $3N/p$ | $3\frac{p-1}{p^2}N$ |
| O AllToAll | $N/p$ | $\frac{p-1}{p^2}N$ |

两次相加，一组 Ulysses Attention 的每卡发送量为：

$$
V_{\mathrm{UP}}=4\frac{p-1}{p^2}N.
$$

### 与 TP+SP 比较时，要对齐计算范围

令两种方案的并行度都为 $p>1$，TP+SP 按 Ring 字节数模型统计。在 [TP+SP](./TP_SP.md) 的经典实现中，Attention 与 MLP 子层各需要一组 AllGather + ReduceScatter；纯 UP 的 MLP 处理本地 token，激活通信集中在 Attention 的两次 AllToAll。

| 前向统计范围 | TP+SP 每 Rank 发送量 | UP 每 Rank 发送量 | UP / TP+SP |
|---|---|---|---|
| Attention 子层 | $2\frac{p-1}{p}N$ | $4\frac{p-1}{p^2}N$ | $2/p$ |
| Attention + MLP 整层 | $4\frac{p-1}{p}N$ | $4\frac{p-1}{p^2}N$ | $1/p$ |

TP+SP 的 AllGather 为每卡收集完整隐藏状态，UP 则收集部分 heads 的完整序列，交换后的单个张量仍只包含全局元素的 $1/p$。加上 MLP 无需激活通信，得到表中的整层比例。

**这些比例比较的是上述模型中的激活发送字节数。** TP+SP 同时切分权重，纯 UP 的权重保持复制；参数显存、训练时的参数梯度同步，以及组合 ZeRO 等方案后的通信，都需要另行计入。字节数比例不能直接解释为耗时比例。[论文通信分析](https://arxiv.org/html/2309.14509v2#S3.SS2)

## 性能限制：字节数之外还要看什么

在串行执行的 Attention 区间内，QKV AllToAll 完成后才能执行本地 Attention，得到 O 后再进行第二次 AllToAll。布局整理也可能夹在通信前后，因此实际耗时还受以下因素影响：

- **AllToAll 的带宽与尾延迟**：多目标交换的完成时间受链路拥塞和较慢 Rank 影响，不能只按总字节数推算。
- **Pack / Unpack 成本**：布局转换若需要物化，可能增加 HBM 读写、临时 buffer 和 kernel 启动。具体成本取决于布局、融合和通信实现。
- **Head 数限制**：均分 heads 的基础方案要求 $p$ 整除 $N_h$，可扩展的并行度受到限制。GQA/MQA 的 KV heads 更少，需要单独处理其分发，不能直接套用本节的 MHA 切分和通信式。

这里讨论的是通信与 Attention 区间。完整 Transformer 层的耗时还包含 QKV、Output Projection 和 MLP 等计算；是否能重叠部分通信，需要结合具体实现判断。
