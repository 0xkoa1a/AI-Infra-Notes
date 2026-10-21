---
title: "Context Parallelism"
order: 7
---

# Context Parallelism

**Context Parallelism（CP）将序列分给多个 Rank，每卡负责一部分 token 的输出。** Linear、LayerNorm 和 MLP 都能直接处理本地 token；Attention 则需要读取其他卡上由 mask 允许的 K、V。

下文介绍每卡保存完整权重的 Ring CP：**Q 留在本卡，KV 沿环逐轮传递。** [Megatron Core 的 CP 说明](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

> Megatron 的 CP 通信方式包括 Ring P2P、AllGather KV、Ulysses 式 AllToAll，以及分层组合。[通信方式定义](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html#megatron.core.transformer.transformer_config.TransformerConfig.cp_comm_type)

## 完整前向：本地 token、完整权重

设输入 $X$ 为 $[B,S,H]$，序列均分到 $p$ 个 Rank。下文使用以下符号：

| 符号 | 含义 |
|---|---|
| $B,S$ | batch size 与完整序列长度 |
| $H$ | 隐藏维度 |
| $N_h,N_{kv}$ | Query head 数与 KV head 数 |
| $D_h$ | 每个 head 的维度，$H=N_hD_h$ |
| $p$ | CP 组中的 Rank 数；图中每个 Rank 对应一张 GPU |

MLP 先将特征从 $H$ 扩展到中间维度 $I$，经过逐元素激活，再投影回 $H$。这些操作只混合同一 token 内的特征，因此每卡可以独立处理自己的 $S/p$ 个 token。

<ContextParallelFlow kind="mlp" />

> 两张前向图均取 $B=1$、$S=4$、$H=4$、$p=2$；MLP 取 $I=8$。Attention 使用 $N_h=N_{kv}=4$、$D_h=1$ 的 MHA，并先画无因果 mask 的情形。矩阵行对应 token，列对应特征或 heads。

LayerNorm 读取一个 token 的完整 $H$ 个特征，Residual 对同一 token 的两路输入逐元素相加，也都能在本卡完成。

> 训练时，每卡从本地 token 得到权重梯度贡献，再在 CP 组内求和。[权重与梯度的分布](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html)

**Attention 的跨卡依赖来自 KV。** 每个本地 Query 需要与远端 KV 计算 Attention，Ring CP 在计算过程中逐块取得这些 KV：

<ContextParallelFlow kind="attention" />

图中 $A$ 是尚未归一化的加权输出，$m,l$ 用于跨块归一化，下一节解释它们的更新方式。一般维度下，各卡的激活形状为：

| 激活 | 每 Rank 形状 |
|---|---|
| 子层输入 X、输出 Y | $[B,S/p,H]$ |
| 本地 Q、Attention 输出 O | $[B,S/p,N_h,D_h]$ |
| 每个 K、V 块 | $[B,S/p,N_{kv},D_h]$ |

O 的 head 维展平后为 $H$，可直接交给本地 O Projection。标准 MHA 中 $N_{kv}=N_h$；GQA/MQA 通过共享 KV heads，使 $N_{kv}<N_h$。

> [TP+SP](./TP_SP.md) 与 [UP](./UP.md) 在 Attention 内按 head 分工，每卡处理部分 heads 的完整序列；Ring CP 按 Query token 分工，每卡通过逐轮接收 KV，完成本地 token 的全部 heads。

## Ring Attention

$p$ 个 Rank 组成一个逻辑 Ring。Rank $r$ 将当前 KV 块发送给 Rank $(r+1)\bmod p$，同时从 Rank $(r-1)\bmod p$ 接收新的 KV 块。

第 0 轮使用本地 KV，随后经过 $p-1$ 次传递，遍历其他 Rank 的 KV。以 $p=4$ 为例：

| 当前参与计算的 KV | Rank 0 | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|---|
| Round 0 | KV₀ | KV₁ | KV₂ | KV₃ |
| Round 1 | KV₃ | KV₀ | KV₁ | KV₂ |
| Round 2 | KV₂ | KV₃ | KV₀ | KV₁ |
| Round 3 | KV₁ | KV₂ | KV₃ | KV₀ |

KV 分块到达，使实现可以复用当前与下一轮 KV 的缓冲。[Ring Attention 原论文](https://arxiv.org/html/2310.01889v4#S3)

**跨 KV 块的 Online Softmax**

**同一个 Query 对全部可见 KV 的分数需要共同归一化。** 分块计算时，各块必须累计同一个归一化分母，以及尚未除以分母的加权输出，才能得到完整 Attention 的结果。

固定一个本地 Query 块 $Q_i$。每轮收到 $K_j,V_j$ 后，计算当前分数块：

$$S_{ij}=\frac{Q_iK_j^\top}{\sqrt{D_h}}$$

对于每个 Query 行、每个 head，维护三个状态：

- $m_i$：已处理分数的最大值。指数计算先减去它，以保持数值稳定。
- $l_i$：这些指数权重之和，即归一化分母。
- $A_i$：用这些权重对 V 加权求和，得到长度为 $D_h$ 的向量。

初始状态为 $m_i=-\infty$、$l_i=0$、$A_i=0$。若当前块的行最大值为 $\widehat m_{ij}$，合并后的最大值为：

$$m_i'=\max(m_i,\widehat m_{ij})$$

最大值改变后，旧状态要乘以 $e^{m_i-m_i'}$，才能与按新最大值计算的当前块使用同一尺度。随后将两部分相加，其中 $k$ 遍历当前 KV 块中的可见 token：

$$l_i'=e^{m_i-m_i'}l_i+\sum_k e^{S_{ij,k}-m_i'}$$

$$A_i'=e^{m_i-m_i'}A_i+\sum_k e^{S_{ij,k}-m_i'}V_{j,k}$$

处理完全部 KV 块后：

$$O_i=\frac{A_i}{l_i}$$

FlashAttention 在单卡内对不同 KV tile 执行同样的 Online Softmax；Ring Attention 将它用于跨卡送达的 KV 块。

对于 Causal Attention，mask 依据 Query 与 Key 的原始全局位置生成，只有当前位置及历史位置参与更新。若某个 Query 行在当前块中没有可见位置，就保留它原有的 $m_i,l_i,A_i$。

## Zig-Zag Ring Attention

对于 Causal Attention，较早的 Query 只能访问少量历史 KV，较晚的 Query 能访问更多位置。因此，即使各卡 token 数相同，连续分片也会把更多有效 Attention 计算交给靠后的 Rank。

Zig-Zag 将序列均分为 $2p$ 个连续块 $C_0,\ldots,C_{2p-1}$，让 Rank $r$ 同时持有一块较早的序列和一块较晚的序列：

$$\operatorname{Rank}(r)=\{C_r,C_{2p-1-r}\}$$

下图对照两种分配方式。[Zig-Zag 实现中的分片说明](https://github.com/zhuzilin/ring-flash-attention/issues/2)

<ContextParallelBalance />

$C_t$ 中的 Query 可以访问前面的 $t$ 个完整 KV 块，以及本块内的下三角区域，其有效计算量约为 $t+1/2$ 个完整块的计算面积。

Rank $r$ 的两个 Query 块的总计算量为：

$$\left(r+\frac{1}{2}\right)+\left(2p-1-r+\frac{1}{2}\right)=2p$$

Attention mask 和位置编码使用 token 的原始全局位置，保持原序列的计算关系。

> 图中按固定长度序列分配计算。变长样本组合后，各卡的有效 token 数和 Attention 计算量还会受样本长度、Padding 与 mask 影响，进一步的动态分配见 [Dynamic Context Parallelism](./Dynamic-CP.md)。

## Ring CP 的通信量与通信计算重叠

设每个元素占 $b$ 字节，完整隐藏状态大小为 $N=BSHb$，完整 K 或 V 的大小为：

$$N_{KV}=BSN_{kv}D_hb$$

以下采用均匀 Ring 传递模型，统计**一次前向中，每 Rank 发往其他 Rank 的激活字节数**。每卡的一个 KV 块包含 K、V 两个张量，共 $2N_{KV}/p$ 字节，经过 $p-1$ 轮转发：

$$V_{\text{CP,fwd}}=(p-1)\cdot2\frac{N_{KV}}{p}
=2\frac{p-1}{p}N_{KV}$$

Zig-Zag 将每卡的 token 拆为早晚两块，保持每卡 KV 总大小与传递轮数，因此发送量相同。

Ring CP 只传输 K 和 V，因此 GQA 和 MQA 可以通过减小 $N_{kv}$ 直接降低通信量。标准 MHA 中 $N_{KV}=N=BSHb$，此时：

$$V_{\text{CP,fwd}}=2\frac{p-1}{p}N$$

在相同并行度和标准 MHA 假设下，UP 的每 Rank 前向发送量为：

$$V_{\text{UP,fwd}}=4\frac{p-1}{p^2}N$$

两者之比为：

$$\frac{V_{\text{CP,fwd}}}{V_{\text{UP,fwd}}}=\frac{p}{2}$$

UP 每次交换各卡持有的激活分片，Ring CP 则将一个 KV 块连续转发 $p-1$ 轮。这种通信方式的差异带来了上面的字节数比例。

UP 需要按 head 分配任务，并承担 AllToAll 的布局重排与尾延迟；Ring CP 可以按序列长度选择并行度，其通信耗时主要取决于逐轮 P2P 的传输与等待。

**双缓冲让当前块计算与下一块传输并行。** 一块缓冲供 Attention 读取，另一块接收下一轮 KV；若下一块在当前计算结束前到达，就能直接进入下一轮，否则 Attention 要等待传输完成。

记第 $t$ 轮的计算和传输耗时分别为 $T_{\text{attention},t}$、$T_{\text{P2P},t}$，$T_{\text{tail}}$ 为额外同步与流水线收尾开销。一次 Ring Attention 的耗时可粗略表示为每轮计算与通信中较长的一项，再加上最后一块计算与收尾：

$$T_{\text{ring}}\approx
\sum_{t=0}^{p-2}\max\left(T_{\text{attention},t},T_{\text{P2P},t}\right)
+T_{\text{attention},p-1}
+T_{\text{tail}}$$

**卡数增加时，每卡计算量下降，发送量却逐渐接近固定值 $2N_{KV}$，通信因此越来越难被计算覆盖。** 同时，$p-1$ 轮 P2P 增加了逐轮启动与同步的开销；本地 Query 块过小，也会降低 Attention kernel 的利用率。

Ring 的推进还受慢链路影响。相邻 Rank 跨越低带宽网络，或多个传输竞争同一链路时，等待会增长。Rank 排布需要结合 NVLink、NVSwitch 和节点间网络拓扑安排。
