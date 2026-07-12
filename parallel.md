- [为什么推理系统需要并行](#为什么推理系统需要并行)
- [推理工作负载的基本特征](#推理工作负载的基本特征)
        - [Transformer 中的主要张量](#transformer-中的主要张量)
        - [KV Cache 显存](#kv-cache-显存)
        - [Prefill 与 Decode](#prefill-与-decode)
                - [Prefill](#prefill)
                - [Decode](#decode)
- [分布式通信基础](#分布式通信基础)
        - [Rank、Process Group 与 Collective](#rankprocess-group-与-collective)
        - [通信成本模型](#通信成本模型)
                - [小消息与大消息](#小消息与大消息)
        - [如何定义“通信量”](#如何定义通信量)
        - [Reduce：什么是归约](#reduce什么是归约)
        - [Gather 与 Scatter](#gather-与-scatter)
                - [Gather](#gather)
                - [Scatter](#scatter)
        - [AllGather](#allgather)
                - [AllGather 的语义](#allgather-的语义)
                - [Ring AllGather 的执行过程](#ring-allgather-的执行过程)
                - [AllGather 通信量](#allgather-通信量)
                - [AllGather 为什么不是发送 $(p-1)N/p$ 给每一个 Rank](#allgather-为什么不是发送-p-1np-给每一个-rank)
                - [AllGather 的常见用途](#allgather-的常见用途)
        - [ReduceScatter](#reducescatter)
                - [ReduceScatter 的语义](#reducescatter-的语义)
                - [将输入切成多个 Chunk](#将输入切成多个-chunk)
                - [Ring ReduceScatter 的执行过程](#ring-reducescatter-的执行过程)
                - [ReduceScatter 通信量](#reducescatter-通信量)
                - [ReduceScatter 的常见用途](#reducescatter-的常见用途)
        - [AllReduce](#allreduce)
                - [AllReduce 的语义](#allreduce-的语义)
                - [为什么 AllReduce 等于 ReduceScatter 加 AllGather](#为什么-allreduce-等于-reducescatter-加-allgather)
                - [为什么可以分 Chunk 归约](#为什么可以分-chunk-归约)
                - [Ring AllReduce 的通信量](#ring-allreduce-的通信量)
                - [Ring AllReduce 的时间模型](#ring-allreduce-的时间模型)
                - [AllReduce 在 TP 中的作用](#allreduce-在-tp-中的作用)
        - [AllToAll](#alltoall)
                - [AllToAll 的语义](#alltoall-的语义)
                - [AllToAll 与 AllGather 的区别](#alltoall-与-allgather-的区别)
                - [AllToAll 通信量](#alltoall-通信量)
                - [AllToAll 在 EP 中的作用](#alltoall-在-ep-中的作用)
        - [Point-to-Point](#point-to-point)
                - [基本语义](#基本语义)
                - [P2P 在 PP 中的作用](#p2p-在-pp-中的作用)
                - [P2P 的同步问题](#p2p-的同步问题)
        - [几种通信原语的统一比较](#几种通信原语的统一比较)
                - [AllGather](#allgather)
                - [ReduceScatter](#reducescatter)
                - [AllReduce](#allreduce)
                - [AllToAll](#alltoall)
                - [Point-to-Point](#point-to-point)
        - [为什么相同通信量不代表相同性能](#为什么相同通信量不代表相同性能)
                - [通信拓扑不同](#通信拓扑不同)
                - [消息切分不同](#消息切分不同)
                - [是否需要同步](#是否需要同步)
                - [是否存在负载不均衡](#是否存在负载不均衡)
                - [是否能与计算重叠](#是否能与计算重叠)
        - [小结](#小结)
- [Data Parallelism](#data-parallelism)
        - [基本原理](#基本原理)
        - [DP 的优点](#dp-的优点)
        - [DP 的局限](#dp-的局限)
        - [Continuous Batching](#continuous-batching)
        - [Chunked Prefill](#chunked-prefill)
- [Tensor Parallelism](#tensor-parallelism)
        - [TP 在并行体系中的位置](#tp-在并行体系中的位置)
        - [统一符号](#统一符号)
        - [为什么线性层可以切分](#为什么线性层可以切分)
        - [Column Parallel Linear](#column-parallel-linear)
                - [切分方式](#切分方式)
                - [为什么 Column Parallel 后可以不通信](#为什么-column-parallel-后可以不通信)
                - [什么时候需要 AllGather](#什么时候需要-allgather)
        - [Row Parallel Linear](#row-parallel-linear)
                - [切分方式](#切分方式)
                - [为什么需要归约](#为什么需要归约)
                - [Row Parallel 的 AllReduce 通信量](#row-parallel-的-allreduce-通信量)
        - [为什么要把 Column Parallel 和 Row Parallel 配对](#为什么要把-column-parallel-和-row-parallel-配对)
        - [Attention 中的 TP](#attention-中的-tp)
                - [QKV Projection](#qkv-projection)
                - [本地 Attention](#本地-attention)
                - [Output Projection](#output-projection)
        - [MLP 中的 TP](#mlp-中的-tp)
        - [每个 Transformer Layer 中有多少次 TP 通信](#每个-transformer-layer-中有多少次-tp-通信)
        - [TP 通信量与张量形状](#tp-通信量与张量形状)
        - [TP 与 Prefill、Decode 的差异](#tp-与-prefilldecode-的差异)
                - [Prefill](#prefill-1)
                - [Decode](#decode-1)
        - [TP 与 MHA、GQA、MQA](#tp-与-mhagqamqa)
        - [Vocab Parallelism](#vocab-parallelism)
                - [输入 Embedding](#输入-embedding)
                - [输出 LM Head](#输出-lm-head)
        - [TP 的硬件拓扑要求](#tp-的硬件拓扑要求)
        - [TP 的性能判断](#tp-的性能判断)
- [Pipeline Parallelism](#pipeline-parallelism)
        - [基本原理](#基本原理)
        - [PP 的显存特点](#pp-的显存特点)
        - [Pipeline Bubble](#pipeline-bubble)
        - [PP 对单 Token 延迟的影响](#pp-对单-token-延迟的影响)
        - [PP 的负载均衡](#pp-的负载均衡)
- [Sequence Parallelism](#sequence-parallelism)
        - [术语范围](#术语范围)
        - [为什么普通 TP 仍然存在激活复制](#为什么普通-tp-仍然存在激活复制)
        - [SP 的核心思想](#sp-的核心思想)
        - [SP 如何与 TP 连接](#sp-如何与-tp-连接)
                - [Sequence-Parallel Layout](#sequence-parallel-layout)
                - [Tensor-Parallel Linear 所需布局](#tensor-parallel-linear-所需布局)
        - [SP 中的 AllGather](#sp-中的-allgather)
        - [SP 中的 ReduceScatter](#sp-中的-reducescatter)
        - [一组 SP + TP 的完整过程](#一组-sp--tp-的完整过程)
        - [为什么 SP 不会增加理论总通信量](#为什么-sp-不会增加理论总通信量)
        - [SP 的真正收益](#sp-的真正收益)
        - [SP 为什么在训练中更重要](#sp-为什么在训练中更重要)
        - [SP 与 CP 的区别](#sp-与-cp-的区别)
- [Context Parallelism](#context-parallelism)
        - [基本原理](#基本原理)
        - [CP 与 SP 的区别](#cp-与-sp-的区别)
- [Ulysses Parallelism](#ulysses-parallelism)
        - [UP 的目标](#up-的目标)
        - [UP 与普通 SP 的区别](#up-与普通-sp-的区别)
        - [初始张量布局](#初始张量布局)
        - [为什么这种布局不能直接独立算完整 Attention](#为什么这种布局不能直接独立算完整-attention)
        - [Ulysses 的 AllToAll 布局转换](#ulysses-的-alltoall-布局转换)
        - [为什么 AllToAll 能实现这个转换](#为什么-alltoall-能实现这个转换)
        - [本地 Attention](#本地-attention)
        - [反向 AllToAll](#反向-alltoall)
        - [UP 的通信量](#up-的通信量)
        - [UP 的通信成本不仅由字节数决定](#up-的通信成本不仅由字节数决定)
        - [UP 的 Head 数限制](#up-的-head-数限制)
        - [UP 与 GQA、MQA](#up-与-gqamqa)
        - [UP 为什么更适合 Prefill](#up-为什么更适合-prefill)
        - [UP 与 Ring Attention 的比较](#up-与-ring-attention-的比较)
- [Ring Attention](#ring-attention)
        - [基本思想](#基本思想)
        - [Online Softmax](#online-softmax)
        - [Ring Attention 的优缺点](#ring-attention-的优缺点)
        - [Ulysses 与 Ring Attention 的区别](#ulysses-与-ring-attention-的区别)
- [Decode 阶段的 Context Parallelism](#decode-阶段的-context-parallelism)
- [Expert Parallelism](#expert-parallelism)
        - [EP 的背景：为什么 MoE 需要不同的并行方式](#ep-的背景为什么-moe-需要不同的并行方式)
        - [MoE Layer 的整体流程](#moe-layer-的整体流程)
        - [Router](#router)
        - [为什么需要 Token Permute](#为什么需要-token-permute)
        - [Dispatch AllToAll](#dispatch-alltoall)
        - [为什么 MoE 通常需要 AllToAllV](#为什么-moe-通常需要-alltoallv)
        - [EP Dispatch 通信量](#ep-dispatch-通信量)
        - [非均匀路由下的通信量](#非均匀路由下的通信量)
        - [本地 Expert 分组](#本地-expert-分组)
        - [Grouped GEMM](#grouped-gemm)
        - [Combine AllToAll](#combine-alltoall)
        - [EP 的两次通信为什么通常无法省掉](#ep-的两次通信为什么通常无法省掉)
        - [EP 与 TP 的区别](#ep-与-tp-的区别)
                - [TP](#tp)
                - [EP](#ep)
        - [Expert Tensor Parallelism](#expert-tensor-parallelism)
        - [EP 的负载不均衡](#ep-的负载不均衡)
        - [Capacity 与 Token Drop](#capacity-与-token-drop)
        - [Expert Replication](#expert-replication)
        - [Expert Placement](#expert-placement)
        - [Hierarchical AllToAll](#hierarchical-alltoall)
        - [Decode 阶段为什么对 EP 特别困难](#decode-阶段为什么对-ep-特别困难)
        - [Attention DP + Expert EP](#attention-dp--expert-ep)
        - [EP 的通信与计算重叠](#ep-的通信与计算重叠)
        - [EP 性能分析](#ep-性能分析)
        - [TP、SP、UP、EP 的统一比较](#tpspupep-的统一比较)
                - [切分对象](#切分对象)
                - [主要通信](#主要通信)
                - [通信数据的性质](#通信数据的性质)
                - [主要目标](#主要目标)
                - [对推理阶段的适用性](#对推理阶段的适用性)
                - [Prefill](#prefill-2)
                - [Decode](#decode-2)
                - [最核心的判断原则](#最核心的判断原则)


# 为什么推理系统需要并行

AI 模型推理使用并行技术，主要是为了解决四类问题：

1. **模型容量问题**：模型权重无法放入单张 GPU。
2. **运行时显存问题**：KV Cache、中间激活和临时 Workspace 无法放入单张 GPU。
3. **单请求延迟问题**：希望多张 GPU 同时计算一个请求，降低 TTFT 或 TPOT。
4. **系统吞吐问题**：希望同时服务更多请求，提高每秒生成的 token 数。

这些目标并不完全一致。

例如：

* Data Parallelism 可以提高总吞吐量，但通常不会降低单请求延迟。
* Tensor Parallelism 可以让多张 GPU 同时计算一个请求，但会引入高频通信。
* Pipeline Parallelism 可以让超大模型跨节点部署，但对单个 token 的延迟帮助有限。
* Context Parallelism 可以支持更长的上下文，但需要额外传输或归约 Attention 中间结果。

因此，选择并行方案时，不能只问“模型怎么切”，还要考虑：

* 请求怎么切；
* 模型参数怎么切；
* token 怎么切；
* KV Cache 怎么切；
* 通信发生在哪里；
* 通信能否和计算重叠；
* 优化目标是延迟、吞吐，还是容量。

---

# 推理工作负载的基本特征

## Transformer 中的主要张量

设 Transformer 的隐藏状态为：

$$
X\in\mathbb{R}^{B\times S\times H}
$$

其中：

* $B$：batch size；
* $S$：序列长度；
* $H$：hidden size；
* $L$：Transformer 层数；
* $N_q$：Query Head 数量；
* $N_{kv}$：KV Head 数量；
* $D_h$：每个 Attention Head 的维度。

通常有：

$$
H=N_qD_h
$$

对于标准 Multi-Head Attention，通常有 $N_q=N_{kv}$；对于 GQA 和 MQA，则有 $N_{kv}<N_q$。

---

## KV Cache 显存

Decoder-only Transformer 在每一层都要保存历史 token 的 Key 和 Value。

忽略 Padding、内存对齐和分页碎片后，KV Cache 大小可以近似表示为：

$$
M_{\text{KV}}=2LBSN_{kv}D_h\cdot b
$$

其中 $b$ 是每个元素占用的字节数，系数 $2$ 来自 Key 和 Value。

KV Cache 会随以下变量线性增长：

* 并发请求数 $B$；
* 上下文长度 $S$；
* 模型层数 $L$；
* KV Head 数量 $N_{kv}$；
* 数据精度。

因此，即使模型权重能够放入单张 GPU，长上下文和高并发产生的 KV Cache 也可能迫使系统使用多卡。

---

## Prefill 与 Decode

LLM 推理一般分为 Prefill 和 Decode 两个阶段。

### Prefill

Prefill 一次处理 Prompt 中的所有 token，输入通常为：

$$
X\in\mathbb{R}^{B\times S\times H}
$$

Prefill 中的矩阵乘法通常较大，具有较高的算术强度和并行度，比较容易充分利用 GPU Tensor Core。

Prefill 主要影响指标是 TTFT，即 Time To First Token。

### Decode

Decode 每一步只为每个请求处理一个新 token，输入通常为：

$$
X\in\mathbb{R}^{B\times 1\times H}
$$

但 Attention 需要读取之前所有 token 的 KV Cache。随着上下文长度增加，每一步需要读取的 KV Cache 也随之增加。

Decode 的主要性能指标包括：

* TPOT：Time Per Output Token；
* ITL：Inter-Token Latency；
* 每秒生成 token 数。

Prefill 通常更偏向 compute-bound，而 Decode 更容易受到以下因素限制：

* 权重读取带宽；
* KV Cache 读取带宽；
* kernel launch latency；
* GPU 间通信延迟；
* batch size 太小导致的 GPU 利用率不足。

因此，一种并行策略可能非常适合 Prefill，却不适合 Decode。

# 分布式通信基础

模型并行最终都需要依赖 GPU 之间的数据交换。不同并行策略的主要区别之一，就是：

* 需要传输什么数据；
* 每次传输多少数据；
* 多久通信一次；
* 哪些 GPU 需要参与；
* 通信是否位于计算的关键路径上。

例如：

* TP（Tensor Parallelism，张量并行）通常需要高频 AllReduce、AllGather 或 ReduceScatter；
* PP（Pipeline Parallelism，流水线并行）通常使用相邻 Stage 之间的 Point-to-Point 通信；
* EP（Expert Parallelism，专家并行）通常依赖 AllToAll；
* CP（Context Parallelism，上下文并行）可能依赖 AllGather、AllToAll、Ring Point-to-Point 或分布式归约。

理解这些通信原语，是理解模型并行性能的基础。

---

## Rank、Process Group 与 Collective

在分布式程序中，每个 GPU 通常由一个独立进程控制，这个进程在通信组中的编号称为 Rank。

假设一个通信组中有 $p$ 张 GPU：

```text
Rank 0 → GPU 0
Rank 1 → GPU 1
...
Rank p-1 → GPU p-1
```

参与同一次通信的一组 Rank 称为 Process Group，即进程组。

Collective Communication，中文通常称为集体通信，是指通信组中的多个 Rank 共同参与一个具有整体语义的通信操作，例如：

* AllReduce；
* AllGather；
* ReduceScatter；
* AllToAll；
* Broadcast。

集体通信描述的是“最终每个 Rank 应该得到什么数据”，并不严格限定底层必须使用哪种算法实现。

例如，AllReduce 可以使用：

* Ring Algorithm；
* Tree Algorithm；
* Recursive Doubling；
* Hierarchical Algorithm；
* NVLink、NVSwitch 和跨节点网络组成的分层算法。

因此，需要区分两个层次：

1. **通信原语的语义**：最终数据如何分布；
2. **通信算法的实现**：数据具体沿什么路径传输。

---

## 通信成本模型

最简单的通信成本模型是：

$$
T_{\text{comm}}=\alpha+\beta N
$$

其中：

* $\alpha$：Latency Term，即一次通信的固定启动延迟；
* $\beta$：每字节传输时间，可以近似理解为带宽的倒数；
* $N$：传输的数据量。

如果有效带宽为 $BW$，则可以近似认为：

$$
\beta\approx\frac{1}{BW}
$$

于是：

$$T_{\text{comm}}\approx\alpha+\frac{N}{BW}$$

但一个 Collective 通常不只包含一步通信。如果算法需要执行 $s$ 个通信步骤，更准确的近似是：

$$T_{\text{comm}}\approxs\alpha+\beta V$$

其中 $V$ 是每个 Rank 在整个 Collective 中发送的数据总量。

这两个部分分别对应：

* $s\alpha$：多轮同步和通信启动带来的延迟；
* $\beta V$：实际传输数据带来的带宽开销。

### 小消息与大消息

当通信量很小时：

$$
s\alpha\gg\beta V
$$

通信主要受启动延迟限制，即 latency-bound。

当通信量很大时：

$$
\beta V\gg s\alpha
$$

通信主要受链路带宽限制，即 bandwidth-bound。

Decode 阶段尤其容易受到 $\alpha$ 的影响，因为每一步 Decode 的计算量较小，但每一层都可能发起一次或多次 Collective。即使每次传输的数据不多，几十层模型累积的通信启动延迟也可能非常明显。

---

## 如何定义“通信量”

在讨论通信量之前，必须明确采用哪一种统计口径。

本文主要采用：

> 每个 Rank 在整个 Collective 中发送到网络的数据量。

在对称通信中，每个 Rank 的接收量通常与发送量相同。

假设每个 Rank 发送 $V$ 字节，则：

* 每 Rank 发送量：$V$；
* 每 Rank 接收量：通常也是 $V$；
* 整个集群的总发送量：$pV$；
* 如果把发送量和接收量同时相加，则是 $2pV$。

不同资料可能采用不同口径。例如，某些资料所说的“通信量”是每 Rank 发送量，另一些资料则计算发送与接收总和。因此比较公式时，必须先确认其定义。

本文默认：

$$
\text{通信量}=\text{每个 Rank 的发送量}
$$

同时，数据在本地 GPU 内部进行的复制不计入网络通信量。

---

## Reduce：什么是归约

Reduce，即归约，是把多个 Rank 上对应位置的数据按照某个运算符合并。

假设有 $p$ 个 Rank，第 $r$ 个 Rank 持有张量：

$$
X^{(r)}\in\mathbb{R}^N
$$

使用求和归约后得到：

$$
Y=\sum_{r=0}^{p-1}X^{(r)}
$$

这是逐元素求和，即：

$$
Y_j=\sum_{r=0}^{p-1}X_j^{(r)}
$$

除了求和，常见归约操作还包括：

* Maximum；
* Minimum；
* Product；
* Bitwise AND；
* Bitwise OR。

归约操作通常要求运算符具有结合性。这样底层算法才能改变计算顺序，例如先局部求和，再把局部结果继续合并。

在浮点数计算中，加法并不严格满足数学上的结合律：

$$
(a+b)+c\neq a+(b+c)
$$

因此，不同通信算法、不同 Rank 数量或不同归约顺序，可能产生很小的浮点误差差异。

---

## Gather 与 Scatter

理解 AllGather 和 ReduceScatter 之前，先理解 Gather 与 Scatter。

### Gather

Gather 将多个 Rank 持有的数据收集到一个指定 Root Rank。

假设每个 Rank 持有一个分片：

$$
X_0,X_1,\ldots,X_{p-1}
$$

Gather 到 Rank 0 后：

```text
Rank 0：[X0, X1, ..., Xp-1]
Rank 1：X1
Rank 2：X2
...
```

只有 Root Rank 得到完整结果。

### Scatter

Scatter 是 Gather 的反向过程。

Root Rank 开始时持有完整张量：

$$
X=[X_0,X_1,\ldots,X_{p-1}]
$$

Scatter 后：

```text
Rank 0：X0
Rank 1：X1
...
Rank p-1：Xp-1
```

每个 Rank 只得到其中一个分片。

Gather 和 Scatter 都存在一个特殊 Root Rank，因此可能产生 Root 端的通信瓶颈。AllGather 和 ReduceScatter 则通常采用更加对称的算法实现。

---

## AllGather

### AllGather 的语义

AllGather 可以理解为：

> 每个 Rank 都把自己的分片分享给所有其他 Rank，最终所有 Rank 都得到完整张量。

假设完整张量大小为 $N$ 字节，并被均匀切分为 $p$ 个分片：

$$
X=[X_0,X_1,\ldots,X_{p-1}]
$$

每个分片大小为：

$$
|X_i|=\frac{N}{p}
$$

初始状态为：

```text
Rank 0：X0
Rank 1：X1
Rank 2：X2
...
Rank p-1：Xp-1
```

执行 AllGather 后：

```text
Rank 0：[X0, X1, ..., Xp-1]
Rank 1：[X0, X1, ..., Xp-1]
...
Rank p-1：[X0, X1, ..., Xp-1]
```

所有 Rank 都得到大小为 $N$ 的完整张量。

---

### Ring AllGather 的执行过程

以 $p=4$ 为例：

```text
初始：
Rank 0：X0
Rank 1：X1
Rank 2：X2
Rank 3：X3
```

将 Rank 连接成一个逻辑环：

```text
Rank 0 → Rank 1 → Rank 2 → Rank 3 → Rank 0
```

第一轮，每个 Rank 向下一个 Rank 发送自己持有的一个分片：

```text
Rank 0 发送 X0 给 Rank 1
Rank 1 发送 X1 给 Rank 2
Rank 2 发送 X2 给 Rank 3
Rank 3 发送 X3 给 Rank 0
```

第二轮，每个 Rank 继续转发上一轮刚收到的分片。

第三轮继续转发。

经过 $p-1=3$ 轮后，每个 Rank 都收到其他三个 Rank 的分片，从而拥有完整张量。

---

### AllGather 通信量

每个分片大小为：

$$
\frac{N}{p}
$$

Ring AllGather 一共需要 $p-1$ 轮。

每一轮中，每个 Rank 发送一个分片，因此每个 Rank 的总发送量为：

$$
V_{\text{AllGather}}=(p-1)\frac{N}{p}
$$

即：

$$
V_{\text{AllGather}}=\frac{p-1}{p}N
$$

当 $p$ 很大时：

$$
\frac{p-1}{p}N\approx N
$$

也就是说，为了让每个 Rank 从大小为 $N/p$ 的本地分片恢复出大小为 $N$ 的完整张量，每个 Rank 大约需要发送和接收 $N$ 字节的数据。

Ring AllGather 的时间可以近似表示为：

$$
T_{\text{AllGather}}\approx(p-1)\alpha+\frac{p-1}{p}N\beta
$$

---

### AllGather 为什么不是发送 $(p-1)N/p$ 给每一个 Rank

一个容易产生的误解是：每个 Rank 要把自己的分片发送给另外 $p-1$ 个 Rank，因此通信量似乎应该更大。

在最朴素的直接发送算法中，确实可以让 Rank 0 分别把 $X_0$ 发送给其他所有 Rank。但 Ring AllGather 会让数据在 Rank 之间逐跳转发。

Rank 0 不需要亲自把 $X_0$ 发送给所有 Rank。它只需要发送给下一个 Rank，后续 Rank 会继续转发。

因此，每个 Rank 每轮只发送一个大小为 $N/p$ 的分片，执行 $p-1$ 轮，总发送量为：

$$
\frac{p-1}{p}N
$$

这体现了 Collective Algorithm 对通信路径的优化。

---

### AllGather 的常见用途

AllGather 常用于：

* 收集沿 Hidden Dimension 切分的激活；
* Sequence Parallelism 中恢复完整序列；
* Ulysses 等分布式 Attention 中转换张量布局；
* 收集分片的 Logits；
* 某些 Tensor Parallel Linear 的输入重构。

---

## ReduceScatter

### ReduceScatter 的语义

ReduceScatter 可以理解为：

> 先把所有 Rank 上的完整张量进行逐元素归约，再把归约结果均匀切分给各个 Rank。

假设每个 Rank 都持有一个大小为 $N$ 的张量：

$$
X^{(0)},X^{(1)},\ldots,X^{(p-1)}
$$

先进行求和归约：

$$
Y=\sum_{r=0}^{p-1}X^{(r)}
$$

再把 $Y$ 切成 $p$ 个分片：

$$
Y=[Y_0,Y_1,\ldots,Y_{p-1}]
$$

其中：

$$
|Y_i|=\frac{N}{p}
$$

最终状态为：

```text
Rank 0：Y0
Rank 1：Y1
...
Rank p-1：Yp-1
```

需要注意，ReduceScatter 的语义等价于“Reduce 后再 Scatter”，但高性能实现不会真的先在某个 Rank 上构造完整归约结果，再执行 Scatter。

底层算法会把归约与分片传输融合起来。

---

### 将输入切成多个 Chunk

为了理解 Ring ReduceScatter，把每个 Rank 的输入张量切成 $p$ 个 Chunk：

$$
X^{(r)}=\left[X_0^{(r)},X_1^{(r)},\ldots,X_{p-1}^{(r)}\right]
$$

其中 $r$ 表示数据来自哪个 Rank，下标 $j$ 表示张量的第几个 Chunk。

最终 Rank $j$ 应该持有：

$$
Y_j=\sum_{r=0}^{p-1}X_j^{(r)}
$$

也就是说：

* Rank 0 最终负责第 0 个 Chunk 的完整归约结果；
* Rank 1 最终负责第 1 个 Chunk；
* 以此类推。

---

### Ring ReduceScatter 的执行过程

在 Ring ReduceScatter 中，各 Rank 沿环发送 Chunk。每收到一个 Chunk，就把它与本地对应 Chunk 相加，然后继续向下一个 Rank 转发。

以某一个 Chunk 为例，它会依次经过不同 Rank：

```text
原始 Chunk
→ 与 Rank A 的对应 Chunk 相加
→ 与 Rank B 的对应 Chunk 相加
→ 与 Rank C 的对应 Chunk 相加
→ 到达最终负责该 Chunk 的 Rank
```

经过 $p-1$ 轮后：

* 每个 Chunk 都完成了来自所有 $p$ 个 Rank 的归约；
* 不同 Chunk 分别停留在不同 Rank；
* 每个 Rank 最终只保存一个大小为 $N/p$ 的归约后 Chunk。

---

### ReduceScatter 通信量

每一轮中，每个 Rank 发送一个大小为 $N/p$ 的 Chunk。

总轮数为 $p-1$，因此每个 Rank 的总发送量为：

$$
V_{\text{ReduceScatter}}=(p-1)\frac{N}{p}
$$

即：

$$
V_{\text{ReduceScatter}}=\frac{p-1}{p}N
$$

时间可以近似表示为：

$$
T_{\text{ReduceScatter}}\approx(p-1)\alpha+\frac{p-1}{p}N\beta
$$

这与 Ring AllGather 的通信量相同。

但二者执行的操作不同：

* AllGather 只转发和收集数据，不进行归约；
* ReduceScatter 在数据传输过程中还要执行逐元素归约。

通常来说，GPU 上的加法计算相对于跨 GPU 数据传输成本较低，因此整体时间仍主要由通信决定。

---

### ReduceScatter 的常见用途

ReduceScatter 常用于：

* Tensor Parallelism 中的 Row Parallel Linear；
* Sequence Parallelism；
* 将 AllReduce 拆成更容易与后续计算衔接的形式；
* 只需要归约结果分片，而不需要每个 Rank 保存完整结果的场景。

例如，Row Parallel Linear 产生的局部结果为：

$$
Y^{(r)}=X_rW_r
$$

完整结果为：

$$
Y=\sum_{r=0}^{p-1}Y^{(r)}
$$

如果后续算子沿序列维度分片执行，就不需要让每个 Rank 得到完整 $Y$，可以直接执行 ReduceScatter，使每个 Rank 只保留一部分归约结果。

---

## AllReduce

### AllReduce 的语义

AllReduce 可以理解为：

> 对所有 Rank 的张量进行归约，并让所有 Rank 都得到完整的归约结果。

假设每个 Rank 持有：

$$
X^{(r)}\in\mathbb{R}^N
$$

求和 AllReduce 的输出为：

$$
Y=\sum_{r=0}^{p-1}X^{(r)}
$$

执行完成后：

```text
Rank 0：Y
Rank 1：Y
...
Rank p-1：Y
```

所有 Rank 都得到同一个完整张量。

---

### 为什么 AllReduce 等于 ReduceScatter 加 AllGather

AllReduce 的目标包括两个部分：

1. 对所有 Rank 的数据进行归约；
2. 让每个 Rank 都得到完整归约结果。

ReduceScatter 恰好完成第一部分，并把完整归约结果分片：

```text
ReduceScatter 后：

Rank 0：Y0
Rank 1：Y1
...
Rank p-1：Yp-1
```

其中：

$$
Y=[Y_0,Y_1,\ldots,Y_{p-1}]
$$

随后执行 AllGather：

```text
AllGather 后：

Rank 0：[Y0,Y1,...,Yp-1]
Rank 1：[Y0,Y1,...,Yp-1]
...
```

于是每个 Rank 都得到：

$$
Y=\sum_{r=0}^{p-1}X^{(r)}
$$

因此，从语义上有：

$$
\text{AllReduce}=\text{ReduceScatter}+\text{AllGather}
$$

这里的加号表示操作顺序，而不是数值加法。

更具体地说，先执行：

$$
\left\{X^{(0)},X^{(1)},\ldots,X^{(p-1)}\right\}
\xrightarrow{\text{ReduceScatter}}
\left\{Y_0,Y_1,\ldots,Y_{p-1}\right\}
$$

再执行：

$$
\left\{Y_0,Y_1,\ldots,Y_{p-1}\right\}
\xrightarrow{\text{AllGather}}
\left\{Y,Y,\ldots,Y\right\}
$$

最终正好得到 AllReduce 的输出。

---

### 为什么可以分 Chunk 归约

AllReduce 通常执行逐元素归约。不同位置的元素之间互不依赖，因此可以先把张量切成 Chunk：

$$
X^{(r)}=[X_0^{(r)},X_1^{(r)},\ldots,X_{p-1}^{(r)}]
$$

然后分别计算：

$$
Y_j=\sum_{r=0}^{p-1}X_j^{(r)}
$$

最后拼接：

$$
Y=[Y_0,Y_1,\ldots,Y_{p-1}]
$$

这与直接对完整张量进行归约完全等价：

$$
Y=\sum_{r=0}^{p-1}X^{(r)}
$$

正是由于逐元素归约可以独立分块，AllReduce 才能够拆成 ReduceScatter 和 AllGather。

---

### Ring AllReduce 的通信量

Ring AllReduce 包含两个阶段。

第一阶段是 ReduceScatter：

$$
V_{\text{ReduceScatter}}=\frac{p-1}{p}N
$$

第二阶段是 AllGather：

$$
V_{\text{AllGather}}=\frac{p-1}{p}N
$$

因此，每个 Rank 的总发送量为：

$$
V_{\text{AllReduce}}=V_{\text{ReduceScatter}}+V_{\text{AllGather}}
$$

即：

$$
V_{\text{AllReduce}}=2\frac{p-1}{p}N
$$

这就是 Ring AllReduce 通信量公式的来源。

当 $p$ 很大时：

$$
2\frac{p-1}{p}N\approx 2N
$$

也就是说，每个 Rank 为了完成一个大小为 $N$ 的 Ring AllReduce，大约需要发送 $2N$ 字节，并接收大约 $2N$ 字节。

整个集群的总发送量为：

$$
V_{\text{cluster}}=p\cdot2\frac{p-1}{p}N
$$

化简得到：

$$
V_{\text{cluster}}=2(p-1)N
$$

如果同时计算发送量和接收量，则整个集群处理的数据总量为：

$$
4(p-1)N
$$

不过工程资料通常更常报告每 Rank 发送量，或者链路上的总发送量，而不会把发送和接收重复相加。

---

### Ring AllReduce 的时间模型

ReduceScatter 和 AllGather 各需要 $p-1$ 轮，因此总通信轮数为：

$$
2(p-1)
$$

Ring AllReduce 的时间可以近似表示为：

$$
T_{\text{AllReduce}}\approx 2(p-1)\alpha+2\frac{p-1}{p}N\beta
$$

该公式揭示了 Ring AllReduce 的特点。

在消息很大时，第二项占主导：

$$
2\frac{p-1}{p}N\beta
$$

Ring 算法可以较充分利用链路带宽。

但在消息较小时，第一项占主导：

$$
2(p-1)\alpha
$$

Rank 数量越多，需要的通信轮数越多，固定延迟越明显。

因此，Ring AllReduce 通常适合大消息，而对于小消息，Tree AllReduce 或 Recursive Doubling 可能具有更低的延迟。

实际的 NCCL（NVIDIA Collective Communications Library，英伟达集体通信库）会根据：

* 消息大小；
* GPU 数量；
* NVLink/NVSwitch 拓扑；
* PCIe 拓扑；
* 跨节点网络；
* Channel 数量；

选择或组合不同算法，而不一定始终使用单一 Ring。

---

### AllReduce 在 TP 中的作用

考虑 Row Parallel Linear：

$$
Y=XW
$$

将输入和权重沿输入维度切分：

$$
X=[X_0,X_1,\ldots,X_{p-1}]
$$

$$
W=
\begin{bmatrix}
W_0 \\
W_1 \\
\vdots \\
W_{p-1}
\end{bmatrix}
$$

第 $r$ 个 Rank 计算：

$$
Y^{(r)}=X_rW_r
$$

完整输出为：

$$
Y=\sum_{r=0}^{p-1}Y^{(r)}
$$

因此，需要对局部输出执行 AllReduce，使每个 TP Rank 都获得完整 $Y$。

如果后续算子只需要 $Y$ 的一个分片，则可以用 ReduceScatter 代替 AllReduce，避免在每个 Rank 上复制完整输出。

---

## AllToAll

### AllToAll 的语义

AllToAll 可以理解为：

> 每个 Rank 都为每个目标 Rank 准备一份不同的数据，并把这些数据发送到对应目标。

假设第 $r$ 个 Rank 的输入为：

$$
X^{(r)}=[
X_{r\rightarrow0},
X_{r\rightarrow1},
\ldots,
X_{r\rightarrow p-1}
]
$$

其中 $X_{r\rightarrow j}$ 表示 Rank $r$ 要发送给 Rank $j$ 的数据。

执行 AllToAll 后，Rank $j$ 得到：

$$
Y^{(j)}=[
X_{0\rightarrow j},
X_{1\rightarrow j},
\ldots,
X_{p-1\rightarrow j}
]
$$

例如，四张 GPU 时：

```text
Rank 0：
发送 A0 给 Rank 0
发送 A1 给 Rank 1
发送 A2 给 Rank 2
发送 A3 给 Rank 3

Rank 1：
发送 B0 给 Rank 0
发送 B1 给 Rank 1
发送 B2 给 Rank 2
发送 B3 给 Rank 3
```

执行后：

```text
Rank 0：[A0, B0, C0, D0]
Rank 1：[A1, B1, C1, D1]
Rank 2：[A2, B2, C2, D2]
Rank 3：[A3, B3, C3, D3]
```

---

### AllToAll 与 AllGather 的区别

AllGather 中，一个 Rank 的同一个本地分片最终会出现在所有 Rank 上：

```text
Rank 0 的 X0
→ 所有 Rank 都获得 X0
```

AllToAll 中，一个 Rank 会向不同目标发送不同的数据：

```text
Rank 0：
X0→0 发给 Rank 0
X0→1 发给 Rank 1
X0→2 发给 Rank 2
```

因此：

* AllGather 是共享和复制；
* AllToAll 是重新分布和置换。

AllGather 执行后，每个 Rank 的结果通常相同；AllToAll 执行后，不同 Rank 得到的结果通常不同。

---

### AllToAll 通信量

假设每个 Rank 的总输入大小为 $N$，并均匀切分为 $p$ 份：

$$
|X_{r\rightarrow j}|=\frac{N}{p}
$$

其中发给自己的那一份不需要经过网络。

因此，每个 Rank 需要发送给其他 $p-1$ 个 Rank 的数据量为：

$$
V_{\text{AllToAll}}=(p-1)\frac{N}{p}
$$

即：

$$
V_{\text{AllToAll}}=\frac{p-1}{p}N
$$

需要注意，这只是均匀 AllToAll 的理想结果。

MoE（Mixture of Experts，混合专家模型）中，不同 Expert 接收到的 token 数可能不同。此时通常使用 AllToAllV，即 Variable-sized AllToAll。

对于 Rank $r$，实际发送量为：

$$
V_r=
\sum_{\substack{j=0\\j\neq r}}^{p-1}
|X_{r\rightarrow j}|
$$

不同 Rank 的 $V_r$ 可能差异很大，通信时间通常由最繁忙的 Rank 和最拥塞的链路决定。

---

### AllToAll 在 EP 中的作用

MoE Router 为每个 token 选择 Top-$k$ Experts。

假设某个 token 当前位于 Rank 0，但其目标 Expert 位于 Rank 3，则需要把该 token 的 Hidden State 从 Rank 0 发送到 Rank 3。

MoE 层通常需要两次 AllToAll：

```text
原始 Token 布局
    │
    ▼
第一次 AllToAll：Dispatch
    │
Token 被发送到对应 Expert
    │
    ▼
本地 Expert GEMM
    │
    ▼
第二次 AllToAll：Combine
    │
Expert 输出返回原 Token 所属 Rank
```

因此，EP 的性能不仅取决于总通信量，还取决于：

* token 是否均匀分布；
* Expert 是否存在热点；
* AllToAll 是否跨节点；
* token 是否需要额外 Permute 和 Pack；
* 通信能否与 Expert GEMM 重叠。

---

## Point-to-Point

### 基本语义

Point-to-Point，简称 P2P，是两个 Rank 之间的直接通信。

主要操作包括：

* Send；
* Recv；
* Isend；
* Irecv。

与 Collective 不同，P2P 不要求整个 Process Group 中的所有 Rank 同时参与。

例如：

```text
Rank 0 --Send--> Rank 1
Rank 2 --Send--> Rank 3
```

这两组通信可以彼此独立。

---

### P2P 在 PP 中的作用

Pipeline Parallelism 将模型层划分到不同 Stage。

```text
Stage 0 → Stage 1 → Stage 2 → Stage 3
```

Stage $i$ 完成计算后，把边界激活发送给 Stage $i+1$：

$$
H_{i+1}=f_i(H_i)
$$

如果边界激活大小为 $N$，一次 Stage 转移中：

* 发送 Rank 发送 $N$ 字节；
* 接收 Rank 接收 $N$ 字节。

对于包含 $p$ 个 Stage 的流水线，一个 Microbatch 的前向传播需要跨越 $p-1$ 个边界。

如果每个边界激活大小相同，则整个集群的总发送量约为：

$$
V_{\text{PP, microbatch}}=(p-1)N
$$

但对于某个具体 Rank，它通常只向下一个 Stage 发送一次，并从上一个 Stage 接收一次。

这与 TP 的 AllReduce 有本质区别：

* PP 通信只发生在 Stage 边界；
* TP 通信可能发生在每个 Transformer Layer 内；
* PP 通信参与的通常只是相邻 Stage；
* AllReduce 需要整个 TP Group 参与。

---

### P2P 的同步问题

如果使用阻塞式 Send 和 Recv，而通信顺序设计不当，可能发生 Deadlock，即死锁。

例如：

```text
Rank 0：等待接收 Rank 1
Rank 1：等待接收 Rank 0
```

双方都在等待对方先发送，程序无法继续。

实际系统通常使用：

* 非阻塞 Isend/Irecv；
* Batch P2P；
* 预定义通信顺序；
* CUDA Stream；
* Event Synchronization；

来避免死锁，并让通信与计算重叠。

---

## 几种通信原语的统一比较

假设通信组中有 $p$ 个 Rank，$N$ 表示完整逻辑张量大小。

### AllGather

初始状态：

$$
\text{每 Rank 持有 } \frac{N}{p}
$$

结束状态：

$$
\text{每 Rank 持有 } N
$$

Ring 算法每 Rank 发送量：

$$
\frac{p-1}{p}N
$$

### ReduceScatter

初始状态：

$$
\text{每 Rank 持有 } N
$$

结束状态：

$$
\text{每 Rank 持有归约结果的 } \frac{N}{p}
$$

Ring 算法每 Rank 发送量：

$$
\frac{p-1}{p}N
$$

### AllReduce

初始状态：

$$
\text{每 Rank 持有 } N
$$

结束状态：

$$
\text{每 Rank 持有完整归约结果 } N
$$

Ring 算法每 Rank 发送量：

$$
2\frac{p-1}{p}N
$$

并且：

$$
\text{AllReduce}=\text{ReduceScatter}+\text{AllGather}
$$

### AllToAll

初始状态：

$$
\text{每 Rank 持有总大小 } N
$$

结束状态：

$$
\text{每 Rank 仍持有总大小 } N
$$

但数据被重新分配到目标 Rank。

均匀情况下，每 Rank 网络发送量为：

$$
\frac{p-1}{p}N
$$

### Point-to-Point

一个 Rank 向另一个 Rank 发送大小为 $N$ 的张量。

发送方通信量为：

$$
N
$$

接收方接收量为：

$$
N
$$

P2P 没有固定的全组通信量，因为参与的 Rank 数量和通信拓扑由程序决定。

---

## 为什么相同通信量不代表相同性能

AllGather、ReduceScatter 和均匀 AllToAll 的理论每 Rank 发送量都可能是：

$$
\frac{p-1}{p}N
$$

但它们的实际性能并不一定相同。

原因包括：

### 通信拓扑不同

AllGather 和 ReduceScatter 很适合使用 Ring 流水传输；AllToAll 则要求不同 Rank 之间进行更复杂的多对多交换，更容易产生链路争用。

### 消息切分不同

同样总计发送 $N$ 字节：

* 一次发送一个大消息；
* 分成几十次小消息；

性能可能完全不同，因为后者会支付更多 $\alpha$。

### 是否需要同步

Collective 往往需要等待所有 Rank 到达对应操作。最快的 Rank 也必须等待最慢的 Rank。

因此，Collective 的实际完成时间近似由最慢 Rank 决定：

$$
T_{\text{collective}}\approx \max_r T_r
$$

### 是否存在负载不均衡

MoE AllToAll 中，不同 Rank 的 token 数量可能不同。某些 Rank 发送量很小，另一些 Rank 发送量很大，最终仍然需要等待最繁忙 Rank。

### 是否能与计算重叠

如果通信完全位于关键路径上，则通信时间直接增加模型延迟。

如果通信能够与 GEMM、Attention 或其他 Kernel 重叠，则暴露出来的通信时间可能小于实际通信时间。

可以定义：

$$
T_{\text{exposed comm}} = T_{\text{comm}} - T_{\text{overlapped}}
$$

优化通信的目标并不一定是让通信本身消失，而是尽可能减少关键路径上暴露的通信时间。

---

## 小结

可以用下面的方式理解这些通信原语：

```text
AllGather
→ 每个 Rank 有一部分数据
→ 所有人收集全部分片

ReduceScatter
→ 每个 Rank 有完整局部结果
→ 一边归约，一边把最终结果分片

AllReduce
→ 每个 Rank 有完整局部结果
→ 归约后所有 Rank 都得到完整结果
→ 等价于 ReduceScatter + AllGather

AllToAll
→ 每个 Rank 为不同目标准备不同数据
→ 数据在 Rank 之间重新分布

Point-to-Point
→ 指定发送方与接收方
→ 常用于 Pipeline Stage 之间传递激活
```

Ring 算法下，通信量的核心推导方式是：

1. 张量被切成 $p$ 个大小为 $N/p$ 的 Chunk；
2. 每个 Rank 每轮发送一个 Chunk；
3. Ring 通常执行 $p-1$ 轮；
4. 因此单阶段每 Rank 发送量为：

$$
(p-1)\frac{N}{p}=\frac{p-1}{p}N
$$

AllReduce 包含 ReduceScatter 和 AllGather 两个阶段，因此：

$$
V_{\text{AllReduce}}=2\frac{p-1}{p}N
$$

这个公式描述的是 Ring AllReduce 中每个 Rank 的发送量，而不是所有 Rank 的总流量，也没有把发送和接收重复计算。



# Data Parallelism

## 基本原理

Data Parallelism，简称 DP，是让不同 GPU 保存完整的模型副本，并处理不同请求。

```text
GPU 0：完整模型副本，处理请求 A、B
GPU 1：完整模型副本，处理请求 C、D
GPU 2：完整模型副本，处理请求 E、F
GPU 3：完整模型副本，处理请求 G、H
```

训练中的 DP 需要同步梯度，而推理没有反向传播，因此不同 Replica 之间通常不需要进行参数同步。

---

## DP 的优点

DP 的主要优势是通信少。

不同副本各自独立执行推理，GPU 之间一般不需要在每个 Transformer Layer 中同步，因此：

* 不会引入逐层 AllReduce；
* 扩展吞吐量比较直接；
* 单请求延迟不会因为模型并行通信而增加；
* 便于弹性扩缩容；
* 便于故障隔离。

如果单个模型实例已经可以放入一张 GPU 或一个模型并行组，DP 通常是提高吞吐量的首选方式。

---

## DP 的局限

DP 不会让多个 GPU 同时计算同一个请求，因此一般不会降低单请求延迟。

此外，每个 Replica 都需要保存完整的模型权重。

如果单个模型实例的权重占用为 $M$，使用 $p$ 个 DP Replica 后，整个集群的权重显存总量约为：

$$
pM
$$

每个 Replica 还会分别维护自己的：

* KV Cache；
* Prefix Cache；
* CUDA Graph；
* 内存池；
* 请求队列。

因此，请求路由必须考虑缓存局部性。例如，一次多轮对话最好始终被路由到同一 Replica，否则可能需要重新进行 Prefill，或者迁移 KV Cache。

---

## Continuous Batching

Continuous Batching 不是严格意义上的模型并行，但它是提高推理吞吐量的关键机制。

传统 Static Batching 会固定一个 Batch，直到所有请求都结束：

```text
Batch:
A：生成 20 个 token
B：生成 200 个 token
C：生成 50 个 token
```

当 A 和 C 提前结束后，它们对应的位置可能长期空闲。

Continuous Batching 会在每个 Decode Iteration 重新组织 Batch：

```text
Step 1：A B C
Step 2：A B C
A 结束
Step 3：D B C
C 结束
Step 4：D B E
```

这样可以不断把新请求加入正在执行的 Decode Batch，提高 GPU 利用率。

---

## Chunked Prefill

长 Prompt 的 Prefill 可能一次占用 GPU 较长时间，导致正在 Decode 的请求无法及时生成下一个 token。

Chunked Prefill 将长 Prompt 切分为多个小块：

```text
Prompt：
[Chunk 0][Chunk 1][Chunk 2][Chunk 3]
```

调度器可以在 Prefill Chunk 和 Decode Step 之间交错执行：

```text
Decode Batch
Prefill Chunk 0
Decode Batch
Prefill Chunk 1
Decode Batch
```

这样可以在 TTFT、TPOT 和吞吐量之间进行权衡。

---

# Tensor Parallelism

## TP 在并行体系中的位置

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

## 统一符号

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

## 为什么线性层可以切分

矩阵乘法：

$$Y=XW$$

既可以沿输出维度切，也可以沿输入维度切。

沿输出维度切分时，不同 Rank 负责产生不同的输出特征；沿输入维度切分时，不同 Rank 产生同一个输出的局部贡献，最后需要把局部贡献相加。

这分别对应：

* Column Parallel Linear；
* Row Parallel Linear。

---

## Column Parallel Linear

### 切分方式

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

### 为什么 Column Parallel 后可以不通信

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

### 什么时候需要 AllGather

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

## Row Parallel Linear

### 切分方式

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

### 为什么需要归约

根据分块矩阵乘法：

$$XW=[X_0,X_1,\ldots,X_{p-1}]\begin{bmatrix}W_0 \\ W_1 \\ \vdots \\ W_{p-1}\end{bmatrix}$$

展开后：

$$Y=\sum_{r=0}^{p-1}X_rW_r$$

也就是：

$$Y=\sum_{r=0}^{p-1}Y^{(r)}$$

因此，不同 Rank 的局部结果必须逐元素相加。

如果所有 Rank 都需要完整输出 $Y$，就执行 AllReduce。

---

### Row Parallel 的 AllReduce 通信量

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

## 为什么要把 Column Parallel 和 Row Parallel 配对

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

## Attention 中的 TP

### QKV Projection

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

### 本地 Attention

每个 Rank 对自己持有的 Head 执行：

$$O_r=\operatorname{Attention}(Q_r,K_r,V_r)$$

其中：

$$O_r\in\mathbb{R}^{B\times S\times N_h/p\times D_h}$$

将 Head 维度展平后：

$$O_r\in\mathbb{R}^{B\times S\times H/p}$$

此时各 Rank 持有 Attention 输出的 hidden-dimension 分片。

---

### Output Projection

Attention Output Projection 为：

$$Y=OW_O$$

输入 $O$ 已沿 hidden dimension 切分，因此 $W_O$ 自然使用 Row Parallel。

每个 Rank 计算：

$$Y^{(r)}=O_rW_{O,r}$$

最后：

$$Y=\sum_{r=0}^{p-1}Y^{(r)}$$

因此需要一次 AllReduce，或者在结合 SP 时使用 ReduceScatter。

---

## MLP 中的 TP

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

## 每个 Transformer Layer 中有多少次 TP 通信

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

## TP 通信量与张量形状

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

## TP 与 Prefill、Decode 的差异

### Prefill

Prefill 中：

$$M=B\times S$$

通常较大。

每张 GPU 上的 GEMM 规模较大，TP 可以有效分摊计算。与此同时，通信张量也较大，更容易充分利用链路带宽。

因此 Prefill 中 TP 往往具有较好的扩展性。

### Decode

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

## TP 与 MHA、GQA、MQA

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

## Vocab Parallelism

Embedding 和 LM Head 的 Vocabulary Dimension 通常很大，也可以沿词表切分。

假设词表大小为 $V$，Embedding Dimension 为 $H$：

$$W_E\in\mathbb{R}^{V\times H}$$

使用 $p$ 个 Rank 时，每个 Rank 保存：

$$W_{E,r}\in\mathbb{R}^{V/p\times H}$$

### 输入 Embedding

每个 Rank 只处理落在自己词表区间内的 token id，其他 token 的本地结果置零。

然后通过 AllReduce 得到完整 Embedding。

### 输出 LM Head

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

## TP 的硬件拓扑要求

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

## TP 的性能判断

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

# Pipeline Parallelism

## 基本原理

Pipeline Parallelism，简称 PP，是沿 Transformer Layer 维度切分模型。

假设模型有 32 层，使用 4 个 Stage：

```text
Stage 0 / GPU 0：Layer 0～7
Stage 1 / GPU 1：Layer 8～15
Stage 2 / GPU 2：Layer 16～23
Stage 3 / GPU 3：Layer 24～31
```

一个请求的激活依次经过所有 Stage：

```text
Stage 0 → Stage 1 → Stage 2 → Stage 3
```

Stage 之间只需要传输边界激活，不需要像 TP 那样在每一层执行 collective。

---

## PP 的显存特点

如果模型总权重为 $M$，使用 $p$ 个均匀 Stage 后，每个 Stage 的权重约为：

$$M_{\text{weight, per stage}}\approx\frac{M}{p}$$

但是，Stage 之间传递的隐藏状态通常仍然具有完整 hidden size：

$$
X\in\mathbb{R}^{B\times S\times H}
$$

也就是说，PP 主要切分模型层数，而不是切分每一层内部的 hidden dimension。

---

## Pipeline Bubble

假设有 $p$ 个 Pipeline Stage 和 $m$ 个 Microbatch。

简单前向流水线的理想利用率可以近似表示为：

$$
U\approx\frac{m}{m+p-1}
$$

当 $m$ 很小时，流水线空泡严重。

例如，只有一个 Microbatch 时：

```text
时间 0：Stage 0 工作
时间 1：Stage 1 工作
时间 2：Stage 2 工作
时间 3：Stage 3 工作
```

多数 Stage 在多数时刻处于空闲状态。

当有多个请求或 Microbatch 时：

```text
时间 →
Stage 0：A B C D
Stage 1：  A B C D
Stage 2：    A B C D
Stage 3：      A B C D
```

流水线才能逐渐被填满。

---

## PP 对单 Token 延迟的影响

对于单个请求的 Decode，第 $t+1$ 个 token 必须等待第 $t$ 个 token 完整通过所有 Stage。

因此，PP 通常不能直接降低单个 token 的端到端延迟。

它的主要作用是：

* 解决模型跨节点部署问题；
* 降低单 GPU 权重显存；
* 在高并发下通过流水线提高吞吐量。

---

## PP 的负载均衡

不能简单认为每个 Stage 放相同数量的层就一定均衡。

不同层的计算量可能不同：

* 某些层是 Dense FFN；
* 某些层是 MoE；
* 首个 Stage 包含 Embedding；
* 最后 Stage 包含 LM Head；
* 不同 MoE 层的 Expert 数和负载不同；
* 某些层具有额外的视觉或多模态模块。

实际系统通常根据 Profiling 结果进行非均匀 Layer Partition。

---

# Sequence Parallelism

## 术语范围

SP（Sequence Parallelism，序列并行）在不同论文和系统中可能指不同技术。

本节所说的 SP 特指 Megatron 风格 Sequence Parallelism：

> 在 TP Group 内，把 LayerNorm、Residual 和逐元素算子之间的激活沿 token 或 sequence 维度切分。

它不同于：

* UP（Ulysses Parallelism，尤利西斯并行）：通过 AllToAll 在序列分片和 Head 分片之间转换；
* CP（Context Parallelism，上下文并行）：对 Attention 上下文和 KV Cache 进行分布式切分；
* Ring Attention：让 KV Block 在设备间环形移动。

Megatron SP 本身通常不能独立存在，而是与 TP 配套使用。

---

## 为什么普通 TP 仍然存在激活复制

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

## SP 的核心思想

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

## SP 如何与 TP 连接

SP 的关键不是单纯把序列切开，而是在两个张量布局之间切换：

### Sequence-Parallel Layout

每个 Rank 持有：

$$X_r\in\mathbb{R}^{T/p\times H}$$

即：

```text
部分 token
完整 hidden dimension
```

### Tensor-Parallel Linear 所需布局

Column Parallel Linear 通常要求每个 TP Rank 获得全部 token：

$$X\in\mathbb{R}^{T\times H}$$

每个 Rank 使用不同的权重分片，对全部 token 计算不同输出特征。

因此，在进入 Column Parallel Linear 前，需要把序列分片恢复为完整 token 集合。

这通过 AllGather 完成。

---

## SP 中的 AllGather

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

## SP 中的 ReduceScatter

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

## 一组 SP + TP 的完整过程

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

## 为什么 SP 不会增加理论总通信量

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

## SP 的真正收益

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

## SP 为什么在训练中更重要

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

## SP 与 CP 的区别

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

# Context Parallelism

## 基本原理

Context Parallelism，简称 CP，是沿上下文或序列 token 维度切分 Attention。

例如，长度为 $S$ 的序列被切到 $p$ 张 GPU：

$$
S=S_0+S_1+\cdots+S_{p-1}
$$

每张 GPU 保存：

$$
X_i\in\mathbb{R}^{B\times S_i\times H}
$$

例如：

```text
GPU 0：token 0～1023
GPU 1：token 1024～2047
GPU 2：token 2048～3071
GPU 3：token 3072～4095
```

CP 的主要目标是：

* 支持超长上下文；
* 分片 KV Cache；
* 分摊 Attention 计算；
* 降低单 GPU 的序列显存占用。

---

## CP 与 SP 的区别

经典 Megatron SP 主要处理：

* LayerNorm；
* Residual；
* Dropout；
* 逐元素操作；
* TP 中间激活。

Context Parallelism 则直接处理：

* Attention 的长序列；
* 分布式 Query、Key 和 Value；
* KV Cache 分片；
* 分布式 Softmax；
* 超长上下文计算。

因此，两者不能简单视为同一个概念。

---

# Ulysses Parallelism

## UP 的目标

UP（Ulysses Parallelism，尤利西斯并行）是一种面向 Attention 的序列并行方法。

它要解决的问题是：

> 序列已经沿 GPU 切分，但标准 Attention Kernel 通常希望获得完整序列上的 Q、K、V。如何在不让每张 GPU 保存全部 Head 的情况下，继续使用高效的本地 Attention Kernel？

UP 的核心方法不是让 KV Block 环形移动，而是通过 AllToAll 在两种布局之间转换：

1. Sequence-Sharded Layout：部分序列、全部 Head；
2. Head-Sharded Layout：完整序列、部分 Head。

转换后，每个 Rank 可以对自己负责的 Head，在完整序列上独立执行 Attention。

---

## UP 与普通 SP 的区别

普通 Megatron SP 主要用于：

* LayerNorm；
* Residual；
* TP 线性层边界；
* 激活显存分片。

UP 直接进入 Attention 内部，重排 Q、K、V：

```text
Megatron SP：
序列分片
→ AllGather 完整序列
→ TP Attention

Ulysses：
序列分片、全部 Heads
→ AllToAll
→ 完整序列、部分 Heads
→ 本地 Attention
```

因此，UP 是一种分布式 Attention 方法，而不仅是逐元素激活分片。

---

## 初始张量布局

设：

* Batch Size 为 $B$；
* 序列长度为 $S$；
* Attention Head 数为 $N_h$；
* 每个 Head 维度为 $D_h$；
* UP Degree 为 $p$。

假设序列均匀切分。

每个 Rank 初始持有：

$$Q_r,K_r,V_r\in\mathbb{R}^{B\times S/p\times N_h\times D_h}$$

也就是：

```text
部分序列
全部 Attention Heads
```

例如：

```text
S   = 4096
N_h = 32
UP  = 4

Rank 0：token 0～1023，Heads 0～31
Rank 1：token 1024～2047，Heads 0～31
Rank 2：token 2048～3071，Heads 0～31
Rank 3：token 3072～4095，Heads 0～31
```

---

## 为什么这种布局不能直接独立算完整 Attention

Rank 0 虽然拥有全部 Heads，但只拥有前 $1/4$ 的 K 和 V。

对于 Rank 0 的 Query，完整 Attention 应该访问整个上下文：

$$K=[K_0;K_1;\ldots;K_{p-1}]$$

$$V=[V_0;V_1;\ldots;V_{p-1}]$$

如果 Rank 0 只使用本地 $K_0,V_0$，它只能看到局部上下文，结果不等价于完整 Attention。

因此必须进行跨 Rank 数据交换。

---

## Ulysses 的 AllToAll 布局转换

UP 将每个 Rank 的 Head 维度切成 $p$ 组：

$$N_h=\frac{N_h}{p}\times p$$

每个源 Rank 将不同 Head Group 发送给不同目标 Rank。

例如：

```text
Rank 0 原有：
token 0～1023，Heads 0～31

切成：
Heads 0～7   → Rank 0
Heads 8～15  → Rank 1
Heads 16～23 → Rank 2
Heads 24～31 → Rank 3
```

其他 Rank 也执行相同操作。

AllToAll 后，Rank 0 收到所有序列分片上的 Heads 0～7：

```text
Rank 0：
token 0～4095，Heads 0～7

Rank 1：
token 0～4095，Heads 8～15

Rank 2：
token 0～4095，Heads 16～23

Rank 3：
token 0～4095，Heads 24～31
```

张量形状变为：

$$Q'_r,K'_r,V'_r\in\mathbb{R}^{B\times S\times N_h/p\times D_h}$$

也就是：

```text
完整序列
部分 Attention Heads
```

---

## 为什么 AllToAll 能实现这个转换

AllToAll 的语义是：

> 每个 Rank 为不同目标 Rank 准备不同数据，通信后每个目标 Rank 收集来自所有源 Rank 的对应数据。

源 Rank 按 Head Group 切分本地序列块：

$$Q_r=[Q_{r\rightarrow0},Q_{r\rightarrow1},\ldots,Q_{r\rightarrow p-1}]$$

其中：

$$Q_{r\rightarrow j}\in\mathbb{R}^{B\times S/p\times N_h/p\times D_h}$$

目标 Rank $j$ 收到：

$$Q'_j=[Q_{0\rightarrow j};Q_{1\rightarrow j};\ldots;Q_{p-1\rightarrow j}]$$

沿序列维度拼接后：

$$Q'_j\in\mathbb{R}^{B\times S\times N_h/p\times D_h}$$

K 和 V 同理。

因此，一次 AllToAll 同时完成：

* 序列维度的 Gather；
* Head 维度的 Scatter。

它不是简单复制完整张量，而是做张量布局转置。

---

## 本地 Attention

布局转换后，每个 Rank 拥有：

* 完整序列；
* 部分 Heads。

不同 Attention Heads 相互独立，因此每个 Rank 可以本地执行：

$$O'_r=\operatorname{Attention}(Q'_r,K'_r,V'_r)$$

输出形状为：

$$O'_r\in\mathbb{R}^{B\times S\times N_h/p\times D_h}$$

这一步可以使用成熟的单卡 FlashAttention（闪存注意力）Kernel，因为对于本地 Head 子集而言，序列是完整的。

---

## 反向 AllToAll

Attention 输出后，通常需要恢复 Sequence-Sharded Layout。

当前布局为：

```text
完整序列
部分 Heads
```

通过第二次 AllToAll，转换回：

```text
部分序列
全部 Heads
```

最终每个 Rank 得到：

$$O_r\in\mathbb{R}^{B\times S/p\times N_h\times D_h}$$

例如：

```text
Rank 0：token 0～1023，Heads 0～31
Rank 1：token 1024～2047，Heads 0～31
...
```

因此 UP 的 Attention 主流程为：

```text
部分序列，全部 Heads
        │
        ▼
QKV AllToAll
        │
        ▼
完整序列，部分 Heads
        │
        ▼
Local Attention
        │
        ▼
Output AllToAll
        │
        ▼
部分序列，全部 Heads
```

---

## UP 的通信量

设每个 Rank 在第一次 AllToAll 前持有的本地 QKV Buffer 大小为：

$$N_{\text{QKV,local}}$$

均匀 AllToAll 中，发给自己的 $1/p$ 数据不经过网络，因此第一次 AllToAll 每 Rank 发送量为：

$$V_{\text{QKV A2A}}=\frac{p-1}{p}N_{\text{QKV,local}}$$

Attention 输出本地 Buffer 大小为：

$$N_{\text{O,local}}$$

第二次 AllToAll 每 Rank 发送量为：

$$V_{\text{O A2A}}=\frac{p-1}{p}N_{\text{O,local}}$$

因此总发送量为：

$$V_{\text{UP}}=\frac{p-1}{p}\left(N_{\text{QKV,local}}+N_{\text{O,local}}\right)$$

对于标准 MHA，如果 Q、K、V 和 O 的元素数量大致相同，则：

$$N_{\text{QKV,local}}\approx3N_{\text{O,local}}$$

于是：

$$V_{\text{UP}}\approx4\frac{p-1}{p}N_{\text{O,local}}$$

实际系统通常会把 Q、K、V 合并到同一个 Buffer 中执行一次 AllToAll，而不是分别发起三次 Collective，从而减少通信启动次数。

---

## UP 的通信成本不仅由字节数决定

AllToAll 的理论发送量看起来与 AllGather 相似，但实际性能往往更敏感。

原因是 AllToAll 需要：

* 每个 Rank 向多个目标 Rank 发送不同数据；
* 对数据进行 Split、Pack 和 Reorder；
* 处理多对多链路争用；
* 等待最慢 Rank 完成。

时间可以粗略表示为：

$$T_{\text{UP}}\approx T_{\text{pack}}+T_{\text{all-to-all}}+T_{\text{attention}}+T_{\text{all-to-all}}+T_{\text{unpack}}$$

其中 Pack 和 Unpack 属于本地显存操作，不计入网络通信量，但仍会消耗 HBM（High Bandwidth Memory，高带宽显存）带宽和 Kernel 时间。

---

## UP 的 Head 数限制

UP 把 Head 维度切成 $p$ 份，因此通常要求：

$$N_h\bmod p=0$$

同时需要：

$$p\leq N_h$$

否则某些 Rank 无法分配到完整 Head Group。

例如：

```text
N_h = 32
UP  = 8
```

每 Rank 分配 4 个 Heads，比较自然。

但如果：

```text
N_h = 8
UP  = 16
```

Head 数不足以支持 16 路标准 Ulysses 切分。

这时通常需要：

* 降低 UP Degree；
* 与 Ring Attention 组合；
* 采用更复杂的 Head/Sequence 二维切分；
* 对 Head 做复制或特殊映射。

---

## UP 与 GQA、MQA

对于 GQA：

$$N_q>N_{kv}$$

Q 可以按较大的 Query Head 数切分，但 K、V 的 Head 数可能不足。

例如：

```text
Query Heads = 64
KV Heads    = 8
UP          = 16
```

Q 可以分成 16 组，但 KV Head 无法自然分成 16 组。

可能的处理方式包括：

* 复制 KV Heads；
* 对 Q 和 KV 使用不同 AllToAll 布局；
* 让多个 Query Head Group 共享同一个 KV Head Group；
* 限制 UP Degree 不超过 $N_{kv}$；
* 使用 Ring Attention 处理 KV 上下文。

因此，标准 UP 在 MHA 上最自然，在 GQA/MQA 上需要额外设计。

---

## UP 为什么更适合 Prefill

Prefill 时：

$$S\gg1$$

序列维度足够大，切分后每个 Rank 仍有较多 token，Attention 计算量也足以覆盖 AllToAll。

Decode 时，每个请求的新 Query 长度通常为：

$$S_q=1$$

此时 Query 序列几乎无法继续切分。

即使历史 KV Cache 很长，标准 Ulysses 的“序列分片到 Head 分片”转换也不再自然，因为本轮只有少量 Query token。

所以 UP 主要适用于：

* 长上下文 Prefill；
* 训练；
* DiT（Diffusion Transformer，扩散 Transformer）；
* 图像和视频 token；
* 具有较大 Query Sequence 的 Attention。

对于 Decode 长 KV Cache，通常更适合使用 KV Context Sharding 或 Ring-Based Context Parallelism。

---

## UP 与 Ring Attention 的比较

UP 的核心是：

```text
序列分片
→ AllToAll
→ Head 分片
→ 本地完整 Attention
```

Ring Attention 的核心是：

```text
Query 留在本地
KV Block 在 Rank 间循环
→ 逐块 Online Softmax
```

UP 的优点：

* 可以直接使用成熟的本地 FlashAttention；
* 通信步骤相对集中；
* 计算逻辑简单；
* 适合 Head 数较多的模型。

UP 的局限：

* UP Degree 受 Head 数约束；
* AllToAll 对网络拓扑敏感；
* GQA/MQA 处理复杂；
* Decode 场景不自然。

Ring Attention 的优点：

* 并行度不严格受 Head 数限制；
* KV 通信可以与 Attention 计算流水化；
* 更适合超长上下文。

Ring Attention 的局限：

* 需要多个 Ring Step；
* Online Softmax 和因果 Mask 处理更复杂；
* 点对点通信调度复杂。

实际系统可以组合二者：

$$P_{\text{context}}=P_{\text{Ulysses}}\times P_{\text{Ring}}$$

在节点内使用 Ulysses AllToAll，在节点间使用 Ring，可以更好地匹配分层硬件拓扑。

---

# Ring Attention

## 基本思想

Ring Attention 中，每张 GPU 保留自己的 Query Block，Key 和 Value Block 在 GPU 之间环形传递。

```text
Step 0：
GPU 0 使用 KV 0
GPU 1 使用 KV 1
GPU 2 使用 KV 2
GPU 3 使用 KV 3

Step 1：
GPU 0 使用 KV 3
GPU 1 使用 KV 0
GPU 2 使用 KV 1
GPU 3 使用 KV 2
```

经过 $p$ 个 Ring Step 后，每张 GPU 的 Query 都与所有 KV Block 完成 Attention。

---

## Online Softmax

对于一个 Query Block，不能简单独立计算每个 KV Block 的 Softmax 后再相加，因为 Softmax 的归一化范围是整个上下文。

Ring Attention 通常使用 Online Softmax。

假设已经处理了一部分 KV，维护：

* 当前最大值 $m$；
* 当前指数和 $l$；
* 当前加权输出 $O$。

当加入新的 KV Block 后，先计算新的局部最大值和局部指数和，再更新全局状态。

如果旧状态为 $(m_{\text{old}},l_{\text{old}},O_{\text{old}})$，新 Block 的状态为 $(m_{\text{new}},l_{\text{new}},O_{\text{new}})$，合并后的最大值为：

$$
m=\max(m_{\text{old}},m_{\text{new}})
$$

归一化因子更新为：

$$
l=e^{m_{\text{old}}-m}l_{\text{old}}+e^{m_{\text{new}}-m}l_{\text{new}}
$$

输出累积值更新为：

$$
O=e^{m_{\text{old}}-m}O_{\text{old}}+e^{m_{\text{new}}-m}O_{\text{new}}
$$

最后输出为：

$$
\operatorname{Attention}(Q,K,V)=\frac{O}{l}
$$

这与 FlashAttention 中的 Online Softmax 思路类似。

---

## Ring Attention 的优缺点

优点：

* KV Cache 和序列长度可以沿设备扩展；
* KV 通信可以与局部 Attention 计算流水化；
* 并行度不完全受 Head 数限制；
* 适合超长上下文和视频 token。

缺点：

* 需要执行多个 Ring Step；
* 每个 Step 都存在点对点通信；
* 实现复杂；
* 对通信和计算重叠要求高；
* 小序列下可能得不偿失。

---

## Ulysses 与 Ring Attention 的区别

Ulysses 的核心是：

> 将序列分片布局通过 AllToAll 转换为 Head 分片布局。

Ring Attention 的核心是：

> Query 保持在本地，KV Block 在 GPU 之间环形移动。

Ulysses 的通信相对集中，Ring Attention 的通信被分散到多个迭代中。

实际系统也可以将二者组合：

$$\text{SP Degree}=\text{Ulysses Degree}\times\text{Ring Degree}$$

---

# Decode 阶段的 Context Parallelism

Prefill 时 Query 长度较大，可以沿 Query 序列维度切分。

但 Decode 每个请求的 Query 长度通常只有 1：

$$
Q\in\mathbb{R}^{B\times 1\times H}
$$

此时已经没有足够的 Query token 可以继续沿序列维度切分。

一种方法是分片历史 KV Cache：

```text
GPU 0：保存前 1/4 KV Cache
GPU 1：保存第 2/4 KV Cache
GPU 2：保存第 3/4 KV Cache
GPU 3：保存最后 1/4 KV Cache
```

每张 GPU 计算局部 Attention：

$$
O_i=\operatorname{Attention}(Q,K_i,V_i)
$$

然后通过分布式 Online Softmax 合并各 GPU 的局部结果。

Decode CP 的本质是：

> 用跨 GPU 通信换取 KV Cache 容量和局部 HBM 读取带宽。

它更适合：

* 超长上下文；
* KV Cache 无法放入单卡；
* 单步 KV Cache 读取成为主要瓶颈；
* GPU 间具有高速互联。

对于较短上下文或极低延迟服务，Decode CP 的额外通信可能反而降低性能。

---

# Expert Parallelism

## EP 的背景：为什么 MoE 需要不同的并行方式

MoE（Mixture of Experts，混合专家模型）层包含多个 Expert，但每个 token 只激活少量 Expert。

设：

* Expert 总数为 $E$；
* 每个 token 选择 Top-$k$ Experts；
* Hidden Size 为 $H$；
* EP Group 中有 $p$ 个 Rank。

如果每个 Rank 保存全部 $E$ 个 Experts，会造成巨大的权重复制。

EP（Expert Parallelism，专家并行）将不同 Expert 放在不同 Rank：

```text
Rank 0：Expert 0～3
Rank 1：Expert 4～7
Rank 2：Expert 8～11
Rank 3：Expert 12～15
```

理想情况下，每个 Rank 保存：

$$\frac{E}{p}$$

个 Experts。

但 token 初始位于处理该请求的 Rank，不一定位于目标 Expert 所在 Rank。

因此 EP 的核心问题是：

> 根据 Router 结果，把 token 动态发送到对应 Expert 所在的 GPU。

---

## MoE Layer 的整体流程

一个完整 MoE Layer 通常包括：

```text
输入 Hidden States
        │
        ▼
Router / Gating
        │
为每个 token 选择 Top-k Experts
        │
        ▼
Token Permute / Pack
        │
按目标 Rank 和 Expert 分组
        │
        ▼
AllToAll / AllToAllV Dispatch
        │
        ▼
本地 Expert Permute
        │
        ▼
Grouped GEMM
        │
        ▼
AllToAll / AllToAllV Combine
        │
        ▼
Token Unpermute
        │
        ▼
按 Router Weight 加权合并
        │
        ▼
输出
```

EP 并不只是两次 AllToAll。

在通信前后，还需要进行大量本地数据重排和元数据处理。

---

## Router

设输入 token 数为 $T$：

$$X\in\mathbb{R}^{T\times H}$$

Router 权重为：

$$W_R\in\mathbb{R}^{H\times E}$$

Router Logits 为：

$$G=XW_R\in\mathbb{R}^{T\times E}$$

对每个 token 选择 Top-$k$ Experts：

$$\mathcal{E}(t)=\operatorname{TopK}(G_t,k)$$

并得到对应路由权重：

$$w_{t,e}$$

如果 $k=2$，每个 token 会生成两个 Routed Token Instances。

总 Routed Token 数近似为：

$$T_{\text{routed}}=kT$$

如果存在 Capacity Limit、Token Drop 或共享 Expert，实际数量可能不同。

---

## 为什么需要 Token Permute

输入 token 通常按照原始请求顺序排列：

```text
token 0
token 1
token 2
token 3
...
```

Router 选择结果可能是：

```text
token 0 → Expert 7
token 1 → Expert 2
token 2 → Expert 7
token 3 → Expert 12
```

这些目标 Expert 分散在不同 Rank 上。

通信库通常希望发送给同一个目标 Rank 的数据位于连续 Buffer 中，例如：

```text
发给 Rank 0 的 token
发给 Rank 1 的 token
发给 Rank 2 的 token
发给 Rank 3 的 token
```

因此需要先执行 Permute 或 Pack：

```text
原始 Token 顺序
        │
        ▼
按目标 Rank 排序
        │
        ▼
同一 Rank 内按 Expert 排序
```

同时记录反向索引，以便计算完成后恢复原 token 顺序。

Permute 不产生网络通信，但会消耗：

* HBM 读写带宽；
* Prefix Sum 或 Histogram；
* Scatter/Gather Kernel；
* 临时 Buffer；
* Kernel Launch。

因此 MoE 中“省掉一次 Permute”可能是非常重要的优化。

---

## Dispatch AllToAll

每个源 Rank 把 Routed Token 发送到目标 Expert 所在 Rank。

假设 Rank $r$ 要发送给 Rank $j$ 的 token Buffer 为：

$$X_{r\rightarrow j}$$

执行 Dispatch AllToAll 后，Rank $j$ 得到：

$$X'_j=[X_{0\rightarrow j},X_{1\rightarrow j},\ldots,X_{p-1\rightarrow j}]$$

这些 token 都将由 Rank $j$ 上的本地 Experts 处理。

---

## 为什么 MoE 通常需要 AllToAllV

普通 AllToAll 假设每个 Rank 发给每个目标 Rank 的数据大小相同。

但 Router 通常产生不均匀分布。

例如：

```text
Rank 0 发给 Rank 1：64 个 token
Rank 0 发给 Rank 2：7 个 token
Rank 0 发给 Rank 3：103 个 token
```

因此实际 MoE 更接近 AllToAllV，即 Variable-Sized AllToAll：

* 不同 Rank 之间的消息大小不同；
* 每个 Rank 需要交换 Send Counts；
* 接收方根据 Counts 分配或定位 Buffer；
* 通信时间取决于最繁忙 Rank 和拥塞链路。

---

## EP Dispatch 通信量

设每个 Rank 原始拥有 $T_{\text{local}}$ 个 token。

每个 token 选择 Top-$k$ Experts，因此本 Rank 产生：

$$kT_{\text{local}}$$

个 Routed Token Instances。

每个 Hidden State 大小为：

$$Hb$$

如果 Expert 在各 Rank 间均匀分布，且路由近似均匀，则一个 Routed Token 的目标 Expert 位于本 Rank 的概率约为：

$$\frac{1}{p}$$

需要发送到远程 Rank 的概率约为：

$$\frac{p-1}{p}$$

因此 Dispatch 阶段每 Rank 的理想发送量近似为：

$$V_{\text{dispatch}}\approx\frac{p-1}{p}kT_{\text{local}}Hb$$

Combine 阶段需要把 Expert 输出发送回原 token 所属 Rank，通信量通常相近：

$$V_{\text{combine}}\approx\frac{p-1}{p}kT_{\text{local}}Hb$$

因此两次 AllToAll 的理想总发送量为：

$$V_{\text{EP}}\approx2\frac{p-1}{p}kT_{\text{local}}Hb$$

这还没有计算：

* Expert Id；
* Router Weight；
* Source Rank；
* Token Index；
* Offset；
* Send Count 和 Receive Count。

元数据相对 Hidden State 通常较小，但在 Decode 小 Batch 下可能不可忽略。

---

## 非均匀路由下的通信量

对于 Rank $r$，实际发送量应写为：

$$V_r^{\text{dispatch}}=\sum_{\substack{j=0 \\ j\neq r}}^{p-1}|X_{r\rightarrow j}|$$

如果每个 token Hidden State 为 $Hb$ 字节，Rank $r$ 发送到 Rank $j$ 的 Routed Token 数为 $n_{r\rightarrow j}$，则：

$$V_r^{\text{dispatch}}=Hb\sum_{\substack{j=0 \\ j\neq r}}^{p-1}n_{r\rightarrow j}$$

Collective 的完成时间通常由最大通信 Rank 决定：

$$T_{\text{dispatch}}\approx\max_rT_r$$

所以平均发送量较小并不能保证 AllToAll 很快。

真正关键的是：

* 最大 Rank 发送量；
* 最大 Rank 接收量；
* 是否有热点链路；
* 是否跨节点；
* 是否存在某个 Expert 的严重拥塞。

---

## 本地 Expert 分组

Dispatch 完成后，一个 Rank 会收到属于多个本地 Experts 的 token。

例如：

```text
Rank 0 收到：
Expert 0：128 个 token
Expert 1：19 个 token
Expert 2：74 个 token
Expert 3：5 个 token
```

为了执行 Expert FFN，需要进一步按本地 Expert 分组。

如果 Dispatch Buffer 已经同时按目标 Rank 和 Expert 排好序，则接收后可以直接形成 Expert-Contiguous Layout。

否则还需要一次本地 Permute。

这也是为什么高性能 EP 实现会尽量融合：

* Router Output；
* Destination Rank 计算；
* Expert Offset；
* Permute；
* Dispatch Buffer 写入。

---

## Grouped GEMM

每个 Expert 的计算通常是一个 MLP：

$$Y_e=\sigma(X_eW_{\text{up},e})W_{\text{down},e}$$

其中：

$$X_e\in\mathbb{R}^{T_e\times H}$$

$T_e$ 是分配给 Expert $e$ 的 token 数。

不同 Expert 的 $T_e$ 往往不同：

```text
Expert 0：128 tokens
Expert 1：19 tokens
Expert 2：74 tokens
Expert 3：5 tokens
```

普通 Batched GEMM（General Matrix Multiplication，通用矩阵乘法）通常要求各矩阵形状相同，不适合这种情况。

Grouped GEMM 允许一次 Kernel Launch 处理多个不同 $M$ 维度的 GEMM：

$$Y_e=X_eW_e,\qquad e=0,1,\ldots,E_{\text{local}}-1$$

Grouped GEMM 的目标是：

* 减少 Kernel Launch 次数；
* 提高小 GEMM 的调度效率；
* 共享 Workspace；
* 在一个 Kernel 中处理多个 Experts；
* 尽量提高 Tensor Core 利用率。

---

## Combine AllToAll

Expert 计算完成后，输出仍位于 Expert 所在 Rank。

但最终输出必须回到原始 token 所属 Rank，并恢复原始 token 顺序。

因此执行第二次 AllToAll：

```text
Expert Rank
    │
    ▼
Combine AllToAll
    │
    ▼
原始 Token Rank
```

随后根据保存的反向索引执行 Unpermute。

对于 Top-$k$ Routing，一个 token 可能收到多个 Expert 输出：

$$Y_t=\sum_{e\in\mathcal{E}(t)}w_{t,e}Y_{t,e}$$

最终还需要按照 Router Weight 进行加权合并。

---

## EP 的两次通信为什么通常无法省掉

Dispatch 是为了把 token 移动到 Expert 所在 Rank。

Combine 是为了把 Expert 输出送回原请求布局。

除非后续层也继续采用完全相同的 Expert-Owned Token Layout，否则必须恢复原布局。

Transformer 的下一层通常还要执行：

* Attention；
* Residual；
* LayerNorm；
* 其他 MoE Layer。

这些操作通常按照请求或 token 原始布局组织，因此需要 Combine。

某些系统会尝试让多个连续 MoE 算子保持 Expert-Owned Layout，但这会显著增加模型结构和调度复杂度。

---

## EP 与 TP 的区别

### TP

TP 把一个 Expert 内部的权重切分：

```text
Expert 0：
Rank 0 持有一部分权重
Rank 1 持有一部分权重
Rank 2 持有一部分权重
Rank 3 持有一部分权重
```

每个被激活的 Expert 需要多个 Rank 共同计算，通常依赖 AllReduce。

### EP

EP 把不同 Experts 放到不同 Rank：

```text
Rank 0：Expert 0～3
Rank 1：Expert 4～7
Rank 2：Expert 8～11
Rank 3：Expert 12～15
```

token 被发送到 Expert 所在 Rank，通常依赖 AllToAll。

因此：

```text
TP：
切单个矩阵
通信对象是 Dense Tensor Partial Result

EP：
切 Expert 集合
通信对象是动态 Routed Tokens
```

---

## Expert Tensor Parallelism

如果单个 Expert 太大，无法放入一张 GPU，还可以在 Expert 内部继续使用 TP。

这种方式称为 ETP（Expert Tensor Parallelism，专家张量并行）。

例如：

```text
EP Group：
不同 Expert 位于不同 Expert Group

每个 Expert Group 内：
TP = 2
```

执行过程变为：

```text
EP Dispatch AllToAll
        │
        ▼
token 到达目标 Expert Group
        │
        ▼
Expert 内部 Tensor Parallel
        │
        ▼
TP Collective
        │
        ▼
EP Combine AllToAll
```

此时同时存在：

* EP AllToAll；
* Expert 内 TP AllReduce；
* 本地 Permute；
* Grouped GEMM 或分片 GEMM。

通信复杂度明显提高。

所以如果单个 Expert 能够完整放入一张 GPU，通常优先采用纯 EP，避免 Expert 内部 TP。

---

## EP 的负载不均衡

理想情况下，每个 Expert 接收相同数量 token：

$$T_e\approx\frac{kT}{E}$$

但实际 Router 可能产生严重偏斜：

```text
Expert 0：300 tokens
Expert 1：280 tokens
Expert 2：15 tokens
Expert 3：8 tokens
```

MoE Layer 的完成时间由最慢 Expert 或最慢 Rank 决定：

$$T_{\text{MoE layer}}\approx\max_rT_r$$

负载不均衡会同时造成：

* 某些 Rank 的通信量更大；
* 某些 Rank 的 Grouped GEMM 更慢；
* 其他 Rank 提前完成后等待；
* GPU 利用率不均；
* Tail Latency 增大。

---

## Capacity 与 Token Drop

一些 MoE 系统会为每个 Expert 设置容量：

$$C=\left\lceil\text{Capacity Factor}\times\frac{kT}{E}\right\rceil$$

当某个 Expert 接收到的 token 数超过 $C$ 时，可能：

* 丢弃超出容量的 token；
* 把 token 路由到次选 Expert；
* 使用 Backup Expert；
* 动态扩展 Buffer；
* 接受负载不均衡。

训练中 Token Drop 可以控制内存和计算上界，但推理通常更难接受丢弃 token，因为会影响输出准确性。

所以推理系统更常使用：

* 动态 Buffer；
* Expert Replication；
* Expert Placement；
* Load-Aware Routing。

---

## Expert Replication

如果某个 Expert 很热门，可以在多个 Rank 上复制该 Expert。

例如：

```text
Expert 7：
Rank 1 有一份
Rank 5 有一份
Rank 9 有一份
```

Router 或调度器把目标为 Expert 7 的 token 分摊到不同副本。

这样可以降低：

* 单个 Rank 的 token 峰值；
* AllToAll 热点；
* Grouped GEMM Straggler；
* Tail Latency。

代价是：

* 额外权重显存；
* Expert Placement 更复杂；
* 副本之间需要保持参数一致；
* 路由需要感知副本位置。

推理阶段权重不更新，因此副本一致性比训练简单。

---

## Expert Placement

即使每个 Rank 保存相同数量的 Experts，也不一定负载均衡。

应根据 Expert 热度和网络拓扑决定放置方式。

例如，不应把多个热点 Experts 全部放到同一 Rank：

```text
不合理：
Rank 0：最热门的 4 个 Experts

更合理：
热点 Experts 分散到不同 Rank
```

跨节点 EP 还要考虑：

* 哪些 Experts 经常被同一类请求共同激活；
* 是否能减少跨节点流量；
* 节点内 NVLink 与节点间网络的差异；
* 是否需要 Hierarchical AllToAll。

---

## Hierarchical AllToAll

在多节点集群中，平坦 AllToAll 会让每张 GPU 与所有其他 GPU 直接交换数据，容易产生跨节点网络压力。

分层 AllToAll 可以分成：

1. 节点内聚合；
2. 节点间交换；
3. 节点内分发。

例如：

```text
GPU
 │
 ▼
节点内 NVLink 聚合
 │
 ▼
节点间 InfiniBand 交换
 │
 ▼
目标节点内 NVLink 分发
```

这种方式可能减少跨节点消息数量，但会增加本地重排和额外阶段。

是否更快取决于：

* 节点内互联带宽；
* 节点间带宽；
* 消息大小；
* Rank 数量；
* Expert 分布。

---

## Decode 阶段为什么对 EP 特别困难

Prefill 中：

$$T=B\times S$$

通常 token 数较多，分配到每个 Expert 的 token 数也相对较大。

Decode 中每个请求每步只有一个新 token：

$$T=B$$

如果 Batch Size 较小，则：

$$T_e\approx\frac{kB}{E}$$

可能远小于 1。

这意味着许多 Expert 每个 Decode Step 只能收到几个甚至零个 token。

结果是：

* Grouped GEMM 非常小；
* Tensor Core 利用率低；
* AllToAll 固定延迟难以摊薄；
* Permute 和 Metadata 开销占比增大；
* Rank 间负载波动更明显。

因此 MoE Decode 往往依赖高并发，把多个请求的 token 汇总成较大的全局 Batch。

---

## Attention DP + Expert EP

现代 MoE 推理常让 Attention 和 Expert 使用不同并行策略。

Attention 部分使用 DP（Data Parallelism，数据并行）：

```text
Rank 0：处理请求 A、B
Rank 1：处理请求 C、D
Rank 2：处理请求 E、F
Rank 3：处理请求 G、H
```

每个 Rank 本地保存 Attention 权重和自己的 KV Cache，不需要逐层 TP AllReduce。

进入 MoE Layer 后，所有 Rank 的 token 进入统一 EP Group：

```text
DP Attention Layout
        │
        ▼
EP Dispatch AllToAll
        │
        ▼
Expert Layout
        │
        ▼
Expert Computation
        │
        ▼
EP Combine AllToAll
        │
        ▼
恢复 DP Attention Layout
```

这种设计的优势是：

* Attention 避免高频 TP Collective；
* 多个 DP Rank 的 token 汇总后，Expert Batch 更大；
* Expert 权重可以分布到更多 GPU；
* Attention KV Cache 与 Expert 权重可以独立扩展。

---

## EP 的通信与计算重叠

EP 优化的关键之一，是让 Dispatch、Expert GEMM 和 Combine 尽可能流水化。

例如把 token 分成多个 Chunk：

```text
Chunk 0 Dispatch
        │
        ├── Chunk 0 Expert GEMM
Chunk 1 Dispatch
        │
        ├── Chunk 1 Expert GEMM
Chunk 0 Combine
```

理想情况下：

* 通信 Stream 执行 AllToAll；
* 计算 Stream 执行 Grouped GEMM；
* 不同 Chunk 形成流水线。

暴露在关键路径上的通信时间为：

$$T_{\text{exposed comm}}=T_{\text{comm}}-T_{\text{overlapped}}$$

但 Chunk 太小会增加：

* Collective 启动次数；
* Kernel Launch 次数；
* Pack/Unpack 次数；
* 小消息延迟。

因此需要在流水粒度与启动开销之间折中。

---

## EP 性能分析

分析 EP 时，不能只看平均 GPU Utilization。

至少需要观察：

* 每个 Expert 的 token 数；
* 每个 Rank 的发送和接收 token 数；
* Dispatch AllToAll 时间；
* Combine AllToAll 时间；
* Permute 和 Unpermute 时间；
* Grouped GEMM Shape；
* 最慢 Rank 的完成时间；
* 跨节点流量；
* Expert Hotness；
* 通信与计算重叠比例。

如果时间线表现为：

```text
某些 Rank Grouped GEMM 很长
其他 Rank 长时间等待
```

说明主要问题可能是 Expert Load Imbalance。

如果表现为：

```text
Grouped GEMM 很短
AllToAll 很长
```

则可能是：

* Batch 太小；
* EP Degree 过大；
* EP 跨节点；
* token 太少；
* Pack/Unpack 成本过高。

---

## TP、SP、UP、EP 的统一比较

### 切分对象

```text
TP：
切权重维度和 hidden/head 维度

SP：
切 TP 线性层之间的 token 激活

UP：
在序列分片与 Head 分片之间转换

EP：
切 Expert 集合，并动态迁移 token
```

---

### 主要通信

```text
TP：
AllReduce / ReduceScatter / AllGather

SP：
ReduceScatter + AllGather

UP：
AllToAll + AllToAll

EP：
AllToAllV Dispatch + AllToAllV Combine
```

---

### 通信数据的性质

TP 传输的是：

```text
Dense Activation Partial Results
```

各 Rank 的张量形状通常规则且相同。

SP 传输的是：

```text
Dense Sequence Activation Shards
```

数据分布规则，通信量容易预测。

UP 传输的是：

```text
QKV 和 Attention Output 的规则 Head/Sequence 分块
```

数据量规则，但 AllToAll 对拓扑敏感。

EP 传输的是：

```text
由 Router 动态决定的 Routed Tokens
```

数据量可能不规则，负载与输入内容相关。

---

### 主要目标

TP 的主要目标：

* 分片模型权重；
* 并行执行单层计算；
* 降低大 GEMM 的单 Rank 计算量。

SP 的主要目标：

* 降低激活复制；
* 减少 LayerNorm、Residual 等重复计算；
* 与 TP 的 ReduceScatter/AllGather 配合。

UP 的主要目标：

* 分布式处理长序列 Attention；
* 让每个 Rank 在完整序列上处理部分 Heads；
* 复用本地高性能 Attention Kernel。

EP 的主要目标：

* 分布 MoE Expert 权重；
* 利用稀疏激活；
* 避免所有 Rank 保存全部 Experts。

---

### 对推理阶段的适用性

### Prefill

Prefill token 数较大：

* TP 容易形成较大 GEMM；
* SP 可以降低长序列激活；
* UP 可以有效切分长上下文；
* EP 的 Expert Batch 较大。

因此四者通常都更容易在 Prefill 中发挥效率。

### Decode

Decode 每步 token 数较少：

* TP 容易受到 Collective Latency 限制；
* SP 的激活显存收益较小；
* 标准 UP 难以切分 Query Sequence；
* EP 容易产生小 Expert GEMM 和 AllToAll 固定开销。

所以 Decode 更依赖：

* 足够大的 Continuous Batch；
* 较低 TP Degree；
* Attention DP；
* 高速节点内互联；
* 专门的 Low-Latency EP 通信；
* 通信计算重叠。

---

### 最核心的判断原则

选择这些并行方式时，应分别回答四个问题。

对于 TP：

> 减少的矩阵计算时间，能否覆盖每层 Collective 的延迟？

对于 SP：

> 激活分片和逐元素计算节省，是否值得引入显式 AllGather/ReduceScatter 边界？

对于 UP：

> 长序列 Attention 的计算和显存收益，能否覆盖两次 AllToAll 和布局转换？

对于 EP：

> Expert 权重分布和稀疏计算收益，能否覆盖动态路由、Permute、两次 AllToAllV 与负载不均衡？

推理 Infra 中不存在脱离负载和拓扑的“最佳并行方式”。

最优方案取决于：

* 模型结构；
* Prefill/Decode 比例；
* Batch Size；
* Context Length；
* Expert 数与 Top-$k$；
* GPU 显存；
* NVLink、NVSwitch、PCIe 和跨节点网络；
* TTFT、TPOT 和吞吐量 SLO。
