# AI 推理 Infra 中的各种并行技术

## 1. 为什么推理系统需要并行

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

# 2. 推理工作负载的基本特征

## 2.1 Transformer 中的主要张量

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

## 2.2 KV Cache 显存

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

## 2.3 Prefill 与 Decode

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

# 3. 分布式通信基础

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

## 3.1 Rank、Process Group 与 Collective

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

## 3.2 通信成本模型

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

$$
T_{\text{comm}}
\approx
\alpha+\frac{N}{BW}
$$

但一个 Collective 通常不只包含一步通信。如果算法需要执行 $s$ 个通信步骤，更准确的近似是：

$$
T_{\text{comm}}
\approx
s\alpha+\beta V
$$

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

## 3.3 如何定义“通信量”

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
\text{通信量}
==========

\text{每个 Rank 的发送量}
$$

同时，数据在本地 GPU 内部进行的复制不计入网络通信量。

---

## 3.4 Reduce：什么是归约

Reduce，即归约，是把多个 Rank 上对应位置的数据按照某个运算符合并。

假设有 $p$ 个 Rank，第 $r$ 个 Rank 持有张量：

$$
X^{(r)}\in\mathbb{R}^N
$$

使用求和归约后得到：

$$
Y
=

\sum_{r=0}^{p-1}X^{(r)}
$$

这是逐元素求和，即：

$$
Y_j
===

\sum_{r=0}^{p-1}X_j^{(r)}
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
(a+b)+c
\neq
a+(b+c)
$$

因此，不同通信算法、不同 Rank 数量或不同归约顺序，可能产生很小的浮点误差差异。

---

## 3.5 Gather 与 Scatter

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

## 3.6 AllGather

### 3.6.1 AllGather 的语义

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

### 3.6.2 Ring AllGather 的执行过程

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

### 3.6.3 AllGather 通信量

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

### 3.6.4 AllGather 为什么不是发送 $(p-1)N/p$ 给每一个 Rank

一个容易产生的误解是：每个 Rank 要把自己的分片发送给另外 $p-1$ 个 Rank，因此通信量似乎应该更大。

在最朴素的直接发送算法中，确实可以让 Rank 0 分别把 $X_0$ 发送给其他所有 Rank。但 Ring AllGather 会让数据在 Rank 之间逐跳转发。

Rank 0 不需要亲自把 $X_0$ 发送给所有 Rank。它只需要发送给下一个 Rank，后续 Rank 会继续转发。

因此，每个 Rank 每轮只发送一个大小为 $N/p$ 的分片，执行 $p-1$ 轮，总发送量为：

$$
\frac{p-1}{p}N
$$

这体现了 Collective Algorithm 对通信路径的优化。

---

### 3.6.5 AllGather 的常见用途

AllGather 常用于：

* 收集沿 Hidden Dimension 切分的激活；
* Sequence Parallelism 中恢复完整序列；
* Ulysses 等分布式 Attention 中转换张量布局；
* 收集分片的 Logits；
* 某些 Tensor Parallel Linear 的输入重构。

---

## 3.7 ReduceScatter

### 3.7.1 ReduceScatter 的语义

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

### 3.7.2 将输入切成多个 Chunk

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

### 3.7.3 Ring ReduceScatter 的执行过程

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

### 3.7.4 ReduceScatter 通信量

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

### 3.7.5 ReduceScatter 的常见用途

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

## 3.8 AllReduce

### 3.8.1 AllReduce 的语义

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

### 3.8.2 为什么 AllReduce 等于 ReduceScatter 加 AllGather

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

### 3.8.3 为什么可以分 Chunk 归约

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

### 3.8.4 Ring AllReduce 的通信量

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

### 3.8.5 Ring AllReduce 的时间模型

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

### 3.8.6 AllReduce 在 TP 中的作用

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
W_0\
W_1\
\vdots\
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

## 3.9 AllToAll

### 3.9.1 AllToAll 的语义

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

### 3.9.2 AllToAll 与 AllGather 的区别

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

### 3.9.3 AllToAll 通信量

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

### 3.9.4 AllToAll 在 EP 中的作用

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

## 3.10 Point-to-Point

### 3.10.1 基本语义

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

### 3.10.2 P2P 在 PP 中的作用

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

### 3.10.3 P2P 的同步问题

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

## 3.11 几种通信原语的统一比较

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

## 3.12 为什么相同通信量不代表相同性能

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

## 3.13 小结

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



# 4. Data Parallelism

## 4.1 基本原理

Data Parallelism，简称 DP，是让不同 GPU 保存完整的模型副本，并处理不同请求。

```text
GPU 0：完整模型副本，处理请求 A、B
GPU 1：完整模型副本，处理请求 C、D
GPU 2：完整模型副本，处理请求 E、F
GPU 3：完整模型副本，处理请求 G、H
```

训练中的 DP 需要同步梯度，而推理没有反向传播，因此不同 Replica 之间通常不需要进行参数同步。

---

## 4.2 DP 的优点

DP 的主要优势是通信少。

不同副本各自独立执行推理，GPU 之间一般不需要在每个 Transformer Layer 中同步，因此：

* 不会引入逐层 AllReduce；
* 扩展吞吐量比较直接；
* 单请求延迟不会因为模型并行通信而增加；
* 便于弹性扩缩容；
* 便于故障隔离。

如果单个模型实例已经可以放入一张 GPU 或一个模型并行组，DP 通常是提高吞吐量的首选方式。

---

## 4.3 DP 的局限

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

## 4.4 Continuous Batching

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

## 4.5 Chunked Prefill

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

# 5. Tensor Parallelism

## 5.1 基本原理

Tensor Parallelism，简称 TP，是把单个算子的权重张量切分到多张 GPU 上。

它属于层内并行，即多张 GPU 同时计算 Transformer 的同一层。

考虑线性层：

$$
Y=XW
$$

其中：

$$
X\in\mathbb{R}^{M\times H},
\qquad
W\in\mathbb{R}^{H\times I}
$$

TP 的核心方式包括 Column Parallel 和 Row Parallel。

---

## 5.2 Column Parallel Linear

将权重 $W$ 沿输出维度切分：

$$
W=[W_0,W_1,\ldots,W_{p-1}]
$$

其中：

$$
W_i\in\mathbb{R}^{H\times I/p}
$$

每张 GPU 计算：

$$
Y_i=XW_i
$$

完整输出为：

$$
Y=[Y_0,Y_1,\ldots,Y_{p-1}]
$$

因为每张 GPU 产生不同的输出特征分片，所以称为 Column Parallel。

如果下一层能够直接消费分片形式的 $Y_i$，则这里不需要立即执行通信。

Transformer 中通常使用 Column Parallel 的位置包括：

* Q Projection；
* K Projection；
* V Projection；
* MLP Gate Projection；
* MLP Up Projection。

---

## 5.3 Row Parallel Linear

将输入 $X$ 沿 hidden dimension 切分：

$$
X=[X_0,X_1,\ldots,X_{p-1}]
$$

同时将权重 $W$ 沿输入维度切分：

$$
W=
\begin{bmatrix}
W_0\
W_1\
\vdots\
W_{p-1}
\end{bmatrix}
$$

每张 GPU 计算局部结果：

$$
Y_i=X_iW_i
$$

完整输出是所有局部结果之和：

$$
Y=\sum_{i=0}^{p-1}Y_i
$$

因此需要执行 AllReduce 或 ReduceScatter。

Transformer 中通常使用 Row Parallel 的位置包括：

* Attention Output Projection；
* MLP Down Projection。

---

## 5.4 Transformer Layer 中的 TP

一个典型 Transformer Layer 的 TP 布局如下：

```text
输入 X
  │
  ├─ QKV Column Parallel
  │      每张 GPU 负责部分 Attention Heads
  │
  ├─ Attention
  │
  ├─ Output Projection Row Parallel
  │
  ├─ AllReduce / ReduceScatter
  │
  ├─ Gate + Up Projection Column Parallel
  │
  ├─ Activation
  │
  ├─ Down Projection Row Parallel
  │
  ├─ AllReduce / ReduceScatter
  │
  └─ 输出
```

经典 Megatron 风格 TP 中，每个 Transformer Layer 通常存在两次主要跨卡同步：

1. Attention Output Projection 之后；
2. MLP Down Projection 之后。

对于一个包含 $L$ 层的模型，一次 Decode Step 中可能需要大约 $2L$ 次 TP collective。

---

## 5.5 TP 的显存收益

理想情况下，TP Degree 为 $p$ 时，每张 GPU 保存约 $1/p$ 的分片权重：

$$
M_{\text{weight, per GPU}}
\approx
\frac{M_{\text{weight}}}{p}
$$

对于可以按 Head 切分的 KV Cache，每张 GPU也可能只保存一部分 KV Head：

$$
M_{\text{KV, per GPU}}
\approx
\frac{M_{\text{KV}}}{p}
$$

但对于 GQA 或 MQA，这个关系未必成立，因为 KV Head 数量可能小于 TP Degree。

---

## 5.6 TP 为什么不能无限增大

TP Degree 增大后，每张 GPU 的计算量会减少，但通信次数通常不会减少。

假设 Decode 中的线性层原本为：

$$
[B,H]\times[H,I]
$$

使用 TP Degree $p$ 后，每张 GPU 计算：

$$
[B,H]\times[H,I/p]
$$

如果 $B$ 很小，$I/p$ 也很小，单张 GPU 上的 GEMM 会变得非常瘦，导致：

* Tensor Core 利用率下降；
* kernel launch 开销占比增大；
* 通信时间超过计算时间；
* GPU 数量增加但延迟不降反升。

因此，TP 的收益取决于：

$$
\text{节省的计算时间}

>

\text{新增的通信与同步时间}
$$

---

## 5.7 TP 与硬件拓扑

TP 通信发生得非常频繁，因此通常应限制在高速互联域中。

优先级通常是：

```text
NVSwitch / NVLink
    优于
PCIe
    优于
跨节点 InfiniBand / RoCE
```

工程上常见的部署方式是：

* 节点内使用 TP；
* 节点之间使用 PP 或 DP。

---

## 5.8 TP 与 MHA、GQA、MQA

对于 MHA，可以自然地按 Attention Head 切分：

```text
GPU 0：Heads 0～7
GPU 1：Heads 8～15
GPU 2：Heads 16～23
GPU 3：Heads 24～31
```

对于 GQA，Query Head 数大于 KV Head 数。例如：

```text
Query Heads = 64
KV Heads    = 8
TP Degree   = 8
```

每张 GPU 可以保存一个 KV Head。

但如果：

```text
Query Heads = 64
KV Heads    = 8
TP Degree   = 16
```

KV Head 数不足以继续均匀分片。系统可能需要：

* 复制 KV Head；
* 让多个 TP Rank 共享同一 KV Head；
* 使用不均匀分片；
* 限制 TP Degree；
* 改用 Context Parallelism。

因此，在 GQA/MQA 模型中，TP Degree 通常不能只根据 Query Head 数决定。

---

# 6. Pipeline Parallelism

## 6.1 基本原理

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

## 6.2 PP 的显存特点

如果模型总权重为 $M$，使用 $p$ 个均匀 Stage 后，每个 Stage 的权重约为：

$$
M_{\text{weight, per stage}}
\approx
\frac{M}{p}
$$

但是，Stage 之间传递的隐藏状态通常仍然具有完整 hidden size：

$$
X\in\mathbb{R}^{B\times S\times H}
$$

也就是说，PP 主要切分模型层数，而不是切分每一层内部的 hidden dimension。

---

## 6.3 Pipeline Bubble

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

## 6.4 PP 对单 Token 延迟的影响

对于单个请求的 Decode，第 $t+1$ 个 token 必须等待第 $t$ 个 token 完整通过所有 Stage。

因此，PP 通常不能直接降低单个 token 的端到端延迟。

它的主要作用是：

* 解决模型跨节点部署问题；
* 降低单 GPU 权重显存；
* 在高并发下通过流水线提高吞吐量。

---

## 6.5 PP 的负载均衡

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

# 7. Sequence Parallelism

Sequence Parallelism，简称 SP，在不同系统中可能指不同技术，需要区分。

---

## 7.1 Megatron Sequence Parallelism

经典 Megatron TP 主要切分线性层，但 LayerNorm、Residual 和逐元素算子可能仍然在每张 GPU 上保存完整的序列激活：

$$
X\in\mathbb{R}^{B\times S\times H}
$$

Sequence Parallelism 沿序列维度切分这些激活：

$$
X_i\in\mathbb{R}^{B\times S/p\times H}
$$

典型过程是：

```text
Row Parallel Linear
        │
ReduceScatter
        │
每张 GPU 得到部分序列
        │
LayerNorm / Dropout / Residual
        │
AllGather
        │
进入下一次 Column Parallel Linear
```

它将某些 AllReduce 拆成 ReduceScatter 和 AllGather，并让中间激活保持序列分片。

---

## 7.2 SP 的主要目的

Megatron SP 的主要目标包括：

* 减少重复保存的激活；
* 降低 LayerNorm、Dropout、Residual 等算子的激活显存；
* 避免每个 TP Rank 都执行完全相同的逐元素计算。

训练中需要保存大量前向激活，因此 SP 的收益非常明显。

推理中没有反向传播，中间激活生命周期较短，因此经典 Megatron SP 的重要性通常低于训练，但在大 Batch、长 Prefill 或显存紧张场景下仍然有价值。

---

# 8. Context Parallelism

## 8.1 基本原理

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

## 8.2 CP 与 SP 的区别

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

# 9. Ulysses Sequence Parallelism

## 9.1 布局转换

Ulysses 初始按序列切分：

```text
GPU 0：部分 token，全部 Heads
GPU 1：部分 token，全部 Heads
GPU 2：部分 token，全部 Heads
GPU 3：部分 token，全部 Heads
```

通过 AllToAll，将布局转换为按 Head 切分：

```text
GPU 0：全部 token，部分 Heads
GPU 1：全部 token，部分 Heads
GPU 2：全部 token，部分 Heads
GPU 3：全部 token，部分 Heads
```

此时每张 GPU 都拥有完整序列，但只负责部分 Attention Heads，可以独立执行本地 Attention。

计算完成后，再执行一次反向 AllToAll，恢复序列分片布局。

---

## 9.2 Ulysses 的通信过程

假设输入 Q、K、V 为：

$$
Q,K,V\in\mathbb{R}^{B\times S/p\times N_h\times D_h}
$$

经过 AllToAll 后，每张 GPU 获得：

$$
Q_i,K_i,V_i
\in
\mathbb{R}^{B\times S\times N_h/p\times D_h}
$$

之后每张 GPU 对自己的 Head 子集计算完整序列 Attention。

---

## 9.3 Ulysses 的特点

优点：

* 可以复用成熟的单卡 FlashAttention Kernel；
* 每张 GPU 负责完整序列上的部分 Head；
* 通信集中在两次 AllToAll；
* 比较容易实现 Attention 计算与通信的结构化组合。

局限：

* 并行度通常受到 Attention Head 数量限制；
* Head 数必须能够被 Ulysses Degree 合理切分；
* AllToAll 对互联带宽和拓扑要求较高；
* Decode 时序列 Query 长度为 1，利用方式与 Prefill 不同。

---

# 10. Ring Attention

## 10.1 基本思想

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

## 10.2 Online Softmax

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
l
=

e^{m_{\text{old}}-m}l_{\text{old}}
+
e^{m_{\text{new}}-m}l_{\text{new}}
$$

输出累积值更新为：

$$
O
=

e^{m_{\text{old}}-m}O_{\text{old}}
+
e^{m_{\text{new}}-m}O_{\text{new}}
$$

最后输出为：

$$
\operatorname{Attention}(Q,K,V)=\frac{O}{l}
$$

这与 FlashAttention 中的 Online Softmax 思路类似。

---

## 10.3 Ring Attention 的优缺点

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

## 10.4 Ulysses 与 Ring Attention 的区别

Ulysses 的核心是：

> 将序列分片布局通过 AllToAll 转换为 Head 分片布局。

Ring Attention 的核心是：

> Query 保持在本地，KV Block 在 GPU 之间环形移动。

Ulysses 的通信相对集中，Ring Attention 的通信被分散到多个迭代中。

实际系统也可以将二者组合：

$$
\text{SP Degree}
================

\text{Ulysses Degree}
\times
\text{Ring Degree}
$$

---

# 11. Decode 阶段的 Context Parallelism

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

# 12. Expert Parallelism

## 12.1 MoE 模型

Mixture-of-Experts 模型包含多个 Expert，但每个 token 只选择少量 Expert。

设：

* Expert 总数为 $E$；
* 每个 token 选择 Top-$k$ Experts；
* GPU 数量为 $p$。

Expert Parallelism，简称 EP，将不同 Expert 放在不同 GPU 上。

例如：

```text
GPU 0：Expert 0～7
GPU 1：Expert 8～15
GPU 2：Expert 16～23
GPU 3：Expert 24～31
```

---

## 12.2 EP 的执行流程

MoE Layer 的典型执行流程如下：

```text
输入 Hidden States
        │
        ▼
      Router
        │
为每个 token 选择 Top-k Experts
        │
        ▼
Token Permute / Pack
        │
        ▼
      AllToAll
        │
        ▼
每张 GPU 获得属于本地 Expert 的 token
        │
        ▼
Grouped GEMM / Expert FFN
        │
        ▼
      AllToAll
        │
        ▼
Token Unpermute / Combine
        │
        ▼
      输出结果
```

第一次 AllToAll 被称为 Dispatch，第二次通常被称为 Combine。

---

## 12.3 EP 与 TP 的区别

对于 TP，一个 Expert 本身被切到多张 GPU：

```text
Expert 0：
GPU 0 保存一部分
GPU 1 保存一部分
GPU 2 保存一部分
GPU 3 保存一部分
```

对于 EP，不同 GPU 保存不同 Expert：

```text
GPU 0：Expert 0～3
GPU 1：Expert 4～7
GPU 2：Expert 8～11
GPU 3：Expert 12～15
```

TP 的主要通信通常是 AllReduce，而 EP 的主要通信通常是 AllToAll。

TP 是对单个 Dense Tensor 做分片；EP 是根据 Router 结果动态迁移 token。

---

## 12.4 Grouped GEMM

每个 Expert 接收到的 token 数通常不同。

假设：

```text
Expert 0：64 个 token
Expert 1：18 个 token
Expert 2：92 个 token
Expert 3：7 个 token
```

不能简单把所有 Expert 当成相同形状的普通 Batched GEMM。

Grouped GEMM 允许一次 Kernel Launch 处理多个不同 $M$ 维度的矩阵乘法：

$$
X_eW_e,\qquad e=0,1,\ldots,E_{\text{local}}-1
$$

其中每个 $X_e$ 的 token 数不同。

Grouped GEMM 的目标包括：

* 减少 kernel launch 数量；
* 提高小 Expert GEMM 的执行效率；
* 共享调度和 Workspace；
* 改善低 token 数 Expert 的利用率。

---

## 12.5 EP 的负载不均衡

Router 可能把大量 token 发送到少数热点 Expert：

```text
Expert 0：300 个 token
Expert 1：280 个 token
Expert 2：15 个 token
Expert 3：8 个 token
```

一次 MoE Layer 必须等待最慢的 Rank 完成，因此热点 Expert 会产生 Straggler。

常见优化包括：

* Router Load Balancing Loss；
* Expert Capacity；
* Expert Replication；
* Redundant Experts；
* Expert Placement；
* 动态 Expert 迁移；
* Expert Parallel Load Balancer；
* 分层 AllToAll；
* 通信与 Grouped GEMM 重叠。

---

## 12.6 Decode 下 EP 的问题

Decode 每个 Step 只有每个请求的一个新 token，因此总 token 数约为 Batch Size $B$。

如果 Batch 很小，分配到单个 Expert 的 token 数可能只有几个，导致：

* Expert GEMM 很小；
* Tensor Core 利用率低；
* Grouped GEMM 效率差；
* AllToAll 固定延迟无法摊薄。

因此，EP 通常更依赖较高并发和较大的全局 Decode Batch。

---

# 13. Attention DP 与 Expert EP

大型 MoE 推理系统常使用不同的并行策略处理 Attention 和 MoE FFN。

```text
Attention：Data Parallel
MoE FFN：Expert Parallel
```

具体过程如下：

```text
不同 DP Rank 处理不同请求
        │
        ▼
各自执行本地 Attention
        │
        ▼
进入 MoE Layer
        │
        ▼
全局 AllToAll Dispatch
        │
        ▼
各 GPU 执行本地 Expert
        │
        ▼
全局 AllToAll Combine
        │
        ▼
回到原来的 DP 请求布局
```

这种设计的优势是：

* Attention 不需要执行高频 TP AllReduce；
* 不同 DP Rank 的 token 可以汇总成更大的 Expert Batch；
* Expert 权重可以分布在更大的 EP Group；
* Attention 和 Expert 可以分别扩展。

这说明现代 MoE 推理不一定使用统一的一个并行维度，而是不同模块采用不同 Process Group。

---

# 14. Prefill–Decode Disaggregation

## 14.1 基本原理

传统部署中，Prefill 和 Decode 共享同一组 GPU：

```text
GPU Pool：
Prefill + Decode
```

长 Prefill 任务可能占用 GPU 较长时间，干扰正在执行的 Decode，导致 TPOT 和 ITL 抖动。

Prefill–Decode Disaggregation，简称 PD 分离，将两类请求放入不同 GPU Pool：

```text
请求
 │
 ▼
Prefill Pool
 │
 │ 生成 KV Cache
 │
 ▼
KV Cache Transfer
 │
 ▼
Decode Pool
 │
 ▼
持续生成 token
```

---

## 14.2 PD 分离的收益

Prefill Pool 和 Decode Pool 可以采用不同的硬件和并行策略。

例如：

```text
Prefill Pool：
TP = 8
适合长 Prompt、大 GEMM

Decode Pool：
TP = 2
DP = 4
适合高并发、小步 Decode
```

Prefill Pool 可以针对以下目标优化：

* TTFT；
* 大矩阵计算；
* 大 token Batch；
* 长序列 Attention；
* 更高算力利用率。

Decode Pool 可以针对以下目标优化：

* TPOT；
* KV Cache 容量；
* HBM 带宽；
* 高并发；
* 更低通信延迟。

---

## 14.3 KV Cache 传输

PD 分离的关键代价是传输 KV Cache。

单个请求的 KV 数据量近似为：

$$
M_{\text{KV, request}}
======================

2LSN_{kv}D_hb
$$

当 Prompt 很长时，KV Cache 传输量可能非常大。

传输路径可能包括：

```text
Prefill GPU HBM
→ RDMA Buffer
→ 网络
→ Decode GPU HBM
```

需要考虑：

* RDMA 带宽；
* GPU Direct RDMA；
* Staging Buffer；
* KV Cache 序列化；
* KV Block Layout；
* 目标 Decode Worker 的可用空间；
* KV 传输与 Decode 调度的重叠。

---

## 14.4 Prefill 与 Decode 的 TP Degree 不同

假设 Prefill 使用 TP $=8$，Decode 使用 TP $=4$。

两侧对 KV Head 的分片布局可能不同。

Prefill 侧可能是：

```text
8 个 Rank：每个 Rank 保存一部分 KV Heads
```

Decode 侧可能是：

```text
4 个 Rank：每个 Rank 保存更大的 KV Head 分片
```

因此，需要执行某种重新分片：

```text
Prefill TP Layout
→ Gather / Repack
→ Network Transfer
→ Scatter
→ Decode TP Layout
```

如果两侧布局完全一致，KV 传输会简单很多。

---

## 14.5 PD 分离并非总是更好

PD 分离减少了 Prefill 和 Decode 的互相干扰，但引入了：

* KV Cache 传输；
* 跨 Pool 排队；
* 额外路由；
* 资源比例配置；
* 故障恢复复杂度；
* Prefix Cache 局部性问题。

其收益取决于：

* Prompt 长度；
* Output 长度；
* TTFT SLO；
* TPOT SLO；
* KV 传输带宽；
* 请求到达模式；
* Prefill 与 Decode 的资源比例。

对于短 Prompt，KV 传输代价较小；对于超长 Prompt，传输成本可能非常显著。

---

# 15. Attention–FFN Disaggregation

Attention–FFN Disaggregation，简称 AFD，是比 PD 分离更细粒度的解耦方式。

它将 Attention 和 FFN 或 MoE Expert 放在不同 GPU Pool：

```text
Attention Pool
      │
      ▼
Hidden States Transfer
      │
      ▼
FFN / Expert Pool
      │
      ▼
Hidden States Transfer
      │
      ▼
下一层 Attention
```

其动机是：

* Attention 主要保存 KV Cache，状态性强；
* Attention 容易受显存容量和 HBM 带宽限制；
* MoE FFN 主要保存 Expert 权重；
* Expert 执行具有动态 AllToAll 和 Grouped GEMM；
* 两类模块适合不同 GPU 配置和扩展方式。

AFD 允许分别配置：

* Attention GPU 数量；
* Expert GPU 数量；
* Attention DP 或 CP；
* Expert EP Degree；
* 两侧 Batch；
* 两侧 GPU 类型。

但它的通信非常频繁，因为每个 Transformer Layer 都可能需要在两个 Pool 之间传输 Hidden States。

因此，AFD 对网络拓扑、延迟和通信计算重叠要求极高。

---

# 16. Speculative Decoding

Speculative Decoding 是算法级并行，而不是传统的模型切分。

普通自回归生成存在严格依赖：

$$
x_t\rightarrow x_{t+1}\rightarrow x_{t+2}
$$

Draft Model 可以一次提出多个候选 token：

```text
Draft Model：
A → B → C → D
```

Target Model 一次性验证多个候选 token：

```text
Target Model：
并行验证 A、B、C、D
```

如果前几个 token 被接受，一次 Target Model Forward 就可以推进多个 token。

假设一次提出 $k$ 个 token，平均接受长度为 $a$，则理想情况下，每次 Target Forward 可以推进约 $a$ 个 token。

实际加速取决于：

* Draft Model 速度；
* Token 接受率；
* Verification Kernel 效率；
* Draft 和 Target 的资源竞争；
* KV Cache Commit 和 Rollback；
* Batch 中不同请求的接受长度差异。

常见方案包括：

* 独立 Draft Model；
* Medusa；
* EAGLE；
* Multi-Token Prediction；
* N-gram Speculation；
* Lookahead Decoding。

---

# 17. DiT 和视频生成中的并行技术

## 17.1 DiT 的计算特征

Diffusion Transformer 通常处理图像或视频 Latent。

对于视频 Latent：

$$
X\in\mathbb{R}^{T\times H\times W\times C}
$$

经过 Patchify 后形成 token 序列：

$$
S=T'\times H'\times W'
$$

视频长度和分辨率增大时，token 数量可能非常大。

与 LLM 不同，Diffusion 模型还需要执行多个去噪步骤：

```text
Step 0 → Step 1 → Step 2 → ... → Step N
```

因此 DiT 可以利用以下维度进行并行：

* 样本；
* CFG 分支；
* 空间 token；
* 时间 token；
* Transformer Layer；
* Patch；
* Diffusion Step 间的时间冗余。

---

## 17.2 DiT Data Parallelism

不同 GPU 生成不同样本：

```text
GPU 0：Prompt A
GPU 1：Prompt B
GPU 2：Prompt C
GPU 3：Prompt D
```

或者对于同一个 Prompt，使用不同 Seed 生成多个候选结果。

这可以提高吞吐量，但不会降低单个视频的生成延迟。

---

## 17.3 CFG Parallelism

Classifier-Free Guidance 通常包含：

* Conditional Branch；
* Unconditional Branch。

传统方法将两个分支组成一个 Batch：

```text
Batch：
[Conditional, Unconditional]
```

CFG Parallelism 将两个分支放到不同 GPU：

```text
GPU 0：Conditional
GPU 1：Unconditional
```

两个分支在大部分 DiT Forward 中彼此独立，只需要在每个 Diffusion Step 结束后汇总结果并进行 Guidance：

$$
\epsilon_{\text{guided}}
========================

\epsilon_{\text{uncond}}
+
w\left(
\epsilon_{\text{cond}}
----------------------

\epsilon_{\text{uncond}}
\right)
$$

CFG Parallelism 的通信频率相对较低，是 DiT 中非常自然的两路并行方法。

---

## 17.4 DiT Sequence Parallelism

DiT 可以沿空间和时间 token 切分：

```text
GPU 0：部分视频 token
GPU 1：部分视频 token
GPU 2：部分视频 token
GPU 3：部分视频 token
```

Attention 可以采用：

* Ulysses；
* Ring Attention；
* Ulysses 与 Ring 的混合并行。

总并行度可以表示为：

$$
P_{\text{SP}}
=============

P_{\text{Ulysses}}
\times
P_{\text{Ring}}
$$

例如：

```text
Ulysses Degree = 2
Ring Degree    = 4
总 SP Degree   = 8
```

Ulysses 可以更多利用节点内高速 AllToAll，Ring 可以扩展到更大的设备范围。

---

## 17.5 PipeFusion

传统 PP 沿模型 Layer 切分，但单个图像或视频可能只有一个大样本，难以形成足够多的 Microbatch。

PipeFusion 将 Latent 进一步切成 Patch，并让 Patch 在 Pipeline 中流动：

```text
Stage 0：Patch A → Patch B → Patch C
Stage 1：          Patch A → Patch B → Patch C
Stage 2：                    Patch A → Patch B → Patch C
```

它同时利用：

* 模型 Layer 分片；
* 图像或视频 Patch 分片；
* Diffusion 相邻 Step 的时间相似性；
* 通信与计算重叠。

PipeFusion 的主要目标是：

* 分片模型权重；
* 分片激活；
* 填充 Pipeline；
* 支持更高分辨率和更长视频。

其难点包括：

* Pipeline Bubble；
* Patch 数量选择；
* Stage 负载不均衡；
* Stale Feature；
* 精度与延迟之间的权衡。

---

## 17.6 DiT 混合并行

DiT 中可以组合多个并行维度：

$$
N_{\text{GPU}}=
DP
\times
CFG
\times
Ulysses
\times
Ring
\times
PipeFusion
$$

例如：

```text
DP         = 1
CFG        = 2
Ulysses    = 2
Ring       = 2
PipeFusion = 2
```

则总 GPU 数为：

$$
1\times2\times2\times2\times2=16
$$

实际选择时通常遵循：

1. 有 CFG 时，优先利用天然的两路 CFG Parallel；
2. token 数很大时增加 Sequence Parallel；
3. 模型权重或激活仍放不下时增加 PipeFusion；
4. 有多个请求或样本时增加 DP。

---

# 18. 混合并行与 Process Group

## 18.1 经典 3D Parallelism

经典 3D Parallelism 通常指：

$$
N_{\text{GPU}}=
DP\times TP\times PP
$$

加入 Context Parallelism 后，可以扩展为：

$$
N_{\text{GPU}}=
DP\times TP\times PP\times CP
$$

---

## 18.2 EP 不一定是简单独立维度

在现代 MoE 系统中，EP 不一定能简单写成上述公式中的一个独立乘数。

例如：

```text
Attention：
DP = 8

MoE：
EP = 64
```

Attention 和 Expert 可能使用不同的 Process Group。

一个 GPU Rank 可能同时属于：

```text
TP Group
DP Group
PP Group
CP Group
EP Group
Attention DP Group
KV Transfer Group
```

因此，实际系统更像是多个重叠通信域，而不是单一规则网格。

---

# 19. 典型配置示例

## 19.1 模型能放入单卡，追求吞吐

```text
DP = 8
TP = 1
PP = 1
```

每张 GPU 保存一个完整模型副本。

优点是通信最少、吞吐扩展简单。

---

## 19.2 70B Dense 模型，单节点 8 卡

```text
TP = 8
DP = 1
PP = 1
```

模型权重和 Attention Heads 分布在 8 张 GPU 上。

适合单节点 NVLink 或 NVSwitch 环境。

---

## 19.3 模型无法放入一个节点

```text
节点 0：
TP = 8
Pipeline Stage 0

节点 1：
TP = 8
Pipeline Stage 1
```

总配置：

```text
TP = 8
PP = 2
```

节点内执行高频 TP 通信，节点间只传输 Pipeline Boundary 激活。

---

## 19.4 长上下文模型

```text
TP = 4
CP = 2
```

TP 分片权重和 Attention Heads，CP 分片序列或 KV Cache。

---

## 19.5 大型 MoE 模型

```text
Attention：Data Parallel
Experts：Expert Parallel
```

Attention 处理不同请求，MoE Layer 中通过 AllToAll 交换 token。

---

## 19.6 视频 DiT

```text
CFG        = 2
Ulysses    = 2
Ring       = 2
PipeFusion = 2
```

同时利用 CFG 分支、序列 token 和 Patch Pipeline。

---

# 20. 如何选择并行方案

## 20.1 第一步：估算显存

总显存占用可以粗略分为：

$$
M_{\text{total}}
================

M_{\text{weights}}
+
M_{\text{KV}}
+
M_{\text{activations}}
+
M_{\text{workspace}}
+
M_{\text{runtime}}
$$

其中 Runtime 部分还可能包括：

* CUDA Context；
* NCCL Buffer；
* CUDA Graph；
* 内存碎片；
* Kernel Workspace；
* 临时 Tensor；
* Prefix Cache 元数据。

不能只根据参数量判断模型是否能够部署。

---

## 20.2 如果模型权重放不下

优先考虑：

* 节点内 TP；
* 跨节点 PP；
* MoE 模型使用 EP；
* DiT 使用 PP 或 PipeFusion；
* 权重量化。

---

## 20.3 如果 KV Cache 放不下

优先考虑：

* GQA 或 MQA；
* KV Cache Quantization；
* TP KV Head Sharding；
* Context Parallelism；
* KV Cache Offload；
* PD 分离；
* 降低最大并发；
* 降低最大上下文长度。

---

## 20.4 如果目标是降低 TTFT

可以考虑：

* 提高 Prefill TP Degree；
* Prefill Context Parallel；
* 更大的 Prefill Token Batch；
* Chunked Prefill；
* 独立 Prefill Pool；
* 优化长序列 Attention Kernel；
* 减少 Prefill 排队时间。

---

## 20.5 如果目标是降低 TPOT

可以考虑：

* 减少 TP Degree；
* 减少逐层 collective；
* 增大 Decode Batch；
* Attention DP；
* 提高 KV Cache 局部性；
* Speculative Decoding；
* 避免长 Prefill 干扰 Decode；
* 将 TP 限制在高速互联域。

---

## 20.6 如果目标是提高吞吐量

可以考虑：

* DP；
* Continuous Batching；
* 更大的全局 Batch；
* EP；
* Prefix Cache；
* Paged KV Cache；
* 在 SLO 允许范围内增加 Batch；
* 请求级调度和负载均衡。

---

## 20.7 如果目标是支持超长上下文

可以考虑：

* Context Parallelism；
* Ring Attention；
* Ulysses；
* KV Cache 分片；
* 分布式 Attention；
* KV Cache Quantization；
* Prefix Cache；
* KV Offload。

---

# 21. 并行策略与网络拓扑

可以采用以下原则：

```text
高频、细粒度通信：
TP、Ulysses、部分 EP
→ 尽量限制在 NVLink / NVSwitch 域内

较低频、较大张量通信：
PP、KV Cache Transfer
→ 可以跨节点

完全独立：
DP Replica
→ 可以跨节点甚至跨集群
```

对于 EP，还要考虑 Expert Placement：

* 热点 Expert 是否位于同一节点；
* 跨节点 AllToAll 是否均衡；
* 是否需要复制热点 Expert；
* 是否可以使用分层 AllToAll；
* 是否能将 Dispatch 与计算重叠。

---

# 22. 常见误区

## 22.1 GPU 越多一定越快

错误。

增加 GPU 会减少单 GPU 的计算量，但可能增加：

* 通信延迟；
* 同步开销；
* kernel launch 占比；
* Pipeline Bubble；
* 调度复杂度；
* Straggler 等待。

只有当：

$$
\text{计算时间减少量}

>

\text{通信和同步增加量}
$$

扩展 GPU 才会获得实际加速。

---

## 22.2 DP 可以降低单请求延迟

通常错误。

DP 让不同请求在不同模型副本上运行，不会让多张 GPU 同时计算同一个请求。

DP 主要提高吞吐量和并发容量。

---

## 22.3 TP Degree 越大越好

错误。

过大的 TP Degree 会让单 GPU GEMM 变小，同时增加通信占比。

低并发 Decode 尤其容易因为过度 TP 而变慢。

---

## 22.4 PP 可以直接降低单 Token 延迟

通常错误。

单个 token 仍然必须顺序经过全部 Stage。

PP 的主要价值是容量扩展和高并发下的流水线吞吐。

---

## 22.5 Sequence Parallelism 就是 Context Parallelism

错误。

经典 SP 主要切分 TP 中的序列激活；CP 直接对长上下文 Attention 和 KV Cache 进行分布式处理。

---

## 22.6 EP 消除了 MoE 通信

错误。

EP 避免了对每个 Expert 使用传统 TP AllReduce，但引入了 Token Dispatch 和 Combine，通常对应两次 AllToAll。

EP 只是把通信模式从 Dense Tensor Synchronization 变成 Sparse Token Routing。

---

## 22.7 PD 分离一定优于聚合部署

错误。

PD 分离减少阶段干扰，但增加 KV Cache 传输、排队和资源调度开销。

是否有收益必须根据 Prompt 长度、输出长度、网络带宽和 SLO 判断。

---

# 23. 性能分析方法

## 23.1 服务层指标

需要观察：

* TTFT；
* TPOT；
* ITL；
* P50、P95、P99 延迟；
* Requests/s；
* Output Tokens/s；
* Goodput；
* SLO 达成率。

Goodput 指在满足延迟 SLO 的前提下，系统真正完成的有效吞吐量。

---

## 23.2 GPU 指标

需要观察：

* SM Utilization；
* Tensor Core Utilization；
* HBM Bandwidth；
* GEMM Shape；
* Kernel Duration；
* Kernel Launch Gap；
* KV Cache Occupancy；
* CUDA Graph 命中率；
* GPU 间执行时间差异。

---

## 23.3 通信指标

需要观察：

* NCCL Kernel 时间；
* AllReduce 时间；
* AllToAll 时间；
* AllGather 时间；
* ReduceScatter 时间；
* 跨节点通信量；
* 通信与计算重叠比例；
* Rank 间同步等待时间。

---

## 23.4 MoE 指标

需要观察：

* 每个 Expert 的 token 数；
* 最大 Expert Load；
* 平均 Expert Load；
* Expert Load 方差；
* Grouped GEMM Shape；
* Dispatch 和 Combine 时间；
* 热点 Expert；
* Expert Replication 命中情况。

---

## 23.5 调度指标

需要观察：

* Running Requests；
* Waiting Requests；
* Batched Tokens；
* Prefill Tokens；
* Decode Tokens；
* Chunked Prefill 大小；
* KV Block 使用率；
* Prefix Cache Hit Rate；
* 请求在各个 Replica 间的分布。

---

# 24. 从 Nsight Systems 时间线判断问题

## 24.1 GEMM 很短，NCCL 很长

可能说明：

* TP Degree 过大；
* Batch 太小；
* GEMM 粒度太小；
* 通信链路太慢；
* TP 跨节点。

可以尝试：

* 降低 TP Degree；
* 增大 Batch；
* 使用 DP 扩吞吐；
* 把 TP 限制在节点内。

---

## 24.2 Pipeline Stage 大量空闲

可能说明：

* Microbatch 太少；
* 并发请求不足；
* PP Degree 过大；
* Stage 划分不均衡。

可以尝试：

* 增加 Microbatch；
* 增加请求并发；
* 调整 Layer Partition；
* 降低 PP Degree。

---

## 24.3 EP Rank 执行时间差异很大

可能说明：

* Expert Load 不均衡；
* 热点 Expert；
* Expert Placement 不合理；
* 跨节点通信不均衡。

可以尝试：

* Expert Replication；
* 动态 Expert Placement；
* Router Load Balancing；
* 增大 Batch；
* 使用分层 AllToAll。

---

## 24.4 Decode 被长 Prefill 阻塞

可能说明 Prefill 和 Decode 存在严重干扰。

可以尝试：

* Chunked Prefill；
* Decode Priority Scheduling；
* 限制单次 Prefill Token 数；
* PD 分离；
* 独立设置 Prefill 和 Decode Batch。

---

## 24.5 GPU 利用率高但延迟仍然很差

GPU Utilization 高并不一定代表系统高效。

可能原因包括：

* NCCL Kernel 占用 GPU；
* 大量无效或重复计算；
* 某些 Rank 计算完后等待其他 Rank；
* Batch 过大导致排队延迟；
* 吞吐提高但 TPOT 恶化；
* Pipeline Bubble 被平均利用率掩盖。

最终应以 TTFT、TPOT、Goodput 和 SLO 为主要评价标准。

---

# 25. 总结

可以用下面的方式记忆各种并行技术。

```text
Data Parallelism
→ 切请求
→ 主要提高吞吐量

Tensor Parallelism
→ 切层内权重和张量
→ 多 GPU 同时计算一层
→ 高频 collective

Pipeline Parallelism
→ 切模型层
→ Stage 间传输激活
→ 依赖多个请求填充流水线

Sequence Parallelism
→ 切序列激活
→ 常与 TP 配合
→ 减少重复激活和逐元素计算

Context Parallelism
→ 切上下文和 KV Cache
→ 支持超长序列
→ 用通信换容量和带宽

Ulysses
→ 序列分片与 Head 分片之间做 AllToAll 转换

Ring Attention
→ Query 留在本地
→ KV Block 在 GPU 间环形移动

Expert Parallelism
→ 切 MoE Experts
→ 通过 AllToAll 路由 token

Prefill–Decode Disaggregation
→ 切推理阶段
→ 独立优化 TTFT 和 TPOT
→ 需要传输 KV Cache

Attention–FFN Disaggregation
→ 切 Transformer 模块
→ 独立扩展 Attention 和 Expert

CFG Parallelism
→ 切 Diffusion 的 Conditional 和 Unconditional 分支

PipeFusion
→ 切 DiT Layer 和 Patch
→ 构造 Patch 级流水线

Speculative Decoding
→ 并行验证多个候选 token
→ 缓解自回归生成的串行性
```

选择并行方案时，最重要的原则是：

> 先使用尽可能少的模型并行解决容量问题，再通过请求级并行扩大吞吐量；只有当节省的计算时间能够覆盖新增通信、同步和调度成本时，才继续增大单请求并行度。

推理 Infra 的核心并不是“尽可能多地并行”，而是在给定模型结构、请求负载、硬件拓扑和延迟 SLO 下，找到计算、显存、通信与调度之间的平衡。
