- [为什么推理系统需要并行](#为什么推理系统需要并行)
- [推理工作负载的基本特征](#推理工作负载的基本特征)
        - [Transformer 中的主要张量](#transformer-中的主要张量)
        - [KV Cache 显存](#kv-cache-显存)
        - [Prefill 与 Decode](#prefill-与-decode)
                - [Prefill](#prefill)
                - [Decode](#decode)
- [Data Parallelism](#data-parallelism)
        - [基本原理](#基本原理)
        - [DP 的优点](#dp-的优点)
        - [DP 的局限](#dp-的局限)
        - [Continuous Batching](#continuous-batching)
        - [Chunked Prefill](#chunked-prefill)
- [Pipeline Parallelism](#pipeline-parallelism)
        - [基本原理](#基本原理-1)
        - [PP 的显存特点](#pp-的显存特点)
        - [Pipeline Bubble](#pipeline-bubble)
        - [PP 对单 Token 延迟的影响](#pp-对单-token-延迟的影响)
        - [PP 的负载均衡](#pp-的负载均衡)
- [Context Parallelism](#context-parallelism)
        - [基本原理](#基本原理-2)
        - [CP 与 SP 的区别](#cp-与-sp-的区别)
- [Ring Attention](#ring-attention)
        - [基本思想](#基本思想)
        - [Online Softmax](#online-softmax)
        - [Ring Attention 的优缺点](#ring-attention-的优缺点)
        - [Ulysses 与 Ring Attention 的区别](#ulysses-与-ring-attention-的区别)
- [Decode 阶段的 Context Parallelism](#decode-阶段的-context-parallelism)

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

