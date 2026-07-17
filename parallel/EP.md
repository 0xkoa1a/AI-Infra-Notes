# Expert Parallelism

Expert Parallelism（EP）沿 MoE Expert 维度切分参数和计算。Token 先由 Router 选择 Expert，再通过 Dispatch AllToAllV 发送到 Expert 所在 Rank；计算完成后通过 Combine AllToAllV 返回原 Rank。

EP 的核心不是规则张量切片，而是数据相关的动态路由。因此分析 EP 时必须同时看通信字节数、消息大小分布、Expert 负载和最慢 Rank。本文统一采用每 Rank 发送量。

## 导航

* [背景与整体流程](#背景与整体流程)
* [路由与数据重排](#路由与数据重排)
* [Dispatch、Combine 与通信量](#dispatchcombine-与通信量)
* [通信量统一总结](#通信量统一总结)
* [本地 Expert 计算](#本地-expert-计算)
* [与其他并行维度组合](#与其他并行维度组合)
* [负载均衡与拓扑优化](#负载均衡与拓扑优化)
* [性能瓶颈总结](#性能瓶颈总结)

---

## 背景与整体流程

### EP 的背景：为什么 MoE 需要不同的并行方式

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

### MoE Layer 的整体流程

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

## 路由与数据重排

### Router

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

### 为什么需要 Token Permute

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

## Dispatch、Combine 与通信量

### Dispatch AllToAll

每个源 Rank 把 Routed Token 发送到目标 Expert 所在 Rank。

假设 Rank $r$ 要发送给 Rank $j$ 的 token Buffer 为：

$$X_{r\rightarrow j}$$

执行 Dispatch AllToAll 后，Rank $j$ 得到：

$$X'_j=[X_{0\rightarrow j},X_{1\rightarrow j},\ldots,X_{p-1\rightarrow j}]$$

这些 token 都将由 Rank $j$ 上的本地 Experts 处理。

---

### 为什么 MoE 通常需要 AllToAllV

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

### EP Dispatch 通信量

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

### 非均匀路由下的通信量

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

### Combine AllToAll

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

### EP 的两次通信为什么通常无法省掉

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

## 通信量统一总结

设每 Rank 初始持有 $T_{\text{local}}$ 个 token，每个 token 选择 Top-$k$ Experts，hidden size 为 $H$，每个元素占 $b$ 字节，EP Degree 为 $p$。

在 Expert 均匀放置且路由均匀的理想条件下，token 路由到本 Rank 的概率为 $1/p$，发送到远端的概率为 $(p-1)/p$：

$$V_{\text{dispatch}}\approx\frac{p-1}{p}kT_{\text{local}}Hb$$

Combine 阶段需要将 Expert 输出送回原 token Rank：

$$V_{\text{combine}}\approx\frac{p-1}{p}kT_{\text{local}}Hb$$

因此 MoE Layer 的理想总发送量为：

$$V_{\text{EP}}\approx2\frac{p-1}{p}kT_{\text{local}}Hb$$

若全局共有 $T$ 个 token 且初始均匀分布，$T_{\text{local}}=T/p$，则：

$$V_{\text{EP}}\approx2\frac{p-1}{p^2}kTHb$$

上述公式未包含 Expert Id、Router Weight、Token Index、Send/Receive Count、Padding 和 Capacity 溢出。更重要的是，它只描述平均理想字节数；实际完成时间通常由最大消息 Rank 和热点链路决定。

---

## 本地 Expert 计算

### 本地 Expert 分组

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

### Grouped GEMM

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

## 与其他并行维度组合

### EP 与 TP 的区别

#### TP

TP 把一个 Expert 内部的权重切分：

```text
Expert 0：
Rank 0 持有一部分权重
Rank 1 持有一部分权重
Rank 2 持有一部分权重
Rank 3 持有一部分权重
```

每个被激活的 Expert 需要多个 Rank 共同计算，通常依赖 AllReduce。

#### EP

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

### Expert Tensor Parallelism

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

### Attention DP + Expert EP

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

## 负载均衡与拓扑优化

### EP 的负载不均衡

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

### Capacity 与 Token Drop

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

### Expert Replication

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

### Expert Placement

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

### Hierarchical AllToAll

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

### Decode 阶段为什么对 EP 特别困难

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

### EP 的通信与计算重叠

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

### EP 性能分析

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

## 性能瓶颈总结

| 瓶颈 | 关键指标 | 常见缓解方式 |
| --- | --- | --- |
| 路由不均衡 | 每 Rank/Expert token 数、最大值与平均值之比 | Load-balance loss、capacity、Expert replication |
| AllToAllV 尾延迟 | 最大 send/recv count、热点链路 | Expert placement、分层 AllToAll、拓扑感知路由 |
| 小 Expert GEMM | 每 Expert token 数、Grouped GEMM 利用率 | Grouped GEMM、token batching、减少过细 EP |
| Decode 小 Batch | 每 Expert 每步 token 数接近 0 或 1 | 增大 batch、复制热门 Expert、降低 EP Degree |
| Token Permute | Pack/Unpack Kernel 和 HBM 时间 | 融合 permute、通信、unpermute，使用连续 Buffer |
| EP + TP 组合 | 通信组重叠和多次同步 | 节点内 TP、节点间 EP，按拓扑规划 Process Group |

EP 的端到端时间更接近：

$$T_{\text{MoE}}\approx T_{\text{route}}+T_{\text{permute}}+T_{\text{dispatch}}+\max_r T_{\text{expert},r}+T_{\text{combine}}+T_{\text{unpermute}}$$

所以平均通信量或平均 Expert 负载都不足以判断性能，必须关注最大值、方差以及通信与 Grouped GEMM 的重叠程度。
