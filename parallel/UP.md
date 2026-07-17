# Ulysses Parallelism

Ulysses Parallelism（UP）是一种 Attention 内部的序列并行方法。它通过 AllToAll 在“部分序列、全部 Heads”和“完整序列、部分 Heads”之间转换，使每个 Rank 能直接使用本地 Attention Kernel。

本文的通信量统一采用每 Rank 发送量；完整 Attention hidden state 的大小记为 $N=BSHb$，UP Degree 记为 $p$。

## 导航

* [目标与适用边界](#目标与适用边界)
* [张量布局与执行流程](#张量布局与执行流程)
* [通信与通信量](#通信与通信量)
* [通信性能分析](#通信性能分析)
* [性能瓶颈总结](#性能瓶颈总结)
* [模型结构与阶段约束](#模型结构与阶段约束)
* [与其他长序列方案的关系](#与其他长序列方案的关系)

---

## 目标与适用边界

### UP 的目标

UP（Ulysses Parallelism，尤利西斯并行）是一种面向 Attention 的序列并行方法。

它要解决的问题是：

> 序列已经沿 GPU 切分，但标准 Attention Kernel 通常希望获得完整序列上的 Q、K、V。如何在不让每张 GPU 保存全部 Head 的情况下，继续使用高效的本地 Attention Kernel？

UP 的核心方法不是让 KV Block 环形移动，而是通过 AllToAll 在两种布局之间转换：

1. Sequence-Sharded Layout：部分序列、全部 Head；
2. Head-Sharded Layout：完整序列、部分 Head。

转换后，每个 Rank 可以对自己负责的 Head，在完整序列上独立执行 Attention。

---

### UP 与普通 SP 的区别

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

## 张量布局与执行流程

### 初始张量布局

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

### 为什么这种布局不能直接独立算完整 Attention

Rank 0 虽然拥有全部 Heads，但只拥有前 $1/4$ 的 K 和 V。

对于 Rank 0 的 Query，完整 Attention 应该访问整个上下文：

$$K=[K_0;K_1;\ldots;K_{p-1}]$$

$$V=[V_0;V_1;\ldots;V_{p-1}]$$

如果 Rank 0 只使用本地 $K_0,V_0$，它只能看到局部上下文，结果不等价于完整 Attention。

因此必须进行跨 Rank 数据交换。

---

### Ulysses 的 AllToAll 布局转换

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

### 为什么 AllToAll 能实现这个转换

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

### 本地 Attention

布局转换后，每个 Rank 拥有：

* 完整序列；
* 部分 Heads。

不同 Attention Heads 相互独立，因此每个 Rank 可以本地执行：

$$O'_r=\operatorname{Attention}(Q'_r,K'_r,V'_r)$$

输出形状为：

$$O'_r\in\mathbb{R}^{B\times S\times N_h/p\times D_h}$$

这一步可以使用成熟的单卡 FlashAttention（闪存注意力）Kernel，因为对于本地 Head 子集而言，序列是完整的。

---

### 反向 AllToAll

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

## 通信与通信量

### UP 的通信量

为了与 TP + SP 使用相同口径，设完整 Attention 输出激活的大小为：

$$N=BSHb$$

其中 $b$ 为每个元素的字节数。序列被 $p$ 个 Rank 均匀切分后，每个 Rank 的本地输出 Buffer 大小为：

$$N_{\text{O,local}}=\frac{N}{p}$$

对于标准 MHA，Q、K、V 的完整大小都约为 $N$，因此每个 Rank 在第一次 AllToAll 前持有的本地 QKV Buffer 为：

$$N_{\text{QKV,local}}=3\frac{N}{p}$$

均匀 AllToAll 中，发给自己的 $1/p$ 数据不经过网络，因此第一次 AllToAll 每 Rank 发送量为：

$$V_{\text{QKV A2A}}=\frac{p-1}{p}N_{\text{QKV,local}}=3\frac{p-1}{p^2}N$$

第二次 AllToAll 每 Rank 发送量为：

$$V_{\text{O A2A}}=\frac{p-1}{p}N_{\text{O,local}}=\frac{p-1}{p^2}N$$

因此总发送量为：

$$V_{\text{UP}}=4\frac{p-1}{p^2}N$$

这里的第二个 $1/p$ 来自本地序列分片的大小。

对于 GQA 或 MQA，K、V 小于 Q。若完整 Q 和 O 的大小均为 $N$，完整 K 和 V 的大小均为 $N_{KV}$，则更一般的结果为：

$$V_{\text{UP}}=2\frac{p-1}{p^2}\left(N+N_{KV}\right)$$

该式假设 Head 能均匀切分，且没有因 KV Head 复制或 Padding 引入额外流量。

实际系统通常会把 Q、K、V 合并到同一个 Buffer 中执行一次 AllToAll，而不是分别发起三次 Collective，从而减少通信启动次数。

---

### TP + SP 与 UP 的通信量比较

比较一个包含标准 MHA 和 MLP 的完整 Transformer Layer，并令两种方案的并行度都为 $p$、各子层边界处的完整隐藏状态大小都为 $N=BSHb$。

TP + SP 在 Attention 子层执行一组 AllGather + ReduceScatter，在 MLP 子层再执行一组 AllGather + ReduceScatter。因此每 Rank 的整层发送量为：

$$V_{\text{TP+SP, layer}}=4\frac{p-1}{p}N$$

纯 UP 始终保持序列分片，MLP 可以直接处理本地 token，不需要通信；只有 Attention 需要交换三个本地 QKV 分片和一个本地输出分片：

$$V_{\text{UP, layer}}=4\frac{p-1}{p^2}N$$

因此两者每 Rank 发送量之比为：

$$\frac{V_{\text{UP, layer}}}{V_{\text{TP+SP, layer}}}=\frac{1}{p}$$

UP 相对 TP + SP 节省的理想通信比例为：

$$1-\frac{V_{\text{UP, layer}}}{V_{\text{TP+SP, layer}}}=1-\frac{1}{p}$$

例如 $p=4$ 时，UP 的整层每 Rank 发送量是 TP + SP 的四分之一；$p=8$ 时是八分之一。

通信量是通过避免完整序列复制省出来的：

* TP + SP 的两次 AllGather 分别在 Attention 和 MLP 前把序列分片复制到所有 Rank，使每个 TP Rank 都获得大小为 $N$ 的完整激活；
* UP 的 AllToAll 不复制完整张量，只把本 Rank 已持有的 $N/p$ 序列分片按 Head 重新分发；
* UP 虽然在 Attention 中交换 Q、K、V、O 四份数据，但每份数据在通信前都只有完整张量的 $1/p$，所以总量带有额外的 $1/p$ 因子；
* UP 的 MLP 保持序列分片布局，无需 TP+SP 在 MLP 边界上的第二组 AllGather + ReduceScatter。

如果只比较 Attention 子层而不计 MLP，TP + SP 的发送量为 $2(p-1)N/p$，此时 UP 与 TP + SP 的比值为 $2/p$；上面的 $1/p$ 是完整 Transformer Layer 的比较结果。

这只是通信字节数的对比。TP 同时切分模型权重，而纯 UP 通常复制 Attention 和 MLP 权重；AllGather/ReduceScatter 与 AllToAll 的实际延迟、拓扑适配和 Pack/Unpack 开销也不同，因此通信量更小不等于端到端一定更快。

---

## 通信性能分析

### UP 的通信成本不仅由字节数决定

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

## 性能瓶颈总结

UP 的理论字节数低，但实际瓶颈通常来自以下方面：

| 瓶颈 | 表现 | 根因或约束 |
| --- | --- | --- |
| AllToAll 尾延迟 | 平均带宽不低但 Collective 完成慢 | 多对多链路争用，由最慢 Rank 或最拥塞链路决定 |
| Pack/Unpack | NCCL 外仍有明显 HBM Kernel | Head/sequence 布局转换需要 Split、Transpose 和 Reorder |
| Head 数限制 | UP Degree 无法继续增大 | 通常要求 $N_h\bmod p=0$ 且 $p\leq N_h$ |
| GQA/MQA | KV 布局复杂或需要复制 | KV Head 数可能小于 UP Degree |
| Decode | AllToAll 难以摊薄 | 新 Query 长度通常为 1，序列维度缺少可切分工作 |
| 跨节点扩展 | 带宽下降、拥塞增大 | AllToAll 比节点内 NVSwitch 更依赖拓扑 |

因此，UP 更适合训练、长 Prompt Prefill 和大规模图像/视频 token。部署时常把 Ulysses 放在节点内，并与节点间 Ring Attention 组合。

---

## 模型结构与阶段约束

### UP 的 Head 数限制

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

### UP 与 GQA、MQA

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

### UP 为什么更适合 Prefill

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

## 与其他长序列方案的关系

### UP 与 Ring Attention 的比较

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
