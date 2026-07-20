# Ulysses Parallelism

Ulysses Parallelism（UP）是一种 Attention 内部的序列并行方法。它通过 AllToAll 在“部分序列、全部 Heads”和“完整序列、部分 Heads”之间转换，使每个 Rank 能直接使用本地 Attention Kernel。


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
