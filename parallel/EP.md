# Expert Parallelism

Expert Parallelism（EP）将不同 MoE Experts 放到不同 Rank。Token 由 Router 选择 Expert，再通过 Dispatch AllToAllV 移动到 Expert 所在 Rank；计算完成后，通过 Combine AllToAllV 返回原 Rank。

EP 不是规则的张量切片，而是由输入内容决定的动态路由。因此分析 EP 时，除了平均通信量，还必须关注 Expert 负载、消息大小分布和最慢 Rank。

---

## 统一符号

设一个 MoE Layer 包含：

* $E$ 个 Experts；
* 每个 token 选择 Top-$k$ Experts；
* Hidden Size 为 $H$；
* $p$ 个 Rank；
* 每个元素占 $b$ 字节；
* 每个 Rank 初始持有 $T_{\text{local}}$ 个 token。

下文只统计 Routed Experts 的一次前向路径；Shared Expert 和训练反向通信另计。

第 $r$ 个 Rank 的输入为：

$$X_r\in\mathbb{R}^{T_{\text{local}}\times H}$$

若 Experts 均匀切分，每个 Rank 保存：

$$E_{\text{local}}=\frac{E}{p}$$

个 Experts。

---

## Router 与 Routed Token

将当前 rank 的全部 token 记为：

$$X\in\mathbb{R}^{T_{\text{local}}\times H}$$


Router 权重为：

$$W_R\in\mathbb{R}^{H\times E}$$

Router Logits 为：

$$G=XW_R\in\mathbb{R}^{T_{\text{local}}\times E}$$

对每个 token 选择 Top-$k$ Experts：

$$\mathcal{E}(t)=\operatorname{TopK}(G_t,k)$$

并得到对应的 Router Weight：

$$w_{t,e},\qquad e\in\mathcal{E}(t)$$

每个 token 会为每个选中的 Expert 生成一个 Routed Token Instance，因此理想 Routed Token 数为：

$$T_{\text{routed}}=kT_{\text{local}}$$

---

## Token Permute

输入 token 按原始请求顺序排列，但 Router 结果通常是无序的：

```text
token 0 → Expert 7  → Rank 1
token 1 → Expert 2  → Rank 0
token 2 → Expert 7  → Rank 1
token 3 → Expert 12 → Rank 3
```

Dispatch 前需要按目标 Rank 和目标 rank 上的 Expert 对 Routed Tokens 进行 Permute：

```text
原始 Token 顺序
        │
        ▼
按目标 Rank 分组
        │
        ▼
同一 Rank 内按 Expert 分组
```

同时保存以下元数据：

* 原始 Token Index；
* Expert Id；
* Router Weight；
* 目标 Rank；
* 反向 Unpermute 索引。


---

## Dispatch AllToAllV

设 Rank $r$ 发给 Rank $j$ 的 Routed Token Buffer 为：

$$X_{r\rightarrow j}$$

Dispatch 后，Rank $j$ 收到来自所有其他 rank 的，目标为本地 Experts 的 Routed Token：

$$X'_j=[X_{0\rightarrow j},X_{1\rightarrow j},\ldots,X_{p-1\rightarrow j}]$$

Router 产生的数据量通常不均匀：

```text
Rank 0 → Rank 1：64 个 token
Rank 0 → Rank 2： 7 个 token
Rank 0 → Rank 3：103 个 token
```

因此 EP 通常需要 AllToAllV，而不是每一对 Rank 消息大小相同的固定 AllToAll。通信前还需要交换或计算 Send Counts 和 Receive Counts。

---

## 本地 Expert 与 Grouped GEMM

Dispatch 后，一个 Rank 会收到属于多个本地 Experts 的 token：

```text
Expert 0：128 tokens
Expert 1： 19 tokens
Expert 2： 74 tokens
```

若 Dispatch Buffer 已经按 Expert 连续排列，接收后可以直接形成 Expert-Contiguous Layout；否则还需要一次本地 Permute。

每个 Expert 通常是一个独立 MLP：

$$Y_e=\sigma(X_eW_{\text{up},e})W_{\text{down},e}$$

其中：

$$X_e\in\mathbb{R}^{T_e\times H}$$

$T_e$ 是 Expert $e$ 收到的 Routed Token 数。

> Grouped GEMM 将多个 Expert GEMM 放入一次 Kernel 调度，以减少 Launch 开销，并提高小 Expert Batch 下的 Tensor Core 利用率。

---

## Combine AllToAllV

Expert 输出仍位于 Expert 所在 Rank，但后续 Attention、Residual 和 LayerNorm 通常按原 token 布局执行，因此输出需要返回原 Rank。

Combine AllToAllV 执行与 Dispatch 相反的通信：

```text
Expert 所在 Rank
        │
        ▼
Combine AllToAllV
        │
        ▼
原 Token 所在 Rank
```

随后根据反向索引恢复原 token 顺序。Top-$k$ Routing 还要按照 Router Weight 合并多个 Expert 输出：

$$Y_t=\sum_{e\in\mathcal{E}(t)}w_{t,e}Y_{t,e}$$

Dispatch 用来进入 Expert-Owned Layout，Combine 用来恢复 Token-Owned Layout。除非后续算子也能继续使用同一个 Expert-Owned Layout，否则这两次通信都不能省掉。

---

## 一组 EP 的完整过程

```text
Token-Owned 输入
每 Rank：[T_local, H]
        │
        ▼
Router / Top-k
        │
每 token 生成 k 个 Routed Token Instances
        │
        ▼
Token Permute
按目标 Rank、Expert 连续排列
        │
        ▼
Dispatch AllToAllV
        │
        ▼
Expert-Contiguous Layout
        │
        ▼
Grouped GEMM
        │
        ▼
Combine AllToAllV
        │
        ▼
Token Unpermute
        │
        ▼
按 Router Weight 加权合并
        │
        ▼
Token-Owned 输出
```

---

## EP 的通信量

以下采用每 Rank 实际经网络发送的数据量，不计算路由到本 Rank Experts 的部分。

每个 Rank 初始持有 $T_{\text{local}}$ 个 token，共生成：

$$kT_{\text{local}}$$

个 Routed Token Instances。每个 Hidden State 的大小为：

$$Hb$$

若 Experts 均匀放置且 Router 近似均匀，一个 Routed Token 的目标 Expert 位于本 Rank 的概率为 $1/p$，位于远端 Rank 的概率为 $(p-1)/p$。

因此 Dispatch 的理想每 Rank 发送量为：

$$V_{\text{dispatch}}\approx\frac{p-1}{p}kT_{\text{local}}Hb$$

Combine 需要发送同样大小的 Expert 输出，理想发送量为：

$$V_{\text{combine}}\approx\frac{p-1}{p}kT_{\text{local}}Hb$$

一组 MoE Layer 的理想总发送量为：

$$V_{\text{EP}}\approx2\frac{p-1}{p}kT_{\text{local}}Hb$$

若全局共有 $T$ 个 token，并且初始均匀分布：

$$T_{\text{local}}=\frac{T}{p}$$

则：

$$V_{\text{EP}}\approx2\frac{p-1}{p^2}kTHb$$

该公式不包含 Expert Id、Router Weight、Token Index、Send/Receive Counts、Padding 和 Capacity 溢出产生的流量。大 Batch 时这些元数据通常小于 Hidden State；Decode 小 Batch 时则可能不可忽略。

---

## 非均匀路由与长尾

理想公式只能描述平均通信量。设 Rank $r$ 发往 Rank $j$ 的 Routed Token 数为 $n_{r\rightarrow j}$，则 Rank $r$ 的实际 Dispatch 发送量为：

$$V_r^{\text{dispatch}}=Hb\sum_{\substack{j=0\\j\neq r}}^{p-1}n_{r\rightarrow j}$$

实际时间还取决于接收量、消息分布和网络拓扑。Collective 必须等待所有 Rank 完成，因此更接近：

$$T_{\text{dispatch}}\approx\max_r T_{\text{dispatch},r}$$

即使平均发送量不大，只要某个热门 Expert 让一个 Rank 接收大量 token，整个 EP Group 仍会被该 Rank 拖慢。

---

## Capacity、Replication 与 Placement

训练系统常为每个 Expert 设置容量：

$$C=\left\lceil\text{Capacity Factor}\times\frac{kT}{E}\right\rceil$$

超过容量的 token 可以被丢弃或改路由，以限制计算和 Buffer 上界。

推理通常不能随意丢弃 token，更常使用动态 Buffer、备用 Expert 或负载感知路由。

热门 Expert 也可以在多个 Rank 上复制，并把 Routed Tokens 分配到不同副本。Replication 能减轻热点和长尾，但会增加权重显存，并让 Router 和 Expert Placement 更复杂。

Expert Placement 不应只保证每个 Rank 的 Expert 数相同，还要避免把多个热门 Experts 放在同一 Rank。跨节点 EP 还要尽量把高频流量留在节点内，并根据 NVLink、NVSwitch 和节点间网络规划 EP Group。

负载均衡问题本身（负载的三重可变性、优化目标、纯重排的下界、硬件前提、四个工作的流变）见 `load-balancing/background.md`；四条具体路线分别见 `load-balancing/EPLB.md`（基于历史统计、周期性离线求解）、`load-balancing/UltraEP.md`（基于精确负载、逐层实时求解）、`load-balancing/MoonEP.md`（把每 Rank 负载钉成常量的 EP 通信库）与 `load-balancing/LPLB.md`（离线定副本拓扑、逐 batch 解 LP 定分流）。

---

## Decode 阶段的小 Batch 问题

Prefill 中参与 MoE Layer 的 token 数为：

$$T=B\times S$$

Decode 中每个请求每步只有一个新 token：

$$T=B$$

当 $B$ 相对 Experts 数量 $E$ 很小时，许多 Experts 每步只收到零个或几个 token。此时：

* Grouped GEMM 过小，Tensor Core 利用率低；
* AllToAllV 固定启动延迟难以摊薄；
* Permute 和元数据成本占比上升；
* 路由随机性会带来更明显的 Rank 间波动。

因此 MoE Decode 通常依赖 Continuous Batching，将多个请求的 token 汇总为更大的全局 Batch。EP Degree 也不是越大越好：切得过细会让每个 Rank 上的 Expert Batch 进一步缩小。

---

## EP 的性能瓶颈

在完全串行的情况下，一组 MoE Layer 的时间可以粗略写成：

$$
\begin{aligned}
T_{\text{MoE}}\approx{}&\max_rT_{\text{route},r}+\max_rT_{\text{permute},r}
+T_{\text{dispatch}}^{\text{collective}}\\
&+\max_rT_{\text{expert},r}
+T_{\text{combine}}^{\text{collective}}
+\max_rT_{\text{unpermute},r}.
\end{aligned}
$$

$T_{\text{dispatch}}^{\text{collective}}$ 和 $T_{\text{combine}}^{\text{collective}}$ 表示整个 Collective 的完成时间，其中同样隐含了最慢 Rank、最大 Send/Receive Count 和热点链路的影响。

如果少数 Rank 的 Grouped GEMM 很长，其他 Rank 大量等待，瓶颈通常是路由不均衡；如果 Grouped GEMM 很短而 Collective 很长，则更可能是最大 Send/Receive Count 过大、存在热点链路、EP 跨节点或消息太小而无法摊薄启动延迟。
