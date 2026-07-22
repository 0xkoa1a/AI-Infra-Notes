# DeepEP

DeepEP 是面向 MoE Expert Parallelism 的通信库。

普通 EP Dispatcher 通常把数据重排和通信拆成多个独立算子：

```text
Router
  │
  ▼
Token Permute / Pack
  │
  ▼
AllToAllV
  │
  ▼
按本地 Expert 再次重排
  │
  ▼
Grouped GEMM
  │
  ▼
反向重排 + AllToAllV + Unpermute
```

这种实现语义清晰，但会产生：

* 多次 HBM 读写和临时 Buffer；
* 多个 Permute、Pack 和 Collective Kernel；
* CPU、通信 Stream 与计算 Stream 之间的同步；
* 对 NVLink 和 RDMA 差异利用不足的扁平通信。

DeepEP 将 EP 的通信抽象为两个专用原语：

* **Dispatch**：根据 Top-$k$ 结果重排 token，并发送到目标 Expert Rank；
* **Combine**：把 Expert 输出送回原 Rank，并恢复原 token 布局。

它的主要收益是融合了数据重排与通信、感知硬件拓扑、支持低精度传输，并可以显式控制通信占用的 SM。

---

## 统一示例 Setup

下面用一个简化但典型的 Top-2 MoE 例子，对照同一个 token 在 DeepEP v1 High-Throughput 和 Low-Latency 模式中的完整路径。两种模式的 Router 结果和 Expert 计算语义相同，区别主要在接收空间如何确定、数据如何传输以及接收端如何组织布局。

以下具体路径以 DeepEP v1 为口径。DeepEP v2 已将两种模式统一到 `ElasticBuffer` 接口。

### 初始拓扑与 Expert 放置

假设有 2 个节点，每个节点 4 张 GPU，共 8 个 rank。每张 GPU 放置两个 expert：

| 节点     | Rank / GPU | 本地 Experts |
| ------ | ---------: | ---------- |
| Node 0 |  r0 / GPU0 | E0、E1      |
| Node 0 |  r1 / GPU1 | E2、E3      |
| Node 0 |  r2 / GPU2 | E4、E5      |
| Node 0 |  r3 / GPU3 | E6、E7      |
| Node 1 |  r4 / GPU0 | E8、E9      |
| Node 1 |  r5 / GPU1 | E10、E11    |
| Node 1 |  r6 / GPU2 | E12、E13    |
| Node 1 |  r7 / GPU3 | E14、E15    |

现在关注一个 token：

```text
token t7
当前位置：Node 0，r1 / GPU1
hidden state：h7
```

Router 对它做 Top-2 路由，得到：

```text
t7 → E3，权重 0.35
t7 → E13，权重 0.65
```

因此，一个逻辑 token 被展开成两个 assignment：

```text
q0 = (token=t7, topk_slot=0) → E3
q1 = (token=t7, topk_slot=1) → E13
```

其中：

* E3 在本地 r1 上；
* E13 在远端 Node 1 的 r6 上。

### 两种模式共同的计算语义

两种模式都需要完成相同的逻辑过程：

```text
Dispatch
  ├── q0：把 h7 交给本地 E3
  └── q1：把 h7 交给远端 E13

Expert Compute
  ├── y7_local  = Expert3(h7)
  └── y7_remote = Expert13(h7)

Combine
  └── 把两个分支恢复到 t7，并按路由权重归约
```

最终结果始终是：

```text
output(t7)
= 0.35 × Expert3(h7)
+ 0.65 × Expert13(h7)
```

无论采用哪种模式，系统都必须保留足够的映射信息，使 Expert 输出能够恢复到：

```text
source rank = r1
token       = t7
Top-K slot  = 0 或 1
weight      = 0.35 或 0.65
```

其中 `q0` 始终只涉及 r1 上的本地 E3。下面重点追踪远程分支 `q1: t7 → E13 → r6`，分别观察两种模式如何完成 Dispatch 和 Combine。

---

## High-Throughput 模式

High-Throughput 主要用于训练和 Prefill。此时每个 Rank 通常持有较多 token，优化目标是获得紧凑 Buffer 和大块连续传输，使 NVLink、RDMA 与 Grouped GEMM 保持较高吞吐。

---

### 本地路由：确定目的 Rank

Router 输出的本质不只是 expert 编号，还要知道 expert 所属的目标 rank：

```text
q0: t7 → E3  → r1
q1: t7 → E13 → r6
```

因此，r1 会把当前批次中的所有 assignment 按目标 rank 分类。

假设 r1 这一批 token 产生的远程 assignment 数量为：

```text
r1 → r4：2 个
r1 → r5：0 个
r1 → r6：3 个
r1 → r7：1 个
```

这里：

```text
count[r1 → r6] = 3
```

意味着 r1 有 3 个 assignment 要发送到 r6，其中一个就是：

```text
t7 → E13
```

此时还没有真正传输 hidden state，只确定了“要发多少”。

---

### Count Exchange：先交换计数

所有 rank 先交换很小的元数据。

r1 告诉 r6：

```text
我会给你发送 3 个 assignment。
```

与此同时，r6 也会收到其他 source rank 发来的 count。

假设 r6 收到的完整计数如下：

| Source rank | 发给 r6 的 assignment 数 |
| ----------: | -------------------: |
|          r0 |                    2 |
|          r1 |                    3 |
|          r2 |                    1 |
|          r3 |                    0 |
|          r4 |                    1 |
|          r5 |                    0 |
|          r6 |                    2 |
|          r7 |                    1 |

总接收数量为：

```text
2 + 3 + 1 + 0 + 1 + 0 + 2 + 1 = 10
```

于是 r6 知道自己需要一个长度为 10 的紧凑接收缓冲区：

```text
compact_recv_buffer[10]
```

这一步只交换 count，不交换 h7。

---

### Prefix-Sum：计算每个 Source 的写入区间

r6 对各 source 的 count 做前缀和：

```text
counts  = [2, 3, 1, 0, 1, 0, 2, 1]
offsets = [0, 2, 5, 6, 6, 7, 7, 9, 10]
```

因此各 source 的接收区间是：

```text
r0 → positions [0, 2)
r1 → positions [2, 5)
r2 → positions [5, 6)
r3 → positions [6, 6)
r4 → positions [6, 7)
r5 → positions [7, 7)
r6 → positions [7, 9)
r7 → positions [9, 10)
```

r1 发来的 3 个 assignment 必须落在：

```text
compact_recv_buffer[2:5]
```

假设 t7 是 r1 发往 r6 的第 2 个 assignment，也就是 r1 局部发送列表中的索引 1，那么：

```text
最终写入位置
= source_offset[r1] + local_index
= 2 + 1
= 3
```

所以 t7 的远程分支最终应该位于：

```text
compact_recv_buffer[3]
```

这就是 count exchange 和 prefix-sum 的核心作用：在传输 payload 之前，先确定每个 source 的合法写入范围，避免互相覆盖。

---

### 发送端打包：把 t7 放入 r6 的发送区域

在 r1 上，所有 assignment 通常先按目标 rank 打包。

例如：

```text
send_to_r4 = [...]
send_to_r5 = []
send_to_r6 = [
    token_a → E12,
    t7      → E13,
    token_b → E13
]
send_to_r7 = [...]
```

t7 的发送 payload 大致包含：

```text
hidden state：h7
目标 expert：E13
原始 token 标识：t7
Top-K 分支：slot 1
路由权重：0.65
```

具体元数据字段会因实现而异，但接收端和 combine 阶段至少需要知道：

* 这个 hidden state 属于哪个 expert；
* 计算结果最终应该返回哪个 source rank；
* 返回后应该恢复到哪个 token、哪个 Top-K 分支。

---
### Payload 传输：同号 GPU 间 RDMA + 目标节点内 NVLink 转发

由于源 rank `r1` 位于 Node 0 的 GPU1，而目标 expert `E13` 位于 Node 1 的 `r6 / GPU2`，该 assignment 需要先跨节点，再在目标节点内转发。

DeepEP v1 高吞吐模式并不是让 `r1 / GPU1` 直接通过 RDMA 写入 `r6 / GPU2`，也不是让每个节点设置一个统一的代理 GPU。它采用分层转发方式：源 GPU 先通过自己的 RDMA Rail，将数据发送到目标节点中具有相同 Local Rank 的 GPU，然后由该 GPU 通过 NVLink 转发给真正的目标 GPU。

因此，`t7 → E13` 的逻辑路径为：

```text
Node 0：r1 / GPU1
   ↓ 写入本 GPU 对应的发送 FIFO / 通信缓冲
   ↓ 跨节点 RDMA / InfiniBand
Node 1：同号的 GPU1
   ↓ 从 GPU1 的 RDMA 接收 FIFO 取出数据
   ↓ 机内 NVLink 转发
Node 1：r6 / GPU2
   ↓
r6 的接收缓冲区
```

也就是：

```text
Node0 GPU1
    ── RDMA ──>
Node1 GPU1
    ── NVLink ──>
Node1 GPU2
```

对于 `t7`，它的 hidden state `h7` 会与目标 Expert、Source Rank、Top-K 分支等元数据一起发送。到达 Node 1 的 GPU1 后，再通过 NVLink 转发到 GPU2，并最终进入 r6 上属于本轮 Dispatch 的接收区域。

### r6 上的初始布局：按 Source 排列

传输完成后，r6 的紧凑接收缓冲区可能是：

```text
位置 0：来自 r0，目标 E12
位置 1：来自 r0，目标 E13

位置 2：来自 r1，目标 E12
位置 3：来自 r1，目标 E13   ← t7
位置 4：来自 r1，目标 E13

位置 5：来自 r2，目标 E12

位置 6：来自 r4，目标 E13

位置 7：来自 r6，目标 E12
位置 8：来自 r6，目标 E13

位置 9：来自 r7，目标 E12
```

可以写成：

```text
by-source layout:

[r0:E12, r0:E13,
 r1:E12, r1:E13(t7), r1:E13,
 r2:E12,
 r4:E13,
 r6:E12, r6:E13,
 r7:E12]
```

这种布局适合通信，因为每个 source rank 可以写入一段连续区域。

但它不适合直接执行 grouped GEMM，因为 E12 和 E13 的 token 混在一起。

---

### 本地 Permute：从 by-source 变成 by-expert

r6 上有两个本地 expert：

```text
E12
E13
```

因此需要进行本地重排：

```text
by-source
    ↓ permute
by-expert
```

重排后可能变成：

```text
E12 group:
[
  r0:E12,
  r1:E12,
  r2:E12,
  r6:E12,
  r7:E12
]

E13 group:
[
  r0:E13,
  r1:E13(t7),
  r1:E13,
  r4:E13,
  r6:E13
]
```

t7 从原来的：

```text
compact_recv_buffer[3]
```

被移动到 E13 分组中的某个位置，例如：

```text
expert_input_E13[1]
```

系统同时需要保存反向映射，例如：

```text
expert_input_E13[1]
    ↔ compact_recv_buffer[3]
    ↔ source rank r1
    ↔ token t7
    ↔ Top-K slot 1
```

这些映射信息会在 combine 阶段使用。

---

### Expert 计算与 Combine

现在 E12 和 E13 的输入已经分别连续排列，可以直接执行 Grouped GEMM。E13 计算完成后，`y7_remote` 位于 r6 的 E13 输出组中。

High-Throughput Combine 首先根据 Dispatch 保存的反向映射，把 by-expert 输出恢复为适合按 Source Rank 返回的布局：

```text
by-expert output
    ↓ inverse permute
by-source output
```

由于 t7 的 Source Rank 是 r1，因此 `y7_remote` 被放入：

```text
send_back_to_r1
```

返回过程采用与 Dispatch 对称的分层路径。r6 位于 Node 1 的 GPU2，因此先通过 GPU2 对应的 RDMA Rail 发往 Node 0 的 GPU2，再由 NVLink 转发给真正的 Source Rank r1：

```text
Node 1：r6 / GPU2
    ── RDMA ──>
Node 0：GPU2
    ── NVLink ──>
Node 0：r1 / GPU1
```

回到 r1 后，根据 token id 和 Top-K slot 恢复：

```text
token t7
slot 1
weight 0.65
output y7_remote
```

本地 E3 分支不需要跨节点返回。r1 最后按照统一示例中的计算语义，将两个 Top-K 分支恢复到 t7 并加权归约。

---

### 整条路径压缩成一行

对于远程分支 `t7 → E13`：

```text
r1 上的 t7
→ Router 选择 E13
→ 确定目标 rank r6
→ count[r1→r6] 加 1
→ r6 根据所有 count 做 prefix-sum
→ 为 r1 分配接收区间 [2,5)
→ t7 被打包为 r1→r6 的第 2 个 assignment
→ 从 Node 0 GPU1 经 RDMA 到 Node 1 GPU1
→ 再经 NVLink 转发到 r6 / GPU2
→ 写入 r6 compact buffer[3]
→ 从 by-source permute 到 E13 输入组
→ Expert13(h7)
→ inverse permute
→ 按 source 打包回 r1
→ 经 Node 1 GPU2、RDMA、Node 0 GPU2 和 NVLink 返回 r1
→ 与本地 Expert3(h7) 按路由权重相加
```

最核心的布局变化是：

```text
原始 token 顺序
→ 按目标 rank 打包
→ 接收端按 source 排列
→ 本地按 expert 排列
→ Expert 计算
→ 按 source 恢复
→ 返回原 rank
→ 恢复原 token 顺序并加权归约
```

因此，DeepEP High-Throughput 模式可以概括为：通信阶段追求按 Source、按 Rank 连续，以便高效执行大块传输；计算阶段追求按 Expert 连续，以便高效执行 Grouped GEMM。它愿意支付 Count Exchange、Prefix-Sum 和本地重排的固定成本，换取紧凑 Buffer 与更高的持续带宽。



---

## Low-Latency 模式

Low-Latency 主要用于 Decode。此时每个 Rank 每步通常只有少量 token，消息很小，Count Exchange、CPU 等待、动态 Shape 和多段转发的固定延迟很难摊薄。因此它不再优先追求恰好容纳本轮 token 的最紧凑 Buffer，而是用更多预留空间换取更短、更稳定的数据路径。

下面继续使用同一个 `q1: t7 → E13 → r6`，并假设各 Rank 产生的 Assignment 与 High-Throughput 示例完全相同。也就是说，r6 最终仍然会收到 10 个 Assignment，其中 E12 和 E13 各接收 5 个；改变的只是这些数据如何到达并形成 Expert 输入。

---

### 根据容量上界预分配接收 Buffer

Low-Latency 模式预先给定每个 Rank 最多处理的 token 数 `T_max`，并据此分配稳定的 RDMA Buffer。对于每个本地 Expert，接收容量按所有 Source Rank 的最坏情况预留，可以概念化为：

```text
recv_x[
    num_local_experts,
    num_ranks × T_max,
    hidden
]
```

r6 上有 E12 和 E13，因此它持有两个 Expert 的接收区域：

```text
recv_x[E12, 0 : 8 × T_max, :]
recv_x[E13, 0 : 8 × T_max, :]
```

这些区域的容量是上界，并不意味着每轮都会填满。实际有效数量由设备端的计数给出：

```text
recv_count[E12] = 5
recv_count[E13] = 5
```

因此 Low-Latency 模式不需要先在 CPU 侧得到总接收量 10，再为本轮动态构造 `compact_recv_buffer[10]`。Kernel 内部仍然需要维护接收计数和源位置等元数据，但发送 Payload 不再依赖“先交换精确 Count，再完成 Prefix-Sum，再确定动态输出 Shape”这条前置路径。

代价是接收 Buffer 大于本轮的真实数据量，部分预留位置无效。它用显存空间换取更少的控制路径和更稳定的地址，这也更适合 Decode 和 CUDA Graph。

---

### 发送端直接按目标 Expert 发起传输

Router 已经给出：

```text
q1: t7 → E13 → r6
```

因此 r1 可以直接根据目标 Expert 和目标 Rank 准备消息，而不必先等待 r6 为所有 Source Rank 计算精确的 Prefix-Sum 区间。

对于 t7，消息至少需要携带或关联：

```text
hidden state：h7
目标 expert：E13
source rank：r1
source token index：t7
Top-K slot：1
```

路由权重可以继续保留在 Source Rank，并在 Combine 时使用。

---

### Pure RDMA：直接到达目标 Rank

DeepEP v1 Low-Latency 模式要求参与 Rank 可以通过 RDMA 访问，并使用 IBGDA 从 GPU 侧发起通信。Toekn 直接写向目标 Rank 的目标 Expert 的 RDMA Buffer：

```text
Node 0：r1 / GPU1
    ── RDMA / IBGDA ──>
Node 1：r6 / GPU2
```

这减少了节点内转发、队列推进以及两段路径之间的协调延迟。对于 Decode 的小消息，这些固定延迟往往比峰值带宽更重要。

---

### r6 直接得到 Expert-major 输入

Low-Latency Dispatch 对外得到的接收结果以本地 Expert 为第一维，因此上层看到的有效数据可以表示为：

```text
E12 valid region:
[
  r0:E12,
  r1:E12,
  r2:E12,
  r6:E12,
  r7:E12
]

E13 valid region:
[
  r0:E13,
  r1:E13(t7),
  r1:E13,
  r4:E13,
  r6:E13
]
```

`recv_count` 标识每组有多少有效 token。


---

### Expert 计算与 Combine

E13 对包含 t7 的有效输入执行计算：

```text
y7_remote = Expert13(h7)
```

Low-Latency Combine 根据 Dispatch 保存的 Source 信息，识别出该输出属于 `r1 / t7 / slot 1`，并通过 Pure RDMA 直接返回 Source Rank：

```text
Node 1：r6 / GPU2
    ── RDMA / IBGDA ──>
Node 0：r1 / GPU1
```

Combine 在返回过程中按照 Top-K Weight 对各 Expert 分支做归约。r1 最终得到统一示例中定义的 t7 输出。

---

### 整条路径压缩成一行

对于远程分支 `t7 → E13`：

```text
r1 上的 t7
→ Router 选择 E13
→ 确定目标 rank r6
→ 使用预分配的容量上界，不等待精确接收 Shape
→ 按 E13 准备 h7 和 Source 元数据
→ 从 r1 / GPU1 经 Pure RDMA 直接写向 r6 / GPU2
→ 进入 r6 的 E13 Expert-major 有效区域
→ Expert13(h7)
→ 根据 Dispatch Handle 找回 r1 / t7 / slot 1
→ 经 Pure RDMA 直接返回 r1
→ 与本地 Expert3(h7) 按路由权重归约
```

最核心的布局变化是：

```text
原始 token 顺序
→ 按目标 Expert 直接发送
→ 接收端得到 Expert-major Buffer
→ Expert 计算
→ 按 Source 信息直接返回并归约
```

因此，DeepEP Low-Latency 模式可以概括为：用容量有上界、地址稳定但不完全紧凑的 Buffer，换掉精确 Shape 同步和分层转发带来的固定延迟，并让 Expert 输入直接按 Expert 组织。它优化的是一次 Decode 通信尽快完成，而不是大批 token 下的最高持续吞吐。

## Hybrid-EP

Hybrid-EP 是 DeepEP 中一种硬件感知的 EP 通信实现：保持“同号 GPU 间 RDMA、节点内 NVLink 转发”的分层通信方式，但用 TMA、persistent kernel、warp specialization 和 chunk pipeline 重构 dispatch/combine，从而减少通信占用的 SM，并将 RDMA、NVLink 以及 permute/unpermute 更细粒度地重叠起来。

为什么能省 SM：因为利用了 TMA，数据不用线程亲自搬运。