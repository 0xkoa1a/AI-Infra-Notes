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

## High-Throughput

下面用一个简化但典型的 Top-2 MoE 例子，追踪单个 token 在 DeepEP v1 高吞吐模式中的完整路径。


## 初始拓扑与 Expert 放置

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

接下来分别追踪这两个分支。

---

## 本地路由：确定目的 Rank

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

## Count Exchange：先交换计数

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

## Prefix-Sum：计算每个 Source 的写入区间

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

## 发送端打包：把 t7 放入 r6 的发送区域

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
## Payload 传输：同号 GPU 间 RDMA + 目标节点内 NVLink 转发

由于源 rank `r1` 位于 Node 0 的 GPU1，而目标 expert `E13` 位于 Node 1 的 `r6 / GPU6`，该 assignment 需要先跨节点，再在目标节点内转发。

DeepEP v1 高吞吐模式并不是让 `r1 / GPU1` 直接通过 RDMA 写入 `r6 / GPU6`，也不是让每个节点设置一个统一的代理 GPU。它采用分层转发方式：源 GPU 先通过自己的 RDMA rail，将数据发送到目标节点中具有相同 local rank 的 GPU，然后由该 GPU通过 NVLink 转发给真正的目标 GPU。

因此，`t7 → E13` 的逻辑路径为：

```text
Node 0：r1 / GPU1
   ↓ 写入本 GPU 对应的发送 FIFO / 通信缓冲
   ↓ 跨节点 RDMA / InfiniBand
Node 1：同号的 GPU1
   ↓ 从 GPU1 的 RDMA 接收 FIFO 取出数据
   ↓ 机内 NVLink 转发
Node 1：r6 / GPU6
   ↓
r6 的接收缓冲区
```

也就是：

```text
Node0 GPU1
    ── RDMA ──>
Node1 GPU1
    ── NVLink ──>
Node1 GPU6
```

对于 `t7`，它的 hidden state `h7` 会与目标 expert、源 rank、Top-K 分支等元数据一起发送。到达 Node 1 的 GPU1 后，再通过 NVLink 转发到 GPU6，并最终进入 GPU6 上属于本轮 dispatch 的接收区域。

## r6 上的初始布局：按 Source 排列

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

## 本地 Permute：从 by-source 变成 by-expert

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

## Expert 计算

现在 E12 和 E13 的输入已经分别连续排列，可以进行 grouped GEMM。

对于 t7 的远程分支：

```text
y7_remote = Expert13(h7)
```

与此同时，t7 的本地分支在 r1 上进入 E3：

```text
y7_local = Expert3(h7)
```

注意两个 expert 接收到的是同一个输入 hidden state h7，但参数不同：

```text
Expert3(h7)  ≠  Expert13(h7)
```

因此产生两个不同的 expert 输出。

---

## Combine：基本沿原路径返回

E13 计算完成后，t7 的输出位于 r6：

```text
y7_remote = Expert13(h7)
```

首先根据之前保存的 inverse permutation，把 by-expert 输出恢复为适合按 source 返回的布局：

```text
by-expert output
    ↓ inverse permute
by-source output
```

由于 t7 原始 source 是 r1，因此它被放入：

```text
send_back_to_r1
```

然后反向跨节点传输：

```text
r6 / Node 1
   ↓ 本地队列
RDMA / InfiniBand
   ↓
Node 0
   ↓ NVLink / 本地搬运
r1
```

回到 r1 后，根据 token id 和 Top-K slot 恢复：

```text
token t7
slot 1
weight 0.65
output y7_remote
```

本地 E3 分支不需要跨节点返回：

```text
slot 0
weight 0.35
output y7_local
```

最后做加权归约：

```text
output(t7)
= 0.35 × Expert3(h7)
+ 0.65 × Expert13(h7)
```

也就是：

```text
o7 = 0.35 y7_local + 0.65 y7_remote
```

---

## 整条路径压缩成一行

对于远程分支 `t7 → E13`：

```text
r1 上的 t7
→ Router 选择 E13
→ 确定目标 rank r6
→ count[r1→r6] 加 1
→ r6 根据所有 count 做 prefix-sum
→ 为 r1 分配接收区间 [2,5)
→ t7 被打包为 r1→r6 的第 2 个 assignment
→ 经 NVLink / RDMA 跨节点传输
→ 写入 r6 compact buffer[3]
→ 从 by-source permute 到 E13 输入组
→ Expert13(h7)
→ inverse permute
→ 按 source 打包回 r1
→ RDMA 返回
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

因此，DeepEP 高吞吐模式可以概括为：通信阶段追求按 source、按 rank 连续，以便高效大块传输；计算阶段追求按 expert 连续，以便高效执行 grouped GEMM。




## High-Throughput 与 Low-Latency

DeepEP 同时面向两种差异很大的负载。V2 使用统一的 `ElasticBuffer` 接口，但两种性能目标仍然存在。

### High-Throughput

High-Throughput 路径主要用于训练和 Prefill：

* token 多，消息较大；
* 重点是持续带宽和紧凑的 Expert 输入布局；
* 可以用异步 Event、多个 Microbatch 或流水调度隐藏通信；
* Pack、量化和通信的固定开销容易摊薄。

此时应优先优化单位时间处理的有效 Routed Token 数，而不是单次 Collective 的最短延迟。

### Low-Latency

Low-Latency 路径主要用于 Decode：

* 每步 token 少，消息很小；
* 固定启动延迟、元数据处理和同步占比更高；
* 通信很难被同一请求的 Expert GEMM 完全隐藏；
* Buffer 上界、稳定地址和可复用 Handle 对 CUDA Graph 更重要。

V2 允许在适用时缓存并复用路由 Handle，减少重复的布局计算与 CPU 同步。但只有路由结果确实不变时才能复用，不能跨不同 token 强行沿用旧路由。

因此，High-Throughput 与 Low-Latency 的区别不是两套 EP 数学公式，而是大消息带宽、Buffer 紧凑度、启动延迟和可重叠性之间的不同取舍。

---

## DeepEP 的通信量

DeepEP 不改变 Top-$k$ Routing 产生的逻辑 Routed Token 数。设：

* 每 Rank 初始有 $T_{\text{local}}$ 个 token；
* 每个 token 选择 $k$ 个 Experts；
* Hidden Size 为 $H$；
* EP Degree 为 $p$；
* Dispatch 和 Combine 每个元素分别占 $b_d$、$b_c$ 字节。

在 Expert 均匀放置且路由均匀时，DeepEP 的理想每 Rank Payload 发送量仍为：

$$V_{\text{DeepEP}}\approx\frac{p-1}{p}kT_{\text{local}}H(b_d+b_c)$$

若 Dispatch 和 Combine 都使用相同精度 $b$：

$$V_{\text{DeepEP}}\approx2\frac{p-1}{p}kT_{\text{local}}Hb$$

这与普通 EP 的逻辑通信量一致。DeepEP 的主要优化来自：

* 减少额外 Permute 和中间 Buffer 的 HBM 流量；
* 将通信映射到更合适的 NVLink / RDMA 路径；
* 使用 FP8 等低精度降低 Dispatch Payload；
* 让通信与独立计算重叠；
* 用更少通信 SM 达到链路饱和。

公式不包含 Expert Id、Router Weight、Scale、Offset 和 Count 等元数据。FP8 Dispatch 仍需要 Scale，因此实际字节数不能只按一个字节直接估算。

还要区分两种带宽：

* **Algorithm Bandwidth**：有效 Routed Payload 除以端到端通信时间；
* **Bus Bandwidth**：NVLink 或 NIC 上实际传输的物理字节数除以时间。

层次化转发会让同一 Payload 依次经过 RDMA 和 NVLink，因此 Algorithm Bandwidth 不能直接当作 NIC 的物理吞吐。

---

## NVLink 与 RDMA 的层次化通信

节点内 NVLink/NVSwitch 和节点间 RDMA 的带宽、延迟和可连接规模差异很大。DeepEP 不把它们简单视为同一种链路，而是支持 Direct 与 Hybrid 两类拓扑模式。

Hybrid Mode 的基本思想是：

```text
源 GPU
  │
  ├── 节点内目标：NVLink
  │
  └── 跨节点目标：RDMA 到目标节点
                         │
                         ▼
                    节点内 NVLink 转发
                         │
                         ▼
                     目标 Expert GPU
```

这样可以把跨节点通信集中到适合访问 NIC 的路径，再利用节点内高速互联完成本地分发，避免让每个 GPU 与所有远端 GPU 建立同等复杂的通信关系。

Direct Mode 不采用相同的层次化转发假设，具体 Peer 映射取决于所使用的 DeepEP 版本。哪种模式更好取决于 GPU–NIC Affinity、NVLink Domain、Rail 映射、节点数量和消息大小，不能只根据 EP Degree 决定。

层次化通信解决的是数据如何穿过不同硬件域，并不能消除热门 Expert。若 Router 让大量 token 聚集到同一 Rank，DeepEP 仍然要面对最大接收量和最慢 Expert。

---

## 为什么通信会占用 SM

GPUDirect RDMA 可以让 NIC 直接读写 GPU HBM，但完整 EP 通信仍然需要 GPU 执行：

* 解析 Routing Map；
* Pack、量化和布局转换；
* 发起或推进设备侧通信；
* 轮询状态与维护同步；
* 在 Combine 中执行 Top-$k$ Reduce；
* 将结果写入最终 Buffer。

因此通信 Kernel 会占用一部分 SM。假设 GPU 共有 $S$ 个 SM，其中 $s$ 个分配给通信，在理想重叠窗口中：

$$T_{\text{overlap}}(s)\approx\max\left(T_{\text{comm}}(s),T_{\text{GEMM}}(S-s)\right)$$

增加 $s$ 通常能先降低通信时间，但当 NVLink 或 RDMA 已经饱和后：

$$\Delta T_{\text{comm}}\approx0$$

继续增加通信 SM 只会减少 Grouped GEMM 可用的 SM，使端到端时间反而上升。

DeepEP V2 根据拓扑、带宽和通信规模解析计算 SM 与 QP 数量。目标是找到接近链路饱和所需的最小 SM 数，而不是让通信微基准单独占满整张 GPU。

---



## DeepEP 的性能瓶颈

在不考虑重叠时，一组 MoE Layer 可以粗略写成：

$$
T_{\text{MoE}}\approx
T_{\text{metadata}}
+T_{\text{dispatch}}^{\text{collective}}
+\max_rT_{\text{expert},r}
+T_{\text{combine}}^{\text{collective}}.
$$

DeepEP 将 Permute、Pack 和部分 Reduce 融合进 Dispatch / Combine，因此这些时间已经包含在两个 Collective 中。通信与计算重叠后，真正需要关注的是：

$$T_{\text{exposed comm}}=\max\left(0,T_{\text{comm}}-T_{\text{overlapped}}\right)$$

主要瓶颈包括：

* **路由不均衡**：最大 Receive Count 和最慢 Expert 仍会决定整个 EP Group 的长尾；
* **通信 SM 过多**：Collective 更快，但 Grouped GEMM 因可用 SM 下降而变慢；
* **通信 SM 过少**：Pack、Reduce 或设备侧通信无法喂满 NVLink / RDMA；
* **拓扑错误**：GPU–NIC Affinity、Rail 或 Process Group 排布错误会形成热点链路；
* **Decode 小消息**：元数据、Kernel Launch 和同步时间难以摊薄；
* **Buffer 上界过大**：提高稳定性和 Graph 兼容性，但会挤占 Expert 权重与 KV Cache 显存；
* **隐藏同步**：D2H Count、动态分配或错误的 Event 依赖会破坏异步重叠。

如果 DeepEP 微基准带宽很高，但模型端到端没有提升，应优先检查通信 SM 是否挤占 GEMM、是否存在额外布局 Kernel，以及 Event 等待是否真的与独立计算重叠。
