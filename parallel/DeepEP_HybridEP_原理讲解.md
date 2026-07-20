# DeepEP 与 HybridEP

DeepEP 是面向 MoE Expert Parallelism 的通信库。EP 决定“哪些 token 应该去哪些 Experts”，DeepEP 则优化这些 token 在 GPU、NVLink 和 RDMA 网络上如何被重排、发送与收回。

本文假设已经理解 [EP](./EP.md) 的 Router、Dispatch、Grouped GEMM 和 Combine，只讨论 DeepEP 在通信实现上的不同。

---

## DeepEP 优化了什么

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

它的主要收益不是改变 Router 或 Expert MLP，而是融合数据重排与通信、感知硬件拓扑、支持低精度传输，并显式控制通信占用的 SM。

---

## Dispatch 与 Combine

### Dispatch

每个 Rank 向 DeepEP 提供：

```text
Hidden States
Top-k Expert Ids
Top-k Weights
```

Dispatch 需要完成三件事：

1. 根据 Expert Id 计算目标 Rank 和目标位置；
2. 将 Routed Tokens 发送到目标 Rank；
3. 生成供本地 Experts 使用的接收布局与 token counts。

概念上，第 $r$ 个源 Rank 仍然发送：

$$X_{r\rightarrow j}$$

给第 $j$ 个目标 Rank。但 DeepEP 会尽量在 fused kernel 内完成 Permute、Pack 和跨 Rank 传输，避免把每一步都物化为独立的大 Buffer。

不同框架对 Expert-Contiguous Layout 的接口边界略有差异：有的直接从 Dispatch 得到按 Expert 排列的输入，有的还会调用一个轻量重排步骤。核心目标都是减少中间搬运，而不是改变 EP 的路由语义。

---

### Expert 计算边界

Dispatch 通常返回：

* 接收到的 Routed Token Hidden States；
* 每个本地 Expert 的 token 数；
* 重排后的 Expert Id 和 Router Weight；
* 一个记录路由与反向映射的 Handle；
* 用于异步依赖管理的 Event。

每个 Expert 的 token 数决定 Grouped GEMM 的实际 $M$ 维：

$$X_e\in\mathbb{R}^{T_e\times H}$$

DeepEP 本身负责通信和布局，不负责解决 Expert 负载不均衡，也不等同于 Grouped GEMM。它只需要把数据交付成 Expert Kernel 能消费的布局。

---

### Combine

Expert 计算完成后，Combine 使用 Dispatch 保存的 Handle 反向执行：

```text
Expert-Contiguous Outputs
        │
        ▼
根据 Handle 找回 Source Rank / Token Index
        │
        ▼
跨 Rank 返回 Expert Outputs
        │
        ▼
恢复 Token-Owned Layout
        │
        ▼
合并 Top-k Expert Outputs
```

对原 token $t$，结果仍然是：

$$Y_t=\sum_{e\in\mathcal{E}(t)}w_{t,e}Y_{t,e}$$

Handle 复用了 Dispatch 已计算的路由、offset 和反向索引，因此 Combine 不需要重新推导一遍 token 去向。

---

### 一组 DeepEP 的完整过程

```text
Router Outputs
        │
        ▼
DeepEP Dispatch
Permute + Pack + NVLink/RDMA Communication
        │
        ▼
Routed Tokens + Tokens per Expert + Handle
        │
        ▼
Grouped Expert GEMM
        │
        ▼
DeepEP Combine
Communication + Reduce + Unpermute
        │
        ▼
原 Rank、原 Token 顺序的输出
```

训练反向中，Dispatch 的反向在通信语义上对应 Combine，Combine 的反向对应 Dispatch。

---

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

## DeepEP Hybrid Mode 与 HybridEP

这两个名字相关，但不是同一个概念：

* **DeepEP Hybrid Mode** 是 DeepEP V2 的一种拓扑模式，强调 NVLink 与 RDMA 的层次化组合；
* **HybridEP** 是 NVIDIA 开发并接入 Megatron Core Flex Dispatcher 的优化后端，使用 TMA、IBGDA 和 Warp-Specialized Pipeline 降低通信 SM 占用。

二者都利用混合互联，但前者描述 DeepEP 的拓扑选择，后者是一套具体的通信 Kernel 实现。

---

### HybridEP Dispatch Pipeline

HybridEP 将每个 CUDA Block 视为一条独立数据通道，通常占用一个 SM。Block 内不同 Warp Group 负责不同流水级：

```text
G2S Warp Group
从 HBM 读取本地或已到达的 token
        │
        ▼
Shared-Memory FIFO
        │
        ├── RDMA Warp Group：使用 IBGDA 推进跨节点传输
        │
        └── S2G Warp Group：使用 TMA 写入节点内目标 GPU
```

不同 Blocks 处理不同 Token Chunks，不需要在 Blocks 之间频繁同步。Shared-Memory FIFO 解耦各流水级的速度，细粒度 Chunk 则用流水深度隐藏 RDMA 延迟。

---

### HybridEP 为什么能省 SM

核心原因不是“通信不需要计算”，而是让少量 Warp 负责控制，让硬件数据通路负责大部分搬运：

* **TMA** 异步执行规则的 HBM、Shared Memory 和 NVLink 数据搬运，减少 Warp 手写 Load/Store；
* **IBGDA** 允许 GPU 侧直接推进 RDMA，减少 CPU Proxy 和 Host 调度；
* **Warp Specialization** 让同一个 SM 内的 Warp Groups 分别负责读入、RDMA 和写出；
* **Chunk Pipeline** 用连续小块填满流水，而不是用大量 SM 忙等一个全局阶段；
* **层次化拓扑** 降低跨节点通信的连接和控制复杂度。

Combine 比 Dispatch 多一步归约。HybridEP 会在数据路径中加入 Reduction Warp，对来自 Top-$k$ Experts 的输出进行分层累加，因此 Combine 的 SM 与精度成本通常不能完全等同于纯数据搬运。

---

## DeepEP V2 的接口边界

DeepEP V2 的主要变化包括：

* 使用统一的 `ElasticBuffer` 接口覆盖 High-Throughput 与 Low-Latency 场景；
* 使用 NCCL Gin 作为主要设备侧通信后端；
* 根据模型和拓扑解析计算 SM 与 QP 数；
* 支持 Direct 与 Hybrid 模式；
* 通过 JIT 编译通信 Kernel；
* Buffer 占用可能高于 V1；
* 不再支持 V1 的 0-SM RDMA Low-Latency EP 语义。

典型调用关系可以简化为：

```python
buffer = ElasticBuffer(
    group,
    num_max_tokens_per_rank=max_tokens,
    hidden=hidden,
    num_topk=topk,
    use_fp8_dispatch=True,
)

num_sms = buffer.get_theoretical_num_sms(num_experts, topk)

recv_x, recv_idx, recv_weight, handle, event = buffer.dispatch(
    x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
    num_max_tokens_per_rank=max_tokens,
    num_sms=num_sms,
    async_with_compute_stream=True,
)

event.current_stream_wait()
expert_out = grouped_expert_mlp(recv_x, handle.num_recv_tokens_per_expert_list)

output, _, event = buffer.combine(
    expert_out,
    handle=handle,
    num_sms=num_sms,
    async_with_compute_stream=True,
)
```

这里最重要的对象是 Handle 和 Event：

* Handle 保存 Combine 所需的路由和布局信息；
* Event 表达通信 Stream 与计算 Stream 之间的依赖；
* `num_sms` 决定通信与 GEMM 如何共享 GPU。

实际 API 会随 DeepEP 版本变化，部署时应以所固定 Commit 的 README 和测试代码为准。

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

---

## 总结

DeepEP 不改变 EP 的路由数学，也不能消除热门 Expert。它优化的是 Routed Tokens 的实际执行路径：

```text
更少中间搬运
+ 拓扑感知 NVLink / RDMA
+ 低精度 Payload
+ 显式 SM 预算
+ 通信计算重叠
= 更低的 Exposed Communication Time
```

High-Throughput 关注训练与 Prefill 的持续带宽，Low-Latency 关注 Decode 的启动延迟。DeepEP Hybrid Mode 是一种拓扑模式；HybridEP 则是使用 TMA、IBGDA 和 Warp Pipeline 的具体优化后端。

评估 DeepEP 时，至少要同时观察 Algorithm Bandwidth、NIC/NVLink Bus Bandwidth、通信 SM 数、最大 Send/Receive Count、Grouped GEMM 时间和 Exposed Communication Time，不能只看单独的 AllToAll 微基准。

版本相关实现以 [DeepEP 官方仓库](https://github.com/deepseek-ai/DeepEP)、[HybridEP 分支](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)和 [Megatron Core MoE 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)为准。
