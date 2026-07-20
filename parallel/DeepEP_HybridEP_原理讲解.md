# DeepEP 原理深度讲解 & Hybrid EP 为何能省 SM

## 0. 一句话直觉

**经典集合通信**像「固定时刻表的公交」：谁发给谁、发多少，在 launch 前就定死了。  
**MoE 的 Expert Parallelism**像「打车」：每个 token 的目的地由 router **每层现场决定**，流量稀疏、不规则、跨节点。

DeepEP 做的事，不是再包一层 NCCL AllToAll，而是把 MoE 的两个原语——**Dispatch（送 token 去见专家）** 与 **Combine（把专家输出加权送回家）**——做成：

1. **感知拓扑**的 GPU 侧通信内核（NVLink 域 vs RDMA 域不对称）；
2. **可显式预算 SM 数量**的通信消费者（把通信当成「占多少 SM 的租户」，而不是黑盒）；
3. 与 DualPipe / 双 microbatch / 解码 overlap **共设计**的 overlap 语义（含历史 LL 的 0-SM hook）。

Hybrid-EP（以及 DeepEP 的 hybrid mode）则进一步回答：**怎样用尽量少的 SM，把 NIC + NVLink 打满**——因为一旦链路打满，多占 SM 只会从 GEMM 手里抢算力。

---

## 1. 问题背景：MoE EP 为什么特别难

### 1.1 MoE 层的数据依赖通信

进入 MoE 层时，每个 rank 通常持有：

- 激活：`(B_i, H)`
- 路由后：`topk_idx / topk_weights`，形状约 `(B_i, K)`

专家分布在 EP 域的各个 GPU 上。token 必须去「见」它被路由到的专家，跑完 FFN 再回来做加权归约。

这与 TP AllReduce / SP AllGather / 静态 AllToAll 的本质差别：

| 维度 | 静态集体通信 | MoE EP |
|------|-------------|--------|
| 目的地 | 架构决定、步步相同 | Router 数据依赖、每层变 |
| 体积 | 可预知 | 热专家/冷专家导致极不均匀 |
| 拓扑利用 | 往往扁平 mesh | NVLink ≫ RDMA，应用层次化 |
| 与计算重叠 | 常见 | 训练要 DualPipe；解码几乎无计算可藏，要低延迟路径 |

DeepSeek-V3 一类细粒度 MoE（大量专家、topk 激活）会把 EP 通信推到墙钟时间的主要矛盾；未优化时通信占比可超过一半。

### 1.2 Infra 视角要优化的「三重约束」

1. **算法带宽**：有效 payload / 墙钟（含本地转发流量时的 logical BW）；
2. **SM 税**：通信内核占用多少 SM——重叠时，这些 SM 不能给 grouped GEMM；
3. **HBM / 延迟权衡**：紧凑布局省显存但要协调 RTT；固定矩形省 RTT 但浪费显存。

DeepEP 的设计轴，基本就是在这三者之间做可配置的工程折中。

---

## 2. DeepEP 是什么

**DeepEP（DeepEveryParallel）** 是 DeepSeek 开源的高性能通信库，核心是 MoE **专家并行**的 GPU kernel：

- **Dispatch**：把 token 激活送到持有对应专家的 GPU；
- **Combine**：把专家输出送回 token 归属 rank，并做（或配合做）**加权 reduce**；
- 支持 **FP8 dispatch / BF16 combine** 等低精度在线传输；
- 显式控制通信占用的 **SM 数量**；
- V2 起统一为 `ElasticBuffer`，后端从 NVSHMEM 转向更轻的 **NCCL Gin**；并实验性提供 0-SM Engram / PP / CP 等原语。

它不是「另一个 NCCL wrapper」，而是 **为不规则 MoE 流量定制的、GPU-initiated、拓扑感知的 all-to-all 实现**。

---

## 3. 端到端数据路径（直觉图）

```
Attention / Router (local)
        │  topk_idx, topk_weights
        ▼
┌─────────────── DISPATCH ───────────────┐
│ 布局：HT=先计数再紧凑；LL=固定最坏矩形 │
│ 打包：可选 FP8 量化                    │
│ 传输：同节点 → NVLink                  │
│       跨节点 → RDMA 到同 rail 对端，   │
│               再 NVLink 转发到目标 GPU │
│ 输出：供 grouped GEMM 的布局           │
└────────────────────────────────────────┘
        │
        ▼
   Expert FFN (grouped GEMM)
        │
        ▼
┌─────────────── COMBINE ────────────────┐
│ 反布局 → 沿原路径送回                   │
│ 对每个 token 的 K 路输出做加权求和      │
└────────────────────────────────────────┘
        │
        ▼
下一层 / Residual
```

### 3.1 层次化转发（Hybrid Mode 的通信拓扑含义）

在多节点、每节点多卡、每卡一条 NIC 的常见 rail 拓扑上：

> Token 要去 **节点 B 的 GPU j** 时，典型路径是：  
> **本节点 GPU i → RDMA → 节点 B 的 GPU i（同 rail）→ NVLink → 节点 B 的 GPU j**。

直觉：

- 每张 NIC 只跟「对轨」的远程 GPU 说话，**连接数 / QP 数可控**，避免 full mesh 爆炸；
- IB 与 NVLink **可以流水重叠**：远端 rail peer 一边收 IB，一边往机内转发；
- DeepSeek-V3 的 **group-limited gating**（限制 token 跨节点的节点数）与此共设计，让跨节点扇出可管。

这就是「Hybrid」一词在 DeepSeek/DeepEP 语境下的第一层含义：**NVLink 域 + RDMA 域的层次化混合**，而不是扁平 GPU 全互连。

---

## 4. 两条内核哲学：High-Throughput vs Low-Latency

这是理解 DeepEP 最关键的分叉。一句话：

> **HT 花一次协调 RTT，换紧凑显存；LL 花最坏情况显存，换掉那次 RTT。**

### 4.1 High-Throughput（训练 / Prefill）：先问再送

**为什么需要「问」？**  
紧凑接收缓冲 `(N_recv, H)` 的大小，以及每个 token 写入的 offset，都依赖「别人要给我多少」。这些在本机 router 跑完前全局未知。

**步骤：**

1. **Count exchange**：每个 rank 已知自己发给每个 peer 的 token 数（整数，极小）；按同样拓扑做一次协调（跨节点 RDMA + 机内 NVLink）。
2. **Prefix-sum layout**：收到各 source 的 count 后，前缀和给出每个 source 块的起始 offset → 紧凑 buffer。
3. **经预注册队列发送**：RDMA 只能写进预先注册的 MR；紧凑 buffer 是本步才定大小的，所以发送方写入 **固定大小的 pre-registered queue**，接收方再 drain 到紧凑 buffer。
4. **按 source 到达 → 本地 permute 成按 expert**：DeepEP 常把「by-source」交给框架再排成「by-expert」喂 grouped GEMM。
5. **Combine** 复用 dispatch 的 handle / 路由信息，**不必再做一轮 count RTT**。

适合：batch 大、GEMM 算得动、可以用 DualPipe / 双 microbatch **藏住协调延迟**；显存紧（KV cache 珍贵）的服务侧 prefill。

### 4.2 Low-Latency（Decode）：不问直接送

Decode 时每 rank 往往只有很少 token，协调 RTT 几乎藏不住，且动态 shape 不利于 CUDA Graph。

**做法：**

1. 预先为每个 `(source_rank, local_expert)` 留 **私有矩形槽位**；
2. 写地址变成闭式公式，例如：

```text
addr = base + (e * R + r) * chunk + slot
```

3. **第一件事就是数据 RDMA**，没有 count RTT；
4. 发送方写完 payload 后，在有序通道上写 **count/flag**（空值可区分「未到达」与「发了 0 个」）；接收方看 flag 得知有效行数；
5. 默认 **FP8 在线 dispatch、BF16 combine**（combine 要累加，精度更敏感）；
6. 缓冲是 `E_local × R × chunk × H` 量级，**大多是空洞**——用显存买延迟。

V1 的 LL 还有著名的 **recv hook**：发送侧 post 之后，NIC/IBGDA 可继续推进；hook 等待时 **不必让 SM 空转 polling**，从而在 overlap 窗口内接近 **0 SM occupation**（V2 说明已不再支持「0 SM RDMA LL」这一旧语义，重叠模型有变化）。

---

## 5. 核心机制拆解（通算融合视角）

### 5.1 进程内：NVLink

- V1：定制 kernel / NVSHMEM LSA 等；
- Hybrid-EP / 新 HT：大量用 **TMA（Tensor Memory Accelerator）** 做 G↔S、跨 GPU 拷贝——拷贝引擎式异步搬运，warp 主要发 descriptor / fence，而不是手写 `LD/ST` 搬字节。

### 5.2 跨节点：GPU-initiated RDMA

| 技术 | 角色 |
|------|------|
| **GPUDirect RDMA** | NIC DMA 直达 GPU HBM，无 bounce buffer |
| **IBGDA** | GPU 映射 NIC UAR，在 device 侧 post WR（V1 / Hybrid-EP） |
| **NVSHMEM** | V1 主后端；GPU↔HCA 映射关键 |
| **NCCL Gin / GDAKI** | V2 默认：更轻、可复用 NCCL communicator |

要点：通信进度尽量 **不经过 CPU proxy**，否则 RTT 与 SM/Host 调度都会变差。

### 5.3 Buffer 与打包

- **V1 `Buffer`**：`num_nvl_bytes` / `num_rdma_bytes` 分离；HT 用 queue（省内存，有死锁边界）；LL 用大固定 RDMA 区域。
- **V2 `ElasticBuffer`**：HT+LL 统一接口；缓冲往往更大；**解析计算 SM/QP**（`get_theoretical_num_sms`），不再靠盲目扫参。
- 在线 **FP8**：直接砍线带宽与延迟（尤其 LL）。

### 5.4 通信为什么会占用 SM？

很多人以为「有 NIC / Copy Engine 就不占 SM」。对 MoE EP 不成立，因为至少有人要做：

- 按 routing map **筛选/打包/量化** token；
- 维护 shared-memory FIFO、阶段同步；
- post RDMA、poll CQ / 等 flag；
- Combine 上的 **分层 reduce**（BF16 累加往往在 CUDA Core 上）。

因此 DeepEP 把通信做成 **显式 SM 预算的持久化/多 warp 内核**：例如 V1 `Buffer.set_num_sms(24)`；V2 `num_sms=` / 理论值。  
目标不是「通信 SM 越多越好」，而是 **刚好打满 NVLink/RDMA 瓶颈，其余 SM 全部留给 GEMM**。

### 5.5 计算–通信重叠的几种形态

1. **流重叠**：`async_with_compute_stream` / `EventOverlap`；
2. **DualPipe（训练）**：双向 PP，把 EP A2A 藏进另一侧计算；
3. **双 microbatch decode**：batch0 在飞，batch1 做 Attention/MoE；
4. **Hybrid-EP 细粒度 chunk pipeline**：64/128 token 一切，把动态路由与 RDMA RTT **藏进流水线气泡**，使 EP 算法带宽逼近静态 AllToAll。

对融合工程师：评估指标应是 **(算法带宽, SM 数, 端到端 GEMM 有效占用)** 三元组，而不是单看微基准 GB/s。

---

## 6. DeepEP V1 → V2：省 SM 的产品化路径

V2 相对 V1 的关键变化（与本文主题直接相关）：

| 项目 | V1 | V2 |
|------|----|----|
| API | HT/LL 分离，`Buffer` | 统一 `ElasticBuffer` |
| 后端 | NVSHMEM + IBGDA | **NCCL Gin**（更轻） |
| SM | 经验调参，常过配（如 24） | **解析资源分配**，V3-like 可到 **4–6 SM** |
| 规模 | 较小 | 宣称可达 EP2048 |
| 性能 | 基线 | 峰值可达约 **1.3×**，SM 可省至约 **1/4** |
| LL 0-SM RDMA | hook 支持 | **不再支持**旧 0-SM LL |

解析分配的直觉：先用拓扑 + `ibstat`/环境变量估计 **RDMA/NVLink 瓶颈带宽**，再反推「打满该瓶颈最少需要多少 SM / QP」。  
过配 SM：微基准可能略好看，但 DualPipe 下会 **直接偷走计算 SM**。

---

## 7. Hybrid EP：两个容易混的概念

实践中「Hybrid EP」常指两件相关但不同的事：

### 7.1 DeepEP 的 Hybrid Mode（拓扑语义）

- **Hybrid**：层次化 NVLink + RDMA（rail 转发）；
- **Direct**：更接近扁平/直连风格的跨 GPU 通信。

V2 两者仍都支持。Hybrid 适合 **机内 NVLink ≫ 机间 RDMA** 的集群（H800/H100 + CX7 等）。

### 7.2 NVIDIA Hybrid-EP（内核实现语义）

[DeepEP `hybrid-ep` 分支](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep) / Megatron `moe_flex_dispatcher_backend=hybridep`：

- 用 **TMA** 做 NVLink 域数据搬运；
- 用 **IBGDA / DOCA / NIXL** 等做 RDMA；
- **Warp-specialized 多级 pipeline**；
- 目标：**极少 SM 打满混合网络**，并支持更细粒度的通算重叠。

Megatron 侧常见配置直觉：

```text
--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep   # 或 hybridep
# 再调 --moe-deepep-num-sms / --moe-hybridep-num-sms
```

下文「为什么 Hybrid EP 能省 SM」主要针对 **7.2 的内核实现**，并兼容 7.1 的层次化拓扑（二者叠加才是完整故事）。

---

## 8. Hybrid-EP 内核解剖：一个 SM 就是一条流水线

NVIDIA 描述的核心抽象：

> **每个 CUDA block ≈ 占用一个 SM，作为一条独立数据通道；block 内不同 warp group 负责不同流水级；不同 block 处理不同 data chunk，block 间无需同步。**

### 8.1 Dispatch 流水线（单 block 内）

典型 warp group 分工（概念模型）：

| Warp Group | 职责 |
|------------|------|
| **G2S** | 用 TMA：本卡（及同 rail 远端已到本卡的）token → SM 内 **shared memory FIFO** |
| **RDMA** | 用 IBGDA：把需要跨节点的数据送到 **同 rail 远程 GPU** |
| **S2G / NVLink** | 用 TMA/LSA：从 SM FIFO 写到 **本节点所有目标 GPU** 的输出 buffer（含本卡） |

数据按 routing map **只搬需要的 token**；chunk 粒度（如 64 tokens）让「等路由元数据 / 等 RDMA RTT」的气泡被后续 chunk 填满。

直觉类比：

> 旧式 EP：很多工人 **亲手搬箱子**（大量 SM 做 LD/ST）。  
> Hybrid-EP：少数调度员操作 **传送带（TMA）+ 码头吊车（IBGDA）**，工人很少，吞吐由传送带和吊车决定。

### 8.2 Combine 流水线：必须分层 Reduce

Combine 要对 K 路专家输出做高精度累加，累加目前主要在 **SM 的 CUDA Core** 上完成，因此 Hybrid-EP 采用 **层次化 reduce**：

1. **机内**相关 warp 先做节点内部分累加；
2. **RDMA warp** 把部分和送到跨节点同 rail GPU；
3. **跨节点相关 warp** 完成全局累加，得到最终结果。

单节点时，机内累加路径可直接落到「跨节点」那组 warp 的角色上。  
路径上仍是：G2S FIFO → Reduction warp → S2G FIFO → TMA 写回 HBM。

层次化 reduce 的副作用：**跨节点流量从「原始 K 路全量」变成「部分和」**，既省带宽，也减少 RDMA 侧需要驱动的 outstanding 工作——间接再省 SM。

---

## 9. 深入：为什么 Hybrid EP 可以省 SM

下面按「因果链」展开，而不是只报「4–16 SM 就打满」。

### 9.1 先搞清：SM 在通信里到底在干什么

GPU 上「搬数据」有几类执行者：

1. **CUDA Core / Tensor Core 上的 warp**：执行 load/store、打包、reduce、轮询；
2. **TMA / Copy Engine**：异步批量拷贝，warp 只发命令；
3. **NIC DMA（GDR）**：HBM↔网络，不消耗「算力指令」，但需要有人 **post/doorbell/completion**。

naive EP 之所以吃很多 SM，是因为把 (1) 当成了主路径：  
**字节搬运的吞吐 ≈ 活跃 warp 数 × 每 warp 有效带宽**。要喂饱 NVLink/IB，就得堆很多 warp → 很多 SM。

Hybrid-EP 的策略是：**把 (1) 缩到「控制面」**，把 (2)(3) 变成「数据面」**。控制面的 SM 需求远低于「手搬字节」的数据面。

### 9.2 机制一：TMA 卸载字节搬运（最大头的 SM 节省）

手写 `LD.global / ST.global` 或甚至 `__ldg`：

- 每个 16B/32B 都要占 issue 槽与寄存器；
- 要靠大量 warp 做延迟隐藏；
- L1/L2 行为还要小心（DeepEP 里常见激进 PTX，如 `L1::no_allocate` 处理 RDMA 可见易失数据）。

TMA：

- warp 写入拷贝描述符后可以去做别的事或等待 barrier；
- 大批连续/张量布局搬运由硬件异步完成；
- **同样 GB/s，活跃 warp / SM 数断崖下降**。

因此 Hybrid-EP 在 **机内 NVLink 扇出**（dispatch 的 S2G、把 token 写到节点内多 GPU）上特别省 SM：这恰恰是 naive 实现最爱堆 SM 的地方。

### 9.3 机制二：层次化拓扑缩小 RDMA 扇出（控制面变瘦）

Flat all-to-all：每个 GPU 可能对 **O(EP_world)** 个 peer 有 outstanding。

Hybrid：

- RDMA 只打 **rail peer ≈ O(节点数)** 量级；
- 机内扇出交给 NVLink/TMA。

后果：

- QP / WR / CQ poll 的并发度下降；
- 卡在「等 NIC」的 warp 变少；
- 同一条 RDMA 流水线用 **更少 SM** 就能把 CX7 打满（NVIDIA 数据：四机 H100 场景约 **4 SM** 就接近 NIC 峰值）。

这与 DeepSeek「group-limited gating + rail 转发」是同一设计哲学：**用算法与拓扑约束，减少设备侧控制复杂度**。

### 9.4 机制三：Warp Specialization + SM 内 FIFO = 用流水深度换 SM 宽度

一个 SM 上同时跑 G2S / RDMA / S2G（及 combine 的 reduce）：

- 形成 **producer–consumer 流水线**；
- shared memory FIFO 解耦各级速率；
- **深度优先于宽度**：先把一条通道的流水填满，再横向加 block（SM）。

经验曲线通常是：

- 1→4 SM：带宽陡升（填满流水与 NIC）；
- 4→8 SM：可能仍有收益；
- 8→16 SM：在 RDMA 瓶颈拓扑上经常 **平台期**——再加 SM 几乎不涨 GB/s，只抢 GEMM。

Hybrid-EP 的产品目标正是：**尽快到达「硬件膝盖」左侧最小 SM 点**。

### 9.5 机制四：瓶颈在链路，不在 ALU —— 多 SM 收益为负

H100 约 132 SM。若通信占 24 SM，重叠窗口里 GEMM 最多用 ~108；若通信用 4–8 SM，GEMM 多出十几到二十个 SM。

关键不等式：

```text
若 Bandwidth(SM) 在 N* 处已饱和物理链路，
则 ∀ N > N*:  ΔBW ≈ 0，但 ΔCompute_SMs = -(N - N*) < 0
⇒ 端到端变慢或 DualPipe 气泡变大
```

所以「省 SM」不是微基准炫技，而是 **通算融合的一阶优化目标**：通信是租户，GEMM 是房东。

### 9.6 机制五：细粒度 Chunk 流式化，减少「屏障式 SM 空转」

若必须等「全局 layout 完全就绪」再开大拷贝：

- 大量 SM 会在 barrier / 等 count 上空转；
- 或被迫用更多 SM 去做无意义的忙等掩盖。

Chunk pipeline：

- 路由元数据与小块 payload 交替推进；
- RDMA RTT 被后续 chunk 掩盖；
- 动态 MoE 的不规则性被「切成接近静态 AllToAll 的流」。

NVIDIA 原文要点：通过 fine-grained chunk streaming，使 EP 带宽 **可比高度优化的静态 all-to-all**——同时保持低 SM。

### 9.7 机制六：Combine 层次化 Reduce 降低跨域工作量

跨节点若回传全部 expert 输出再在 home 累加，RDMA 字节与控制复杂度都高。  
机内先 reduce 再跨节点，相当于 **在高带宽域做归约，在低带宽域传摘要**。

这同时：

- 降低 RDMA 负载 → 更少 RDMA warp 压力；
- 把一部分累加留在已经占用的那几个通信 SM 上，而不是另开大规模 reduce kernel 抢 SM。

### 9.8 机制七：与 DeepEP V2 解析 SM 分配同向

Hybrid-EP 的「少 SM 打满」与 DeepEP V2 的 `get_theoretical_num_sms` 是同一思想的两端：

- Hybrid-EP：用 TMA+流水线 **把 N\*** 往左推（同样拓扑，膝盖更早到来）；
- DeepEP V2：用带宽模型 **算准 N\***，避免过配。

公开数字锚点（量级，非你集群保证值）：

| 场景 | 大约 SM | 现象 |
|------|---------|------|
| Hybrid-EP，单机 H100 NVLink | ~8 | 填满 NVLink |
| Hybrid-EP，4×DGX H100 + CX7 | ~4 | 接近 NIC 峰值 |
| Hybrid-EP，GB200 NVL36 | ~16 | 填满大规模 NVLink 域 |
| DeepEP V1，V3-like 训练 | ~24 | 经验过配常见 |
| DeepEP V2，V3-like | **4–6** | 性能持平或更好 |
| DeepEP V2，SM90 EP 8×2 | 12 | RDMA ~90 GB/s logical |

Megatron 实测量级（Grace Blackwell，相对基线 dispatcher）：DeepSeek-V3 上 Hybrid-EP 相对 DeepEP 约 **+14%** TFLOPS/GPU 等——收益来自 **更低通信 SM 税 + 更高有效带宽**，而不只是单一因素。

### 9.9 反例：什么情况下「Hybrid / 少 SM」会失效

1. **Rail / HCA 映射错误**：流量跨轨拥塞，NIC 打不满，看起来像「要更多 SM」，实际是拓扑问题；
2. **ACP bonding 等导致 `ibstat` 带宽读半**：V2 解析 SM 算少，需 `EP_RDMA_GBS` 等手动校正；
3. **纯 NVLink 域、追求绝对峰值**：可能故意用更多 SM（V2 表：SM100 EP8 峰值 64 SM vs 最小 24 SM）——那是 **吞吐微基准** 目标，不是 DualPipe 目标；
4. **Combine 累加极重、chunk 过小**：控制开销占比上升，N\* 右移；
5. **误用 HT/LL**：decode 走 HT 协调、prefill 走巨大 LL 矩形，都会让「省 SM」的故事崩盘。

---

## 10. 与传统方案对比（选型直觉）

| 方案 | 模式 | 相对 DeepEP/Hybrid-EP |
|------|------|----------------------|
| NCCL AllToAll + permute | 主机/通用集体 | 易用；不规则 MoE、FP8 融合、SM 预算弱 |
| Megatron AllGather dispatcher | 聚集到本地 | 小 EP/大 topk 机内尚可；宽 EP 爆内存带宽 |
| Megatron AllToAll | 标准 A2A | 基线；缺域感知与 SM 精细控制 |
| Flex + DeepEP | 设备侧 EP | 跨节点 EP、FP8、SM API；HT 仍非零 SM |
| Flex + HybridEP | TMA+IBGDA HT | NVL72/GB200 等上少 SM；部署更重、吃拓扑 |
| Tutel / FasterMoE | 早期 MoE 系统优化 | 调度/容量强；非同一套 GPU-initiated RDMA+hook 栈 |

融合工程师的差异化检查清单：

1. 有没有 **SM-number API**？
2. 是否 **NVLink↔RDMA 融合转发**？
3. 是否 **FP8 dispatch**？
4. decode 路径如何重叠（hook / graph / cached handle）？
5. 是否与 **routing 约束（group-limited）** 共设计？

---

## 11. 使用形态速览（便于对照源码）

### 11.1 DeepEP V2 `ElasticBuffer`

```python
from deep_ep import ElasticBuffer

buf = ElasticBuffer(
    group,
    num_max_tokens_per_rank=...,
    hidden=...,
    num_topk=...,
    use_fp8_dispatch=True,
)
num_sms = buf.get_theoretical_num_sms(num_experts, num_topk)

recv_x, recv_topk_idx, recv_topk_weights, handle, event = buf.dispatch(
    x, topk_idx=..., topk_weights=..., num_experts=...,
    num_max_tokens_per_rank=..., num_sms=num_sms,
    async_with_compute_stream=True,
)
# ... 独立计算 ...
event.current_stream_wait()

combined_x, _, event = buf.combine(
    x, handle=handle, num_sms=num_sms,
    async_with_compute_stream=True,
)
```

### 11.2 V1 LL hook（历史重要语义）

```python
Buffer.set_num_sms(24)
recv, count, handle, event, hook = buffer.low_latency_dispatch(
    ..., return_recv_hook=True,
)
# 此处可跑另一 batch / GEMM；NIC 推进，通信侧接近 0 SM
hook()  # 物化接收缓冲
```

---

## 12. 给通算融合工程师的设计原则（收束）

1. **把 EP 当成带 SM 配额的租户**，不是黑盒 collective；配额目标是链路饱和点 N\*，不是越大越好。
2. **Hybrid（层次化）优先于 flat**，当且仅当 NVLink ≫ RDMA；并用 gating 限制跨节点扇出。
3. **HT vs LL 按 regime 切开**：训练/prefill 要紧凑与可藏 RTT；decode 要去 RTT、可 graph，接受显存空洞。
4. **Hopper+ 上投资 TMA + GPU-initiated RDMA**：把通信从「算力搬砖」变成「控制面 + 硬件数据面」。
5. **端到端以 GEMM 占用为准绳**：微基准 GB/s 上升但 SM 翻倍，在 DualPipe 下可能是负优化。
6. **测量三元组**：算法带宽、通信 SM 数、重叠窗口内有效 TFLOPS；再辅以 NIC rail 正确性与拥塞控制策略。

---

## 13. 术语速查

| 术语 | 含义 |
|------|------|
| Dispatch / Combine | MoE EP 去程 / 回程原语 |
| HT / LL | High-throughput / Low-latency 内核族 |
| Hybrid Mode | NVLink+RDMA 层次化转发 |
| Hybrid-EP | NVIDIA/DeepEP 分支：TMA+IBGDA 的少 SM 实现 |
| IBGDA | GPU 直接访问 IB 门铃/队列的技术路径 |
| TMA | Hopper+ 张量内存加速拷贝 |
| Gin | NCCL 设备侧通信 API（DeepEP V2 后端）
| Rail | 多机多卡下「同下标 GPU + 对应 NIC」对齐的路径 |
| N\* | 打满物理链路所需的最小通信 SM 数 |
| DualPipe | DeepSeek 双向流水，用于藏 EP 通信 |

---

## 附录：假设与边界

- 本文综合公开 README、NVIDIA Blog、社区解剖文章与 V3 报告中的设计叙述；**具体指令级实现以你使用的 DeepEP commit / Hybrid-EP 分支源码为准**。
- 「Hybrid EP 省 SM」在工程上常同时受益于：**层次化拓扑 + TMA 卸载 + warp pipeline + 解析/经验选 N\***；单独强调任一机制都会不完整。
- 日期与版本演进较快（V2、Gin、0-SM LL 语义变更）；落地前请核对当前仓库 News / legacy 文档。

---

*文档生成位置：`/kl_infra_infer_intern/yangyixin03/DeepEP_HybridEP_原理讲解.md`*
