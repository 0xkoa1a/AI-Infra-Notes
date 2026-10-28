---
title: "MegaMoE：Expert-Wave 实现与 Dispatch 设计（分享讲解）"
order: 10
---

# MegaMoE：Expert-Wave 实现与 Dispatch 设计

> 本文将两份分享讲解按主题合并，保留技术正文、公式、代码示例与图注，图示按原结构用 Vue 重绘。原文主要讨论 DeepGEMM `88965b0` 的早期 Expert-Wave 实现。
>
> 来源：[Kernel 的通信与计算融合](https://chatgpt.com/s/t_6ae1b3c4a510819195d906a3842c9015)、[Dispatch 的 metadata push 与 payload pull](https://chatgpt.com/s/t_6ae1b42a45988191bd5a5dbd7e81dc53)。

MegaMoE 是 DeepSeek 在 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM "DeepSeek GPU GEMM 算子库") 中实现的一个 **Persistent MoE Mega-Kernel**。它将 Expert Parallelism（EP）中的 Dispatch、两次 Grouped GEMM、SwiGLU、Combine 全部融合进一次 kernel launch，并利用 Blackwell 的 Tensor Core、TMA 和 NVLink 实现细粒度流水线。

我查阅了 DeepGEMM 的实际 CUDA 实现。建议重点看以下四个文件：

- [sm100_fp8_fp4_mega_moe.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)：Kernel 本体，包含 warp specialization、GEMM、dispatch 和 epilogue。
- [scheduler/mega_moe.cuh](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)：Persistent CTA 如何选择下一块 L1/L2 tile。
- [csrc/apis/mega.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/csrc/apis/mega.hpp)：输入、输出、workspace 和 token pool 的布局。
- [heuristics/mega_moe.hpp](https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/jit_kernels/heuristics/mega_moe.hpp)：Block size、pipeline stages、SMEM 的选择。

这里以早期的 Expert-Wave 实现（commit `88965b0`）为主，因为它的 scheduler 正是前面 X-Stage 工作所针对的设计。DeepGEMM 后续主分支已经引入 ring buffer、shared expert 等额外机制，具体实现存在版本差异。

## 背景：MegaMoE 实际上在计算什么？

考虑 EP8，每张 GPU 保存一部分专家权重。每个 token 经过 router，选出 top-k 个专家。

对于一个 token $x$，每个被选中的专家执行：

$$
\begin{aligned}
[g,u]&=xW_1\\
h&=\operatorname{SiLU}(g)\odot u\\
o&=hW_2
\end{aligned}
$$

随后根据 router 的权重 $p_e$，合并专家输出：

$$
y=\sum_{e\in\operatorname{TopK}(x)}p_eo_e
$$

其中 $W_1$ 包含 gate/up 两路投影，因此实际的第一层输出维度是 $2I$，SwiGLU 之后变成 $I$。

普通 MoE 推理一般拆成：

<MegaMoESharedDiagram view="moe-flow" />

MegaMoE 的输出仍然是这些运算的结果，但运行方式发生了很大变化。

比如原先必须等整个 Dispatch kernel 完成才能启动 L1。而 MegaMoE 允许：

- Dispatch 收到专家 A 的一块 token，就让 L1 开始计算这一块。
- L1 完成一个输出 tile，就立即执行 SwiGLU 和 FP8 量化，送入 L2 的输入池。
- L2 对已经就绪的输入 tile 进行计算，完成后直接向原 token 所在 GPU 发出 NVLink remote stores。
- 以上工作由同一个 persistent kernel 内不同的 warp 推进。

因此，MegaMoE 实现的关键问题是：**如何让不同 warp 独立推进各自负责的工作，同时满足 Dispatch → L1 → SwiGLU → L2 → Combine 的数据依赖。**

## 整体架构：一个 SM 上的 Persistent CTA

原始 kernel 使用 SM100 的 warp specialization，典型的计算 CTA 内可以理解为：

<MegaMoESharedDiagram view="cta-roles" />

源码最外层基本就是根据 `warp_idx` 选择长期运行的角色。每个角色执行自己的循环，由 barrier、arrival counter 协调。

这也是 `__launch_bounds__(kNumThreads, 1)` 的目的之一：限制每个 SM 的驻留 CTA 数量，配合 persistent scheduling 管理固定的计算与通信角色。实际计算采用 2-CTA cluster，两个 CTA 在 Tensor Memory 和 MMA 执行中有协作关系。（[MegaMoE 参考讲解](https://llm-infra-atlas.github.io/moe/07_megamoe/)；[调度器源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)）

下面从输入如何进入 token pool 开始，沿着 kernel 的数据路径逐步看实现。

## 数据与内存布局：为什么需要 Token Pool？

这是理解 MegaMoE 代码的第一个关键。

普通 GEMM 的输入矩阵是规则的二维数组，但 MoE 不同：每个 expert 实际分到多少 token，是运行时 router 决定的。

假设某张 GPU 拥有三个专家，本次收到的 token 数分别是：

| Expert | Token 数 | `BLOCK_M=16` 时需要的 M tiles |
|---|---:|---:|
| E0 | 27 | 2 |
| E1 | 8 | 1 |
| E2 | 19 | 2 |

MegaMoE 把它们按 expert 顺序摆进一个连续的 L1 token pool，每个 expert 占用整数个 `BLOCK_M`：

<MegaMoESharedDiagram view="token-pool" />

对于第 $e$ 个专家，其 pool 起点就是前面所有专家占用 M blocks 数量的前缀和：

$$
\operatorname{offset}_e
=B_M\sum_{j<e}
\left\lceil\frac{M_j}{B_M}\right\rceil
$$

其中 $M_j$ 是专家 $j$ 的实际 token 数，$B_M$ 是 `BLOCK_M`。

这让所有专家能够共用一套 GEMM kernel。每个 CTA 只需知道 `(expert_idx, m_block_idx, n_block_idx)`，即可计算 token、权重和输出的地址。

MegaMoE 同时维护几类重要的全局显存结构：

| 结构 | 保存什么 | 谁生产 / 谁消费 |
|---|---|---|
| `l1_acts` | Dispatch 后的 FP8 tokens | Dispatch / L1 |
| `l1_topk_weights` | Router weights | Dispatch / L1 epilogue |
| `l2_acts` | SwiGLU 后的 FP8 intermediate | L1 epilogue / L2 |
| `token_src_metadata` | 原 rank、原 token、top-k 槽位 | Dispatch / L2 epilogue |
| `combine_token_buffer` | 每个 top-k 槽位的 partial output | L2 epilogue / Reduce |
| Arrival counters | 数据就绪状态 | 不同 warp 之间的同步 |

这些 pool 位于 Global Memory。L1/L2 计算时通过 TMA 从 pool 搬运 tile 到 SMEM，结果由 Tensor Core 累加在 TMEM 中。

这里需要留意一个实现细节：**MegaMoE 的 L1→L2 intermediate 仍然可以经过 Global Memory**。单 kernel fusion 的价值主要是消除独立 kernel 之间的全局阶段等待，使中间数据能够按 tile 流水化生产和消费，并提高缓存复用。它并没有把整个 MoE 层的 intermediate 都放进 SMEM。（[API 与工作区布局](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/csrc/apis/mega.hpp)；[MegaMoE 参考讲解](https://llm-infra-atlas.github.io/moe/07_megamoe/)；[Kernel 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)）

## GPU 如何直接访问其他 Rank 的显存？

底层使用 PyTorch Symmetric Memory。

每个 rank 都分配具有一致逻辑布局的显存块，经过 rendezvous 后获取其他 rank 对应区域的地址。

源码中的 `SymBuffer::map()` 本质上通过地址偏移将本地指针映射到远端：

```cpp
remote_ptr = local_ptr
           + (remote_base - local_base);
```

随后 kernel 可以通过远端指针完成 P2P 访存，而不必为每个 token 再调用一次 NCCL 或 DeepEP 通信 API。

当然，peer memory 映射、访问权限和设备拓扑必须事先满足条件。（[MegaMoE 参考讲解](https://llm-infra-atlas.github.io/moe/07_megamoe/)；[Kernel 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)）

## Dispatch：为什么是先 Push Metadata，再 Pull Token？

这是 MegaMoE 一个相当特别的设计。

一般我们想到 EP Dispatch，都会理解为：一个 GPU 主动把本地 token push 给目标专家所在的 GPU。

**MegaMoE 采用「源端 push routing metadata，目标端 pull token payload」的设计，核心目的是让 Expert Owner 自己组织 GEMM 输入，将跨 GPU 的数据搬运与按 Expert 的 permute 融合在一起。**

MegaMoE 的 dispatch 实际分两部分：

<MegaMoESharedDiagram view="push-pull" />

其具体步骤是：

1. **Routing count。** Dispatch warps 读取本地 `topk_idx`，统计需要送给每个 expert 的 token 数，通过 atomic 更新发送计数。
2. **Push metadata。** 将源 token 的索引和 top-k 槽位写到 expert owner rank 的对称显存工作区。
3. **交换计数并同步。** 各 rank 获得自己需要接收的 token 数，确保远端 metadata 已写完。
4. **Pull payload。** Expert owner 根据 metadata 主动通过 NVLink/TMA 读取远端 FP8 token 和 scale factor，并按 expert-major layout 写进 L1 pool。
5. **发布数据就绪状态。** 每写完一个 token，更新对应 pool block 的 `l1_arrival_count`。

为什么采用这种设计？

因为 expert owner 已经知道自己希望如何组织 token pool。它可以直接把远端 token 拉到适合 Grouped GEMM 的位置，避免发送方之间竞争目标位置所带来的复杂管理。

此外，一个 expert 的 token 可能来自多个 rank。源码里还用了 round-robin 式的来源选择，交错处理不同 rank 的 token，以分散远端数据搬运负载。

不过这里有两个值得区分的问题：

- 为什么 metadata 要由源端 push？因为只有源端天然知道每个 token 被 router 分配给了哪些 expert。
- 为什么 token payload 要由目标端 pull？因为目标端负责组织自己的 expert-major token pool，可以在接收数据时直接写入 GEMM 所需的布局。

我核对了 [MegaMoE 原始 CUDA 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)，下面具体解释。

### Push metadata：具体发送了什么？

假设 EP8，每个 rank 拥有 4 个 expert，总共 32 个 expert，每个 token 的 `topk=2`。

Rank 0 有一个 token：

```text
token_idx = 17

topk_idx[17] = [9, 23]
topk_weight[17] = [0.7, 0.3]
```

意味着它需要由 Expert 9 和 Expert 23 处理。

此时 Rank 0 会分别向这两个 expert 所在的 GPU 发送路由信息。

**但真正发送的不是整个 `topk_idx` 数组，也不是 token vector，而是一个 32-bit 的 flattened index：**

$$
\text{token\_topk\_idx}
=\text{token\_idx}\times K_{\text{top}}+\text{topk\_slot}
$$

在这个例子中：

| 目标 Expert | 目标 Rank | Push 的值 |
|---|---|---|
| Expert 9 | Rank 2 | 34 |
| Expert 23 | Rank 5 | 35 |

由于 `topk=2`，目标端拿到 35 就能够反推出：

```cpp
src_token_idx = 35 / 2;  // 17
src_topk_idx  = 35 % 2;  // 1
```

源 Rank 是由 metadata 所在的分区隐式表示的，目标 Expert 由 metadata 所在的 expert 分区表示。

也就是说，每条 routing metadata 中真正需要传输的只有一个 uint32。

此外，源端还会传输**每个源 Rank 发往各个 Expert 的 token 数量**，用于让目标端知道每个分区有多少有效条目。（[Kernel 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)；[布局源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/layout/mega_moe.cuh)）

#### 为什么不让目标端主动 pull `topk_idx`？

因为此时目标端只知道自己有哪些 expert，还不知道哪些 token 被分配给了自己。

如果直接从各个源 GPU 拉取原始 routing 表，目标端就需要扫描各源 Rank 的 `topk_idx`，过滤出指向本地 expert 的条目。

例如 Rank 2 要处理 Expert 8–11，但它并不知道 Rank 0 的哪些 token 选择了这四个 expert。

而源 Rank 0 只需要扫描一次自己的 routing 结果，就能把条目发送到对应的 expert。

**这相当于让 Routing Producer 构建一个按 Expert 分类的稀疏反向索引。**

当然，也可以先让源端建立反向索引，再由目标端 pull 这个索引。但依然需要源端执行分类、计数和地址发布。MegaMoE 直接选择了 push metadata。

### Metadata 写入目标端的哪个位置？

源码使用三个索引组织 metadata：

```cpp
get_src_token_topk_idx_ptr(
    local_expert_idx,
    src_rank_idx,
    slot_idx
);
```

因此，目标端的 metadata workspace 逻辑上是：

<MegaMoESharedDiagram view="metadata-layout" />

源端扫描 `topk_idx` 时，通过 atomic 预先分配每个 expert 的写入偏移量，再把 flattened index 写到目标 GPU 的对应分区。

关键源码是：

```cpp
const auto dst_rank_idx =
    expert_idx / kNumExpertsPerRank;

const auto dst_slot_idx =
    atomicAdd_block(
        shared_storage.expert_token_count + expert_idx, 1
    );

const auto dst_ptr =
    workspace.get_src_token_topk_idx_ptr(
        expert_idx % kNumExpertsPerRank,
        sym_buffer.rank_idx,
        dst_slot_idx
    );

*sym_buffer.map(dst_ptr, dst_rank_idx) =
    token_topk_idx;
```

注意，这里的 `dst_slot_idx` 是通过计数器分配的。这样，多个 SM/warp 产生的 routing entries 可以写入同一个 Expert 的连续 metadata 区域。

每个源 Rank 还有独立分区，所以不需要让来自不同 GPU 的条目争抢同一条 metadata 写入队列。（[Kernel 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)；[布局源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/layout/mega_moe.cuh)）

### Pull token：确实是在完成 GEMM 所需要的 permute

目标端收到 metadata 和计数、经过 NVLink barrier 后，便可以开始组织自己的 L1 input pool。

举个例子，Rank 2 有两个 expert：

- Expert 8 需要来自 Rank 0、Rank 1 的 token。
- Expert 9 也需要来自多个源 Rank 的 token。

这些 token 在源端的原始位置可能是完全随机的。

<MegaMoESharedDiagram view="token-permute" />

源码中的关键地址计算是：

```cpp
// 根据 metadata 找源端 token
src_token_idx = src_token_topk_idx / kNumTopk;

// 根据目标 Expert 和当前 token 在 Expert 内的编号
// 计算它应该放在 L1 pool 的哪里
pool_token_idx =
    expert_pool_block_offset * BLOCK_M
    + token_idx_in_expert;

// 通过 NVLink 从源端读取
src_ptr = sym_buffer.map(
    input_token_buffer[src_token_idx],
    src_rank
);

// 直接存入 GEMM 的 L1 pool
dst_ptr = l1_token_buffer[pool_token_idx];
```

实际 payload 由 `tma_load_1d` 从源端搬到 SMEM staging buffer，再由 `tma_store_1d` 写入目标端的 L1 token pool。

它还会从源端读取对应的 FP8 scale factors、router weight，构造 Combine 所需的 source metadata，最后发布 L1 tile 的 arrival count。（[Kernel 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)）

因此，你所说的「直接 Grouped GEMM，不需要 permute」需要稍微精确一点：

**Permute 仍然发生了，只是融合在 Dispatch 的远端 gather 和本地 pool packing 中，省去了一次独立的 permute pass。**

而且 pool 也并非绝对紧凑。每个 Expert 的实际 token 数量需要按 `BLOCK_M` 向上对齐，最后一个 M tile 可能存在 padding。

### 为什么 Payload 不也 Push 过去？

实际上，源端也可以直接把 token payload 写入目标端预先分配好的 Expert Pool。

这种方案在功能上完全可行，也能够产生同样适合 Grouped GEMM 的布局。

但它需要解决一个额外问题：**源端必须协调好每个 token 的最终目标地址。**

MegaMoE 当前的设计中：

- 源端只负责根据 router 决定哪些 token 发往哪个 expert，并把小型索引发送出去。
- 目标端掌握各专家的接收计数，自己决定 token 在 L1 pool 中的顺序。
- 目标端还可以按源 Rank round-robin 处理 token，分散 NVLink pull 请求。
- FP8 token、scale factors 和 router weight 能够由同一目标端 dispatch 逻辑一起组织，并在写入完成后发布 tile readiness。

这样目标端对 GEMM 输入缓冲区和数据就绪状态的管理更集中。

不过，源码并没有提供一个严格的 `push payload` 与 `pull payload` 性能对照实验，所以不能断言 pull 的 NVLink 带宽必然更高，或者该选择一定优于设计良好的 push-permute 实现。

### 一个进一步的性能取舍

MegaMoE 这个设计也有代价。

因为源端先 push metadata，目标端才能知道要 pull 哪些 token，所以 Dispatch 开始时存在一次元数据同步依赖。源码明确使用全 Rank 的 NVLink barrier，保证 metadata 和计数已经可见，然后才开始 payload pull。

此外，pull 的远端地址可能是不规则的，具体的请求大小和并发度会影响 NVLink 吞吐。

因此，MegaMoE 实际是在权衡 metadata 同步成本、跨 GPU 访问模式、目标端 buffer 管理，以及与 GEMM 的流水线融合程度。

在这个设计中，**最有价值的收益是目标端可以边接收、边整理成 Expert-major tiles，并让已经就绪的 tiles 立即进入 L1 GEMM。** 这使 Dispatch 和后面的计算能够真正重叠。

## Dispatch 和 L1 怎样真正地重叠？

假设一个 expert 有 50 个 token，`BLOCK_M=16`。

当 Dispatch 已经拉回前 16 个 token，而其余 34 个还在 NVLink 上传输时，L1 就应该开始处理第一个 tile。

MegaMoE 为每个 pool block 准备一个计数器：

```cpp
// Dispatch warp：某个 token 已经写入 L1 pool
publish_token_to_l1_pool(token);

red_add_release(l1_arrival_count[pool_block], 1);
```

L1 的 TMA Load A warp 在读取相应 token tile 之前检查计数器：

```cpp
// L1 TMA producer
while (load_acquire(l1_arrival_count[pool_block])
       != valid_m) {
    // 等待这一块所有 token 到齐
}

tma_load_A(...);
```

这里的 `valid_m` 对普通完整 tile 是 16，对最后一个不满的 tile 则可能是 2。

只有对应 tile 的数据全部写入 pool，L1 才会发起 TMA load。

<MegaMoESharedDiagram view="dispatch-overlap" />

于是，在后续 token 仍在通过 NVLink 进入 L1 pool 时，Tensor Core 已经开始处理此前到达的 tile。

L1→L2 则使用类似的机制，只是从计数器变成了完成掩码 `l2_arrival_mask`：L1 epilogue 为每个完成的输出 N tile 设置一个 bit，L2 在读取完整的 intermediate 行时等待相应 bits 全部到齐。

原始 Expert-Wave 的实现中，L2 主要在本 wave 的 L1 工作分配结束后启动。源码注释还指出，它曾尝试按 K block 更细粒度地等待 L1 结果，但在已有 wave 大小能够提供足够重叠的场景下，这种做法收益为负，因此采用整行就绪检查。（[Kernel 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)；[MegaMoE 参考讲解](https://llm-infra-atlas.github.io/moe/07_megamoe/)）

## GEMM Mainloop：如何用一套计算管线运行 L1 和 L2？

接下来进入真正的计算 kernel。

MegaMoE 的 L1 和 L2 都是 GEMM：

$$
\begin{aligned}
L1 &: (M_e,H)\times(H,2I)\\
L2 &: (M_e,I)\times(I,H)
\end{aligned}
$$

对一个 GEMM tile，调度器需要提供：

```cpp
struct TileTask {
    Phase phase;       // Linear1 / Linear2
    int expert_idx;
    int m_block_idx;
    int n_block_idx;
};
```

随后同一套计算代码根据 `phase` 切换 activation pool、weight tensor、K dimension 和 epilogue。

### Block 划分

原始实现中，`BLOCK_N=128`。`BLOCK_M` 则由输入规模决定，面向单个专家 token 数较少的 MoE 场景，可以选择较小的 M tile。

对每个 GEMM tile：

$$
C_{B_M\times B_N}
=
\sum_{k=0}^{K/B_K-1}A_kB_k
$$

注意，`BLOCK_K` 只决定 MMA mainloop 如何迭代 reduction dimension。一个 `(m_block, n_block)` task 必须完成整个 K reduction，才产生最终输出 tile。

### Warp Specialization

这部分最值得结合 Blackwell UMMA 的执行方式来看。

每个 CTA 中，计算相关的 warp 有以下职责：

| Warp | 工作 |
|---|---|
| TMA Load A | 从 L1/L2 pool 加载 activation 和 scale |
| TMA Load B | 根据 expert_idx 加载对应的 weight 和 scale |
| MMA Issue | 等待数据就绪，发射 block-scaled UMMA |
| Epilogue | 从 TMEM 取出 accumulator，执行后处理 |

原始实现使用两个 CTA 组成 cluster，配合 Blackwell 的 2-CTA UMMA 指令。只有 leader CTA 的 MMA warp 发射指令，两个 CTA 参与数据加载、计算及 epilogue。相邻 CTA 对应的 N tile 会被配对，以满足协作要求。

其数据流水线大致如下：

<MegaMoESharedDiagram view="gemm-pipeline" />

具体地说，TMA Load A 和 B warp 各自遍历相同的任务序列，使用不同的 TMA descriptor 加载对应的数据。

MMA warp 等待 `full_barrier`，确认 A、B 和 scale factors 已经进入 SMEM，然后通过 UTCCP 将 scale factors 搬入 TMEM，并发出 `tcgen05` 系列的 block-scaled UMMA。

相关的真实源码接口包括：

```cpp
cute::UMMA::make_instr_desc_block_scaled<...>();

shared_storage.full_barriers[stage_idx].wait(phase);

ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(...);
```

MMA 发射后，通过 mbarrier 通知 TMA producer 该 SMEM stage 可以复用；完成整个 tile 的累加后，再通知 epilogue 读取 TMEM。

### 这里为什么需要两套 pipeline？

源码实际涉及两个相互衔接的 producer-consumer pipeline。

**SMEM pipeline** 负责 A/B/SF 数据的加载和消费。其 full/empty barriers 控制 TMA producer 什么时候可以复用 SMEM stage。

**TMEM accumulator pipeline** 负责 MMA producer 和 epilogue 的交接。只有 epilogue 消费完前一个 accumulator，MMA 才能安全复用对应的 TMEM 区域。

源码中的 `tmem_full_barriers` 和 `tmem_empty_barriers` 就在协调这种关系，原始实现采用两个 epilogue accumulator stages。（[Kernel 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)）

后面 X-Stage 论文发现的瓶颈，恰恰发生在 TMEM accumulator pipeline 的下游：epilogue 被 remote-store backpressure 阻塞，最终占住 staging slot，使 MMA producer 等待 `tmem_empty`。

## L1 Epilogue：SwiGLU、Router Weight 和 FP8 量化

L1 GEMM 结束之后，得到的是 FP32 accumulator。

普通实现可能把 GEMM 输出写回 Global Memory，再启动单独的 SwiGLU kernel。但 MegaMoE 的 epilogue warp 直接读取 TMEM accumulator。

为了高效执行 SwiGLU，源码还将 gate/up 权重预先按固定粒度交错排列，使 TMEM load 获得的数据更容易配成 `(gate, up)`。

Epilogue 执行：

$$
h=\operatorname{SiLU}(g)\odot u\cdot p_e
$$

这里有一个有意思的优化：**Router Weight 提前乘到了 L1 的激活结果上。**

为什么可以这样做？

因为 L2 是线性变换，忽略量化舍入误差：

$$
(p_eh)W_2=p_e(hW_2)
$$

因此，原本在 Combine 阶段乘的 router weight，可以提前乘到 SwiGLU 输出上。这样 Combine 阶段只需要做 top-k 求和。FP8 再量化会引入舍入误差，因此这是一种数值上近似等价的实现。

执行完 SwiGLU 后，还需要进行 FP8 量化。

Epilogue warp 根据数值求 amax，生成对应的 UE8M0 scale factor，将输出量化为 FP8 E4M3，再通过 SMEM 和 TMA store 写入 L2 activation pool。

最后，它更新 `l2_arrival_mask`，通知 L2 对应的数据已经就绪。

这部分融合了原本单独的 activation、router weight multiplication 和量化步骤。（[Kernel 源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh)；[MegaMoE 参考讲解](https://llm-infra-atlas.github.io/moe/07_megamoe/)）

## L2 Epilogue：为什么它是 X-Stage 优化的重点？

L2 GEMM 完成后，epilogue 从 TMEM 读取 FP32 accumulator，并转换为 BF16。

对于每个有效 token，Dispatch 阶段已经保存了：

```cpp
struct TokenSrcMetadata {
    int src_rank;
    int src_token_idx;
    int src_topk_idx;
};
```

因此，L2 epilogue 不需要重新执行 unpermute。它能够直接找到源 token 的地址，并将结果写回：

```cpp
auto meta = token_src_metadata[pool_token_idx];

remote_ptr = map_remote(
    combine_buffer,
    meta.src_rank
);

remote_ptr[meta.src_topk_idx]
          [meta.src_token_idx]
          [hidden_offset] = output;
```

上面是逻辑伪代码，真实实现会按 tile 和 vector 组织内存写入。

Combine buffer 的组织方式很重要。假设一个 token 激活三个专家，那么其原始 rank 上有三个不同的输出槽位：

<MegaMoESharedDiagram view="combine-slots" />

每个 expert 的输出写到不同槽位，所以即使来自多个 rank，也不会发生同一地址的写入冲突。

等所有 rank 的 L2 输出写回完成后，kernel 执行一次跨 rank 的 NVLink barrier，随后在原 token 所在 GPU 上读取各个 top-k 槽位，以 FP32 累加，最后输出 BF16。

这里的跨 rank barrier 仍然很重要。**Remote store 发射返回只意味着写入请求进入发送路径，不能直接认为接收端数据已经完成可见。** Combine reduce 必须遵循正确的完成与可见性同步。（[MegaMoE 参考讲解](https://llm-infra-atlas.github.io/moe/07_megamoe/)；[X-Stage 论文](https://arxiv.org/html/2607.23264v2)）

## Persistent Scheduler：所有 SM 如何分配 L1/L2 工作？

前面解释了一个 CTA 怎样计算 tile，现在来看所有 CTA 怎样分配这些 tile。

原始代码的核心是 `MegaMoEScheduler`。

每个计算角色都通过 `scheduler.for_each_block(...)` 遍历同一套逻辑任务序列。调度器给出任务，TMA Load A/B、MMA 和 epilogue 分别执行自己负责的部分。

这种设计减少了跨角色分发任务所需的额外全局队列，但不同角色的推进速度仍然可以由流水线 barrier 解耦。

### Expert-Wave 的调度规则

假设本 GPU 有四个 expert，一个 wave 包含两个 expert。

原始执行顺序是：

<MegaMoESharedDiagram view="expert-wave" />

为什么不逐个 expert 完成 L1→L2？

因为在 MoE 的小 M GEMM 场景中，单个 expert 产生的 tile 数量可能太少，无法让全部 SM 都保持忙碌。

将多个 expert 组成一个 wave，有利于获得足够的 tile parallelism，同时限制同时活跃的 expert 权重数量，以维持较好的 L2 cache locality。

Wave 大小由启发式方法决定，核心考虑包括：

- 每个 expert 预计有多少 token。
- 每个 expert 的 L1/L2 总共可以产生多少 tile。
- GPU 有多少 SM。
- Wave 过大时权重工作集与缓存局部性如何变化。

### 任务怎样分配到每个 CTA？

源码采用一个很直接的 strided scheduler。

假设 GPU 有 $S$ 个计算 CTA。一个 CTA 从自己的 `blockIdx.x` 开始，遍历逻辑任务序列中的：

$$
k=\operatorname{blockIdx.x}+tS
$$

其中 $t=0,1,2,\ldots$。

源码中的核心就是：

```cpp
block_idx = blockIdx.x;

// 取出当前逻辑任务
get_next_block();

// 下次领取隔 S 个任务的工作
block_idx += kNumSMs;
```

例如一个 wave 有 400 个逻辑 tile、200 个 CTA，那么 CTA 0 负责 tile 0、200，CTA 1 负责 tile 1、201，以此类推。

这里原始实现的相邻 CTA 会配对参与 2-CTA MMA，所以逻辑 tile 序列还需要保证相邻任务的 M/N 坐标满足协作约束。

这个方案不需要每执行一个 tile 都向全局 work queue 发起 atomic fetch。

相应的代价是，在 routing skewed、不同专家 tile 耗时差别较大时，静态 strided 分配仍可能出现负载不均衡。（[调度器源码](https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh)）

## X-Stage 论文到底修改了 Scheduler 的哪一部分？

现在就可以很准确地定位前面论文的优化了。

原始 `get_next_block()` 的核心是一个两阶段状态机：

```cpp
if (next_phase == Linear1) {
    if (fetch_next_l1_block()) {
        return Linear1;
    }

    next_phase = Linear2;
}

if (next_phase == Linear2) {
    if (fetch_next_l2_block()) {
        return Linear2;
    }

    next_phase = Linear1;
}
```

它在一个 wave 内先发出 L1 任务，再发出 L2 任务。

X-Stage 论文则改成两个独立的逻辑任务流，并根据 L1 对 L2 的 dependency lead 决定下一次调度哪个阶段：

```cpp
while (has_L1 || has_L2) {
    if (has_L2 && L1_is_far_enough_ahead()) {
        emit(next_L2_tile);
    } else {
        emit(next_ready_L1_tile);
    }
}
```

具体算法见论文 [Appendix A](https://arxiv.org/html/2607.23264v2#A1)。

它还必须检查 L1 输入是否已经被 Dispatch 准备好，以及输出 ring slot 是否可以写入。

### 为什么这会影响 MMA？

考虑一个 L2 tile 完成后，epilogue 需要发射很多 remote stores。

原始 Expert-Wave 连续安排下一个 L2：

<MegaMoESharedDiagram view="x-stage" />

原本较短的 L2 mainloop 可能无法完全覆盖上一个 epilogue 的 remote-store span。若 staging slot 迟迟无法释放，MMA producer 就会在 `tmem_empty` 停顿。

新调度可以在此处安排一个已经 ready 的 L1 tile。由于 L1 没有 L2 那样的 Combine remote stores，并且它的计算仍然是必要工作，便可以利用这段计算时间让远端存储请求继续排空。

最终，它同时改变了两个因素：L2 remote stores 的发射间隔，以及下一次复用本地 staging slot 之前能够执行的有用计算量。（[X-Stage 论文](https://arxiv.org/html/2607.23264v2)）

## 把整个 Kernel 还原成一个 CUDA 程序

如果把细节收起来，MegaMoE 大致可以写成下面这个结构。**这是根据真实实现整理的教学版伪代码，保留数据流与同步语义，省略了 CTA cluster、mbarrier phase、scale factor layout 等细节。**

```cpp
__global__ void mega_moe_kernel(Context ctx) {
    int warp = threadIdx.x / 32;

    // 不同 warp 各自运行长期存在的角色循环
    if (warp < NUM_DISPATCH_WARPS) {
        count_routed_tokens(ctx);
        push_routing_metadata(ctx);

        nvlink_barrier_before_pull(ctx);

        for (auto token : assigned_tokens(ctx)) {
            pull_remote_token_to_l1_pool(ctx, token);
            save_source_metadata(ctx, token);
            publish_l1_arrival(ctx, token);
        }

        cleanup_workspace(ctx);
    }
    else if (warp == LOAD_A_WARP) {
        for (auto task : scheduler(ctx)) {
            wait_activation_ready(ctx, task);

            for (int k = 0; k < task.num_k_tiles; ++k) {
                wait_smem_empty(k);
                tma_load_activation_and_sf(ctx, task, k);
                signal_smem_full(k);
            }
        }
    }
    else if (warp == LOAD_B_WARP) {
        for (auto task : scheduler(ctx)) {
            for (int k = 0; k < task.num_k_tiles; ++k) {
                wait_smem_empty(k);
                tma_load_weight_and_sf(ctx, task, k);
                signal_smem_full(k);
            }
        }
    }
    else if (warp == MMA_WARP) {
        for (auto task : scheduler(ctx)) {
            wait_tmem_empty();

            for (int k = 0; k < task.num_k_tiles; ++k) {
                wait_smem_full(k);
                issue_umma(ctx, task, k);
                release_smem_stage(k);
            }

            signal_tmem_full();
        }
    }
    else if (warp >= EPILOGUE_START) {
        for (auto task : scheduler(ctx)) {
            wait_tmem_full();

            if (task.phase == Linear1) {
                swiglu_and_quantize(ctx, task);
                store_l2_activations(ctx, task);
                publish_l2_arrival(ctx, task);
            } else {
                convert_to_bf16(ctx, task);
                remote_store_combine_partials(ctx, task);
            }

            release_tmem_stage();
        }

        nvlink_barrier_before_reduce(ctx);
        reduce_topk_partials(ctx);
    }
}
```

真正的代码当然更加复杂，因为这里所有的 warp 都在同时推进：

- TMA producer 可能已经在准备第 $n+1$ 个 tile。
- MMA producer 正在计算第 $n$ 个 tile。
- Epilogue 正在处理第 $n-1$ 个 tile。
- Dispatch warp 还在拉取后续 experts 的 tokens。
- Remote-store 请求可能仍在 GPU 的通信路径中排空。

整个 kernel 的吞吐由这些生产者、消费者和异步资源共同决定。

## 我认为最值得深入源码的三个地方

从 CUDA kernel 优化的角度，MegaMoE 最有研究价值的是以下三个接口。

**`MegaMoEScheduler::get_next_block()`** 决定 task 的先后顺序。它影响权重 cache locality、计算连续性、L1/L2 依赖等待和 remote-store burst pattern。X-Stage 的主要改动就在这一层。

**`l1_arrival_count` / `l2_arrival_mask`** 决定 task 何时真正可以被执行。它们是让跨 rank 通信、跨 CTA 生产与消费得以流水化的关键，也给调度器施加了正确性约束。

**`tmem_empty_barriers` 和 L2 epilogue** 决定了 remote-store latency 是否会沿着 staging buffer 反压到 MMA。要理解 X-Stage 的性能收益，仅看 NVLink 吞吐是不够的，必须把这个环节也分析清楚。

把这三个部分连起来，你就能理解 MegaMoE 的一个核心设计问题：**Scheduler 应该以什么顺序提交已经就绪的 L1/L2 tiles，才能同时保持 Tensor Core 忙碌、减少权重缓存失效，并避免 remote-store 背压占用计算关键路径？**

这也是为什么我认为，若要在 MegaMoE 上进一步做硬件感知调度研究，最合适的切入点是 scheduler 与 epilogue 的交界处：它既包含实际执行的自由度，也能观察到通信背压对计算流水线的真实影响。
