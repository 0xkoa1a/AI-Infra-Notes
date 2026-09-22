---
title: "Blackwell 架构 MatMul 算子实现精读"
order: 1
---

# Blackwell 架构 MatMul 算子实现精读

这份 Level 9 实现把 GEMM 拆成三条长期运行的工作流：TMA warp 加载输入，MMA warp 发射矩阵计算，epilogue warps 写回结果。**读懂它的关键，是弄清每份数据由谁持有、何时完成，以及何时允许下一轮覆盖同一块存储。**

本文精读 [MatmulTutorial 的 Level 9 源码](https://github.com/KnowingNothing/MatmulTutorial/blob/69c6886b93c3d569f1ba49185615c7fce9f19df0/examples/matmul/this-sm100/level9/matmul.cu)，固定 commit 为 `69c6886b93c3d569f1ba49185615c7fce9f19df0`。下面的 C++ 片段保留实现中的控制关系，调整了断行并补充中文注释；未展开的指针设置和 helper 可对照原文件与文末附录。性能数字来自同版本作者报告，未在本文中复测。

## 先建立数据与执行图景

### 计算任务与两个 CTA 的分工

算子把两份 BF16 输入相乘，以 FP32 累加，最后写出 BF16 结果：

$$
D=AB^{\mathsf T}.
$$

- $M$：输出行数；$N$：输出列数；$K$：相乘求和的维度。
- $A$：行主序的 $[M,K]$ 矩阵，元素为 BF16。
- $B$：行主序的 $[N,K]$ 矩阵，计算时逻辑转置。
- $D$：行主序的 $[M,N]$ 输出，元素为 BF16。

**CTA** 是 CUDA 线程块。此实现把两个 CTA 组成一个 cluster，协作执行 `cta_group::2` UMMA；UMMA 是这里使用的 Blackwell 异步 Tensor Core 矩阵乘加指令。两个 CTA 分别负责不同的输出行，共同使用同一段输出列对应的 B 数据。

| 层级 | 尺寸 | 在代码中的作用 |
|---|---|---|
| 每 CTA 的输出 tile | $256\times256$ | `BLOCK_M × BLOCK_N` |
| 两 CTA 合计输出 | 两块 $256\times256$ | 相同 N 区间，不同 M 区间 |
| 每轮 K block | 64 个 K 元素 | 每个 CTA 加载 A $[256,64]$、B $[128,64]$ |
| 每次 UMMA 的 K-step | 16 个 K 元素 | 一个 K block 分 4 次推进 K |
| 每个 CTA 的 M-wave | 128 行 | 两个 wave 覆盖本 CTA 的 256 行 |

<BlackwellGemmDiagram view="cluster" />

每次 UMMA 由 leader CTA 发射，两边各产生 128 行本地结果，所以指令的逻辑 `UMMA_M=256`。每个 CTA 自己的输出仍有 256 行，需要再做两个 M-wave。这里的“2”分别指 cluster 中的 CTA 数与本地输出的 wave 数。

> 下文先按两个 CTA 能配对到相同 N 区间的情况阅读；scheduler 小节会推导这一前提。B 的原始布局是 `[N,K]`，图中 B 的前后两半沿 N 行划分，对应 D 的前后两组列。

### 存储路径与 TMEM 映射

TMA 将全局输入搬到各 CTA 的共享内存（SMEM），UMMA 读取它们并把累加结果放进 Tensor Memory（TMEM）。结果随后经过线程寄存器和 CD SMEM，由 TMA 写回 D。`CD` 是代码对输出暂存区的命名。

<BlackwellGemmDiagram view="storage" />

此实现每个 CTA 分配完整的 `128 lanes × 512 columns` TMEM，每个单元存一个 32-bit 累加值。逻辑输出有 256 行，必须明确它与物理 lane 的对应关系：

<BlackwellGemmDiagram view="tmem" />

| M-wave | 本 CTA 的逻辑输出行 | TMEM lane | TMEM column |
|---|---|---|---|
| `w=0` | `[0,128)` | `[0,128)` | `[0,256)` |
| `w=1` | `[128,256)` | `[0,128)` | `[256,512)` |

两组结果同时保存在 TMEM 中。Epilogue 中 `local_tid` 标识本 CTA 的 128 个读取线程，每个线程处理一条 lane；`w * 128` 将这条 lane 还原到对应输出行。MMA 代码里的 `w * BLOCK_N` 则选择该 wave 使用的 TMEM column 区间。

### 角色、缓冲与完成通知

每个 CTA 有 8 个 warp，共 256 个线程。Warp 是 GPU 的 32 线程执行组；这里通过不同分支分配职责，即 warp specialization。

| Warp | 角色 | 参与 CTA | 工作 |
|---|---|---|---|
| 0 | TMA producer | 两个 CTA | 各选 1 个线程加载本地 A/B |
| 1 | MMA consumer/producer | leader CTA | 全 warp 等待、shuffle；选 1 个线程发 UMMA 和 commit |
| 2 | TMEM 管理 | 两个 CTA | 整个 warp 协作分配、释放 TMEM |
| 3 | 无主流水线工作 | 两个 CTA | 参与末尾集合 |
| 4–7 | Epilogue consumer | 两个 CTA | 128 线程读取本地 TMEM，写回各自的 D tile |

TMA 与 MMA 按 **K block** 交接输入，MMA 与 epilogue 按**完整输出 tile** 交接累加结果。写回时，一个 tile 再分成 8 个 $[128,64]$ chunk。三者使用的缓冲粒度不同：

| 缓冲 | 数量与容量（每 CTA） | 复用者 | 保护方式 |
|---|---|---|---|
| A/B SMEM stage | 4 × 48 KiB | TMA、UMMA | `full_bar[s]` / `empty_bar[s]` |
| TMEM accumulator stage | 1 × 256 KiB | UMMA、epilogue | `tmem_full_bar[0]` / `tmem_empty_bar[0]` |
| CD SMEM store stage | 2 × 16 KiB | epilogue、TMA Store | bulk-group `wait_group` |

`full` 表示本轮数据已可消费，`empty` 表示旧数据已用完、允许覆盖。Barrier 的 **phase** 区分同一同步对象的连续轮次；下文用 stage 表和代码跟踪它的翻转。

| Barrier | 等待者 | 完成条件 | 允许继续的操作 |
|---|---|---|---|
| leader 的 `full_bar[s]` | leader MMA warp | 两 CTA 的 arrival 与合计 TMA 字节数都满足 | UMMA 读取输入 |
| 各 CTA 的 `empty_bar[s]` | 各自 TMA warp | UMMA commit 完成后 multicast arrival | TMA 覆盖该输入 stage |
| 各 CTA 的 `tmem_full_bar[0]` | 各自 epilogue | 最后 K block 的 UMMA commit 完成 | 读取完整累加结果 |
| leader 的 `tmem_empty_bar[0]` | leader MMA warp | 两 CTA 各 128 个读取线程完成 arrival | 下一 tile 覆盖 TMEM |
| named barrier 1 | 本 CTA 的 128 个 epilogue 线程 | 这 128 个线程集合 | 复用 CD stage 或发出 Store |

## 沿 Level 9 代码阅读

### 配置与内存预算

配置先确定空间分工，再决定能够同时保留几轮数据。每 CTA 的 SMEM 共约 224 KiB，其中大部分留给四轮 A/B 输入，剩下两块 16 KiB 给输出暂存。

```cpp
static constexpr uint32_t BLOCK_M = 256;
static constexpr uint32_t BLOCK_N = 256;
static constexpr uint32_t BLOCK_K = 64;
static constexpr uint32_t CLUSTER_SIZE = 2;
static constexpr uint32_t NUM_STAGES = 4;
static constexpr uint32_t WAVE_BLOCK_M = 128;
static constexpr uint32_t NUM_M_WAVES = BLOCK_M / WAVE_BLOCK_M;
static constexpr uint32_t LOAD_N_PER_CTA = BLOCK_N / CLUSTER_SIZE;
static constexpr uint32_t UMMA_M = WAVE_BLOCK_M * CLUSTER_SIZE;
static constexpr uint32_t UMMA_N = BLOCK_N;
static constexpr uint32_t UMMA_K = 16;
static constexpr uint32_t NUM_EPILOGUE_STAGES = 1;
static constexpr uint32_t TMEM_COLS =
    NUM_EPILOGUE_STAGES * NUM_M_WAVES * BLOCK_N;
```

| SMEM 区域（按地址从低到高） | 计算 | 容量 |
|---|---|---:|
| CD store buffers | $2\times128\times64\times2$ bytes | 32 KiB |
| A stages | $4\times256\times64\times2$ bytes | 128 KiB |
| B stages | $4\times128\times64\times2$ bytes | 64 KiB |
| Barrier 与 TMEM 地址槽 | 11 个 8-byte 槽与 4-byte 地址 | 92 bytes |
| 合计 | `SMEM_SIZE` | 229,468 bytes |

A stages 和 B stages 各自连续排列。相邻 stage 的地址步长固定，后面的 MMA warp 因而可以用各 lane 保存 descriptor，再通过 shuffle 取出当前 stage 所需的值。TMEM 单独分配，不计入这张 SMEM 预算表。

### Persistent scheduler 与配对条件

Host 只启动数量受限的 CTA，每个 CTA 反复领取不同 tile，形成 persistent loop。Scheduler 是确定性的编号计算，不需要一个额外线程维护全局任务队列。

```cpp
uint32_t tile_idx =
    static_cast<uint32_t>(++current_iter) * num_ctas + cta_id;
if (tile_idx >= num_tiles) return false;

uint32_t tpg = num_n_blocks * SWIZZLE_GROUP_SIZE;
uint32_t gi  = tile_idx / tpg;
uint32_t fm  = gi * SWIZZLE_GROUP_SIZE;
uint32_t ig  = tile_idx % tpg;
uint32_t mg  = min(SWIZZLE_GROUP_SIZE, num_m_blocks - fm);
m_block = fm + ig % mg;
n_block = ig / mg;
return true;
```

这里 `SWIZZLE_GROUP_SIZE=16`。先在一组 M block 内变化行号，再推进 N，使相邻 CTA 更容易复用同一段 B。以 `num_ctas=192` 为编号示例，CTA 0 枚举 `0,192,384,…`，CTA 1 枚举 `1,193,385,…`；随后才用上面的公式映射到二维坐标。

**两个 CTA 必须得到相同 `n_block`，并且执行相同数量的迭代。** 只把 launch 的 CTA 数向下取偶数，还不能保证任意 shape 都满足这件事。

令 `num_m_blocks=ceil(M/256)`。它为正偶数、`num_ctas` 为正偶数时，每组 M block 的长度都是偶数（完整组为 16，末组也为偶数），相邻的偶/奇编号不会跨过 N 边界，总 tile 数也为偶数。这给出了当前映射保持配对的充分条件。若 M block 数为奇数，则需要逐项分析，不能直接套用总览中的分工。

例如取 `M=768、N=512`，第一轮有以下映射：

| CTA 编号 | 所属 cluster | `m_block` | `n_block` |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 |
| 2 | 1 | 2 | 0 |
| 3 | 1 | 0 | 1 |

第二个 cluster 的两半 B 已来自不同 N 区间。这是对索引公式的推导；TMA 的 OOB 处理只能处理访存边缘，无法修复这种配对。完整可运行条件还包括 tensor map 的地址/stride 对齐、硬件资源和有效 launch 配置。原 host 没有为这些 shape 情况提供通用回退，后文的性能表也只代表作者测试的尺寸。

**各角色为什么各有一份 scheduler？** TMA、MMA、epilogue 在同一个 CTA 中枚举相同序列，却可以推进到不同位置：TMA 正在预取后续 K block 时，MMA 可能仍在计算当前 tile，epilogue 仍在处理前一个结果。它们分别持有线程私有状态，由 barrier 对齐数据依赖。即使把构造语句移到分支外，各线程得到的也仍是独立实例。

<details>
<summary>完整 TileScheduler 定义</summary>

```cpp
struct TileScheduler {
    uint32_t num_m_blocks, num_n_blocks, num_tiles, num_ctas, cta_id;
    int current_iter;
    __device__ TileScheduler(uint32_t M, uint32_t N, uint32_t nc)
        : num_m_blocks((M + BLOCK_M - 1) / BLOCK_M),
          num_n_blocks((N + BLOCK_N - 1) / BLOCK_N),
          num_tiles(num_m_blocks * num_n_blocks),
          num_ctas(nc), cta_id(blockIdx.x), current_iter(-1) {}
    __device__ bool get_next_block(uint32_t& m_block, uint32_t& n_block) {
        uint32_t tile_idx = static_cast<uint32_t>(++current_iter) * num_ctas + cta_id;
        if (tile_idx >= num_tiles) return false;
        // Group M-blocks in chunks of SWIZZLE_GROUP_SIZE, sweep N within each group
        uint32_t tpg = num_n_blocks * SWIZZLE_GROUP_SIZE;
        uint32_t gi  = tile_idx / tpg;
        uint32_t fm  = gi * SWIZZLE_GROUP_SIZE;
        uint32_t ig  = tile_idx % tpg;
        uint32_t mg  = min(SWIZZLE_GROUP_SIZE, num_m_blocks - fm);
        m_block = fm + ig % mg;
        n_block = ig / mg;
        return true;
    }
};
```

</details>

### 初始化：发布 barrier，分配 TMEM

每个 CTA 都有自己的 SMEM 和 barrier 副本。初始化代码先计算地址，随后设置计数。`full_bar` 和 `tmem_empty_bar` 的实际消费发生在 leader，部分通知需要从 peer 路由过去；另外两类 barrier 则在两边各自等待。

```cpp
if (warp_idx == 1 && elect_one()) {
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        barrier_init(full_bar[s], CLUSTER_SIZE);
        barrier_init(empty_bar[s], 1);
    }
    for (uint32_t e = 0; e < NUM_EPILOGUE_STAGES; ++e) {
        barrier_init(tmem_full_bar[e], 1);
        barrier_init(tmem_empty_bar[e],
                     CLUSTER_SIZE * NUM_EPILOGUE_THREADS);
    }
    fence_barrier_init();
}
if (warp_idx == 2)
    tmem_alloc_2sm(tmem_addr, TMEM_COLS);
cluster_sync();
```

`elect_one()` 在当前 warp 中选一个活跃线程发出操作；TMEM 分配要求整个 warp 参与，所以 warp 2 的分支没有套这个条件。Warp 0 还会预取 A、B、D 的 TMA descriptor，这时取的是描述信息，张量加载发生在主循环。

`fence_barrier_init()` 发布本线程完成的 mbarrier 初始化，`cluster_sync()` 让整个 cluster 集合后再进入工作阶段。这两个调用共同保证后面的跨 CTA 通知有已初始化的目标。Barrier 和 TMEM 的初始化都在 persistent loop 之外，后续通过 phase 区分轮次。

<details>
<summary>SMEM 指针与 barrier 副本的地址设置</summary>

```cpp
    // --- Barrier setup ---
    uint64_t* full_bar[NUM_STAGES];
    uint64_t* empty_bar[NUM_STAGES];
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        full_bar[s]  = reinterpret_cast<uint64_t*>(smem_buf + OFF_BAR + s * 8);
        empty_bar[s] = reinterpret_cast<uint64_t*>(smem_buf + OFF_BAR + NUM_STAGES * 8 + s * 8);
    }
    uint64_t* tmem_full_bar[NUM_EPILOGUE_STAGES];
    uint64_t* tmem_empty_bar[NUM_EPILOGUE_STAGES];
    for (uint32_t e = 0; e < NUM_EPILOGUE_STAGES; ++e) {
        tmem_full_bar[e]  = reinterpret_cast<uint64_t*>(smem_buf + OFF_BAR + NUM_STAGES * 16 + e * 8);
        tmem_empty_bar[e] = reinterpret_cast<uint64_t*>(smem_buf + OFF_BAR + NUM_STAGES * 16 + NUM_EPILOGUE_STAGES * 8 + e * 8);
    }
    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(smem_buf + OFF_TMEM);

    // SMEM pointers (separated layout)
    uint8_t* smem_cd_base = smem_buf + OFF_CD;
    auto get_smem_a = [&](uint32_t s) -> void* { return smem_buf + OFF_A + s * SMEM_A_SIZE; };
    auto get_smem_b = [&](uint32_t s) -> void* { return smem_buf + OFF_B + s * SMEM_B_SIZE; };
```

</details>

### TMA producer：等待空位，再加载一个 K block

每个 CTA 的 warp 0 选一个线程发出加载。该线程先等待本地 `empty_bar[stage]`，保证上次 UMMA 已经用完这一格缓冲；随后向 leader 的 `full_bar[stage]` 登记本轮工作。

```cpp
barrier_wait(empty_bar[stage], phase ^ 1);
if (is_leader) {
    barrier_arrive_expect_tx(full_bar[stage],
                            TMA_BYTES * CLUSTER_SIZE);
} else {
    barrier_arrive_cluster(full_bar[stage], 0);
}

int32_t kc = kb * BLOCK_K;
tma_load_2d_cg2(get_smem_a(stage), &tma_a,
                 full_bar[stage], kc, m_coord);
tma_load_2d_cg2(get_smem_b(stage), &tma_b,
                 full_bar[stage], kc, n_coord);

stage = (stage + 1) % NUM_STAGES;
phase ^= (stage == 0);
```

上述片段位于每个 tile 的 `kb` 循环内。A 的起始行是 `m_block * 256`，B 的起始 N 行是 `n_block * 256 + cta_rank * 128`。每个 CTA 加载 48 KiB，两个 CTA 合计 96 KiB；leader 登记这份合计字节数，每个 CTA 各贡献一次 arrival。只有 arrival 与 TMA 的字节完成条件都满足，本轮 `full_bar` 才完成。

TMA 的目标 SMEM 保持本地地址，完成通知则路由到 leader。这个区别决定了“数据各放各处、MMA 统一等就绪”的组织方式，相关 helper 见附录。

**四格缓冲怎样绕回？** `stage` 与 `phase` 在 persistent loop 外初始化为 0；跨输出 tile 时继续使用，不清零。以下按所有 K block 的累计处理序号展示首次绕回：

| 累计 K block 序号 | 使用的 stage | 本轮 phase | TMA 等待的 empty parity | 含义 |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 1 | 首次使用，尚无旧 UMMA 占用 |
| 1 | 1 | 0 | 1 | 首次使用 |
| 2 | 2 | 0 | 1 | 首次使用 |
| 3 | 3 | 0 | 1 | 首次使用；之后 stage 绕回 0 |
| 4 | 0 | 1 | 0 | 等第 0 个 K block 的 UMMA 归还 stage 0 |
| 5 | 1 | 1 | 0 | 等第 1 个 K block 的 UMMA 归还 stage 1 |

初始化后的首次 `phase ^ 1` 等待立即通过；之后它表示等待该格上一轮的完成。MMA 等待 `full_bar` 时使用本轮 `phase`。四个 stage 限制了尚未归还的输入缓冲数量，不能理解为可以无条件预取四个完整输出 tile。

### MMA consumer：等输入、发计算、归还缓冲

Leader 的 warp 1 全部参与等待与 shuffle，实际 UMMA 和 commit 由一个 elected thread 发射。本节将这一分支按数据依赖拆开。

**缓存四个 stage 的 descriptor。** descriptor 的低位包含以 16 bytes 为单位的 SMEM 地址，高位保留布局信息。Lane 0–3 各存一个 stage 的低位，计算时从对应 lane 广播：

```cpp
uint64_t desc0_a = make_smem_desc(get_smem_a(0), UMMA_SBO);
uint64_t desc0_b = make_smem_desc(get_smem_b(0), UMMA_SBO);
uint32_t ahi = static_cast<uint32_t>(desc0_a >> 32);
uint32_t bhi = static_cast<uint32_t>(desc0_b >> 32);
uint32_t a_lo_base = static_cast<uint32_t>(desc0_a);
uint32_t b_lo_base = static_cast<uint32_t>(desc0_b);
uint32_t my_a_lo = a_lo_base
    + (lane_idx < NUM_STAGES ? lane_idx * (SMEM_A_SIZE / 16) : 0u);
uint32_t my_b_lo = b_lo_base
    + (lane_idx < NUM_STAGES ? lane_idx * (SMEM_B_SIZE / 16) : 0u);
```

**开始新 tile 前，等 TMEM 归还。** TMEM 只有一个 accumulator stage，所以 `accum_idx` 总为 0；`accum_phase` 每个输出 tile 翻转一次。下一 tile 必须等两个 CTA 的所有 epilogue 线程完成旧结果读取。

```cpp
uint32_t accum_idx = scheduler.current_iter % NUM_EPILOGUE_STAGES;
uint32_t accum_phase =
    (scheduler.current_iter / NUM_EPILOGUE_STAGES) & 1;
barrier_wait(tmem_empty_bar[accum_idx], accum_phase ^ 1);
tcgen05_fence_after();
```

随后进入 `kb` 循环，等待该 K block 的 A/B 数据：

```cpp
barrier_wait(full_bar[stage], phase);
tcgen05_fence_after();
uint32_t cur_a_lo = __shfl_sync(0xFFFFFFFF, my_a_lo, stage);
uint32_t cur_b_lo = __shfl_sync(0xFFFFFFFF, my_b_lo, stage);
```

`barrier_wait()` 等待目标 phase 完成，`tcgen05_fence_after()` 则约束后续 tcgen05 操作与这个线程同步点的顺序。二者配合后，Tensor Core 才按这里要求的依赖读取或覆盖存储。

| 交接 | 等待条件 | fence 之后的操作 |
|---|---|---|
| TMA → UMMA | `full_bar` | UMMA 读取 A/B SMEM |
| UMMA → epilogue | `tmem_full_bar` | `tcgen05.ld` 读取 TMEM |
| epilogue → 下一 tile | `tmem_empty_bar` | UMMA 覆盖同一 TMEM |

**每轮 K block 发出 8 次 UMMA。** 外层 `k` 划分四个 K-step，内层 `w` 选择两个 M-wave。代码保持 K-step 在外、M-wave 在内的实际顺序：

```cpp
if (elect_one()) {
    #pragma unroll
    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
        uint32_t b_lo = cur_b_lo + k * 2;
        uint64_t bd = ((uint64_t)bhi << 32) | b_lo;
        #pragma unroll
        for (uint32_t w = 0; w < NUM_M_WAVES; ++w) {
            uint32_t a_lo = cur_a_lo
                + w * (WAVE_BLOCK_M * BLOCK_K
                       * sizeof(__nv_bfloat16) / 16)
                + k * 2;
            uint64_t ad = ((uint64_t)ahi << 32) | a_lo;
            uint32_t tmem_offset =
                accum_idx * NUM_M_WAVES * BLOCK_N + w * BLOCK_N;
            uint32_t accum = (kb > 0 || k > 0) ? 1u : 0u;
            umma_f16_cg2(tmem_offset, ad, bd, idesc, accum);
        }
    }
}
```

`k * 2` 来自每步 16 个 BF16，共 32 bytes，相当于两个 16-byte 地址单位。A 还加上一个 wave 的偏移：`128 × 64 × 2 / 16 = 1024`。B 的两半由两 CTA 提供，不随本地 `w` 改变。

两个 wave 写入不同 TMEM column 区间，因此首个 K-step 都要从零开始累加。`accum` 只看 `kb`、`k`，不看 `w`；后续 K-step 才读取已有累加值。

**提交完成点，允许输入 stage 复用。** 发出 UMMA 后不会立即覆盖 A/B。Commit 跟踪之前的异步计算，完成后向两个 CTA 的 `empty_bar` multicast arrival；最后一个 K block 还要通知输出结果就绪。

```cpp
if (elect_one()) {
    umma_commit_2sm(empty_bar[stage]);
    if (kb == num_k_blocks - 1)
        umma_commit_2sm(tmem_full_bar[accum_idx]);
}
stage = (stage + 1) % NUM_STAGES;
phase ^= (stage == 0);
```

| 异步工作 | 完成通知的来源 | 消费者在哪里等待 |
|---|---|---|
| TMA Load | 拷贝完成产生 `complete_tx(bytes)`，结合 arrival 计数 | MMA 的 `full_bar` |
| UMMA | 显式 commit 为此前操作建立完成点，完成后 arrival | TMA 的 `empty_bar` 或 epilogue 的 `tmem_full_bar` |

Commit 本身不阻塞发射线程；等待发生在消费这些通知的位置。

### Epilogue：按 chunk 搬出结果，再交还 TMEM

Epilogue 的 128 个线程先等待完整 tile 的 UMMA 累加完成。它们按 `w=0,1`、每个 wave 内 `s=0,1,2,3` 的顺序处理 8 个 $[128,64]$ chunk。**每个 chunk 搬到 CD SMEM 后就发出对应 Store，边搬边写回。**

```cpp
const uint32_t local_tid = threadIdx.x - 128;
const uint32_t epi_warp = local_tid / 32;
uint32_t tma_store_stage = 0;
```

这三个变量在 epilogue 的 persistent loop 外定义。每个 tile 开始时，用与 MMA 相同的 tile 序号求 `accum_idx`、`accum_phase`，再等待结果：

```cpp
barrier_wait(tmem_full_bar[accum_idx], accum_phase);
tcgen05_fence_after();
```

**先取得可覆盖的 CD stage。** 以下片段位于 `w`、`s` 两层循环内。只有负责 Store 的线程等待自己的 bulk groups，然后通过 named barrier 让其余 epilogue 线程一起继续：

```cpp
if (epi_warp == 0 && elect_one())
    tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
named_barrier_sync(NUM_EPILOGUE_THREADS, EPILOGUE_BAR_ID);

uint32_t smem_stage_base = static_cast<uint32_t>(
    __cvta_generic_to_shared(
        smem_cd_base + tma_store_stage * SMEM_CD_PER_STAGE));
```

`wait_group<1>` 最多保留一个较新的未完成 group。由于 CD0、CD1 交替使用，准备覆盖 CD0 时，它此前对应的较老 Store 已完成，CD1 的较新 Store 可以仍在进行。若较老 Store 尚未完成，当前 chunk 的 TMEM 读取也会被这次等待推迟。

**读取 TMEM，转换并按 swizzle 写入 SMEM。** 一次 `tmem_load_8x` 为每线程取出 8 个 FP32；调用 `tmem_load_fence()` 等寄存器结果可用后，再转成 8 个 BF16、打包到四个 32-bit 寄存器。每个 chunk 沿 64 列重复 8 次：

```cpp
for (uint32_t i = 0; i < STORE_BLOCK_N / ELEMS_PER_BANK_GROUP; ++i) {
    uint32_t tmem_col = accum_idx * NUM_M_WAVES * BLOCK_N
        + w * BLOCK_N + s * STORE_BLOCK_N + i * ELEMS_PER_BANK_GROUP;
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
    tmem_load_8x(tmem_col, r0, r1, r2, r3, r4, r5, r6, r7);
    tmem_load_fence();
    uint32_t p0 = pack_bf16(r0, r1);
    uint32_t p1 = pack_bf16(r2, r3);
    uint32_t p2 = pack_bf16(r4, r5);
    uint32_t p3 = pack_bf16(r6, r7);
    uint32_t swizzled_col = i ^ (local_tid % BANK_GROUPS_PER_SWIZZLE);
    uint32_t smem_addr = smem_stage_base
        + local_tid * SWIZZLE_CD_BYTES + swizzled_col * BANK_GROUP_BYTES;
    st_shared_128(smem_addr, p0, p1, p2, p3);
}
```

这里 `ELEMS_PER_BANK_GROUP=8`、`BANK_GROUP_BYTES=16`、`SWIZZLE_CD_BYTES=128`。XOR 改变不同输出行中 16-byte 组的落点，以匹配 TMA descriptor 的 128-byte swizzle，并减少 bank conflict。它重排的是 CD SMEM 地址，输出矩阵的逻辑列序保持不变。

**最后一个 chunk 读完，立即通知 TMEM 可以复用。** 此时前七个 chunk 的 Store 已经发出，最后一个 chunk 的数据也已离开 TMEM。每个读取线程对 leader 的 `tmem_empty_bar` 做一次 arrival，两个 CTA 共 256 次：

```cpp
if (w == NUM_M_WAVES - 1 && s == NUM_STORES - 1) {
    tcgen05_fence_before();
    barrier_arrive_cluster(tmem_empty_bar[accum_idx], 0);
}
__syncwarp();
```

`fence_before` 把本线程此前的 tcgen05 读取排在同步通知之前。下一 tile 的 UMMA 只需等这些读取全部结束，就能覆盖 TMEM；旧 tile 的 Global Store 可以尚未结束。

**发布 CD SMEM 数据并发出 Store。** 每个 chunk 都执行以下片段，最后一个 chunk 也一样。Proxy fence 让普通线程写出的 SMEM 数据对 TMA async proxy 可见，named barrier 则确保全部 128 行都已经填好。

```cpp
tma_store_fence();
named_barrier_sync(NUM_EPILOGUE_THREADS, EPILOGUE_BAR_ID);
if (epi_warp == 0 && elect_one()) {
    const int32_t n_idx = n_block * BLOCK_N + s * STORE_BLOCK_N;
    const int32_t m_idx = m_block * BLOCK_M + w * WAVE_BLOCK_M;
    tma_store_2d(smem_cd_base + tma_store_stage * SMEM_CD_PER_STAGE,
                 &tma_d, n_idx, m_idx);
    tma_store_commit();
}
tma_store_stage = (tma_store_stage + 1) % NUM_TMA_STORE_STAGES;
```

TMA Store 从 SMEM 取数据，所以这条写回路径需要经过寄存器和 CD SMEM。两块 CD 缓冲使线程填充与较早的 Store 有机会重叠；归还 TMEM 又允许下一 tile 的 Tensor Core 计算与写回尾部重叠。

<BlackwellGemmDiagram view="overlap" />

单个 TMEM accumulator stage 带来的限制是：旧 tile 的 TMEM 读取完成前，下一 tile 不能写这块 TMEM。重叠发生在归还之后的计算与写回之间；CD Store 若阻塞了最后一批读取，也会推迟归还。

### 排空流水线与释放资源

最后一个 tile 之后，没有“下一 tile 开始时”的等待来保护资源。MMA 分支因此补等最后一轮 `tmem_empty`；epilogue 的 Store 发射线程等待全部 Store 完成。它们分别保护 TMEM 生命周期和最终输出写回。

```cpp
// MMA 分支退出 persistent loop 后
int last_iter = scheduler.current_iter - 1;
if (last_iter >= 0) {
    uint32_t li = last_iter % NUM_EPILOGUE_STAGES;
    uint32_t lp = (last_iter / NUM_EPILOGUE_STAGES) & 1;
    barrier_wait(tmem_empty_bar[li], lp);
}
```

```cpp
// Epilogue 分支退出 persistent loop 后
if (epi_warp == 0 && elect_one())
    tma_store_wait<0>();
```

随后所有角色回到共同的清理路径。CTA 内先集合，再进行 cluster 同步，最后由两个 CTA 的 warp 2 协作释放 TMEM：

```cpp
__syncthreads();
cluster_sync();
if (warp_idx == 2)
    tmem_dealloc_2sm(0, TMEM_COLS);
```

## 优化取舍与性能范围

### 从 Level 2 到 Level 9 留下了什么

不同 Level 用于逐步引入机制。阅读最终实现时，保留下面这些差异即可；Level 9 的代码解释只使用当前配置。

| Level | 关键变化 | 对最终实现的作用 |
|---|---|---|
| 2 | A/B SMEM `full/empty` 与 2SM 完成通知 | 建立输入缓冲交接 |
| 3 | TMA/MMA 各自独立循环 | 让加载与计算按各自进度推进 |
| 4 | TMEM→SMEM 后合并写回 | 重排输出数据 |
| 5 | Persistent scheduler、2D swizzle | 多 tile 复用 CTA，改善 B 的 L2 复用 |
| 6 | 三角色、M-wave、TMA Store | 将计算、结果搬出与写回分工 |
| 7 | 两 CTA 计算不同输出 tile | 避免重复计算相同结果 |
| 8 | Swizzled CD、提前通知 `tmem_empty` | 改善输出访问，提前归还 TMEM |
| 9 | 大 tile、4 输入 stage、descriptor shuffle、256 次读取完成通知 | 在容量限制下协调输入、累加与输出路径 |

Level 9 把 tile 扩大到 $256\times256$，减少需要调度的 tile 数，但也占用了全部 512 个 TMEM column，因此只保留一个 accumulator stage。它通过 CD 双缓冲和提前归还 TMEM 获得部分跨 tile 重叠，不能把前一版本的 TMEM double buffering 解释直接沿用过来。

### 性能数字对应哪些条件

同版本 [Level 9 README](https://github.com/KnowingNothing/MatmulTutorial/blob/69c6886b93c3d569f1ba49185615c7fce9f19df0/examples/matmul/this-sm100/level9/README.md#performance-results-gb200-bf16) 报告 GB200、BF16 的结果。下表中的比值均为作者列出的本实现吞吐除以 DeepGEMM 吞吐：

| M | N | K | 本实现 TFLOPS | DeepGEMM TFLOPS | 比值 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 4096 | 4096 | 1559 | 1660 | 0.94 |
| 6144 | 6144 | 6144 | 1590 | 1512 | 1.05 |
| 8192 | 8192 | 8192 | 1480 | 1499 | 0.99 |
| 10240 | 10240 | 10240 | 1425 | 1500 | 0.95 |
| 12288 | 12288 | 12288 | 1463 | 1466 | 1.00 |
| 4096 | 576 | 7168 | 497 | 964 | 0.52 |
| 4096 | 4096 | 7168 | 1084 | 1398 | 0.78 |

“约 98%”对应前五个方阵测量点的平均比值。后两行展示固定 tile 在其他 shape 上的局限；作者将部分差距归因于 DeepGEMM 按 shape 选择不同配置。整体对比同时改变了多项设计，不能据此给某一项优化分配独立加速比。

> README 的概括范围与表内部分数字存在出入，本文保留具体测量点。表中结果也不能替代边缘 shape 的正确性检查；本文新增的 scheduler 条件来自索引推导。

## 附录：descriptor、helper 与 host launch

### UMMA 的两个 descriptor

SMEM descriptor：

```cpp
// 描述 operand tile 在 SMEM 中“从哪里开始、按什么 layout 解释”。
__device__ __forceinline__ uint64_t
make_smem_desc(void* smem_ptr, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr)) >> 4;

    d |= (uint64_t)(addr & 0x3FFF);              // start address，16B 单位
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; // stride byte offset
    d |= (uint64_t)1 << 46;                     // SM100 descriptor version
    d |= (uint64_t)2 << 61;                     // SWIZZLE_128B layout
    return d;
}
```

本算子使用 K-major、128B swizzle。不同 pipeline stage 主要改变 start address；`k * 2` 和 wave offset 也都是在移动这个 start-address field。

Instruction descriptor：

```cpp
// 描述“做什么 UMMA”，不包含 A/B 的 SMEM 地址。
__device__ __forceinline__ uint32_t
make_instr_desc(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);           // C accumulator = FP32
    d |= (1u << 7);           // A = BF16
    d |= (1u << 10);          // B = BF16
    d |= ((N / 8) << 17);     // N dimension
    d |= ((M / 16) << 24);    // M dimension
    return d;
}
```

这里 `idesc = make_instr_desc(256,256)`。K-step 的 16 来自所选 `kind::f16` UMMA 形式和本算子的 `UMMA_K=16` 循环组织，不是通过这段 builder 的 M/N field 传入。

### TMA tensor map 的创建

```cpp
static void create_tma_desc(
    CUtensorMap* map, const void* ptr,
    uint64_t dim0, uint64_t dim1,
    uint32_t box0, uint32_t box1,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B) {

    uint64_t dims[2]     = {dim0, dim1};
    uint64_t strides[1]  = {dim0 * sizeof(__nv_bfloat16)};
    uint32_t box[2]      = {box0, box1};
    uint32_t estrides[2] = {1, 1};

    cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        const_cast<void*>(ptr), dims, strides, box, estrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

// A、B 在 C++ 中是 row-major，但 tensor map 把连续维 K/N 放在 dim0。
create_tma_desc(&tma_a, A, K, M, BLOCK_K, BLOCK_M);
create_tma_desc(&tma_b, B, K, N, BLOCK_K, LOAD_N_PER_CTA);
create_tma_desc(&tma_d, D, N, M, STORE_BLOCK_N, STORE_BLOCK_M,
                CU_TENSOR_MAP_SWIZZLE_128B);
```

因此实际 TMA box 为：

| Map | Global 逻辑 shape | 每次 box |
|---|---|---|
| A load | `(K,M)` view | `(64,256)` = 32 KiB |
| B load | `(K,N)` view | `(64,128)` = 16 KiB/CTA |
| D store | `(N,M)` view | `(64,128)` = 16 KiB |

### 关键 helper 的高层含义

| Helper | 高层作用 |
|---|---|
| `elect_one()` | 在调用它的 warp 中选一个 issuer |
| `barrier_init()` | 建立 mbarrier 的初始 phase/count |
| `barrier_arrive_expect_tx()` | 一次 arrival，并把待完成 TMA 字节数加入 tx-count |
| `barrier_arrive_cluster()` | 向 cluster 内指定 CTA 的 barrier 做远程 arrival |
| `barrier_wait()` | 轮询当前或紧邻上一 parity，直到 phase 完成 |
| `tma_load_2d_cg2()` | Global→调用 CTA 的 SMEM，并向指定 barrier complete-tx |
| `umma_f16_cg2()` | 发异步 2SM BF16 UMMA，结果累加到两边 TMEM |
| `umma_commit_2sm()` | 为此前 UMMA 建完成点，完成后 multicast arrival |
| `tmem_load_8x()` | epilogue warp 从 TMEM 集体加载 8 个 FP32 column |
| `tmem_load_fence()` | 等待上述异步 TMEM load 的寄存器结果可用 |
| `tma_store_fence()` | 使 generic SMEM 写对 TMA async proxy 可见 |
| `tma_store_commit/wait()` | 提交和限制 TMA Store bulk groups |

特别注意 2SM TMA helper 的两个地址：

- **SMEM destination**：保持调用 CTA 的本地地址。
- **Completion barrier**：通过 peer-bit/mapping 路由到 leader CTA。

若把 peer mask 错误地应用到 destination，数据会被写进错误 CTA 的 SMEM。

### Launch 为什么是 persistent cluster kernel

```cpp
uint32_t num_tiles = num_m_blocks * num_n_blocks;

int num_sms = 0;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

// 最多一 CTA/SM，并向下取偶数，以组成 2-CTA cluster。
uint32_t num_ctas = min((uint32_t)num_sms, num_tiles);
num_ctas = (num_ctas / CLUSTER_SIZE) * CLUSTER_SIZE;

config.gridDim          = dim3(num_ctas, 1, 1);
config.blockDim         = dim3(NUM_THREADS, 1, 1);
config.dynamicSmemBytes = SMEM_SIZE;

attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim = {CLUSTER_SIZE, 1, 1};
```

grid 提供数量受限的常驻 CTA。每个 CTA 在 `TileScheduler` 的 while-loop 中处理多个 tile；TMEM 和 barrier 只初始化一次，phase 跨 tile 连续推进。
