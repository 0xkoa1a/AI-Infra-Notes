# Blackwell 架构 MatMul 算子实现精读

算子实现来自 https://github.com/KnowingNothing/MatmulTutorial/tree/main/examples/matmul/this-sm100

这里节选 level 9 实现，其可达到 98% 的 DeepGEMM 性能。

## 1. 先建立整体图景

### 1.1 计算任务与 tile

算子计算：

```text
D[M, N] = A[M, K] @ B[N, K]^T

A、B：BF16
TMEM 累加器：FP32
D：BF16
```

Level 9 使用固定的主 tile：

| 层级 | 形状/数量 | 含义 |
|---|---:|---|
| 每个 CTA 的输出 tile | `256 × 256` | 一个 CTA 最终写回一块 D |
| 每个 K block | `64` | 主循环每轮消费 64 个 K 元素 |
| 每条 UMMA 的 K | `16` | 一个 K block 需要 4 个 K-step |
| M-wave | `2 × 128` 行 | 每个 CTA 的 256 行分两次进入本地 TMEM |
| CTA cluster | `2` 个 CTA | 两个 SM 协作执行 `cta_group::2` UMMA |

同一 cluster 中的两个 CTA 被 scheduler 分配到相同的 `n_block`、不同的 `m_block`：

```text
CTA 0 / SM 0                           CTA 1 / SM 1

A tile 0: [256, 64]                    A tile 1: [256, 64]
B 左半:   [128, 64]                    B 右半:   [128, 64]
          \                               /
           \------ cta_group::2 UMMA ----/
                          │
             两边共同提供完整的 256 列 B
                          │
CTA 0 TMEM: tile 0 的结果                CTA 1 TMEM: tile 1 的结果
```

两个 CTA 共享一条 2SM UMMA 的发射过程和 B 的两个半块，但各自使用本地 A，并在本地 TMEM 中得到不同的输出 tile。

### 1.2 数据经过哪些存储层级

```text
Global A/B
    │
    │ TMA Load：异步搬运
    ▼
SMEM A/B（4 个流水线 stage）
    │
    │ UMMA：读取 SMEM descriptor，执行 BF16 × BF16 → FP32
    ▼
TMEM（Tensor Core 累加器）
    │
    │ tcgen05.ld：128 个 epilogue 线程各读一条 TMEM lane
    ▼
寄存器 FP32 → 打包为 BF16 → swizzled SMEM CD（2 个 store stage）
    │
    │ TMA Store
    ▼
Global D
```

不要把三个 descriptor 混在一起：

| 对象 | 描述什么 | 谁消费 |
|---|---|---|
| `CUtensorMap tma_a/tma_b/tma_d` | Global tensor 的基址、shape、stride、TMA box、swizzle、OOB 等 | TMA load/store |
| `desc_a/desc_b` | 当前 A/B tile 在 SMEM 中的起始位置和 UMMA layout | UMMA |
| `idesc` | A/B/C 数据格式和本条 UMMA 的 M、N 等指令属性 | UMMA |

### 1.3 三角色 warp specialization

每个 CTA 有 8 个 warp，共 256 个线程：

| Warp | 角色 | 参与 CTA | 工作 |
|---|---|---|---|
| 0 | TMA producer | 两个 CTA | 各选 1 个线程持续加载 A/B |
| 1 | MMA consumer/producer | 仅 leader CTA | 全 warp 等 barrier/shuffle，选 1 个线程发 UMMA 和 commit |
| 2 | TMEM 管理 | 两个 CTA | 集体分配/释放 TMEM |
| 3 | 空闲 | 两个 CTA | 不参与主流水线 |
| 4–7 | Epilogue consumer | 两个 CTA | 128 线程把本地 TMEM 写回各自的 D tile |

三个角色分别运行自己的 persistent loop：

```text
时间 ─────────────────────────────────────────────────────────────►

TMA warp:       load tile 0     load tile 1     load tile 2 ...
                       │full            │full
MMA warp:              MMA tile 0       MMA tile 1       ...
                              │tmem_full
Epilogue warps:               read tile 0 / TMA store
                                     │tmem_empty
```

它们不是每处理一个 tile 就重新集合，而是通过 barrier 组成长期运转的生产者—消费者流水线。TMA 可以领先 MMA 最多 4 个 K block；epilogue 把结果读出 TMEM 后便尽快释放 TMEM，使后续 MMA 能与尚未结束的 TMA Store 重叠。

### 1.4 三种“stage”不要混淆

| 缓冲 | 数量 | 被谁复用 | 保护它的同步 |
|---|---:|---|---|
| A/B SMEM pipeline stage | 4 | TMA 与 MMA | `full_bar[s]` / `empty_bar[s]` |
| TMEM accumulator stage | 1 | MMA 与 epilogue | `tmem_full_bar[0]` / `tmem_empty_bar[0]` |
| CD SMEM TMA-store stage | 2 | epilogue 填充与 TMA Store | bulk-group `wait_group` |

Level 9 没有 TMEM double buffering：`accum_idx` 恒为 0，但 `accum_phase` 每个 tile 翻转，仍然可以安全地重复使用同一块 TMEM。它有两级 CD SMEM buffering，用于让 TMEM→SMEM 和先前的 TMA Store 重叠。

### 1.5 Barrier 协议总表

| Barrier | 等待者 | 完成者 | 含义 |
|---|---|---|---|
| `full_bar[s]` | leader 的 MMA warp | 两个 CTA 的 TMA + tx-count | 两边 stage `s` 的 A/B 都已进入 SMEM |
| `empty_bar[s]` | 两个 CTA 的 TMA warp，各等本地副本 | UMMA commit multicast | UMMA 已读完 stage `s`，SMEM 可以覆盖 |
| `tmem_full_bar[0]` | 两个 CTA 的 epilogue warps | 最后一个 K block 的 UMMA commit | 完整 FP32 tile 已在 TMEM 中 |
| `tmem_empty_bar[0]` | leader 的 MMA warp | 两 CTA × 128 个 epilogue 线程 | 所有线程都已读完 TMEM，可以写下一 tile |
| named barrier 1 | 同一 CTA 的 128 个 epilogue 线程 | 这些线程自己 | CD SMEM 填充和 TMA Store 发射前的 CTA 内集合 |

`full/empty` 是 A/B SMEM 的循环队列；`tmem_full/tmem_empty` 是 TMEM 的循环队列。名称都从消费者视角理解：`full` 表示有数据可消费，`empty` 表示可被生产者覆盖。

## 2. Level 9 核心代码精读

下面保留 Level 9 的核心控制流，并把 Level 2、Level 6 中已有的中文注释迁移到对应位置。helper 的 PTX 封装和 host launch 放在附录。

### 2.1 配置与内存预算

```cpp
static constexpr uint32_t BLOCK_M      = 256;
static constexpr uint32_t BLOCK_N      = 256;
static constexpr uint32_t BLOCK_K      = 64;
static constexpr uint32_t CLUSTER_SIZE = 2;
static constexpr uint32_t NUM_STAGES   = 4;

// 每个 CTA 的 TMEM 只有 128 条 lane；一个 256 行 tile 分成两个 M-wave。
static constexpr uint32_t WAVE_BLOCK_M = 128;
static constexpr uint32_t NUM_M_WAVES  = BLOCK_M / WAVE_BLOCK_M;  // 2

// 每个 CTA 只加载 B 的一半；cta_group::2 UMMA 组合两个半块。
static constexpr uint32_t LOAD_N_PER_CTA = BLOCK_N / CLUSTER_SIZE; // 128
static constexpr uint32_t UMMA_M = WAVE_BLOCK_M * CLUSTER_SIZE;    // 256
static constexpr uint32_t UMMA_N = BLOCK_N;                         // 256
static constexpr uint32_t UMMA_K = 16;

static constexpr uint32_t SMEM_A_SIZE =
    BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);          // 32 KiB/stage
static constexpr uint32_t SMEM_B_SIZE =
    LOAD_N_PER_CTA * BLOCK_K * sizeof(__nv_bfloat16);   // 16 KiB/stage
static constexpr uint32_t TMA_BYTES = SMEM_A_SIZE + SMEM_B_SIZE; // 48 KiB/CTA

// TMEM: 1 个 accumulator stage × 2 个 M-wave × 256 列 = 512 columns。
// 分配一个 column 时会同时得到该 column 的全部 128 条 lane。
static constexpr uint32_t NUM_EPILOGUE_STAGES = 1;
static constexpr uint32_t TMEM_COLS =
    NUM_EPILOGUE_STAGES * NUM_M_WAVES * BLOCK_N;        // 512

static constexpr uint32_t NUM_THREADS          = 256;
static constexpr uint32_t NUM_EPILOGUE_THREADS = 128;

// 每个 TMA Store 搬 128 行 × 64 个 BF16 = 16 KiB。
// 一个 CTA tile 有 2 个 wave × 4 个 N-chunk = 8 次 store。
static constexpr uint32_t STORE_BLOCK_M = 128;
static constexpr uint32_t STORE_BLOCK_N = 64;
static constexpr uint32_t NUM_STORES    = BLOCK_N / STORE_BLOCK_N; // 4
static constexpr uint32_t NUM_TMA_STORE_STAGES = 2;
```

SMEM 采用 `[CD][所有 A stage][所有 B stage][barriers][tmem_ptr]`：

```text
CD store buffers       2 × 16 KiB =  32 KiB
A pipeline             4 × 32 KiB = 128 KiB
B pipeline             4 × 16 KiB =  64 KiB
barriers + tmem_ptr                  < 1 KiB
                                      --------
总计约                              224 KiB
```

A/B stage 分离后，相邻 stage 的 descriptor 低 32 位具有固定步长，MMA warp 才能用 `__shfl_sync` 缓存 descriptor，而不必让每个线程持有数组。

### 2.2 Persistent 2D swizzle scheduler

```cpp
// 只启动有限数量的 cluster：num_clusters = min(num_sms / 2, num_tiles);
// 核心是 uint32_t tile_idx = (++current_iter) * num_clusters + cluster_id;
// 假设有 32 个 cluster，则 cluster 0 负责 tile 0, 32, 64, 96, ...，cluster 1 负责 tile 1, 33, 65, 97, ...，以此类推。
// tile 映射还做了 swizzle，提高 L2 Cache locality
//
// 上面是 Level 6 的原注。Level 9 已改为 per-CTA scheduler：
//   num_ctas = min(num_sms, num_tiles)，然后向下取偶数；
//   cta_id   = blockIdx.x；
// 因而相邻 CTA 属于同一 cluster，并分别得到不同的 M tile。

struct TileScheduler {
    uint32_t num_m_blocks, num_n_blocks, num_tiles, num_ctas, cta_id;
    int current_iter;

    __device__ TileScheduler(uint32_t M, uint32_t N, uint32_t nc)
        : num_m_blocks((M + BLOCK_M - 1) / BLOCK_M),
          num_n_blocks((N + BLOCK_N - 1) / BLOCK_N),
          num_tiles(num_m_blocks * num_n_blocks),
          num_ctas(nc),
          cta_id(blockIdx.x),
          current_iter(-1) {}

    __device__ bool get_next_block(uint32_t& m_block, uint32_t& n_block) {
        // persistent 分配：每个 CTA 每轮跨过 num_ctas 个线性 tile。
        uint32_t tile_idx =
            static_cast<uint32_t>(++current_iter) * num_ctas + cta_id;
        if (tile_idx >= num_tiles) return false;

        // 每 16 个 M block 组成一组；组内先变化 M，再变化 N。
        // 这样一批相邻 CTA 共享 n_block，B 的相同区域更容易留在 L2。
        uint32_t tiles_per_group = num_n_blocks * SWIZZLE_GROUP_SIZE;
        uint32_t group_idx       = tile_idx / tiles_per_group;
        uint32_t first_m         = group_idx * SWIZZLE_GROUP_SIZE;
        uint32_t in_group        = tile_idx % tiles_per_group;
        uint32_t m_in_group      = min(SWIZZLE_GROUP_SIZE,
                                       num_m_blocks - first_m);

        m_block = first_m + in_group % m_in_group;
        n_block = in_group / m_in_group;
        return true;
    }
};
```

例如 `num_ctas=192` 时，CTA 0 处理线性 tile `0, 192, 384, ...`，CTA 1 处理 `1, 193, 385, ...`。swizzle 再把线性编号映射成 `(m_block,n_block)`。在本教程针对的合法 shape 上，同一 cluster 的偶/奇 CTA 保持相同 `n_block`，从而能共同组成 B 的完整 256 列。

### 2.3 Kernel 初始化

```cpp
__global__ void
__cluster_dims__(2, 1, 1)
__launch_bounds__(NUM_THREADS, 1)
bf16_gemm_2sm_kernel(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_d,
    uint32_t M, uint32_t N, uint32_t K,
    uint32_t num_ctas)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    const uint32_t warp_idx  = get_warp_id();
    const uint32_t lane_idx  = get_lane_id();
    const uint32_t cta_rank  = get_cluster_rank();
    const bool     is_leader = (cta_rank == 0);

    extern __shared__ __align__(1024) uint8_t smem_buf[];

    // 每个 CTA 都有自己的 SMEM 和 barrier 副本。
    // full_bar 在 leader CTA arrive 和 wait（因为 leader 负责 issue UMMA）
    // empty_bar 则各 CTA 等自己的副本。
    uint64_t* full_bar[NUM_STAGES];
    uint64_t* empty_bar[NUM_STAGES];
    for (uint32_t s = 0; s < NUM_STAGES; ++s) {
        full_bar[s]  = reinterpret_cast<uint64_t*>(
            smem_buf + OFF_BAR + s * 8);
        empty_bar[s] = reinterpret_cast<uint64_t*>(
            smem_buf + OFF_BAR + NUM_STAGES * 8 + s * 8);
    }

    uint64_t* tmem_full_bar[NUM_EPILOGUE_STAGES];
    uint64_t* tmem_empty_bar[NUM_EPILOGUE_STAGES];
    for (uint32_t e = 0; e < NUM_EPILOGUE_STAGES; ++e) {
        tmem_full_bar[e] = reinterpret_cast<uint64_t*>(
            smem_buf + OFF_BAR + NUM_STAGES * 16 + e * 8);
        tmem_empty_bar[e] = reinterpret_cast<uint64_t*>(
            smem_buf + OFF_BAR + NUM_STAGES * 16
                     + NUM_EPILOGUE_STAGES * 8 + e * 8);
    }

    uint32_t* tmem_addr = reinterpret_cast<uint32_t*>(smem_buf + OFF_TMEM);
    uint8_t* smem_cd_base = smem_buf + OFF_CD;

    auto get_smem_a = [&](uint32_t s) -> void* {
        return smem_buf + OFF_A + s * SMEM_A_SIZE;
    };
    auto get_smem_b = [&](uint32_t s) -> void* {
        return smem_buf + OFF_B + s * SMEM_B_SIZE;
    };

    // Prefetch TMA descriptors
    // 从当前 warp 的活跃线程中选出一个线程
    if (warp_idx == 0 && elect_one()) {
        // 这里预取的是 TMA descriptor，而不是 A、B 矩阵本身

        // TMA descriptor 主要包含 A/B 的:
        // global memory base addr.,
        // shape,
        //每个维度的 stride,
        // data type,
        // 每次 TMA load 的 tile shape
        // SMEM swizzle 方式
        // 越界处理方式

        // TMA descriptor 是在 host 端创建的，作为 __grid_constant__ 传入 kernel
        prefetch_tma(&tma_a);
        prefetch_tma(&tma_b);
        prefetch_tma(&tma_d);
    }

    // Q：为什么不直接指定 threadIdx.x == 0？
    // A：这里需要“当前角色 warp 中恰好一个活跃线程”，并不总是 warp 0。
    //    warp 1、warp 4 等角色根本不包含 thread 0。完整 warp 时可以手写
    //    lane_idx == 0，但 elect_one() 更直接表达 one-issuer 契约，也不绑定固定 lane。

    // barrier 初始化在 persistent loop 之外，因为 pipeline 是跨 tile 连续运转的
    // 不需要在 tile 边界重新归零
    if (warp_idx == 1 && elect_one()) {
        for (uint32_t s = 0; s < NUM_STAGES; ++s) {
            barrier_init(full_bar[s], CLUSTER_SIZE); // 两个 CTA 各贡献一次 arrival
            barrier_init(empty_bar[s], 1);           // 一次 UMMA commit arrival
        }
        for (uint32_t e = 0; e < NUM_EPILOGUE_STAGES; ++e) {
            barrier_init(tmem_full_bar[e], 1);
            // 两边每个 epilogue 线程都要证明自己的 TMEM load 已结束。
            barrier_init(tmem_empty_bar[e],
                         CLUSTER_SIZE * NUM_EPILOGUE_THREADS); // 256
        }

        // 把 barrier 初始化正式*发布*给 cluster 内其他线程、其他 CTA，以及后续使用 mbarrier 的异步硬件操作
        // 后面的 cluster_sync 的确同时有执行同步和 release/acquire；
        // 但它的内存可见性保证针对 arrive 前的普通 memory access，mbarrier.init 不是普通 st.shared
        // PTX 为它单独定义了 fence.mbarrier_init：只发布同一线程此前执行的 init。

        // 因此这里两者分工是：
        //   fence_barrier_init：把 mbarrier 的初始状态发布到 cluster 范围；
        //   cluster_sync：让所有 CTA 集合，并在返回后才开始远程 arrive/TMA/UMMA。
        // 两者都是必要的
        fence_barrier_init();
    }

    // tcgen05.alloc 是 warp-collective，因此整个 warp 2 参与，而不是 elect_one。
    if (warp_idx == 2)
        tmem_alloc_2sm(tmem_addr, TMEM_COLS);

    // 封装了 barrier.cluster.arrive.aligned 和 barrier.cluster.wait.aligned
    // 同时包含：
    // 执行同步：所有线程/CTA 都到达后才能继续
    // release：发布 arrive 之前的操作
    // acquire：wait 返回之后可以观察到其他线程发布的操作
    cluster_sync();

    const uint32_t num_k_blocks = (K + BLOCK_K - 1) / BLOCK_K;

    // instruction descriptor 告诉 tensor core A/B 矩阵的 data type、累加值 data type、每次 UMMA 的 tile shape
    const uint32_t idesc = make_instr_desc(UMMA_M, UMMA_N);
```


### 2.4 TMA warp：生产 A/B SMEM stage

```cpp
    if (warp_idx == 0 && elect_one()) {
        // ======================== TMA WARP ========================
        // 每个 CTA 由 1 个 elected thread 发射 TMA；两个 CTA 独立运行此分支。
        TileScheduler scheduler(M, N, num_ctas);
        uint32_t m_block, n_block;
        uint32_t stage = 0, phase = 0;

        while (scheduler.get_next_block(m_block, n_block)) {
            // 两个 CTA 的 n_block 相同、m_block 不同。
            const int32_t m_coord = m_block * BLOCK_M;
            const int32_t n_coord =
                n_block * BLOCK_N + cta_rank * LOAD_N_PER_CTA;

            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                // phase ^ 1 即当前 phase 的上一轮。
                // 第一轮等待 phase ^ 1，立即返回
                // 其后每轮等待上一轮 UMMA commit multicast 到 empty_bar[stage_idx]，才会返回
                barrier_wait(empty_bar[stage], phase ^ 1);

                // 先登记期待 byte count，再发射 TMA，防止 TMA 很快完成，但 barrier 还没登记 complete_tx
                // 之后 leader CTA 会 wait full_bar 以确保 TMA 完成，然后发射 UMMA，所以这里把 tx-count 的登记集中在 leader CTA
                if (is_leader) {
                    barrier_arrive_expect_tx(
                        full_bar[stage], TMA_BYTES * CLUSTER_SIZE);
                } else {
                    barrier_arrive_cluster(full_bar[stage], 0);
                }

                const int32_t kc = kb * BLOCK_K;

                // 每个 CTA 加载：
                //   A[256,64] = 32 KiB（不同 m tile）
                //   B[128,64] = 16 KiB（完整 N tile 的不同一半）
                // 目的地是自己的 SMEM buffer
                tma_load_2d_cg2(
                    get_smem_a(stage), &tma_a, full_bar[stage], kc, m_coord);
                tma_load_2d_cg2(
                    get_smem_b(stage), &tma_b, full_bar[stage], kc, n_coord);

                stage = (stage + 1) % NUM_STAGES;
                phase ^= (stage == 0); // 4 个 stage 绕一圈后切换 parity
            }
        }
```

`full_bar[stage]` 完成需要同时满足：两次 arrival 已发生，且两个 CTA 总计 `2 × 49152` 字节的 TMA tx-count 已归零。

### 2.5 MMA warp：消费 SMEM，生产 TMEM

```cpp
    } else if (warp_idx == 1 && is_leader) {
        // ======================== MMA WARP ========================
        // Level 9 与 Level 2/6 的区别：warp 1 的 32 个线程都参与 barrier_wait
        // 和 __shfl_sync；只有 UMMA/commit 的实际发射由 elect_one() 限定为一个线程。

        // 每个流水线阶段都需要 A/B 的 SMEM descriptor
        // SMEM desc 的低 32 位是 SMEM 基址（每个 tile 不一样），高 32 位是 layout 等（每个阶段一样）
        // 因此这里只每个阶段存低 32 位（alo[NUM_STAGES] / blo[NUM_STAGES]），高 32 位统一保存为 ahi/bhi
        //
        // 上三行是 Level 2 的原注。Level 9 不再真的保存 alo[]/blo[]：
        // stage 分离布局让低 32 位可以由 stage 0 加固定 stride 得到，再用 shuffle 广播。
        uint64_t desc0_a = make_smem_desc(get_smem_a(0), UMMA_SBO);
        uint64_t desc0_b = make_smem_desc(get_smem_b(0), UMMA_SBO);
        uint32_t ahi = static_cast<uint32_t>(desc0_a >> 32);
        uint32_t bhi = static_cast<uint32_t>(desc0_b >> 32);

        // ad/bd：A/B 的 SMEM 基址和 layout
        // lane 0..3 分别缓存 stage 0..3 的 descriptor low；其余 lane 的值不用作源。
        uint32_t a_lo_base = static_cast<uint32_t>(desc0_a);
        uint32_t b_lo_base = static_cast<uint32_t>(desc0_b);
        uint32_t my_a_lo = a_lo_base
            + (lane_idx < NUM_STAGES ? lane_idx * (SMEM_A_SIZE / 16) : 0u);
        uint32_t my_b_lo = b_lo_base
            + (lane_idx < NUM_STAGES ? lane_idx * (SMEM_B_SIZE / 16) : 0u);

        // 注意这里的 scheduler 和 warp 0 分支的 scheduler 是独立的实例
        // 流水线稳定运行时，每个 warp 分支的 current_iter 不同，因此 scheduler 也需要是不同的实例
        TileScheduler scheduler(M, N, num_ctas);
        uint32_t m_block, n_block;
        uint32_t stage = 0, phase = 0;

        while (scheduler.get_next_block(m_block, n_block)) {
            // level 6 的 epilogue 做了 double buffering，于是可以和 MMA 重叠
            //
            // 这里 Level 9 的 NUM_EPILOGUE_STAGES=1，没有 TMEM
            // double buffering；它靠尽早把 TMEM 搬入两级 CD SMEM，让 MMA 与
            // 后续 TMA Store 重叠。
            uint32_t accum_idx =
                scheduler.current_iter % NUM_EPILOGUE_STAGES; // 恒为 0
            uint32_t accum_phase =
                (scheduler.current_iter / NUM_EPILOGUE_STAGES) & 1; // 每 tile 翻转

            // 首 tile 等待 phase^1 会立即返回；之后等两边 epilogue 的 256 次
            // arrival，保证上一 tile 的所有 TMEM load 已结束。
            barrier_wait(tmem_empty_bar[accum_idx], accum_phase ^ 1);

            // Q：barrier_wait 已经同步，fence_after 是否冗余？
            // A：不冗余。wait 证明 barrier phase 完成并提供 acquire；
            //    tcgen05_fence_after 把“后续 tcgen05 操作”排在该同步点之后。
            //    一个管 barrier 状态/内存可见性，一个管异步 Tensor Core pipeline 顺序。
            tcgen05_fence_after();  // 保证后续的 tcgen05 发生在线程同步（wait）之后

            for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
                // 两个 CTA 的 TMA 数据都完成后，full_bar[stage] 才完成。
                barrier_wait(full_bar[stage], phase);
                tcgen05_fence_after();

                // 32 个线程从保存 stage descriptor 的对应 lane 取值。
                uint32_t cur_a_lo =
                    __shfl_sync(0xFFFFFFFF, my_a_lo, stage);
                uint32_t cur_b_lo =
                    __shfl_sync(0xFFFFFFFF, my_b_lo, stage);

                if (elect_one()) {
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
                        // descriptor 地址以 16 B 为单位；一个 K-step 前进
                        // 16 个 BF16 = 32 B，因此 low field 增加 32/16 = 2。
                        uint32_t b_lo = cur_b_lo + k * 2;
                        uint64_t bd = ((uint64_t)bhi << 32) | b_lo;

                        #pragma unroll
                        for (uint32_t w = 0; w < NUM_M_WAVES; ++w) {
                            // A descriptor 同时沿 M-wave 和 K-step 移动。
                            // wave offset = 128 * 64 * 2 B / 16 B = 1024。
                            uint32_t a_lo = cur_a_lo
                                + w * (WAVE_BLOCK_M * BLOCK_K
                                     * sizeof(__nv_bfloat16) / 16)
                                + k * 2;
                            uint64_t ad = ((uint64_t)ahi << 32) | a_lo;

                            // wave 0 占 TMEM columns [0,256)，
                            // wave 1 占 TMEM columns [256,512)。
                            uint32_t tmem_offset =
                                accum_idx * NUM_M_WAVES * BLOCK_N
                                + w * BLOCK_N;

                            // 这里的 ad/bd 是 SMEM descriptor，idesc 是 UMMA instruction descriptor
                            // accum 指定是否累加：首轮为 0 表示初始化，后续轮次为 1 表示累加
                            // 两个 wave 写不同 TMEM columns，所以它们在首个 K-step
                            // 都应清零；accum 不依赖 w。
                            uint32_t accum = (kb > 0 || k > 0) ? 1u : 0u;

                            // leader CTA 的 warp 1 的一个线程发射 UMMA 指令，其他线程不发射
                            umma_f16_cg2(tmem_offset, ad, bd, idesc, accum);
                        }
                    }
                }

                // commit 不是另一条 MMA，而是为此前发出的异步 UMMA 建立完成点。
                // 完成后 multicast arrival 到两个 CTA 的 empty_bar[stage]，TMA
                // 才能覆盖此 stage。
                if (elect_one()) {
                    umma_commit_2sm(empty_bar[stage]);

                    // 最后一轮通知 TMEM 的累加结果可以读取
                    if (kb == num_k_blocks - 1)
                        umma_commit_2sm(tmem_full_bar[accum_idx]);
                }

                stage = (stage + 1) % NUM_STAGES;
                phase ^= (stage == 0);
            }
        }

        // 最后一个 tile 后不再有下一轮 tmem_empty wait，因此显式补等一次，
        // 确保 epilogue 结束后才进入 TMEM/barrier 的销毁路径。
        int last_iter = scheduler.current_iter - 1;
        if (last_iter >= 0) {
            uint32_t last_idx = last_iter % NUM_EPILOGUE_STAGES;
            uint32_t last_phase =
                (last_iter / NUM_EPILOGUE_STAGES) & 1;
            barrier_wait(tmem_empty_bar[last_idx], last_phase);
        }
```

一轮 `kb` 共发出 `4 个 K-step × 2 个 M-wave = 8` 次 UMMA。TMA 与 UMMA 的完成协议不同：

```text
TMA：issue copy → complete_tx(bytes) → full_bar phase 完成
UMMA：issue mma  → explicit commit → 完成后 arrival → empty/tmem_full phase 完成
```

`commit` 与 MMA 分开，使一批 UMMA 可以共享完成点，也能把同一批计算完成通知给不同消费者。它不是等待；真正等待发生在消费者的 `barrier_wait()`。

### 2.6 Epilogue：消费 TMEM，生产 D

```cpp
    } else if (warp_idx >= 4) {
        // ======================== EPILOGUE WARPS ========================
        // 两个 CTA 都执行 epilogue，因为它们的本地 TMEM 是不同的输出 tile。
        // 128 个线程与 TMEM 的 128 条 lane 对应；每个线程处理一个本地结果行。
        TileScheduler scheduler(M, N, num_ctas);
        uint32_t m_block, n_block;

        const uint32_t local_tid = threadIdx.x - 128; // 0..127
        const uint32_t epi_warp  = local_tid / 32;    // 0..3
        uint32_t tma_store_stage = 0;

        while (scheduler.get_next_block(m_block, n_block)) {
            uint32_t accum_idx =
                scheduler.current_iter % NUM_EPILOGUE_STAGES; // 0
            uint32_t accum_phase =
                (scheduler.current_iter / NUM_EPILOGUE_STAGES) & 1;

            // 等最后一个 K block 的 commit，证明完整 tile 已写入 TMEM。
            barrier_wait(tmem_full_bar[accum_idx], accum_phase);
            tcgen05_fence_after();
            // 此时 TMEM 中有 256x256 个 FP32
            // 但是本地 TMEM 只有 128 个 lane，因此 M 方向分成 NUM_M_WAVES=2 个 wave
            // TMA store 每次只写 64 列，因此 N 方向分成 NUM_STORES=4 个 store
            // 共有 128 个 epilogue 线程，每个线程负责一个 TMEM lane，即 2 x 4 = 8 次 TMA store

            #pragma unroll
            for (uint32_t w = 0; w < NUM_M_WAVES; ++w) {
                #pragma unroll
                for (uint32_t s = 0; s < NUM_STORES; ++s) {
                    // CD SMEM 有两个 stage，每个 stage 16 KiB，足够存放 128 行 × 64 列的 BF16。
                    // 这里 wait_group<NUM_TMA_STORE_STAGES - 1> 的意思是
                    // 最多允许 NUM_TMA_STORE_STAGES - 1 个较新的 TMA Store group 仍然 on flight。
                    // 这保证当 wrap-around 回某个 stage 时，之前使用该 stage 的 store 已经完成。
                    // 只有一个线程执行 wait，之后用 named barrier 通知另外 127 个 epilogue 线程当前 CD stage 已经可以覆盖。
                    if (epi_warp == 0 && elect_one())
                        tma_store_wait<NUM_TMA_STORE_STAGES - 1>();
                    named_barrier_sync(NUM_EPILOGUE_THREADS,
                                       EPILOGUE_BAR_ID);

                    uint32_t smem_stage_base = static_cast<uint32_t>(
                        __cvta_generic_to_shared(
                            smem_cd_base
                            + tma_store_stage * SMEM_CD_PER_STAGE));

                    // 从 TMEM 读取一个 128x64 的 tile，转成 BF16 并 pack 成 u32，写入到 CD SMEM
                    // 每轮读取 8 个 FP32，共循环 8 次
                    #pragma unroll
                    for (uint32_t i = 0;
                         i < STORE_BLOCK_N / ELEMS_PER_BANK_GROUP;
                         ++i) {
                        uint32_t tmem_col =
                            accum_idx * NUM_M_WAVES * BLOCK_N
                            + w * BLOCK_N
                            + s * STORE_BLOCK_N
                            + i * ELEMS_PER_BANK_GROUP;

                        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                        tmem_load_8x(tmem_col,
                                     r0, r1, r2, r3, r4, r5, r6, r7);

                        // tcgen05.ld 是异步的；必须等寄存器结果可用后才能转换。
                        tmem_load_fence();

                        // 8 个 FP32 → 8 个 BF16，两个 BF16 打包进一个 u32。
                        uint32_t p0 = pack_bf16(r0, r1);
                        uint32_t p1 = pack_bf16(r2, r3);
                        uint32_t p2 = pack_bf16(r4, r5);
                        uint32_t p3 = pack_bf16(r6, r7);

                        // 这里的 swizzle 是为了避免 bank conflict
                        // XOR swizzle：不同 TMEM lane 写不同的 16B bank group，
                        // 与 tma_d 的 SWIZZLE_128B descriptor 相匹配。
                        uint32_t swizzled_col =
                            i ^ (local_tid % BANK_GROUPS_PER_SWIZZLE);
                        uint32_t smem_addr =
                            smem_stage_base
                            + local_tid * SWIZZLE_CD_BYTES
                            + swizzled_col * BANK_GROUP_BYTES;

                        st_shared_128(smem_addr, p0, p1, p2, p3);
                    }

                    // 读完最后一个 wave 的最后一个 N-chunk 之后，尽早释放 TMEM，以被 UMMA 复用
                    if (w == NUM_M_WAVES - 1 &&
                        s == NUM_STORES - 1) {
                        // before_thread_sync 把“本线程此前的 tcgen05 load”排在
                        // 随后的 arrival 前。随后两 CTA 的 128 个线程分别 arrive，
                        // 共同满足 leader tmem_empty 的 count=256。
                        tcgen05_fence_before();
                        barrier_arrive_cluster(tmem_empty_bar[accum_idx], 0);
                    }
                    __syncwarp();

                    tma_store_fence();    // proxy fence 使刚写入 CD SMEM 的 BF16 对 TMA async proxy 可见。
                    named_barrier_sync(NUM_EPILOGUE_THREADS, EPILOGUE_BAR_ID);  // 保证所有 128 个线程都已经填完自己的行

                    // 128 个线程填好 CD stage 后，仅一个线程发射 TMA Store。
                    if (epi_warp == 0 && elect_one()) {
                        const int32_t n_idx =
                            n_block * BLOCK_N + s * STORE_BLOCK_N;
                        const int32_t m_idx =
                            m_block * BLOCK_M + w * WAVE_BLOCK_M;

                        tma_store_2d(
                            smem_cd_base
                                + tma_store_stage * SMEM_CD_PER_STAGE,
                            &tma_d, n_idx, m_idx);
                        tma_store_commit();
                    }

                    tma_store_stage =
                        (tma_store_stage + 1) % NUM_TMA_STORE_STAGES;
                }
            }
        }

        // 最后离开 persistent loop 时，等待所有尚未完成的 TMA Store，确保 kernel 退出前 Global D 已写完。
        if (epi_warp == 0 && elect_one())
            tma_store_wait<0>();
    }
```


### 2.7 清理

```cpp
    // 每个 CTA 内先等所有角色退出 persistent loop，再做 cluster 集合。
    __syncthreads();
    cluster_sync();

    // cta_group::2 TMEM 需要两个 peer warp 协同释放。
    if (warp_idx == 2)
        tmem_dealloc_2sm(0, TMEM_COLS);
#endif
}
```

## 3. 几个需要单独展开的问题

### 3.1 Scheduler 为什么在每个角色分支中各建一个实例

CUDA 局部变量属于线程私有寄存器。不存在“warp 0 调一次 scheduler，然后 warp 1 自动拿到结果”这种共享：若要共享，就必须另外写入 SMEM 并同步，反而把三个独立角色重新串行化。

因此每个角色都维护一个便宜的、确定性的本地 scheduler：

```text
TMA scheduler：可能已经枚举到 tile 3，正在预取
MMA scheduler：可能正在 tile 1
EPI scheduler：可能仍在写 tile 0
```

它们在真实时间上的 `current_iter` 可以不同，但用同一公式枚举同一 tile 序列。`full/empty/tmem_*` barrier 保证消费者不会越过生产者。

从 C++ 语义说，可以把 `TileScheduler scheduler(...)` 的构造语句写到角色分支外；但每个线程得到的仍是独立实例，并没有变成“只调用一次”。这样还会让不需要 scheduler 的 warp 也持有相关状态，扩大变量 live range。放在分支里更准确地表达所有权。

### 3.3 为什么需要 M-wave；为什么每个 SM 一次只有 128 行

SM100 每个 CTA 的 TMEM 固定为 `128 lanes × 512 columns`，每个 cell 为 32 bit。`cta_group::2` 同时触及当前 CTA 和 peer CTA 的 TMEM，所以一次 2SM UMMA 的逻辑 `M=256` 可以理解成两边各产生 128 行本地结果：

```text
一次 wave：CTA 0 本地 128 行 + CTA 1 本地 128 行
```

但 Level 9 每个 CTA 自己要完成 256 行，因此每个 CTA 还要沿 A 的 M 方向做两次 wave：

```text
wave 0：本 CTA A[  0:128, :] → TMEM lanes 0..127, columns   0..255
wave 1：本 CTA A[128:256, :] → TMEM lanes 0..127, columns 256..511
```

第二个 wave 不是使用“TMEM 第 128～255 行”——本地并没有这些 lane。它复用同样的 128 条 lane，把结果放进另一组 columns。epilogue 用 `w * WAVE_BLOCK_M` 恢复全局输出行偏移。

这是 SM100 暴露的 TMEM/UMMA 数据通路组织，不是 tutorial 随意设定的常数。公开编程模型说明了 128 lanes，但没有公开晶体管级为什么选择 128；读代码时把它当作硬件固定维度即可。

### 3.4 为什么 `barrier_wait()` 后还要 `tcgen05_fence_after()`

它们回答不同的问题：

```text
barrier_wait：目标 phase 是否已经完成？生产者的数据/完成信号是否可观察？
fence_after：后续 tcgen05 指令能否被排到这个线程同步点之前？
```

本算子中的三处典型组合是：

```text
wait full_bar      → fence_after → UMMA       // TMA/SMEM 交给 Tensor Core
wait tmem_full     → fence_after → TMEM load  // UMMA/TMEM 交给 epilogue
wait tmem_empty    → fence_after → next UMMA  // epilogue 交还 TMEM
```

`fence_after` 通常不等待 UMMA 完成；UMMA 完成由 `commit + mbarrier` 跟踪。反过来，仅有 wait 也没有建立 tcgen05 pipeline 所要求的专用代码移动顺序。因此两者不是重复同步。


## 4. 从 Level 2 到 Level 9：优化主线

只保留理解最终代码所需的演进：

| Level | 关键变化 | 最终留下的思想 |
|---|---|---|
| 2 | 2-stage A/B SMEM，`full/empty`，2SM TMA completion 路由 | TMA 与 UMMA 用细粒度 barrier 交接 |
| 3 | TMA/MMA 各自独立循环 | 真正的 warp specialization |
| 4 | TMEM→SMEM 后合并写回 | epilogue 先重排数据再写 Global |
| 5 | persistent scheduler + 2D swizzle | 少量常驻 CTA 处理多 tile，提升 L2 reuse |
| 6 | TMA/MMA/epilogue 三角色、M-wave、TMA Store | 计算与写回重叠 |
| 7 | cluster 两 CTA 计算不同 tile | 消除 2SM 重复计算 |
| 8 | swizzled CD、ASAP `tmem_empty` | 减少 bank conflict，尽早归还 TMEM |
| 9 | `256×256` tile、4 stage、descriptor shuffle、256-thread TMEM release | 与 DeepGEMM 的大矩阵路径接近 |

性能不是由某一条 UMMA 指令单独决定的。Level 9 的核心是让四条数据通路各自连续工作：TMA Load、Tensor Core、TMEM epilogue、TMA Store；barrier/fence 只建立最小必要依赖。

## 5. 附录：descriptor、helper 与 host launch

### 5.1 UMMA 的两个 descriptor

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

### 5.2 TMA tensor map 的创建

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

### 5.3 关键 helper 的高层含义

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

```text
SMEM destination：保持调用 CTA 的本地地址
completion barrier：通过 peer-bit/mapping 路由到 leader CTA
```

若把 peer mask 错误地应用到 destination，数据会被写进错误 CTA 的 SMEM。

### 5.4 Launch 为什么是 persistent cluster kernel

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

grid 只提供常驻工作者，不是一 tile 一 CTA。每个 CTA 在 `TileScheduler` 的 while-loop 中处理多个 tile；TMEM 和 barrier 只初始化一次，phase 跨 tile 连续推进。
