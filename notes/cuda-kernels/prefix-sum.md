---
title: "Prefix Sum"
order: 2
---

# Prefix Sum

## 任务定义

对一维数组求前缀和，每个输出包含当前位置及其之前的全部输入。例如，输入 `[1, 2, 3, 4]`，输出 `[1, 3, 6, 10]`。这种包含当前位置的定义称为 **inclusive scan**；若只包含当前位置之前的输入，则称为 **exclusive scan**，同一输入的输出为 `[0, 1, 3, 6]`。本文实现 inclusive scan。

- 输入：GPU 显存中的连续 FP32（`float`）数组 `x[n]`，`n >= 1`；输入为有限数，计算过程不溢出 FP32。
- 输出：GPU 显存中的连续 FP32 数组 `y[n]`，其中 `y[i]` 是 `x[0]` 到 `x[i]` 的和；输入保持不变，输入与输出不重叠。
- 精度：累加器、中间结果与输出均使用 FP32，允许不同加法顺序产生舍入差异。以同一输入的 FP64 前缀和为参考，逐位置比较绝对误差。

代码使用 CUDA C++，`x`、`y` 均已分配。先用单个 block 讲清块内扫描，再扩展到任意长度；块内版本要求 `1 <= n <= 256`，统一启动 256 个线程。

## 最朴素：一个线程顺序累加

一个线程从左到右读取输入，每累加一项就写出一个前缀和。它可以处理任意长度，也直接体现了任务定义。

```cpp
__global__ void scan_serial(const float* x, float* y, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
        y[i] = sum;
    }
}

scan_serial<<<1, 1>>>(x, y, n);
```

所有加法串在同一条依赖链上，只有一个线程工作。下一步将不同位置的前缀并行计算。

## Shared memory 扫描：每轮将覆盖范围翻倍

让每个线程维护一个位置的部分和。第一轮加上左边相邻位置的值，第二轮加上左边距离 2 的部分和，第三轮加上距离 4 的部分和。每轮合并两个相邻区间，覆盖范围随之翻倍；处理 256 项需要 8 轮。这种算法称为 Hillis–Steele scan。

<PrefixSumDiagram view="doubling" />

```cpp
__global__ void scan_shared(const float* x, float* y, int n) {
    constexpr int B = 256;
    __shared__ float s[B];
    int tid = threadIdx.x;

    // 越界位置补 0，所有线程继续参加后续同步。
    s[tid] = tid < n ? x[tid] : 0.0f;
    __syncthreads();  // 等初始值全部写入。

    for (int offset = 1; offset < B; offset *= 2) {
        float left = tid >= offset ? s[tid - offset] : 0.0f;
        __syncthreads();  // 所有线程读完旧值后，才能覆盖 s。
        s[tid] += left;
        __syncthreads();  // 等本轮写完，再读取下一轮的输入。
    }

    if (tid < n)
        y[tid] = s[tid];
}

// 本节处理 1 <= n <= 256。
scan_shared<<<1, 256>>>(x, y, n);
```

每轮大部分线程都做一次加法。对块容量 `B`，加法总量随 `B log₂B` 增长；并行轮数减少了，却重复计算了许多区间的和。

## 树形扫描：先求区间和，再分发前缀

重复计算来自不同位置分别合并相互重叠的区间。**Blelloch scan 先让每个树节点求一次子树总和，再把子树之前的和向下传递**，将加法总量降到与块容量成正比。

向下传递时，左子树继承父节点的前缀；右子树的前缀还要加上左子树总和。根节点之前没有输入，因此前缀置为 0。传到叶子时得到 exclusive scan，再各自加上原始输入，得到 inclusive scan。

<PrefixSumDiagram view="tree" />

代码用 shared memory 数组保存树节点：一个区间的总和存到该区间最右端的位置。

```cpp
__global__ void scan_tree(const float* x, float* y, int n) {
    constexpr int B = 256;
    __shared__ float s[B];
    int tid = threadIdx.x;
    float original = tid < n ? x[tid] : 0.0f;
    s[tid] = original;
    __syncthreads();  // 等叶子值写完。

    // 向上：合并左右区间，总和写入右端。
    for (int stride = 1; stride < B; stride *= 2) {
        int right = (tid + 1) * (2 * stride) - 1;
        if (right < B)
            s[right] += s[right - stride];
        __syncthreads();  // 下一层使用这一层的区间和。
    }

    if (tid == 0)
        s[B - 1] = 0.0f;  // 根的前缀为 0。
    __syncthreads();

    // 向下：左子树继承父前缀，右子树再加上左子树总和。
    for (int stride = B / 2; stride > 0; stride /= 2) {
        int right = (tid + 1) * (2 * stride) - 1;
        if (right < B) {
            float left_sum = s[right - stride];
            float parent_prefix = s[right];
            s[right - stride] = parent_prefix;
            s[right] = parent_prefix + left_sum;
        }
        __syncthreads();  // 等这一层前缀写完，再向下传递。
    }

    if (tid < n)
        y[tid] = s[tid] + original;  // exclusive → inclusive。
}

// 本节处理 1 <= n <= 256。
scan_tree<<<1, 256>>>(x, y, n);
```

对 256 项，向上和向下各有 8 轮。加法总量减少了，但每轮仍读写 shared memory，并同步整个 block。

## Warp shuffle：先扫 warp，再连接各 warp

前面的实现每轮通过 shared memory 交换数据。Warp 内可以用 shuffle 直接传递寄存器中的值，按距离 1、2、4、8、16 完成扫描。每个 warp 只处理固定的 32 项。

一个 256 线程 block 有 8 个 warp。它们先各自扫描，把各自总和交到 shared memory；warp 0 扫描这 8 个总和后，每个 warp 将**前面所有 warp 的总和**加到自己的局部前缀上。这样只需要两次 block 同步。

```cpp
__device__ __forceinline__ float warp_scan(float v) {
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        // 32 个 lane 全部执行 shuffle，取回左边 offset 处的旧值。
        float left = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset)
            v += left;  // 左边没有对应位置的 lane 保留自身值。
    }
    return v;
}

__global__ void scan_shuffle(const float* x, float* y, int n) {
    constexpr int B = 256;
    __shared__ float warp_prefix[B / 32];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;

    float v = tid < n ? x[tid] : 0.0f;
    v = warp_scan(v);
    if (lane == 31)
        warp_prefix[warp] = v;  // warp 最后一个 lane 持有该 warp 总和。
    __syncthreads();  // 等 8 个 warp 交出总和。

    if (warp == 0) {
        float total = lane < B / 32 ? warp_prefix[lane] : 0.0f;
        total = warp_scan(total);  // 其余 24 个 lane 补 0 并参与 shuffle。
        if (lane < B / 32)
            warp_prefix[lane] = total;
    }
    __syncthreads();  // 等 warp 总和的前缀全部写完。

    if (warp > 0)
        v += warp_prefix[warp - 1];
    if (tid < n)
        y[tid] = v;
}

// 本节处理 1 <= n <= 256。
scan_shuffle<<<1, 256>>>(x, y, n);
```

Shared memory 只保存 8 个 warp 总和，warp 内交换全部在寄存器之间完成。这个 kernel 仍只处理一个 block；更长的输入还需要连接不同 block 的结果。

## 多 block：扫描块总和，再加回偏移

将输入按连续的 256 项分块，各 block 独立扫描。此时每块结果只包含块内输入，缺少前面所有块的总和。先收集并扫描各块总和，就能得到每块应加的偏移。

<PrefixSumDiagram view="blocks" />

若块总和数组本身超过 256 项，就继续按相同方式分块，直到只需一个 block。逐层返回时，再将偏移加回对应层的结果。下面是完整实现：

```cpp
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_scan_multi(float v) {
    int lane = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        // 每个 warp 的 32 个 lane 全部参加，越界输入已补 0。
        float left = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset)
            v += left;
    }
    return v;
}

__global__ void scan_blocks(const float* x, float* y,
                            float* block_sums, int n) {
    constexpr int B = 256;
    __shared__ float warp_prefix[B / 32];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    size_t i = size_t(blockIdx.x) * B + tid;

    float v = i < size_t(n) ? x[i] : 0.0f;
    v = warp_scan_multi(v);
    if (lane == 31)
        warp_prefix[warp] = v;
    __syncthreads();  // 等所有 warp 的总和写完。

    if (warp == 0) {
        float total = lane < B / 32 ? warp_prefix[lane] : 0.0f;
        total = warp_scan_multi(total);
        if (lane < B / 32)
            warp_prefix[lane] = total;
    }
    __syncthreads();  // 后续线程需要读取前面 warp 的总和。

    if (warp > 0)
        v += warp_prefix[warp - 1];
    if (i < size_t(n))
        y[i] = v;
    // 尾部补 0，所以线程 255 即使越界，其前缀仍是整块总和。
    if (tid == B - 1 && block_sums != nullptr)
        block_sums[blockIdx.x] = v;
}

__global__ void add_block_prefix(float* y, const float* scanned_sums,
                                 int n) {
    constexpr int B = 256;
    size_t i = size_t(blockIdx.x) * B + threadIdx.x;
    if (blockIdx.x > 0 && i < size_t(n))
        y[i] += scanned_sums[blockIdx.x - 1];
}

// 返回 workspace 所需的 float 数量；n >= 1。
size_t scan_workspace_floats(int n) {
    size_t count = 0;
    while (n > 256) {
        int blocks = 1 + (n - 1) / 256;
        count += size_t(2) * blocks;
        n = blocks;
    }
    return count;
}

// Host 函数：workspace 与 x、y 不重叠；仅提交工作，返回时不等待 GPU 完成。
void scan_hierarchical(const float* x, float* y, int n,
                       float* workspace, cudaStream_t stream) {
    int blocks = 1 + (n - 1) / 256;
    if (blocks == 1) {
        scan_blocks<<<1, 256, 0, stream>>>(x, y, nullptr, n);
        return;
    }

    float* sums = workspace;
    float* scanned_sums = workspace + blocks;
    float* next_workspace = workspace + size_t(2) * blocks;

    // 同一 stream 按启动顺序执行：块内扫描 → 块总和扫描 → 加回偏移。
    scan_blocks<<<blocks, 256, 0, stream>>>(x, y, sums, n);
    scan_hierarchical(sums, scanned_sums, blocks, next_workspace, stream);
    add_block_prefix<<<blocks, 256, 0, stream>>>(y, scanned_sums, n);
}

// 预先分配 workspace：容量至少为 count * sizeof(float) 字节。
// count 为 0 时可传 nullptr；x、y、workspace 均保留到 stream 执行完成。
size_t count = scan_workspace_floats(n);
scan_hierarchical(x, y, n, workspace, stream);
```

每一层的临时空间包含两段等长数组：各块总和 `sums`，以及它们的 inclusive 前缀 `scanned_sums`；后面紧接递归层的临时空间。例如，`n = 100000` 时，先产生 391 个块总和，再产生 2 个；共需 `2 × 391 + 2 × 2 = 786` 个 `float`。

该实现将跨 block 的依赖放在同一 stream 的多次 kernel 启动之间。代价是写出局部结果后，还要读取并加回块偏移，多了一遍输出数组的读写。
