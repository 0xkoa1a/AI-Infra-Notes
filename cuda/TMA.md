# TMA

TMA 位于 SM 内部，专门负责在 Shared Memory 和 Global Memory 之间高效、异步搬运大块数据，从而使 SM 线程不必亲自搬运数据。

---

以一个矩阵乘法 C[M, N] = A[M, K] × B[K, N] 为例。

一次 TMA load 的输入包括：
- A/B 矩阵的 TMA descriptor
- 此次搬运的 A/B 矩阵的坐标
- Shared Memory 中的目标地址
- 用于同步的 mbarrier

TMA descriptor（Tensor Map） 描述 A/B 矩阵的:
- global memory 基址
- shape,
- 每个维度的 stride,
- data type,
- 每次 TMA load 的 tile shape
- SMEM swizzle 方式
- 越界处理方式

一次 TMA load 会从 global memory 中 A/B 矩阵的指定坐标处，搬运一个 tile 到 Shared Memory 中的指定地址，并在完成时更新 mbarrier 中的 tx_count。TMA load 是异步的，线程可以在发射 TMA load 后继续执行其他计算。

---

一次 TMA store 的输入包括：
- C 矩阵的 TMA descriptor
- 此次搬运的 C 矩阵的坐标
- Shared Memory 中的源地址

它会从 Shared Memory 中的指定地址，将一个 tile 搬运到 global memory 中 C 矩阵的指定坐标处。TMA store 同样是异步的。

和 load 的不同在于，store 不用 mbarrier 保证同步。[TODO]: 如何同步？

> 如果 shared memory 是 CUDA 线程刚写好的，发起 TMA Store 前通常需要：
> `fence.proxy.async.shared`
> 以确保普通线程的 SMEM 写入对 TMA 的 async proxy 可见。