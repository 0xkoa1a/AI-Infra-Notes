---
title: "TMA"
order: 3
---

# TMA

TMA 位于 SM 内部，专门负责在 Shared Memory 和 Global Memory 之间高效、异步搬运大块数据，从而使 SM 线程不必亲自搬运数据。

***

## TMA Load

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

一次 TMA load 会从 global memory 中 A/B 矩阵的指定坐标处，搬运一个 tile 到 Shared Memory 中的指定地址。TMA load 是异步的，线程可以在发射后继续执行其他工作。

在循环开始前可以通过 `prefetch.tensormap` 预取 TMA descriptor。

TMA Load 使用 mbarrier 跟踪完成状态：
- 发射线程先执行 `arrive_expect_tx(bytes)`，表明自己已到达，并登记本次 TMA load 的字节数。然后发射绑定到同一个 mbarrier 的 TMA load。
    - 必须先登记 expected bytes，再发射异步搬运，避免搬运在事务登记之前完成。
- 本次 TMA load 完成时，硬件将相应字节数从 barrier 的 `tx_count` 中减去。
    - 当 `pending_arrival_count == 0` 且 `tx_count == 0` 时，当前 phase 才完成。
- 消费线程对当前 phase 执行 `wait`。
    - 带 acquire 语义的 wait 还保证 TMA 写入的 SMEM 数据随后可以被普通 CUDA load 正确读取。

TMA 通过 async proxy 写 SMEM，CUDA 线程通过 generic proxy 读取 SMEM。TMA load 完成时会建立规定的 generic-async proxy 可见性，因此消费者正确等待对应的 mbarrier phase 后，不需要再额外执行一次 `fence.proxy.async.shared::cta`。这里不能用普通 CTA barrier 代替 mbarrier wait：CTA barrier 只能等待 CUDA 线程集合，不能判断 TMA 是否已经完成搬运。

在多 stage 流水中，每个 SMEM stage 对应循环使用的 mbarrier phase。生产者向某个 stage 发射新的 TMA load 之前，必须先确认上一次使用该 stage 的消费者已经读完；消费者则等待本轮 load 对应的 phase 完成后才能读取。消费结束后再通知生产者该 stage 已经可以复用。

完整 TMA load 流程如下：

```text
初始化并发布 mbarrier
→ 可选地预取 TMA descriptor
→ 等待目标 SMEM stage 可以复用
→ 发射线程执行 arrive_expect_tx(bytes)
→ 发射绑定到该 mbarrier 的 TMA Load
→ TMA 搬运数据并执行 complete_tx(bytes)
→ arrival count 和 tx_count 同时归零，当前 phase 完成
→ 消费线程 wait/acquire 对应 phase
→ 消费线程读取 SMEM
→ 消费结束后通知生产者复用该 stage
```

***

## TMA Store

一次 TMA store 的输入包括：
- C 矩阵的 TMA descriptor
- 此次搬运的 C 矩阵的坐标
- Shared Memory 中的源地址

它会从 Shared Memory 中的指定地址，将一个 tile 搬运到 global memory 中 C 矩阵的指定坐标处。TMA store 同样是异步的。

在 TMA store 发射之前，每一个写过源 SMEM 的线程需要执行 `fence.proxy.async.shared::cta`，把本线程此前的 SMEM 写排在后续 TMA 读取之前。然后所有这些生产线程需要通过 barrier 同步。这是为了确保 TMA 读取源 SMEM 时，能够看到所有生产线程此前写入的结果（CUDA 线程写 SMEM 通过 generic proxy，TMA 读取 SMEM 通过 async proxy，弱内存模型不保证它们之间的内存顺序）

TMA Store 使用线程私有的 **bulk async-group**；发射线程通过 `commit_group` 和 `wait_group` 跟踪完成状态。
- `cp.async.bulk.commit_group` 把当前线程此前发射但尚未提交的 bulk async 操作归入一个新的 bulk async-group。**N stage 的 TMA store 可以使用 `wait_group(N - 1)`** 同步。
- `cp.async.bulk.wait_group N` 等待，直到至多只剩最近的 `N` 个 group 仍未完成。（`N=0` 表示等待所有 group 完成。）

bulk async-group 是 **per-thread** 的：哪个线程发射并提交 TMA Store，就应由哪个线程等待这些 group。其他线程不能直接用自己的 `wait_group` 等待该发射线程的 group；如果它们需要知道 SMEM stage 已经可复用，应由发射线程完成 wait，再通过 barrier 通知它们。

> 普通的 `wait_group` 等待完整完成，包括读完源 SMEM、写完目标 Global Memory，以及结果对发射线程可见。带 `.read` 的形式只等待 TMA 读完源地址：
> 读完成之后，已经可以安全复用源 SMEM
> 但在 kernel 结束前确认最终 Global Memory 写入完成时，应等待完整完成。

完整 TMA store 流程如下：

```text
所有生产线程写 SMEM
→ 每个生产线程执行 async proxy fence
→ 所有生产线程集合
→ 单线程发射 TMA Store
→ 同一线程 commit bulk async-group
→ 同一线程 wait 到旧 group 已读完源 SMEM
→ 线程集合后复用相应 SMEM stage
```
