# Hopper H100 MatMul 优化：核心 Takeaway

这篇文章最核心的 takeaway 是：

> 在 H100 上，高性能矩阵乘法的关键已经不只是“让更多线程做更多乘加”，而是把整个 kernel 组织成一条持续运转的异步流水线：Tensor Core 负责计算，TMA 负责搬运，warp specialization 负责分工，shared memory 和寄存器负责缓冲，软件调度负责维持 L2 局部性。
>
> 真正的优化目标，是让数据持续、及时地喂给 Tensor Core，同时避免寄存器、shared memory、同步和缓存成为新的瓶颈。

作者最终在一个特定 benchmark 上从 32 TFLOPS 做到 764 TFLOPS，达到当时 cuBLAS 的约 107%。但文章的价值并不是“手写 kernel 普遍优于 cuBLAS”，而是完整展示了怎样针对 Hopper 架构逐层识别并消除瓶颈。

## 1. 首先理解作者到底在计算什么

矩阵乘法是：

$$
C=A\times B
$$

输出矩阵 $C$ 中的每个元素，都是 $A$ 的一行与 $B$ 的一列做点积。

GPU 不会一次处理整个大矩阵。一般会把 $C$ 切成许多二维小块，即 output tiles。一个 thread block 负责其中一个 tile。

要计算一个 $C$ tile，需要读取：

- $A$ 的一块行面板；
- $B$ 的一块列面板。

因为完整的 K 维数据太大，不能一次装进片上存储，所以还要沿 K 维分段：

1. 从显存取一小段 A、B；
2. 用它们更新输出 tile；
3. 再取下一段；
4. 重复累加，直到遍历完整个 K 维；
5. 最后把结果写回显存。

因此，一个 GEMM kernel 的主体其实是：

$$
\text{反复 Load A/B} \rightarrow \text{Compute} \rightarrow \text{最后 Store C}
$$

文章后面的所有优化，本质上都是在优化这条数据流。

## 2. 初学者需要的 Hopper 心智模型

可以把一个 H100 的 SM 想象成一座小工厂。

| Hopper 部件 | 工厂类比 | 在文章中的作用 |
|---|---|---|
| HBM/global memory | 远处的大仓库 | 存放完整的 A、B、C，容量大但访问成本高 |
| L2 cache | 所有工厂共享的中转仓 | 缓存不同 SM 可能重复使用的 A、B 数据 |
| Shared memory | 单个工厂内的货架 | 暂存即将参与计算的 A、B tile |
| Registers | 工人手边的工作台 | 保存正在累加的 C tile，最快但非常有限 |
| Tensor Core | 专用矩阵乘法生产线 | 以很高吞吐率完成小矩阵乘加 |
| TMA | 自动搬运设备 | 在显存和 shared memory 之间异步搬运二维 tile |
| Warp group | 一组协作工人 | Hopper 上由 4 个 warp、共 128 个线程组成 |
| Thread-block cluster | 相邻工厂组成的协作组 | 允许多个 SM 同步并共享或广播数据 |

Hopper 的特殊之处，是拥有一组能够相互配合的异步硬件：

- WGMMA 让一个 warp group 协作驱动 Tensor Core；
- TMA 独立进行大块、多维数据传输；
- 异步 barrier 协调生产者与消费者；
- thread-block cluster 进一步把协作范围扩展到多个 SM。

TMA 可以在不占用大量普通线程和寄存器的情况下搬运多维 tensor，使数据传输与计算重叠；cluster 则为多个 SM 之间的数据协作增加了一层新的执行层级。

所以，在 Hopper 上写高性能 kernel，越来越像设计一个并行数据处理流水线，而不只是编写一组执行算术的线程。

## 3. 第一大飞跃：必须使用正确的计算单元

作者最初使用传统 CUDA Core 方式做 BF16 矩阵乘法，只得到约 32 TFLOPS。换成 Hopper Tensor Core 后，性能一下提升到 317 TFLOPS，接近 10 倍。

这告诉我们：

> 对于现代 GPU，选择正确的硬件执行路径，通常比低层次地优化普通算术代码重要得多。

Tensor Core 不是“更快的普通乘法器”。它是专门完成小矩阵乘加的硬件。一次操作会让 128 个线程协作，把一小块 A 和 B 相乘，并把结果累加到分布在这些线程寄存器中的 C。

因此：

- A、B 必须以 Tensor Core 要求的布局放进 shared memory；
- 输出累加值必须长期留在寄存器中；
- 线程之间必须按照 warp group 的粒度协作；
- 必须连续提交足够多的矩阵操作，才能填满 Tensor Core 流水线。

TMA 在这里非常重要，因为它既能搬运二维 tile，又能直接生成 Tensor Core 需要的 shared-memory 布局。否则，大量线程会被消耗在地址计算、数据重排和普通 load 上。

## 4. 第二大飞跃：提高数据复用，而不是只增加计算

作者随后把每个 thread block 负责的 output tile 扩大，性能从 317 提升到 423 TFLOPS。

原因是更大的 tile 可以提高 arithmetic intensity，即“每搬运一个字节，能够进行多少次计算”。

例如，同一块 A 可以和更宽的 B 区域相乘，同一块 B 也可以服务更多输出行。这样，一次从显存取回的数据可以被重复使用更多次。

但更大的 tile 不是免费午餐：

- 更多输出元素需要更多 accumulator registers；
- 更大的 A、B tile 需要更多 shared memory；
- 每个 block 使用的资源变多，能够同时驻留的 block 可能减少；
- 每个线程的寄存器超过上限时，会发生 register spilling。

文章中有一个特别有教育意义的失败：直接把 tile 扩大后，性能不是上升，而是从约 500 TFLOPS 跌到 123 TFLOPS。

原因是每个线程需要保存太多输出累加值，超过了单线程寄存器容量。部分状态被迫溢出到更慢的存储层，而且原本可以并行提交的 Tensor Core 操作被串行化。

作者的解决方案是增加 consumer warp group，把大 output tile 分给两组消费者计算。总寄存器需求没有消失，但被分散到了更多线程上。随后再把生产者不需要的寄存器额度让给消费者，最终达到约 631 TFLOPS。

这里最重要的认识是：

> GPU 资源限制不仅有“整个 SM 一共有多少”，还有“单个线程能使用多少”。总量足够，不代表分配方式合理。

这也是为什么 occupancy、线程数和性能之间没有简单的单调关系。

## 5. 最核心的 Hopper 技巧：把 Load 和 Compute 重叠起来

最初的执行过程是串行的：

```text
加载第 1 块 → 计算第 1 块 → 加载第 2 块 → 计算第 2 块 → …
```

Tensor Core 很快，所以它经常计算完后等待下一块输入。此时理论计算能力再高也没有意义。

作者将线程分成不同角色：

- producer warp group：使用 TMA 预取后续的 A、B tile；
- consumer warp group：使用 Tensor Core 处理已经到达的数据；
- shared-memory circular buffer：存放多个处于不同阶段的 tile；
- barrier：标记某个缓冲槽是“已填满”还是“已消费”。

于是执行过程变成：

```text
Producer: Load 1 | Load 2 | Load 3 | Load 4 ...
Consumer:        Compute 1 | Compute 2 | Compute 3 ...
```

这叫 latency hiding。

它没有让一次内存访问本身变快，而是让内存访问发生时，Tensor Core 同时进行有用计算。只要 producer 能够及时供货，consumer 几乎感受不到 load latency。

性能由此从 423 提升到 498 TFLOPS。

这是整篇文章最通用的性能思想：

> 高性能系统往往不是把每一步都做到最快，而是让不同硬件单元同时工作，使慢步骤不再出现在关键路径上。

## 6. 优化范围从单个 block 扩展到整个 GPU

完成单个 SM 内部的流水线后，作者发现还可以在一个 tile 写回 C 时，提前加载下一个 tile 的输入。

为此，kernel 采用了类似 persistent scheduling 的结构：大致只启动与 SM 数量相当的长期运行 block，每个 block 在同一个 SM 上依次处理多个 output tile。这样软件可以直接控制每个 SM 接下来处理哪个 tile。

第一次尝试反而让性能从约 640 跌到 400 TFLOPS。

原因不是单个 SM 内部的流水线出错，而是全 GPU 的 tile 调度破坏了 L2 cache 局部性：

- 同一时间运行的 SM 在访问相距很远的 output tiles；
- 它们需要的 A、B 区域几乎没有重叠；
- L2 中的数据无法被其他 SM 重用；
- HBM 流量增加。

作者重新安排 tile，让同时执行和相继执行的 tiles 在二维矩阵上尽量接近。相邻 output tile 通常会共享 A 或 B 的一部分，因此一个 SM 从 HBM 取回的数据可以被其他 SM 从 L2 复用。

重新调度后，性能提升到 660 TFLOPS，文中测得 L2 hit rate 达到 83%，高于该 benchmark 下 cuBLAS 的 70%。

甚至只使用 128 个 SM 比使用全部 132 个 SM 更快，因为 128 能更规则地匹配 tile 分组，形成更好的缓存复用。

这给出了一个很反直觉、但非常重要的结论：

> “使用更多 SM”不一定等于“更快”。如果额外并行度破坏了数据局部性，少用一点计算资源反而可能提高整体吞吐率。

## 7. Hopper 的跨 SM 协作

相邻 output tiles 可能需要同一块 A 或 B。即使 L2 命中，从 L2 向两个 SM 分别传输两次，仍然存在重复工作。

Hopper 的 thread-block cluster 可以把相邻 SM 上的 block 组织起来，再使用 TMA multicast：

- 某个输入 tile 只取一次；
- 然后同时发送到 cluster 中多个 SM 的 shared memory；
- 多个 SM 各自计算不同的 output tile。

作者使用两个相邻 block 组成 cluster，将性能从 704 提升到 734 TFLOPS，首次超过该测试中的 cuBLAS。

但更大的 cluster 反而更慢，因为跨 SM 同步和资源约束也会增长。因此，cluster 的意义不是“尽量把更多 SM 绑在一起”，而是：

> 只在确实存在高价值数据复用时，建立最小、最便宜的跨 SM 协作关系。

## 8. 后半段为什么只剩小幅提升

当主要瓶颈都被解决之后，作者继续优化：

- 降低频繁 barrier 的同步开销；
- 调整写回顺序，让相邻线程写入相邻地址；
- 避免输出写回污染原本用于缓存 A、B 的 L1/L2；
- 避免不必要地清零 accumulator registers；
- 用 TMA 异步写回结果；
- 用 Hilbert curve 安排 tile 顺序，使跨 tile group 的访问也保持空间局部性。

性能逐步从 704 提升到 764 TFLOPS，但每次收益越来越小。

大致的性能演进如下：

| 阶段 | 文中性能 | 解决的问题 |
|---|---:|---|
| 普通 CUDA Core baseline | 32 TFLOPS | 尚未使用专用矩阵硬件 |
| Tensor Core + TMA | 317 | 计算路径不匹配 H100 |
| 更大的 output tile | 423 | 输入数据复用不足 |
| Producer/consumer 流水线 | 498 | Load latency 暴露 |
| 两组 consumer + 寄存器重分配 | 631 | 单线程寄存器压力 |
| L2-aware 全局调度 | 660 | SM 之间缺少缓存局部性 |
| 更轻量的同步 | 704 | 高频 barrier 开销 |
| Cluster + TMA multicast | 734 | 跨 SM 的重复数据传输 |
| 写回、异步 store、Hilbert 调度 | 764 | 剩余的小型开销 |

最值得注意的是，优化过程中发生过多次严重性能倒退。作者不是靠直觉一次写出最终 kernel，而是不断：

1. profile；
2. 找到当前最显著的瓶颈；
3. 提出一个硬件层面的解释；
4. 修改设计；
5. 重新测量；
6. 接受某些直觉是错的。

这其实比任何单个优化技巧都更值得学习。
