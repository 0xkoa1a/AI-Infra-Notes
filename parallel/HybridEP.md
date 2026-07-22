
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