---
title: "DeepGEMM MegaMoE：从 token 路由到通信与计算流水线"
order: 9
---

# DeepGEMM MegaMoE：从 token 路由到通信与计算流水线

MegaMoE 把一次 MoE 前向中的 Dispatch、两次矩阵乘法、SwiGLU 和 Combine 放进常驻 GPU kernel。**它将依赖细化到 token block、输出列块和输入 K-block，让数据搬运、矩阵计算和结果回传按各自的进度推进。** 读懂这份实现，需要同时追踪数据放在哪里、谁能消费它，以及什么时候允许覆盖缓冲。

本文以 DeepGEMM 官方仓库 commit [`057ca596`](https://github.com/deepseek-ai/DeepGEMM/tree/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7)（2026-09-30 发布）的 SM100 FP8/FP4 路径为主，覆盖 FP8 激活配 FP4 或 FP8 权重。读者应熟悉 CUDA、MoE 与 EP；TMA、TMEM 和 UMMA 的基本路径可参照 [Blackwell MatMul 精读](./blackwell-matmul.md)。本文就近解释 MegaMoE 新增的分工与依赖，源码位置统一固定到这一版本。

## 先看一个 token 怎样往返

以两 GPU、四专家、top-k=2 为教学例子：GPU 0 保存专家 E0、E1，GPU 1 保存 E2、E3。GPU 0 的一个 token 选择 E0 和 E2。两个专家分别计算，结果回到 GPU 0 的两个槽位，最后相加。

<MegaMoEDiagram view="flow" />

图中只追踪一个 token。实际 kernel 将同一专家接收的多条 token 组成矩阵，并且两 GPU 同时处理各自的输入和专家任务。

单个专家先把输入投影到 gate、up 两份特征，经过 SwiGLU 后，再投影回 hidden 维度。为对齐代码，本文把这两次投影称为 **Linear1（L1）** 和 **Linear2（L2）**。

- $P$：EP rank 数；本文按每 GPU 一个 rank 解释。
- $E$：routed expert 总数，每 rank 持有 $E/P$ 个连续编号的专家。
- $T_r$：rank $r$ 的输入 token 数，$T_{\max}$ 是预分配容量。
- $H$：hidden 维度；$F$：每个专家的 intermediate hidden 维度。
- $k$：每个 token 的路由槽数；$\alpha_{t,s}$：token $t$ 在槽 $s$ 上的路由权重。
- $m_e$：专家 $e$ 实际接收的 token 数，包含来自不同 rank 的路由项。

| 数据 | 逻辑形状 | 用途 |
|---|---|---|
| 本 rank 的输入 $X_r$ | $[T_r,H]$ | token 原始位置 |
| 专家编号、路由权重 | 均为 $[T_r,k]$ | 确定去向和贡献权重 |
| 专家收到的 $X_e$ | $[m_e,H]$ | L1 输入 |
| 专家权重 $W_{1,e}$ | $[2F,H]$ | 同时产生 gate、up |
| L1 输出 $G_e,U_e$ | 各为 $[m_e,F]$ | SwiGLU 的两路输入 |
| L2 输入 $A_e$ | $[m_e,F]$ | 加权后的 SwiGLU 输出 |
| 专家权重 $W_{2,e}$ | $[H,F]$ | 投影回 hidden 维度 |
| 专家结果 $Z_e$ | $[m_e,H]$ | 回传到源 token 的路由槽 |
| 本 rank 最终输出 $Y_r$ | $[T_r,H]$ | 沿 top-k 槽求和 |

这里的形状描述逻辑元素，FP4 权重的物理存储会将两个 4-bit 值打包到一个字节；scale 另行存储。

### 路由权重放在第二次 GEMM 之前

实现把路由权重乘到 SwiGLU 输出上，再执行 L2。同一专家收到的路由权重组成列向量 $\boldsymbol{\alpha}_e$，形状为 $[m_e,1]$。暂时省略量化，有：

$$
\begin{aligned}
[G_e,U_e]&=X_eW_{1,e}^{\mathsf T},\\
A_e&=\boldsymbol{\alpha}_e\odot\bigl(\operatorname{SiLU}(G_e)\odot U_e\bigr),\\
Z_e&=A_eW_{2,e}^{\mathsf T}.
\end{aligned}
$$

$\operatorname{SiLU}(g)=g/(1+e^{-g})$，$\odot$ 表示逐元素相乘。每行的路由权重沿 F 轴广播。在实数运算下，线性投影允许把每行的这个标量从 L2 输出侧移到输入侧；实际实现还包含舍入和量化，改变乘权重的位置会改变数值路径。

L1 的 FP32 累加结果先转为 BF16 gate/up，执行可选 clamp 和 SwiGLU，再乘路由权重、量化成 FP8。L2 以 FP32 累加，回传 BF16；最终 top-k 求和又使用 FP32 寄存器，写出 BF16。理解正确性时要对齐这一顺序。[L1/L2 epilogue 源码](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1057)

> Gate/up 权重按每组 8 行交错排列。这样，epilogue 读取一个局部累加块就能同时拿到对应的 gate/up，生成 F 维中间结果的一个片段。权重和 scale 在执行前一起变换，详见 [权重布局变换](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/mega/__init__.py#L110)。

## 启动前：把远端数据变成可访问的地址

### Symmetric memory 承担什么工作

每 rank 分配相同布局的缓冲，其中包含输入、路由信息、中间数据、回传槽和同步信号。PyTorch 的 `rendezvous` 交换并建立 peer buffer 的可访问映射。Kernel 接收各 rank 的基址；已知本地某个字段的地址，就能通过基址差得到远端对应字段的地址。[Python 缓冲准备](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/mega/__init__.py#L45)、[设备侧地址映射](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/layout/sym_buffer.cuh#L17)

“对称”指各 rank 的分配与布局彼此对应，各缓冲的虚拟地址可以不同。通过映射地址，GPU 线程可以直接读写 peer memory；数据跨 GPU 时仍需要互联传输，并且消费者仍需等待正确的完成信号。[PyTorch symmetric memory 说明](https://docs.pytorch.org/docs/main/symmetric_memory.html)

> NVSHMEM 也提供 GPU 发起通信的接口。本文的实现使用 PyTorch symmetric memory 和设备侧 peer load/store，阅读它时关注地址映射与同步即可。

Host 还会准备三类信息：

- 将输入、输入 scale、专家编号和路由权重放入约定的缓冲视图；Router 的打分与 top-k 选择在此次 kernel 之前完成。
- 变换专家权重和 scale，创建 TMA tensor map，描述张量的地址、形状和 swizzle。
- 根据 token 数和模型形状选择分块、共享内存 stage 数等参数，通过 DeepJIT 编译对应的 kernel 实例。

用 $B_M$、$B_N$、$B_K$ 分别表示一次分块计算中的 token 行数、输出列数和输入特征数。在本路径中，$B_N=128$、$B_K=128$；$B_M$ 随预计每专家 token 数选择。每 GPU 的 grid 包含与 SM 数相同数量的 CTA，两个 CTA 组成一个 cluster。Kernel 内的软件 grid barrier 依赖这些常驻 CTA 能继续推进，因此资源和 launch 配置也是同步正确性的前提。[配置选择](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/jit_kernels/heuristics/mega_moe.hpp#L99)、[JIT 与 launch](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp#L148)

### 一个 CTA 内的线程分工

当前配置每 CTA 有 512 个线程，即 16 个 warp。不同角色通过完成信号协调，分别推进自己的循环。

| Warp | 持续执行的职责 |
|---|---|
| 0–3 | 统计路由、发送源索引、拉取输入；后期清理 workspace |
| 4 | 按任务加载激活与激活 scale |
| 5 | 按任务加载专家权重与权重 scale |
| 6 | leader CTA 等待输入、发射 UMMA |
| 7 | leader CTA 生成并发布 L1/L2 任务 |
| 8–15 | 读取 TMEM；L1 做 SwiGLU 与量化，L2 回传结果；末尾做 top-k 求和 |

初始化时另有 warp 协作建立 barrier 和分配 TMEM；表格描述进入主路径后的职责。角色入口集中在 [kernel 的分支部分](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L341)。

## Dispatch：先发送索引，再按专家拉取输入

Dispatch 分为路由元数据准备和 token 搬运。源 GPU 先告诉目标 GPU“哪些本地 token 选择了你的专家”；目标 GPU 据此拉取输入，把同一专家的路由项放成连续的 token block。

### 从源 token 找到目标专家的槽位

每个 Dispatch CTA 先统计自己处理的路由项。它对全局每专家计数做原子累加，原子操作返回的旧计数给这份局部列表分配一段互不重叠的槽位；线程再将源索引写到专家所属 rank。

源索引把 token 位置和路由槽编码在一起：

$$
i_{\mathrm{src}}=t\,k+s,\qquad
t=\lfloor i_{\mathrm{src}}/k\rfloor,\qquad s=i_{\mathrm{src}}\bmod k.
$$

$t$ 是源 token 编号，$s$ 是该 token 的 top-k 槽。专家编号决定目标 GPU；目标表按“本地专家、来源 rank、来源列表槽”组织。接收方既能找到输入，也能记住结果应回到哪个 token 的哪个槽。

随后各 rank 发布专家计数，并通过 grid/NVLink barrier 确认这些元数据已准备好。计数的高 32 位累计统计完成次数，低 32 位累计 token 数：调度器等到全部 rank、全部 CTA 的统计贡献到齐，才能确定每个专家的 M-block 数。这里的计数完成表示路由列表可用，实际 token 搬运发生在下一阶段。[路由统计与发布](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L346)

### 接收 GPU 拉取并发布 token block

接收方在同一专家内轮流选择不同来源 rank。例如两来源分别有 2 条、1 条路由项时，顺序可以表示为“rank 0 第 0 条 → rank 1 第 0 条 → rank 0 第 1 条”。源码用各 rank 的剩余计数推导这种排列，将对多个来源的访问交错开。

一个 Dispatch warp 每次负责一条路由项：

- 由源索引得到源 token 地址，通过 TMA 将输入片段从 peer memory 搬到本地 SMEM，再写入本地 L1 激活缓冲。
- 同时复制输入 scale、路由权重，并保存来源 rank、token 编号与 top-k 槽。
- 等待该 token 的最后一片输入写入完成，再更新 L1 block 的就绪计数。

FP8 输入的每 32 个 K 元素共用一个 UE8M0 scale。这种一字节格式只存指数，用 2 的幂表示缩放系数；四个 scale 打包为一个 int32。因此输入视图 `x_sf` 的物理形状为 $[T_{\max},H/128]$，逻辑上有 $H/32$ 个 scale/行。L1/L2 激活 scale 还按 TMA、UTCCP 需要的布局排列；它们随激活一起交接。[输入拉取与发布](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L433)

L1 的激活加载 warp 等一整个 $B_M$ block 就绪后开始读取。末块不足 $B_M$ 行时，Dispatch 将计数补到整块目标值；任务另带 `valid_m`，表示真正有效的行数。UMMA 使用按 16 行向上对齐的有效 M，epilogue 跳过无效行。这样既能使用固定的缓冲布局，也能处理专家收到不同数量 token 的情况。

## 沿一对 CTA 跟完两次 GEMM

### 这里的两个 CTA 按输出列分工

MegaMoE 的配对 CTA 处理相同的 token block，各自负责不同的 128 列输出。激活的 M 行分成两半加载并 multicast，两个 CTA 都需要完整的激活块；权重则各取自己负责的 N 行。下面以 $B_M=64$ 的一个 L1 K-block 为例。

<MegaMoEDiagram view="partition" />

逻辑运算仍为 $XW^{\mathsf T}$。实现发射 UMMA 时交换两操作数，以 $WX^{\mathsf T}$ 的方向计算，所以输出逻辑轴与 TMEM 的存储轴互换。

| 范围 | 逻辑结果 | 实现中的累加方向 |
|---|---|---|
| 单 CTA | $[B_M,128]$ | 128 条 TMEM lane × $B_M$ 个 column |
| 两 CTA 合计 | $[B_M,256]$ | 两份本地 TMEM，各对应 128 个输出特征 |
| 一个 L1 CTA 完成 SwiGLU 后 | $[B_M,64]$ | gate/up 配对，特征宽度减半 |

这个分工与前一篇 Blackwell MatMul 示例不同：前一篇两 CTA 各负责不同 M 行；这里两 CTA 共享 M 范围、分担 N 列。GPU 间的 EP 通信与 GPU 内的 2CTA UMMA 协作属于不同层次，cluster 始终位于单 GPU 内。[UMMA 配置](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L158)

### 从任务信息到 TMA、UMMA 与 epilogue

Scheduler 发布的任务包含阶段、专家、M-block、N-cluster、逻辑 pool block、有效行数和 N/K 形状。两 CTA 各有两个 SMEM 任务槽；scheduler 把同一份任务发布到两 CTA 的对应槽，轮流使用槽位。激活加载、权重加载、MMA 和 epilogue 角色分别按相同顺序消费任务。

一次任务内部的执行顺序是：

- 两个 TMA warp 选择该任务的激活与权重 tensor map，沿 K 读取激活、权重及 scale；先等输入 stage 空闲，再登记加载。
- Leader MMA warp 等该 stage 的数据到齐，将 scale 从 SMEM 经 UTCCP 复制到 TMEM，然后发射 block-scaled UMMA。
- 每个 K-block 的计算完成后归还输入 stage；最后一块还发布 TMEM 结果就绪。
- Epilogue 等结果就绪，读取本地累加值，执行对应阶段的后处理。

输入 `full` barrier 的 arrival 数为 4，来自“两个 CTA × 两个 TMA 加载角色”，还需满足登记的传输字节数。输入 stage 数由 SMEM 预算选择；TMEM 有两个 accumulator stage；任务信息也有两个槽。它们分别保护输入数据、累加结果与任务元数据，不能将其中一种 stage 数套到其他缓冲上。[激活/权重加载与 MMA](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L690)

### L1 输出的列块成为 L2 输入的 K-block

L1 epilogue 从 TMEM 读取 gate/up，执行 SwiGLU 和路由乘权，并计算每行、每 32 个特征的最大绝对值，生成 scale 后量化为 FP8。量化数据经 SMEM、TMA Store 写到本地 L2 激活缓冲；scale 也写到对应区域。L1 输出与 L2 输入使用同一份中间数据。

关键依赖来自维度：L1 每个 128 列块生成 64 列 SwiGLU 输出，而 L2 每个 K-block 要 128 个输入特征，所以 **两个 L1 N-block 完成，才能供给一个 L2 K-block**。

<MegaMoEDiagram view="readiness" />

图例采用 $H=512,F=256,B_M=64$：L1 总宽度 $2F=512$，拆成四个 128 列块；SwiGLU 得到四个 64 列片段。前两个片段已写好时，L2 可以加载 $K\in[0,128)$ 并开始累加。后两个片段准备好后，继续 $K\in[128,256)$；完整的 L2 输出依赖两次累加都完成。

实现用 `l2_full_mask` 的每一位表示一个 L1 N-block 的发布状态。L1 epilogue 等待该块的 TMA Store 完成，并以 release 语义更新位；L2 加载 warp 用 acquire 读取，只等待当前 K-block 需要的位。release/acquire 保证先写数据、再通知，观察到通知后才能消费相应数据。环形槽复用时用 generation 奇偶区分新旧状态。[逐 K-block 依赖](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L139)

## 多个任务如何持续推进

### L1 预热，再交错派发 L2/L1

每个 cluster 的 scheduler 先获取专家接收计数，构造逻辑 M-block 池，再用原子任务计数领取工作。当前版本先派发若干轮 L1，随后在仍有工作时交替领取 L2 与 L1。这里的“wave”表示全 GPU 的 cluster 各领取一项任务所形成的一轮派发，实际完成时间由数据依赖决定。

L1 预热用于保证首次 L2 所涉及的 M-block 的生产任务都已派发。L2 任务还会检查各调度队列已经领走所需的 L1 工作；否则等待中的 L2 可能占住流水线，而它所等待的 L1 尚未进入流水线。预热轮数考虑两阶段每 M-block 的任务数差异，因此依赖模型形状和本次接收量。[预热计算](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L15)、[任务领取](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L389)

“生产任务已派发”只满足调度层的条件。激活加载 warp 仍通过前文的 L1 就绪计数和 L2 K-block mask 等待实际数据。两层检查分别保证工作能推进，以及计算读取的数据已完成。

> 当前版本还按 GPU locality domain 分配 N-cluster 工作，队列耗尽后按条件从另一个 domain 领取任务，以兼顾局部访问和负载均衡。这是 GPU 内的任务安排；专家所属 EP rank 仍由路由编号确定。

### L1、L2 环形缓冲各有自己的释放条件

所有专家的 token block 形成逻辑池，但 L1/L2 激活只保存在容量有限的物理 ring 中。

- $q$：逻辑 block 编号。
- $R$：ring 的物理 block 槽数。
- $q\bmod R$：该 block 使用的物理槽。
- $\lfloor q/R\rfloor$：generation，即该槽的使用轮次，用来区分同一槽的新旧内容。

<MegaMoEDiagram view="ring" />

同一个物理编号对应两份不同的存储：L1 ring 放 $[B_M,H]$ 输入，L2 ring 放 $[B_M,F]$ 中间值。Dispatch 等所有 L1 N-block 消费旧输入后，才能将下一代 token 搬入 L1 槽；L1 epilogue 等所有 L2 N-block 消费旧中间值后，才能将新中间值写进 L2 槽。

L2 消费结束的计数在完整 UMMA 结果就绪后更新，此时后续 epilogue 依赖的是 TMEM，不再需要 L2 输入。L1 ring 的归还计数则在 L1 epilogue 发布中间结果时更新。它们是该实现采用的具体释放点。[L1 发布与 L2 释放](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1240)

来源 metadata 按完整逻辑池保存，保留 rank、token、top-k 槽三个字段。这样，激活槽被下一代覆盖后，旧任务仍能找到正确的回传地址。Ring 容量由 host 按预热、L1/L2 推进速度和流水余量估计；主 kernel 另有容量断言。任意缩小 R 会改变复用与 generation 判断成立的前提。

### TMEM 释放后，回传仍可能继续

L2 epilogue 读取 FP32 累加值，转换成 BF16，经 SMEM 重排后用向量 global store 写到源 GPU 的 Combine 槽。线程发出 remote store 后，已发出的请求可以继续通过互联传输；接收端要等系统范围的就绪通知才能使用结果。

<MegaMoEDiagram view="overlap" />

每 CTA 有两个 TMEM accumulator stage，任务 $j$ 和 $j+2$ 复用同一槽。Epilogue 等最后一次有效 TMEM load 完成后就发布 `tmem_empty`，后续同槽任务才可覆盖它。$B_M=240$ 时还会预取下一 store block 到寄存器，进一步提前释放 TMEM；较小配置的预取深度为 0。[读取与释放 TMEM](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1006)

通信仍可能通过有限缓冲拖慢计算：remote store 发射变慢，会占住 epilogue；epilogue 推进变慢，又可能使 TMEM 槽或任务信息槽迟迟不能归还。后续 MMA 或 scheduler 因此等待。图中的重叠是依赖允许的执行关系，收益取决于路由分布、计算量和互联压力。

## 回传完成后，源 GPU 沿 top-k 槽求和

每条 routed expert 结果写入源 rank 的 `combine_token_buffer[s,t,:]`，逻辑形状为 $[k,T_{\max},H]$。不同路由槽有各自的位置，回传过程无需对同一个输出向量做原子加法。最终由源 GPU 沿槽轴相加，保持 token 顺序：

$$
Y_r[t,h]=\sum_{s\,\text{有效}}\operatorname{combine}_r[s,t,h].
$$

槽内已经包含路由权重的贡献。无效专家编号 `-1` 对应的槽被跳过。

这个版本的完成通知按 **rank 的全部 L2 写回**发布，而最终求和按本 token 需要的专家 rank 等待：各 GPU 的 epilogue 先做本地 grid sync，汇集本 GPU 的写回；再向 peer 写 system-scope release 信号。源 GPU 用 system-scope acquire 等所选专家所在 rank 的本轮标记，然后读取对应槽。system scope 将这组顺序与可见性保证扩展到 GPU 之间。标记使用 kernel launch 的 grid index，区分连续调用。[回传与最终求和](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L1405)

最终求和复用前面计算阶段的 SMEM 区域。每个 epilogue warp 使用两个 TMA 加载槽和一个输出槽，一边预取下一个 expert 贡献，一边在 FP32 寄存器中累加当前贡献，最后转换并写出 BF16。Dispatch warp 此时清理专家计数、任务计数和 ring 信号，使缓冲能够用于下一次调用。

> Shared expert 在本地直接读取输入，不依赖 routed Dispatch。它的 L1/L2 任务加入同一调度路径，结果使用额外的 Combine 槽；多个 shared expert 在实现中合成更宽的中间投影。FP8/FP4 主路径中的 shared 权重使用 FP8，BF16 路径单独实现。主线的形状表省略了这个可选分支。

## 性能证据与实现边界

MegaMoE 减少了 kernel 边界，并允许中间数据就绪后直接推进下游工作。它仍会读写显存：L1 输入 ring、L2 中间 ring、Combine 槽都是全局内存；融合收益需要连同这部分流量、warp 占用和同步开销一起衡量。

官方 [PR #316 的历史测量](https://github.com/deepseek-ai/DeepGEMM/pull/316) 报告 EP=8，各数字按八 rank 平均。下面保留两个 8,192 tokens/rank 的测量点；它们来自 2026-04 的发布，不能作为本文固定的 2026-09 实现的复测结果。

| 配置 | E / k / H / F | MoE 调用耗时 | 相对 legacy 加速比 |
|---|---|---:|---:|
| DeepSeek-V4-Flash 形状 | 256 / 6 / 4096 / 2048 | 1,283.1 μs | 1.56× |
| DeepSeek-V4-Pro 形状 | 384 / 6 / 7168 / 3072 | 2,818.5 μs | 1.50× |

这里计量的是 MoE 执行，不能外推为完整模型端到端加速。当前测试脚本另外会按估算字节数除以时间输出 HBM/NVLink GB/s，并用假设带宽估算 reduction 时间；这些数字用于计算口径，不能直接当成硬件计数器测得的实际流量或独立阶段耗时。[测试与计量逻辑](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/tests/test_mega_moe.py#L354)

[X-Stage 论文](https://arxiv.org/html/2607.23264v1) 进一步研究 MegaMoE 的 remote-store 压力：集中连续执行 L2 容易形成回传突发，穿插 L1 计算可为已发出的请求提供传输时间。论文在旧 commit `7f2a703` 上将交错调度与 Expert-Wave 基线比较，报告 84 个配置的几何平均 1.18×、最大 1.62× **kernel 加速**。当前源码已经包含 L1 预热与 L2/L1 交错逻辑；论文结果应按其版本与基线解读。

使用和进一步优化时，以下限制会直接影响行为：

- **形状与量化布局**：本路径固定 SwiGLU 和 32 元素 scale 粒度；H、F、分块和对齐必须满足 host/kernel 检查。L1 N-block 的 readiness mask 最多 64 位，当前实现据此限制 $2F/128\le64$。
- **通信环境**：本文机制依赖 peer 地址映射与对应的 NVLink 通信路径。`SymBuffer` 的 rank 上限只是布局检查，不足以证明任意 GPU 数、跨节点或拓扑都能采用相同实现。
- **常驻资源**：Dispatch、两个加载 warp、MMA、scheduler 和 epilogue 共用 SM、寄存器与 SMEM。小任务可能更受固定同步与调度开销影响，热点专家也会改变实际队列进度。
- **数值路径**：FP4/FP8、BF16 舍入、clamp 和 fast-math 都会影响结果。仓库测试在 legacy 依赖可用时检查逐元素相等；这说明测试设计的比较标准，本教程只完成源码核查，未运行 GPU 正确性或性能实验。

<details>
<summary>源码导航：继续阅读每个机制</summary>

| 文件 | 阅读目标 |
|---|---|
| [Python API](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/mega/__init__.py) | SymmBuffer、权重变换、调用边界 |
| [Host API](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/apis/mega_moe.hpp) | 张量检查、缓冲视图和 ring 容量 |
| [配置 heuristic](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/jit_kernels/heuristics/mega_moe.hpp) | 分块、线程数、SMEM 预算 |
| [SM100 主 kernel](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) | 各 warp 分支、TMA/UMMA、epilogue、回传与求和 |
| [Scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh) | 任务池、预热、交错、逐 K-block 依赖 |
| [Buffer layout](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/layout/mega_moe.cuh) | ring、完整 metadata 池和信号地址 |
| [通信 barrier](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/comm/barrier.cuh) | grid 与跨 rank 的 release/acquire 关系 |

</details>
