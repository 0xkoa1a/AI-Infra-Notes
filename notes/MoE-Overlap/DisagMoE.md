---
title: "DisagMoE"
order: 1
---

# DisagMoE

## 引言

这篇是 **DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated AF-Pipe Parallelism**，2026 年 5 月提交。和前面的 MoE-Prefill 不同，它研究的是**大规模 MoE 训练**，核心问题仍然是跨节点 EP 的 dispatch/combine 通信。([arXiv][1])

它的核心思路可以先压缩成一句话：

> **不要继续让 Attention 和 FFN 共用同一批 GPU，再费力把 All-to-All 塞进有限的 FFN 计算窗口；直接把 Attention 和 FFN 拆到两组 GPU 上，把它们变成流水线的两个独立 stage，再重新分配 GPU 和网络资源，让 Attention compute、FFN compute、MoE communication 长时间并行。**

传统 EP 的一层是：

**Attention → dispatch → FFN → combine → 下一层 Attention**

同一组 GPU 顺序承担这些工作，因此当前 FFN 的计算时间天然限制了 dispatch/combine 能隐藏多少。Comet、Tutel 等工作可以把它切得很细，但**如果通信本来就比 FFN compute 长，那么怎么切都会剩下一截 communication tail**。论文在跨节点实验中就观察到了这一点。

DisagMoE 改成：

**Attention workers ⇄ FFN workers**

然后同时处理很多 microbatch。稳态下可以出现：

**A 组算 microbatch 2 的 Attention + F 组算 microbatch 1 的 FFN + 网络传另外 microbatch 的 token/gradient**

因此它真正扩大的是 **module-level overlap window**，而不是继续优化单个 MoE operator 内的 overlap。([arXiv][2])

---

## 背景 / 相关工作

## 跨节点 EP 的通信为什么严重

这篇论文讨论的主要场景是大模型训练中常见的 **DP + EP**：

Attention 等 dense component 按 DP 复制，而 experts 按 EP 分布。Attention 产生的 token 必须从当前 DP rank 重排到对应 expert rank，FFN 算完以后再重排回来，因此每层有 dispatch 和 combine 两次 All-to-All。([arXiv][2])

问题在于，大模型的 expert 数量和参数量继续增长以后，一个 EP group 很容易跨 node。

这时：

* node 内通信走 NVLink；
* node 间通信走 InfiniBand / Ethernet；
* 后者带宽远低于 node 内链路。

论文在 H800 集群上的 profiling 中，随着 EP 从单节点扩展到 8 节点，All-to-All 占训练时间的比例从大约 **22% 增长到接近 78%**。Top-$k$ 增大也会显著增加通信，因为更多 token-expert assignments 必须跨设备传输。

因此这里的问题不是：

> 怎么把 collective kernel 再优化几个百分点？

而是：

> **怎么让跨节点通信不再直接占据训练 critical path？**

---

## 相关工作：从 FFN 内 overlap 到 whole-graph overlap

论文实际上把此前 MoE communication overlap 的工作分成三条路线：

* **FasterMoE / Tutel / Comet：FFN 内部 overlap。** 把 token 或 GroupGEMM 切成更小 chunk/tile，一个 chunk 做 FFN 时，另外 chunk 做 dispatch/combine。Comet 已经把这个粒度推进到了很细的 tile-level producer-consumer pipeline，但 overlap window 最终还是由 FFN compute 提供。
* **Lancet / DualPipe：扩大到 microbatch / computation graph。** 不再只要求当前 token 的 FFN 去隐藏自己的通信，可以利用另一个 microbatch、其他 operator，DualPipe 甚至利用 forward/backward 之间的独立工作进一步扩大窗口。
* **Attention–FFN disaggregation：直接拆资源。** MegaScale-Infer、StepFun 等 serving 系统已经把 Attention 与 FFN 放到不同设备上；HeterMoE 将这种思路扩展到 training。但已有训练方案的规模受到参数、optimizer state 和流水结构限制，论文指出 HeterMoE 展示的模型规模约为 4.3B，难以直接扩展到数百 B MoE。

前两类 overlap 有一个共同的结构性问题：

**Attention 和 FFN 仍然绑定在同一个 resource pool 中。**

假设一张 GPU：

* 一会儿做 Attention；
* 一会儿做 FFN；
* 同时尝试把网络通信插到空隙里。

那么你无法单独回答：

> Attention 到底应该得到多少 GPU compute？
>
> FFN 到底应该得到多少 GPU compute 和 NIC bandwidth？

因为它们共享同一个 GPU/NIC 配置。

DisagMoE 最核心的变化，就是把这个绑定关系拆掉。([arXiv][2])

---

## 为什么 Attention 和 FFN 不应该共享相同的 compute/communication 配比

这是理解整篇论文最重要的背景。

设序列长度为 $s$。论文首先给出一个非常简单的 scaling observation。

Attention 的 FLOPs 可以近似为：

$$
F_A(s)=\alpha_1s^2+\alpha_2s.
$$

FFN 的 FLOPs 可以近似为：

$$
F_F(s)=\beta s.
$$

EP communication volume 则近似为：

$$
V(s)=\gamma s.
$$

自然语言就是：

> **序列增长时，Attention 的计算量有一个 $s^2$ 项；FFN 计算和 MoE token communication 都基本只随 $s$ 线性增长。** ([arXiv][2])

这意味着长序列下发生了一件很特殊的事情。

Attention：

$$
\frac{\text{compute}}{\text{communication}}
$$

会越来越大。

但 FFN：

$$
\frac{\text{compute}}{\text{communication}}
$$

基本不变。

论文在 Qwen3、8 节点实验中，从 4K 增长到 32K sequence length 时：

* Attention 占计算的比例从 **28.38% → 50.26%**；
* FFN 从 **19.22% → 13.96%**；
* Attention compute/communication ratio 从 **1.08 → 2.78**；
* FFN 则基本停留在 **0.73 → 0.77**。([arXiv][2])

这意味着长序列以后：

**Attention 越来越 compute-bound，而 FFN/EP 一侧依然 network-bound。**

如果两者强行共用相同 GPU:NIC 比例，就很难让二者同时工作在高效率区域。

---

## Roofline 视角

论文把前面的现象放到一个 compute-communication roofline 中。

定义 arithmetic intensity：

$$
I=\frac{\text{Compute FLOPs}}{\text{Communication Bytes}}.
$$

自然语言就是：

> **每传 1 Byte 数据，可以做多少 FLOPs。**

硬件有一个 turning point：

$$
\hat I=\frac{\text{Peak FLOPs}}{\text{Peak Network Bandwidth}}.
$$

当 $I<\hat I$ 时，网络先成为瓶颈；当 $I>\hat I$ 时，计算更可能成为瓶颈。([arXiv][2])

对论文使用的模型近似，可以进一步得到：

$$
I_{\rm attn}
============

\frac{H(2+2/g)+4S}{2k},
$$

而：

$$
I_{\rm ffn}=2D_e.
$$

这里 $H$ 是 hidden size，$g$ 是 GQA group 参数，$k$ 是 Top-$k$，$D_e$ 是 expert FFN hidden dimension。([arXiv][2])

这里甚至不需要细看公式，最重要的是：

> **$I_{\rm attn}$ 中有 $S$，所以序列越长，Attention 越向 compute-bound 移动；$I_{\rm ffn}$ 与 $S$ 无关，所以长序列不会自动拯救 FFN 的 communication bottleneck。**

这也是 DisagMoE 相比一般“长序列给更多 compute，所以 overlap 会更好”的说法更精确的地方。

长序列增加的额外 compute **主要在 Attention，不在 FFN**。

---

## 动机

## 为什么 Comet/Tutel 仍然盖不住通信

考虑最理想的 FFN-level pipeline：

**dispatch chunk 1 → FFN chunk 1 → combine chunk 1**

与此同时可以传 chunk 2。

如果一次 MoE layer 中：

$$
T_{\rm FFN}<T_{\rm dispatch}+T_{\rm combine},
$$

自然语言就是：

> **所有 FFN compute 加起来都没有通信长，那么 FFN 自己不可能完全隐藏这些通信。**

无论 chunk 切得多细，最后仍然会剩：

$$
T_{\rm comm}-T_{\rm FFN}
$$

这一段裸露 communication tail。

论文的跨节点测量中，单个 FFN operation 的计算时间只有两次 All-to-All 总时间的大约 **40%–60%**，因此 Tutel/Comet 这种 operator-level overlap 必然存在剩余通信。([arXiv][2])

---

## 为什么仅仅跨 microbatch overlap 也不够

Lancet、DualPipe 把 Attention 等其他 computation 引进来了，这已经比只用 FFN 好很多。

但又产生一个问题：

> **Attention : FFN : Communication 的相对时间会随着 sequence length 改变。**

例如 4K：

**Attention 和 FFN computation 比较接近。**

到了 32K：

**Attention 明显比 FFN 重很多。**

如果所有 GPU 固定按相同方式执行 Attention 和 FFN，即使你安排出了一个看起来很漂亮的流水线，也可能变成：

**Attention stage 非常长 → FFN/communication stage 很短**

或者反过来。

于是 pipeline throughput 最终取决于最慢 stage：

$$
T_{\rm pipeline}\approx\max(T_A,T_F).
$$

自然语言就是：

> **不是 overlap 出来了就结束了；Attention 与 FFN 两边必须速度匹配，否则快的一边还是要等慢的一边。** ([arXiv][2])

这就引出 DisagMoE 最重要的设计自由度：

> **既然 Attention 和 FFN 的 scaling law 不同，那就不要规定它们必须拿相同数量的 GPU 和相同的 NIC/GPU 比例。**

---

## 方法

整个方法可以分成三个依赖关系明确的部分：

**先把 Attention / FFN 拆到不同 worker → 再让这些 worker 构成 AF-Pipe → 最后根据 workload 给两边分配不同数量的 GPU/NIC。**

## Disaggregated component placement

传统一个 Transformer layer 的 $A_i$ 和 $F_i$ 基本生活在同一组 GPU。

DisagMoE 改成两类 worker：

**A-Worker**

负责 Attention。组内 GPU 对 Attention 使用 DP。

**F-Worker**

负责 MoE FFN。组内 experts 使用 EP 分布。

于是一个 token 的 forward 数据流变成：

**A-Worker → M2N → F-Worker → N2M → 下一 A-Worker**

而 backward 则沿相反方向传 gradient。([arXiv][2])

注意这里的“disaggregation”不是简单地准备：

**一组 GPU 保存所有 Attention + 一组 GPU 保存所有 FFN。**

如果这样做，训练时所有 layer 的 Attention parameters、FFN parameters、optimizer states 都堆在各自设备上，很容易 OOM，这正是论文认为已有 AFD training 不够 scalable 的原因之一。([arXiv][2])

DisagMoE 继续对 layer 做 pipeline grouping。

假设有 8 层、每种 component 有 2 个 group：

A-group 0 保存：

$A_0,A_2,A_4,A_6$

A-group 1 保存：

$A_1,A_3,A_5,A_7$

FFN 同理。

因此每个 worker 只保存自己负责的部分 layer，而不是保存全模型同类 component。

这实际上把：

**component disaggregation**

和：

**interleaved pipeline parallelism**

组合起来了。

---

## AF-Pipe

仅仅把 Attention 和 FFN 放到不同 GPU 上是不够的。

如果只跑一个 microbatch：

**A GPU 算的时候 F GPU 闲着 → F GPU 算的时候 A GPU 闲着**

利用率反而会很差。

因此 batch 被拆成多个 microbatch，让它们形成流水。

例如稳态可以想象为：

| 资源       | 同一个时刻                                    |
| -------- | ---------------------------------------- |
| A-worker | microbatch 3 的 Attention                 |
| F-worker | microbatch 2 的 FFN                       |
| Network  | microbatch 1/2 的 hidden state 或 gradient |

于是 Attention compute、FFN compute、communication 不再有“一层必须完全做完才能开始下一件事”的全局串行关系。论文 Figure 8 展示的就是这种多 microbatch、forward/backward 交错的稳态 execution。

> 这里最容易误解的一点是：**DisagMoE 并不是让同一个 token 的 Attention 和 FFN 同时计算。**
>
> 这两者仍有严格数据依赖。
>
> 它 overlap 的是**不同 microbatch / 不同 layer / forward-backward 阶段**的工作。

---

## 把 All-to-All 变成 M-to-N / N-to-M

拆开 A/F worker 后，传统 EP 的 collective 形式也发生变化。

假设：

* A-worker 有 $M$ 张 GPU；
* F-worker 有 $N$ 张 GPU。

A 侧按 DP 拥有 token，F 侧按 expert placement 拥有 experts。

因此 dispatch 不再是同一个 EP group 内：

**N → N All-to-All**

而是：

**M 个 Attention ranks → N 个 FFN ranks**

即 M-to-N，简称 M2N。

算完 expert 后再：

**N → M**

即 N2M。([arXiv][2])

这不是换个 collective 名字而已。

传统 PP + EP 可能出现：

**FFN combine → 得到完整 activation → pipeline P2P → 下一 stage**

也就是 combine 已经搬了一次数据，随后 P2P 又搬一次。

AF-Pipe 把这些 stage boundary communication 融合为统一的 M2N/N2M communication。论文声称这消除了冗余 transfer，并在相应场景下把相关 communication cost 降低约 $1/k$。([arXiv][2])

---

## 把 communication 当作 pipeline stage

这是 AF-Pipe 和普通 PP 的另一个区别。

普通 PP 往往认为 stage 是：

**Compute A → communication → Compute B**

通信只是 stage 边界上的附属动作。

AF-Pipe 明确把 MoE communication 当成一个 **first-class pipeline stage**。

每个 A/F worker 使用独立的：

* forward stream；
* backward stream；
* communication stream。

send / receive 也使用独立 ProcessGroup 异步推进。([arXiv][2])

所以某个 F-worker 在一个时刻可能：

正在计算 microbatch $j$ 的 FFN；

同时：

向前面的 A-worker 发送另一个 microbatch 的 backward gradient；

同时：

从后面的 A-worker 接收另一个 backward result。

它真正追求的是：

> **只要当前 compute 所需输入已经 ready，就让 compute stream 继续走；communication stream 独立处理其他依赖链。**

因此 steady state 能接近连续 GPU utilization。([arXiv][2])

---

## Adaptive worker allocation

到这里已经有 overlap 了，但还缺最后一个问题：

> Attention 和 FFN 各应该分多少 GPU？

这恰好是论文最值得关注的部分之一。

设整个系统有 $W$ 张 GPU。

分成：

$$
M+N=W,
$$

其中：

* $M$ 张用于 Attention；
* $N$ 张用于 FFN。

网络 NIC 也分别给 A/F 两侧预算。([arXiv][2])

Attention stage 的时间近似由“计算”和“通信”较慢的一项决定：

$$
T_A=
\max
\left(
\frac{C_A}{PM},
\frac{V}{M_AB_{\rm IB}}
\right).
$$

自然语言就是：

> **A-worker GPU 不够时是 compute-bound；NIC bandwidth 不够时是 network-bound，stage 时间取较慢者。**

FFN 同理：

$$
T_F=
\max
\left(
\frac{C_F}{PN},
\frac{V}{M_FB_{\rm IB}}
\right).
$$

([arXiv][2])

AF-Pipe 是 producer-consumer pipeline，所以真正优化的第一目标不是简单最大化总 MFU，而是：

$$
\min \max(T_A,T_F).
$$

自然语言就是：

> **先让 Attention stage 和 FFN stage 尽量一样快，消除流水线长尾。**

在达到最小 bottleneck latency 的多个 allocation 中，再选择 MFU 更高的 GPU/NIC 配置。论文把这个问题写成 MILP，先用 roofline model 得到一个候选配置，再用少量实际 profiling 在附近做 local refinement，补偿 analytical model 没有建模准确的 kernel 和系统效应。([arXiv][2])

---

## 为什么长序列会改变 GPU 配比

这也是 DisagMoE 与这条 long-sequence survey 关系最密切的地方。

短序列时：

$$
C_A\approx C_F
$$

Attention 和 FFN 的 stage 比较接近，因此大致 **1:1 的 GPU allocation** 就不错。

随着 sequence length 增长：

$$
C_A
$$

因为 Attention 的 $S^2$ 项越来越大，而：

$$
C_F
$$

仍只线性增长。

因此必须：

> **给 Attention 更多 compute resource，但 FFN 并不需要同比增加 GPU；FFN 更需要的是 network bandwidth。**

论文的 ablation 很直观。

4K sequence 时，Attention : FFN GPU ratio 最优大约在：

**16 : 16**

附近。

16K 时，最优变成：

**16 : 10**

此时相对于 Megatron baseline 提升约 **1.56×**，相比固定均匀的 16:16 allocation 也提升 **1.29×**。([arXiv][2])

所以 sequence length 在这里已经不只是：

> “给多少 overlap window”

而是直接成为：

> **“应该怎样切分整个 GPU/NIC 集群”**

的输入变量。

这是这篇论文相比很多 overlap 工作更进一步的地方。

---

## 效果

论文在最多 **16 节点 × 8 H800 = 128 GPU** 的集群上实现 DisagMoE，基于 Megatron-LM / PyTorch 2.6，使用 GPUDirect/GDRCopy 实现 M2N/N2M communication；测试 DeepSeek-MoE、GPT-OSS-120B 和 Qwen3 类配置，sequence length 覆盖 **4K–32K**。([arXiv][2])

端到端结果：

相对于 Megatron 1F1B/interleaved training：

**1.59–1.81×**。

相对于当时 SOTA MoE overlap：

最高 **1.34×**。([arXiv][2])

更具体地：

相对 Tutel / Comet：

大约 **1.2–1.5×**。

相对 DualPipe：

平均约 **1.05–1.13×**。([arXiv][2])

它降低的主要就是**剩余未被隐藏的通信**。8K DeepSeek-MoE 实验中，non-overlapped communication 相比：

Tutel 减少最多 **88%**；

Comet 减少 **75%**；

DualPipe 减少 **45%**。([arXiv][2])

论文也测试了 Top-$k$ 和 EP size：Top-$k$ 增大会同时增加 FFN compute 和 communication，EP 跨更多 GPU/node 也会增加通信；DisagMoE 在这些配置下仍获得约 **1.08–1.92×** 的相对加速范围。([arXiv][2])

---

## 延伸

## 与 MoE-Prefill 的区别

前一篇 MoE-Prefill 和这篇其实是在利用**完全不同的自由度**。

MoE-Prefill：

**改变通信对象。**

从：

**move activation**

改成：

**move expert weights**

然后：

**当前 layer compute ↔ 下一 layer weight transfer**

而 DisagMoE：

**仍然移动 activation，但改变 model placement 和 schedule。**

从：

**Attention + FFN colocated**

变成：

**Attention workers ⇄ FFN workers**

再通过 microbatch pipeline 让：

**Attention compute ↔ FFN compute ↔ activation communication**

并行。([arXiv][2])

所以两篇其实可以放在 survey 的两个不同分支：

**MoE-Prefill：communication-object freedom**

**DisagMoE：placement/resource-allocation freedom**

---

## 对长序列 MoE overlap 更重要的启发

这篇论文给出的一个很有用的结论是：

> **不能只用整个 Transformer layer 的 compute/communication ratio 判断能不能 overlap。**

因为一个 layer 内：

**Attention 的 ratio 随 $S$ 快速增长**

而：

**FFN 的 ratio 基本不随 $S$ 改变。**

把它们平均成一个 ratio，会掩盖这种结构差异。([arXiv][2])

因此更合适的建模方式可能是分别描述：

$$
\rho_A(S)=\frac{T_A^{\rm compute}}{T^{\rm comm}},
$$

和：

$$
\rho_F(S)=\frac{T_F^{\rm compute}}{T^{\rm comm}}.
$$

自然语言就是：

> **分别问 Attention 有多少能力隐藏通信、FFN 有多少能力隐藏通信，而不是给整个 layer 一个平均值。**

然后让 placement / scheduling / GPU allocation 根据 $(\rho_A,\rho_F)$ 变化。

这也解释了一个表面上有点反直觉的现象：

> **长序列并没有让 MoE FFN 自己变得更容易隐藏 EP communication；长序列真正增加的是 Attention 这一大块可利用的计算资源。**

DisagMoE 的核心就是想办法**解除 Attention compute 与 FFN communication 之间原本过强的执行绑定，把这块 $O(S^2)$ 的计算真正变成系统可调度的 overlap resource**。([arXiv][2])

## 这篇论文的边界

它主要针对 **pretraining-style、固定 sequence length 和固定 microbatch shape**。正因为 workload 可以提前知道，GPU/NIC allocation 才可以 offline 求一次然后长期使用。对于 RL training、动态 sequence、动态 batch 等 workload，需要在线重新做资源 allocation，论文没有解决。([arXiv][2])

另外，pipeline virtual stage 越多通常 bubble 越小，但每个 worker 需要保存更多 layer parameters / activations。论文实验中增加 virtual stages 能持续改善吞吐，但超过一定程度后会 OOM；其测试中 $v>16$ 已触及这一问题。([arXiv][2])

因此如果把 **DisagMoE** 在我们的 survey 中压缩成一个最核心的理解，我会写成：

> **传统 MoE overlap 在问：“怎样把 A2A 塞进已有的计算空隙？”**
>
> **DisagMoE 则进一步问：“为什么 Attention、FFN 和网络必须被绑定在同一组 GPU 上？如果把资源拆开，能不能人为塑造一个更适合 overlap 的 compute/communication ratio？”**

而长序列正是让这种重新分配资源变得越来越有价值的 workload。

[1]: https://arxiv.org/abs/2605.11005 "[2605.11005] DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated AF-Pipe Parallelism"
[2]: https://arxiv.org/pdf/2605.11005 "DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated AF-Pipe Parallelism"


--- 

对。你的调度**完全合理**，而且更接近 Lancet / DualPipe 这类 aggregated cross-microbatch overlap：

* 所有 GPU 做 `mb1 Attention`，同时网络做 `mb0 dispatch`
* `mb0 dispatch` 完成后，所有 GPU 做 `mb0 FFN`
* 然后所有 GPU 做 `mb2 Attention`，同时网络做 `mb0 combine`
* 如此交错

因此，**并不是因为 combine 必然出现在关键路径上，所以必须把 GPU 拆成 A/F 两组。** 如果下一段 Attention 足够长，combine 一样可以藏在里面。论文也明确说，Lancet/DualPipe 已经利用不同 microbatch 的计算去隐藏通信。([arXiv][1])

## 你的方案什么时候已经足够好

把同一组 GPU 上的稳态简化成：

```text
GPU:      A1 ───────── F0 ── A2 ───────── F1 ── A3 ...
Network:  D0 ────          C0/D1 ────          C1/D2 ...
```

如果 Attention 足够长，使得：

$$
T_A \ge T_D + T_C
$$

自然语言就是：**相邻 Attention 的时间足够容纳需要执行的 dispatch/combine 网络工作。**

那么 EP communication 可以几乎完全从关键路径消失。

此时：

$$
T_{\text{step}}\approx T_A+T_F.
$$

这两项全都是有效 GPU 计算，没有通信 bubble。

**在这个理想场景下，你问“为什么还要 DisagMoE”是很合理的。**

而且我需要修正我上一条回答中的一个过度简化：我之前说 disaggregation 可以把 $A+F$ 变成 $\max(A,F)$，容易让人误以为这是免费的 throughput gain。

如果总 GPU 数固定，并且 Attention、FFN 都能**完美线性扩展**，其实没有这种免费收益。

假设总 GPU 数 $W$，总 Attention 工作量为 $a$，FFN 工作量为 $f$。

所有 GPU colocate：

$$
T_{\rm colocated}=\frac{a+f}{W}.
$$

拆成 $M$ 张 Attention GPU、$N$ 张 FFN GPU，且 $M+N=W$：

$$
T_{\rm disag}=
\max\left(\frac{a}{M},\frac{f}{N}\right).
$$

最佳分配恰好是让两边一样长，即 $M:N=a:f$，此时仍然得到：

$$
T_{\rm disag}^{*}=\frac{a+f}{W}.
$$

自然语言就是：**如果 GPU 是完全可分割、算子完美扩展、通信已经完全隐藏，那么“所有 GPU 轮流算 A/F”和“GPU 分成 A/F 两组同时算”理论吞吐一样。**

所以 DisagMoE 真正的理由不在这里。

## 真正的问题：GPU 和 NIC 不能独立分配

这是论文真正想解决的东西。

长序列下：

* Attention 越来越 **compute-bound**
* FFN + EP 侧仍然更 **network-bound**

论文测量中，sequence 从 4K → 32K 时：

* Attention compute/communication ratio：`1.08 → 2.78`
* FFN：`0.73 → 0.77`

也就是说，Attention 多出来的主要是算力需求，而 FFN 的 compute/network 性质基本没变。([arXiv][1])

但在你提出的 colocated 方案中，每张 GPU 都同时承担 A 和 F：

```text
GPU 0: Attention + FFN + 一份 NIC
GPU 1: Attention + FFN + 一份 NIC
...
GPU W: Attention + FFN + 一份 NIC
```

Attention 和 FFN 被迫共享**相同的 GPU:NIC 配比**。

假设一台 node 有：

* 8 张 H800
* 8 张 GPU 对应固定总量的 IB bandwidth

对于 Attention：

> 我主要缺 FLOPs，网络其实没那么缺。

对于 FFN：

> 我 FLOPs 没那么缺，真正缺的是跨节点网络。

但是 colocated placement 下你没办法说：

> “从 FFN 那边拿一点 GPU compute 给 Attention，同时把更多 NIC bandwidth 留给每张 FFN GPU。”

因为根本不存在独立的“A 资源池”和“F 资源池”。

这正是论文 Figure 9 的 roofline argument：aggregated architecture 中 Attention 与 FFN 使用相同的 compute/network roof；拆开后可以分别改变两个 worker group 的 GPU 数量和有效 NIC bandwidth，使 network-bound 的 FFN 得到更高的 per-GPU network bandwidth，而 Attention 继续靠近 compute roof。([arXiv][1])

---

## 一个具体例子

假设有 32 张 GPU。

长序列 workload 所需要的总工作：

* Attention：22 个 GPU-equivalent
* FFN：10 个 GPU-equivalent

理想情况下：

## 你的方案

32 张 GPU 全部：

```text
32 GPUs → Attention
32 GPUs → FFN
32 GPUs → Attention
32 GPUs → FFN
```

纯 compute 层面完全没毛病。

甚至这可能比 22/10 静态切分更灵活。

但 FFN 执行的时候：

* 32 张 GPU 一起算的 FFN kernel 可能并不能很好地扩到 32 张；
* expert parallelism 反而跨更多节点；
* 每张 GPU 对应的 token 更少；
* collective domain 更大；
* FFN 真正需要的 network bandwidth 并没有因为多给 GPU 就按比例增加。

这时“多给 FFN GPU”可能并不划算。

## DisagMoE

改成：

```text
22 GPUs → Attention
10 GPUs → FFN
```

然后：

```text
22 A-GPUs: A0 ───── A1 ───── A2 ───── A3
                     ↓        ↓
10 F-GPUs:       F0 ───── F1 ───── F2
```

此时 FFN 只占较少 GPU，因此相对于 FFN compute capacity，它可以获得更高的有效 network bandwidth；Attention 则吃掉更多 compute resources。

论文的 adaptive allocation 正是在搜索这个 balance。它不是简单追求 `A || F`，而是同时调整 **A/F GPU 数和 NIC bandwidth allocation**。([arXiv][1])

## 那 combine 到底是不是一个问题？

**是问题，但不是你这里的根本问题。**

你设计：

```text
A1 || dispatch0
F0
A2 || combine0
```

如果 $A_1$、$A_2$ 都足够长，完全成立。

但实际还有几个限制。

**通信窗口未必总能刚好拼起来。** 论文指出，已有 microbatch-level pipeline 已经这么做，但 FlashAttention 并不像 GroupGEMM 那样容易任意 chunk；当 A2A 很长时仍可能留下 communication tail。([arXiv][1])

**combine0 和 dispatch1 可能同时想占网络。** 从稳态看，网络任务不只是一个 `C0`：

```text
D0
C0
D1
C1
...
```

你真正需要覆盖的是整个网络 workload，而不是单独证明某个 Attention 比某个 combine 长。

不过如果长 Attention 确实足够覆盖它们的**总网络工作**，这仍然不构成问题。

**DisagMoE 还有一个额外的通信优化。** 在带 PP 的 conventional setup 中，FFN 输出可能先做 combine，随后又通过 pipeline P2P 发送到下一 stage，相当于多搬一次。AF-Pipe 把传统 P2P 与 combine 融合进统一的 M2N/N2M stage。论文称这减少了冗余传输。([arXiv][1])

但如果我们讨论的是你这个**没有额外 PP boundary**的简化场景，这一点确实不是决定性因素。

## 所以可以这样理解两种设计

你的方案解决的是：

> **我有一整池 GPU，怎样通过跨 microbatch 调度，让 network 在 GPU 做有用计算时工作？**

这是 **temporal scheduling freedom**。

DisagMoE 再增加一个问题：

> **Attention 和 FFN 的 compute/network 需求根本不同，我为什么必须给它们同一种 GPU:NIC resource ratio？**

于是增加了 **spatial placement / resource allocation freedom**。论文明确把其贡献描述为 separate A/F worker groups，并自适应调整两边 GPU/NIC allocation。([arXiv][1])

所以我会把你的问题回答成：

> **不是因为 combine 必然进入关键路径。**
>
> 如果长 Attention 已经能把 dispatch + combine 全部盖住，你提出的 cross-microbatch schedule 非常合理，甚至在理想硬件模型下已经达到最优。
>
> **DisagMoE 的主要额外收益在于现实系统里 Attention 和 FFN 对 GPU 与 NIC 的需求比例不同、扩展效率也不同。分组之后才能分别给它们匹配资源；AF-Pipe 再负责让两组资源持续工作并隐藏 M2N/N2M 通信。**

这也是为什么论文反复强调的关键词是 **compute-communication imbalance**，而不仅仅是“communication overlap”。([arXiv][1])

[1]: https://arxiv.org/html/2605.11005 "DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated AF-Pipe Parallelism"
