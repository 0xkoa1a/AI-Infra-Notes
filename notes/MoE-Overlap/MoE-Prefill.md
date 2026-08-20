# 引言

你给的 arXiv ID `2605.02960` 当前是 **v2（2026-05-15）**，标题为 **MoE-Prefill: Zero Redundancy Overheads in MoE Prefill Serving**；早期版本常被称为 **ZeRO-Prefill**。它研究的不是一般在线 Chat/Decode，而是 **MoE 模型的 prefill-only serving**。([arxiv.org](https://arxiv.org/abs/2605.02960))

一句话概括：

> **传统 EP 固定 expert 的位置、移动 token activation；MoE-Prefill 反过来固定 token 所在 GPU，在层间异步移动 expert weight，并用长 prefill 的计算把 weight communication 完全盖住。**

传统 EP：

```text id="2f30y6"
GPU 上的 token
    ↓
AllToAll dispatch
    ↓
远端 expert compute
    ↓
AllToAll combine
    ↓
token 回来
```

MoE-Prefill：

```text id="40rl3d"
GPU 上的 token + 当前层全部 experts
    ↓
本地完成当前层 MoE compute
    │
    └── 同时 AllGather 下一层全部 experts
    ↓
下一层继续本地计算
```

它真正改变的不是“AllToAll 怎么做得更快”，而是：**对于长、大 batch、compute-bound 的 prefill，为什么一定要移动 activation，而不能移动 weight？** ([arxiv.org](https://arxiv.org/pdf/2605.02960))

# 背景 / 相关工作

## Prefill-only workload

论文关注大量**根本不需要 decode** 的任务，例如 classification、moderation、recommendation/ranking、factual verification、multiple-choice selection。二分类只需一次 forward 后比较候选 token 的 logits，不必 autoregressive 生成答案。作者把它抽象为 **prefill-as-a-service**。其匿名生产集群中，这类 workload 占输入 token 流量的 **65.3%**。([arxiv.org](https://arxiv.org/pdf/2605.02960))

它和交互式 decode 的系统特征不同：

* 目标主要是 throughput，而非单请求 latency；
* 可以积累很大的 batch；
* 输入可达数万 token；
* 因而形成很长的 compute-bound forward pass；
* production workload 中还常有 system prompt、user profile、document header 等 prefix sharing。([arxiv.org](https://arxiv.org/pdf/2605.02960))

这提供了两个机会：一是用长计算窗口隐藏 weight streaming；二是把相同 prefix 的请求放到同一 GPU，使 prefix attention 和 KV 只计算、保存一次。([arxiv.org](https://arxiv.org/pdf/2605.02960))

## 传统分布式 MoE 的三个问题

论文把瓶颈分成 **memory、compute、communication**。

**Memory。** MoE total parameters 很大，即使单 token 只激活少量 expert，所有 expert weight 仍需存储。长 prefill 下 KV 还随 batch size $B$ 和 sequence length $S$ 线性增长：

$$
V_{\mathrm{KV}}=2LBSN_{\mathrm{kv}}d_hb.
$$

自然语言就是：**batch 或 context 越大，KV memory 越大；长序列 aggressive batching 时，KV 甚至可能成为主要 HBM 压力。** 对 Qwen3-235B-A22B，传统 baseline 至少需要约 **4 张 H100** 才能容纳模型。([arxiv.org](https://arxiv.org/pdf/2605.02960))

**Compute。** 大 prefill 本来适合形成大 GEMM，但 distributed execution 会把 global token batch 分散到各 GPU，chunked prefill 又进一步切小 kernel。论文实测中，小 chunk 会降低 MFU，增大或关闭 chunking 能恢复 MFU。([arxiv.org](https://arxiv.org/pdf/2605.02960))

此外 routing skew 会产生不规则 per-expert GEMM 和 GPU straggler。论文对 Qwen3-30B-A3B 的测量中，聚合 48 个 MoE layer 后，expert token load 的 max/min 达 **16.15×**，单层更偏斜。MoE-Prefill并不会让 expert 收到的 token 数变均匀；它解决的是**固定 expert placement 导致的跨 GPU straggler**。([arxiv.org](https://arxiv.org/pdf/2605.02960))

**Communication。** 典型 DP+EP 中，每层有一次 AllToAll dispatch 和一次 AllToAll combine。论文给出的每设备、每层 activation communication 近似为：

$$
V_{\mathrm{EP}}
\approx
4k\frac{P-1}{P}BSHb.
$$

自然语言就是：**传统 EP 的通信随 token 数 $BS$ 和 Top-$k$ fan-out 一起增长。长 context 带来更多计算，也同时带来更多 activation traffic。** 此外 token permutation、reshape、alignment、inverse permutation 等辅助 kernel 也在 critical path。([arxiv.org](https://arxiv.org/pdf/2605.02960))

## 相关工作：MoE overlap

此前大量工作仍接受 **activation-routed EP**：

* **FasterMoE**：定制通信和 pipeline，让 AllToAll 与 expert computation 重叠；
* **Tutel**：优化 hierarchical AllToAll、parallelism 与 MoE kernels；
* **Lancet**：认为 expert GEMM 提供的 overlap window 不够，把 overlap 扩到 whole computation graph；
* **EPS-MoE**：面向 inference，根据 workload 选择 GroupGEMM/DenseGEMM，并让 computation 与 communication 自适应 overlap。([arxiv.org](https://arxiv.org/html/2502.19811v1?utm_source=chatgpt.com))

它们的共同点是：**expert placement 静态，activation 根据 routing 去找 expert。** 因此两次 activation movement 仍是 execution semantics 的一部分。MoE-Prefill 的区别不是“更好地隐藏 activation AllToAll”，而是**取消 activation AllToAll，把通信对象换成下一层 expert weight 的 AllGather**。([arxiv.org](https://arxiv.org/pdf/2605.02960))

## 相关工作：prefill、prefix cache、expert offloading

vLLM、SGLang、continuous batching、chunked prefill、prefill/decode disaggregation 等，大多仍服务于完整 prefill+decode。**PrefillOnly（SOSP 2025）** 已专门研究 prefill-only workload，尤其优化 KV-cache lifetime 和 job-level scheduling，但基本不修改 MoE execution stack。([arxiv.org](https://arxiv.org/pdf/2605.02960))

PagedAttention、RadixAttention 等把 prefix reuse 视为 cache hit；MoE-Prefill进一步把它变成 scheduling signal：优先把请求发到拥有最长 matching prefix 的 GPU。([arxiv.org](https://arxiv.org/pdf/2605.02960))

Expert offloading 的典型问题是：Router 之后才知道需要哪些 expert，如果 expert 不在 GPU，CPU→GPU transfer 会直接阻塞执行。**Pre-gated MoE** 等工作通过预测下一层 expert 来提前 prefetch；Fiddler、SwapMoE、MoE-Infinity 等从 orchestration、cache、offloading 角度优化。([arxiv.org](https://arxiv.org/abs/2308.12066?utm_source=chatgpt.com))

MoE-Prefill 不预测“下一层哪些 expert 会热门”，而利用**层执行顺序确定**这一点，直接准备下一层**完整 expert set**。因此不存在 cache miss，也不依赖 popularity；代价是即使某个 expert 没有 token 命中，也会被搬运。论文追求的不是最少 byte，而是让这些 byte **离开 critical path**。([arxiv.org](https://arxiv.org/pdf/2605.02960))

# 动机

论文的核心动机是：

> **传统 EP 默认 expert weight 是静态资产、activation 是自然应该移动的对象；长、大 batch 的 prefill 改变了这个权衡。**

传统 activation EP：

$$
V_{\mathrm{activation}}\propto BS.
$$

而给定模型时，一层 expert weight 的大小基本固定；$BS$ 增大却会让当前层计算窗口越来越长。于是会出现：

```text id="ssf9p3"
短 prefill

compute:  =========
weight:   =================
           ↑ 盖不住


长 prefill

compute:  ========================================
weight:   =================
           ↑ 完全藏进去
```

论文据此给 backend 提出三个约束：

* **C1 Feasibility**：weight、KV、activation 必须装得下；
* **C2 Performance**：per-layer critical path 上不能有 synchronous collective；
* **C3 Balance**：routing skew 不能让某张 GPU 成为 layer straggler。([arxiv.org](https://arxiv.org/pdf/2605.02960))

# 方法

MoE-Prefill 由 **frontend scheduler + AsyncEP backend** 组成。Frontend 决定“哪些请求去哪张 GPU、如何成 batch”；backend 决定“这批请求怎样执行”。二者通过一个 **saturation threshold $T$** 联动。([arxiv.org](https://arxiv.org/pdf/2605.02960))

## AsyncEP

以 $P=4$ 为例。每张 GPU 长期保存各层约 $1/4$ 的 expert shard，同时第一层完整 expert set 预先复制到各 GPU。执行第 $l$ 层时：

1. GPU 已有第 $l$ 层完整 experts；
2. Attention 本地执行；
3. Router 得到 Top-$k$；
4. 因为所有 experts 都在本 GPU，dispatch 变成本地操作；
5. 本地完成 MoE compute；
6. 同时独立 CUDA stream 通过 NVLink AllGather 第 $l+1$ 层完整 expert weights；
7. 第 $l$ 层结束时，第 $l+1$ 层 weights 已 ready。([arxiv.org](https://arxiv.org/pdf/2605.02960))

依赖变成：

```text id="v1ovs7"
Layer l compute
├─────────────────────────────┤

         Layer l+1 weight AllGather
         ├──────────────┤

                              Layer l+1 compute
                              ├───────────────────┤
```

因此论文把 AsyncEP 的 **on-path communication** 近似视为 0。不是没有通信，而是通信被移出 critical path。([arxiv.org](https://arxiv.org/pdf/2605.02960))

## Saturation threshold

要完全隐藏通信，**当前层计算时间至少要覆盖下一层 weight transfer**。

设最慢 transfer latency 为 $t_{\mathrm{EP}}$，GPU 峰值算力为 $F_{\mathrm{GPU}}$，$\gamma\ge 1$ 留 jitter 余量，则：

$$
T=t_{\mathrm{EP}}F_{\mathrm{GPU}}\gamma.
$$

自然语言就是：**每张 GPU 的 batch 必须提供足够 FLOPs，计算窗口才能完整覆盖 transfer。** 所以这里的 saturation threshold 更像“必须攒够的最小工作量”，不是通常意义上的最大 batch 限制。([arxiv.org](https://arxiv.org/pdf/2605.02960))

## CPU offloading

如果仅 D2D AsyncEP 仍然占太多 HBM，论文进一步把大部分 expert weight 放入 CPU pinned memory，并形成两级 pipeline：

```text id="ptlqst"
当前 layer compute
─────────────────────────────

下一 layer D2D AllGather
      ────────────────

更后面的 layer H2D prefetch
      ────────────────────────
```

PCIe H2D 和 NVLink D2D 可以并行，较慢的 channel 决定前面的 $t_{\mathrm{EP}}$。([arxiv.org](https://arxiv.org/pdf/2605.02960))

## Frontend scheduling

Frontend 同时做三件事：

* **Prefix-aware routing**：优先发给拥有最长 matching prefix KV 的 GPU；
* **True-FLOPs accounting**：共享 prefix 的请求不能简单按 token 数重复计费；
* **Overlap-aware batching**：每张 GPU 都要获得足够 FLOPs，使其达到 $T$。([arxiv.org](https://arxiv.org/pdf/2605.02960))

完整闭环是：

```text id="23ztoq"
硬件 bandwidth / model
        ↓
测出 weight transfer latency
        ↓
得到 saturation FLOPs T
        ↓
scheduler 给每张 GPU 分配足够工作
        ↓
current-layer compute ≥ next-layer transfer
        ↓
AsyncEP communication 被隐藏
```

所以 **AsyncEP 提供 overlap 的可能性，scheduler 保证实际 workload 真能提供 overlap window**。([arxiv.org](https://arxiv.org/pdf/2605.02960))

# 效果

论文在 **Qwen3-235B-A22B（128 experts，Top-8，约 22B active parameters）** 上测试 A100/H100/H200、BF16/FP8 等配置。相对最强 distributed baseline：

* 真实 prefill-only workload throughput 提升 **1.35–1.37×**；
* long-context synthetic workload 最高 **1.59×**；
* per-GPU MFU 达 **29.8–36.2%**。([arxiv.org](https://arxiv.org/pdf/2605.02960))

长 context 下优势更明显：传统 EP 的 activation AllToAll 随 $BS$ 增长，而 AsyncEP 的 layer weight transfer 不随 token 数同比增长；一旦 transfer 被 compute window 完全覆盖，context 再增长不会让这段通信重新暴露。([arxiv.org](https://arxiv.org/pdf/2605.02960))

通过 weight offloading / KV 策略，它还把 Qwen3-235B-A22B 的部署范围从传统方案需要的 **至少约 4 GPU** 扩展到 **1–8 GPU**。([arxiv.org](https://arxiv.org/pdf/2605.02960))

# 延伸

从“长序列计算通信掩盖”的视角，它和 FasterMoE、Lancet、Comet 的根本区别是：

```text id="nqnhav"
传统 overlap：

communication object = activation
activation A2A ↔ expert / other compute
```

```text id="kqimns"
MoE-Prefill：

communication object = weight
当前 layer compute ↔ 下一 layer weight transfer
```

因此它引入了一种更高层的自由度：**不仅优化 schedule，还重新选择 distributed execution 中到底移动 data 还是移动 parameters。** ([arxiv.org](https://arxiv.org/pdf/2605.02960))

一个自然的后续建模问题是：**什么时候移动 expert weights 比移动 activation 更划算？** 如果一层全部 expert weight 大小记作 $W_E$，weight AllGather 主要由 $W_E$ 决定，而传统 DP+EP activation traffic 为：

$$
V_{\mathrm{act}}
\approx
4k\frac{P-1}{P}BSHb.
$$

所以随着 $BS$ 增大，会出现 communication crossover；而即使 weight byte 数更大，只要当前计算能完全覆盖 transfer，AsyncEP 仍可能更优。这个“**通信量 crossover + overlap feasibility**”二维边界，是理解 MoE-Prefill 与其他 long-sequence MoE overlap 工作关系时很有用的抽象。这里是基于论文 Table 1 和 AsyncEP 数据流作出的进一步推导。([arxiv.org](https://arxiv.org/pdf/2605.02960))

论文自身的适用边界也很明确：主要针对 **throughput-oriented、batch-driven、prefill-only** workload；不适合强 latency-sensitive 的在线交互、无法稳定积累足够 batch 达到 $T$ 的 bursty arrival。低带宽 interconnect 也会增大 $t_{\mathrm{EP}}$，使 weight transfer 更难完全隐藏。([arxiv.org](https://arxiv.org/pdf/2605.02960))

放回长序列 MoE survey 中，它最值得保留的问题转换是：

> **与其只问“怎样把现有 AllToAll 藏进长序列带来的 compute window”，还可以问“能否换一种数据流，让通信量本身不再随 sequence length 增长，再把这段通信藏进窗口里？”**
