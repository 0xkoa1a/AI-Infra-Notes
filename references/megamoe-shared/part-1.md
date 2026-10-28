# MegaMoE Kernel：从源码理解通信与计算如何融合

MegaMoE 是 DeepSeek 在 <Entity category="software" value="DeepGEMM" disambig="DeepSeek GPU GEMM 算子库"/> 中实现的一个 **Persistent MoE Mega-Kernel**。它将 Expert Parallelism（EP）中的 Dispatch、两次 Grouped GEMM、SwiGLU、Combine 全部融合进一次 kernel launch，并利用 Blackwell 的 Tensor Core、TMA 和 NVLink 实现细粒度流水线。

我查阅了 DeepGEMM 的实际 CUDA 实现。建议重点看以下四个文件：

- <Link url="https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh" title="sm100_fp8_fp4_mega_moe.cuh"/>：Kernel 本体，包含 warp specialization、GEMM、dispatch 和 epilogue。
- <Link url="https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh" title="scheduler/mega_moe.cuh"/>：Persistent CTA 如何选择下一块 L1/L2 tile。
- <Link url="https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/csrc/apis/mega.hpp" title="csrc/apis/mega.hpp"/>：输入、输出、workspace 和 token pool 的布局。
- <Link url="https://github.com/deepseek-ai/DeepGEMM/blob/main/csrc/jit_kernels/heuristics/mega_moe.hpp" title="heuristics/mega_moe.hpp"/>：Block size、pipeline stages、SMEM 的选择。

这里以早期的 Expert-Wave 实现（commit `88965b0`）为主，因为它的 scheduler 正是前面 X-Stage 工作所针对的设计。DeepGEMM 后续主分支已经引入 ring buffer、shared expert 等额外机制，具体实现存在版本差异。

## 背景：MegaMoE 实际上在计算什么？

考虑 EP8，每张 GPU 保存一部分专家权重。每个 token 经过 router，选出 top-k 个专家。

对于一个 token \(x\)，每个被选中的专家执行：

\[
\begin{aligned}
[g,u]&=xW_1\\
h&=\operatorname{SiLU}(g)\odot u\\
o&=hW_2
\end{aligned}
\]

随后根据 router 的权重 \(p_e\)，合并专家输出：

\[
y=\sum_{e\in\operatorname{TopK}(x)}p_eo_e
\]

其中 \(W_1\) 包含 gate/up 两路投影，因此实际的第一层输出维度是 \(2I\)，SwiGLU 之后变成 \(I\)。

普通 MoE 推理一般拆成：

<box border radius="lg" padding={3} gap={1} align="center">
  <box background="surface-secondary" padding={2} radius="md" width="100%" align="center">
    <text weight="medium" size="sm">Dispatch All-to-All</text>
    <caption>把 token 发送到专家所在 GPU</caption>
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="rgba(75,136,208,0.10)" radius="md" padding={2} width="100%" align="center">
    <text weight="medium" size="sm">Grouped GEMM L1</text>
    <caption>Gate / Up projection</caption>
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="surface-secondary" radius="md" padding={2} width="100%" align="center">
    <text weight="medium" size="sm">SwiGLU + Quantization</text>
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="rgba(75,136,208,0.10)" radius="md" padding={2} width="100%" align="center">
    <text weight="medium" size="sm">Grouped GEMM L2</text>
    <caption>Down projection</caption>
  </box>
  <icon name="arrow-down" color="tertiary"/>
  <box background="surface-secondary" radius="md" padding={2} width="100%" align="center">
    <text weight="medium" size="sm">Combine All-to-All + Reduce</text>
  </box>
</box>

MegaMoE 的输出仍然是这些运算的结果，但运行方式发生了很大变化。

比如原先必须等整个 Dispatch kernel 完成才能启动 L1。而 MegaMoE 允许：

- Dispatch 收到专家 A 的一块 token，就让 L1 开始计算这一块。
- L1 完成一个输出 tile，就立即执行 SwiGLU 和 FP8 量化，送入 L2 的输入池。
- L2 对已经就绪的输入 tile 进行计算，完成后直接向原 token 所在 GPU 发出 NVLink remote stores。
- 以上工作由同一个 persistent kernel 内不同的 warp 推进。

因此，MegaMoE 实现的关键问题是：**如何让不同 warp 独立推进各自负责的工作，同时满足 Dispatch → L1 → SwiGLU → L2 → Combine 的数据依赖。**

## 整体架构：一个 SM 上的 Persistent CTA

原始 kernel 使用 SM100 的 warp specialization，典型的计算 CTA 内可以理解为：

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium">单个 SM 上的 Persistent CTA</text>
  <box background="surface-secondary" radius="md" padding={3} gap={2}>
    <row justify="between" align="center">
      **Dispatch warps**
      <text color="secondary" size="xs">4 warps</text>
    </row>
    <text size="sm">Routing 计数、远端 token 拉取、写入 L1 token pool</text>
  </box>
  <grid columns={2} gap={2}>
    <grid-item>
      <box background="rgba(58,139,211,0.10)" border radius="md" padding={3} gap={1}>
        <text weight="medium" size="sm">TMA Load A</text>
        <caption>Activation / scale</caption>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(58,139,211,0.10)" border radius="md" padding={3} gap={1}>
        <text weight="medium" size="sm">TMA Load B</text>
        <caption>Weight / scale</caption>
      </box>
    </grid-item>
  </grid>
  <box background="rgba(58,139,211,0.10)" border radius="md" padding={3} gap={1}>
    <text weight="medium" size="sm">MMA Issue warp</text>
    <text size="xs" color="secondary">提交 Blackwell UMMA 指令，结果进入 TMEM</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="tertiary"/></row>
  <box background="rgba(212,145,73,0.10)" border radius="md" padding={3} gap={1}>
    <row justify="between" align="center">
      <text weight="medium" size="sm">Epilogue warps</text>
      <text color="secondary" size="xs">多个 warpgroup</text>
    </row>
    <text size="sm">L1：SwiGLU + FP8 quantize + L2 pool 写入</text>
    <text size="sm">L2：BF16 转换 + NVLink remote store</text>
  </box>
  <caption>逻辑分工示意。原始实现还使用两个 CTA 组成 cluster 协同执行 MMA。</caption>
</box>

源码最外层基本就是根据 `warp_idx` 选择长期运行的角色。每个角色执行自己的循环，由 barrier、arrival counter 协调。

这也是 `__launch_bounds__(kNumThreads, 1)` 的目的之一：限制每个 SM 的驻留 CTA 数量，配合 persistent scheduling 管理固定的计算与通信角色。实际计算采用 2-CTA cluster，两个 CTA 在 Tensor Memory 和 MMA 执行中有协作关系。<Cite refs={["turn285302view1","turn643185view1"]}/>

下面从输入如何进入 token pool 开始，沿着 kernel 的数据路径逐步看实现。

## 数据与内存布局：为什么需要 Token Pool？

这是理解 MegaMoE 代码的第一个关键。

普通 GEMM 的输入矩阵是规则的二维数组，但 MoE 不同：每个 expert 实际分到多少 token，是运行时 router 决定的。

假设某张 GPU 拥有三个专家，本次收到的 token 数分别是：

| Expert | Token 数 | `BLOCK_M=16` 时需要的 M tiles |
|---|---:|---:|
| E0 | 27 | 2 |
| E1 | 8 | 1 |
| E2 | 19 | 2 |

MegaMoE 把它们按 expert 顺序摆进一个连续的 L1 token pool，每个 expert 占用整数个 `BLOCK_M`：

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium" size="sm">L1 Token Pool（按 M 维连续组织）</text>
  {#each [{name:"E0",a:27,b:32,c:"#6C9ED4"},{name:"E1",a:8,b:16,c:"#E4AE71"},{name:"E2",a:19,b:32,c:"#7BB9A1"}] as e}
    <row align="center" gap={2}>
      <box width="30px"><text size="xs" weight="medium">{e.name}</text></box>
      <box flex="1" background="surface-tertiary" radius="xs" height="29px" direction="row" clip>
        <box width={`${e.a/e.b*100}%`} background={e.c} justify="center" align="center">
          <text size="2xs" color="#142531" weight="medium">{e.a} valid</text>
        </box>
        <box flex="1" justify="center" align="center">
          <text size="2xs" color="secondary">{e.b-e.a} pad</text>
        </box>
      </box>
    </row>
  {/each}
  <caption>示例中的 5 个 M tiles 总共占据 80 行。不同专家的 tile 不会混在一起，方便直接按 expert 索引读取权重。</caption>
</box>

对于第 \(e\) 个专家，其 pool 起点就是前面所有专家占用 M blocks 数量的前缀和：

\[
\operatorname{offset}_e
=B_M\sum_{j<e}
\left\lceil\frac{M_j}{B_M}\right\rceil
\]

其中 \(M_j\) 是专家 \(j\) 的实际 token 数，\(B_M\) 是 `BLOCK_M`。

这让所有专家能够共用一套 GEMM kernel。每个 CTA 只需知道 `(expert_idx, m_block_idx, n_block_idx)`，即可计算 token、权重和输出的地址。

MegaMoE 同时维护几类重要的全局显存结构：

| 结构 | 保存什么 | 谁生产 / 谁消费 |
|---|---|---|
| `l1_acts` | Dispatch 后的 FP8 tokens | Dispatch / L1 |
| `l1_topk_weights` | Router weights | Dispatch / L1 epilogue |
| `l2_acts` | SwiGLU 后的 FP8 intermediate | L1 epilogue / L2 |
| `token_src_metadata` | 原 rank、原 token、top-k 槽位 | Dispatch / L2 epilogue |
| `combine_token_buffer` | 每个 top-k 槽位的 partial output | L2 epilogue / Reduce |
| Arrival counters | 数据就绪状态 | 不同 warp 之间的同步 |

这些 pool 位于 Global Memory。L1/L2 计算时通过 TMA 从 pool 搬运 tile 到 SMEM，结果由 Tensor Core 累加在 TMEM 中。

这里需要留意一个实现细节：**MegaMoE 的 L1→L2 intermediate 仍然可以经过 Global Memory**。单 kernel fusion 的价值主要是消除独立 kernel 之间的全局阶段等待，使中间数据能够按 tile 流水化生产和消费，并提高缓存复用。它并没有把整个 MoE 层的 intermediate 都放进 SMEM。<Cite refs={["turn215171view2","turn285302view3","turn289378view0"]}/>

## Dispatch：为什么是先 Push Metadata，再 Pull Token？

这是 MegaMoE 一个相当特别的设计。

一般我们想到 EP Dispatch，都会理解为：一个 GPU 主动把本地 token push 给目标专家所在的 GPU。

MegaMoE 的 dispatch 实际分两部分：

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium" size="sm">假设 Rank 0 的 token x 需要送给 Rank 1 的 Expert E</text>
  <grid columns={2} gap={2}>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
        **Rank 0**
        <text size="xs" color="secondary">Token owner</text>
        <text size="sm">x、topk_idx、weight</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
        **Rank 1**
        <text size="xs" color="secondary">Expert owner</text>
        <text size="sm">E 的权重、L1 pool</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center" gap={2} align="center">
    <icon name="arrow-right" color="info"/>
    <text size="sm">Rank 0 push token 索引与路由元数据到 Rank 1</text>
  </row>
  <row justify="center" gap={2} align="center">
    <icon name="arrow-left" color="info"/>
    <text size="sm">Rank 1 根据元数据，从 Rank 0 pull token 内容</text>
  </row>
  <box background="rgba(58,139,211,0.08)" radius="md" padding={2} align="center">
    <text size="sm">Rank 1 的 Dispatch warp 将 x 放进 E 的 L1 token pool</text>
  </box>
</box>

其具体步骤是：

1. **Routing count。** Dispatch warps 读取本地 `topk_idx`，统计需要送给每个 expert 的 token 数，通过 atomic 更新发送计数。
2. **Push metadata。** 将源 token 的索引和 top-k 槽位写到 expert owner rank 的对称显存工作区。
3. **交换计数并同步。** 各 rank 获得自己需要接收的 token 数，确保远端 metadata 已写完。
4. **Pull payload。** Expert owner 根据 metadata 主动通过 NVLink/TMA 读取远端 FP8 token 和 scale factor，并按 expert-major layout 写进 L1 pool。
5. **发布数据就绪状态。** 每写完一个 token，更新对应 pool block 的 `l1_arrival_count`。

为什么采用这种设计？

因为 expert owner 已经知道自己希望如何组织 token pool。它可以直接把远端 token 拉到适合 Grouped GEMM 的位置，避免发送方之间竞争目标位置所带来的复杂管理。

此外，一个 expert 的 token 可能来自多个 rank。源码里还用了 round-robin 式的来源选择，交错处理不同 rank 的 token，以分散远端数据搬运负载。

### GPU 如何直接访问其他 Rank 的显存？

底层使用 PyTorch Symmetric Memory。

每个 rank 都分配具有一致逻辑布局的显存块，经过 rendezvous 后获取其他 rank 对应区域的地址。

源码中的 `SymBuffer::map()` 本质上通过地址偏移将本地指针映射到远端：

```cpp
remote_ptr = local_ptr
           + (remote_base - local_base);
```

随后 kernel 可以通过远端指针完成 P2P 访存，而不必为每个 token 再调用一次 NCCL 或 DeepEP 通信 API。

当然，peer memory 映射、访问权限和设备拓扑必须事先满足条件。<Cite refs={["turn285302view2","turn507185view0"]}/>

## Dispatch 和 L1 怎样真正地重叠？

假设一个 expert 有 50 个 token，`BLOCK_M=16`。

当 Dispatch 已经拉回前 16 个 token，而其余 34 个还在 NVLink 上传输时，L1 就应该开始处理第一个 tile。

MegaMoE 为每个 pool block 准备一个计数器：

```cpp
// Dispatch warp：某个 token 已经写入 L1 pool
publish_token_to_l1_pool(token);

red_add_release(l1_arrival_count[pool_block], 1);
```

L1 的 TMA Load A warp 在读取相应 token tile 之前检查计数器：

```cpp
// L1 TMA producer
while (load_acquire(l1_arrival_count[pool_block])
       != valid_m) {
    // 等待这一块所有 token 到齐
}

tma_load_A(...);
```

这里的 `valid_m` 对普通完整 tile 是 16，对最后一个不满的 tile 则可能是 2。

只有对应 tile 的数据全部写入 pool，L1 才会发起 TMA load。

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium" size="sm">时间线示意</text>
  <svg viewBox="0 0 340 146" width="100%">
    <text x="5" y="22" fontSize="11" fill="currentColor">Dispatch</text>
    <text x="5" y="69" fontSize="11" fill="currentColor">L1 MMA</text>
    <text x="5" y="115" fontSize="11" fill="currentColor">L1 Epi</text>
    {#each [{x:83,n:0},{x:163,n:1},{x:243,n:2}] as e}
      <rect x={e.x} y="8" width="68" height="25" rx="4" fill="#A9C6E8"/>
      <text x={e.x+34} y="25" textAnchor="middle" fontSize="11" fill="#132C42">Block {e.n}</text>
      <rect x={e.x+18} y="54" width="68" height="25" rx="4" fill="#81B4DC"/>
      <text x={e.x+52} y="71" textAnchor="middle" fontSize="11" fill="#132C42">Block {e.n}</text>
      <rect x={e.x+37} y="100" width="68" height="25" rx="4" fill="#E5B77D"/>
      <text x={e.x+71} y="117" textAnchor="middle" fontSize="11" fill="#432914">Block {e.n}</text>
      <path d={`M${e.x+40} 33 V54`} stroke="currentColor" strokeWidth="1.2" strokeDasharray="3 3" opacity=".55"/>
    {/each}
    <path d="M83 139 H326" stroke="currentColor" opacity=".5"/>
    <path d="M321 135 L327 139 L321 143" stroke="currentColor" fill="none" opacity=".5"/>
    <text x="325" y="136" fontSize="10" fill="currentColor">t</text>
  </svg>
  <caption>同一时间，不同角色可以处理不同 tile。这里展示了依赖关系，方块长度不表示实际运行时长。</caption>
</box>

于是，在后续 token 仍在通过 NVLink 进入 L1 pool 时，Tensor Core 已经开始处理此前到达的 tile。

L1→L2 则使用类似的机制，只是从计数器变成了完成掩码 `l2_arrival_mask`：L1 epilogue 为每个完成的输出 N tile 设置一个 bit，L2 在读取完整的 intermediate 行时等待相应 bits 全部到齐。

原始 Expert-Wave 的实现中，L2 主要在本 wave 的 L1 工作分配结束后启动。源码注释还指出，它曾尝试按 K block 更细粒度地等待 L1 结果，但在已有 wave 大小能够提供足够重叠的场景下，这种做法收益为负，因此采用整行就绪检查。<Cite refs={["turn289378view0","turn285302view4"]}/>

## GEMM Mainloop：如何用一套计算管线运行 L1 和 L2？

接下来进入真正的计算 kernel。

MegaMoE 的 L1 和 L2 都是 GEMM：

\[
\begin{aligned}
L1 &: (M_e,H)\times(H,2I)\\
L2 &: (M_e,I)\times(I,H)
\end{aligned}
\]

对一个 GEMM tile，调度器需要提供：

```cpp
struct TileTask {
    Phase phase;       // Linear1 / Linear2
    int expert_idx;
    int m_block_idx;
    int n_block_idx;
};
```

随后同一套计算代码根据 `phase` 切换 activation pool、weight tensor、K dimension 和 epilogue。

### Block 划分

原始实现中，`BLOCK_N=128`。`BLOCK_M` 则由输入规模决定，面向单个专家 token 数较少的 MoE 场景，可以选择较小的 M tile。

对每个 GEMM tile：

\[
C_{B_M\times B_N}
=
\sum_{k=0}^{K/B_K-1}A_kB_k
\]

注意，`BLOCK_K` 只决定 MMA mainloop 如何迭代 reduction dimension。一个 `(m_block, n_block)` task 必须完成整个 K reduction，才产生最终输出 tile。

### Warp Specialization

这部分最值得结合 Blackwell UMMA 的执行方式来看。

每个 CTA 中，计算相关的 warp 有以下职责：

| Warp | 工作 |
|---|---|
| TMA Load A | 从 L1/L2 pool 加载 activation 和 scale |
| TMA Load B | 根据 expert_idx 加载对应的 weight 和 scale |
| MMA Issue | 等待数据就绪，发射 block-scaled UMMA |
| Epilogue | 从 TMEM 取出 accumulator，执行后处理 |

原始实现使用两个 CTA 组成 cluster，配合 Blackwell 的 2-CTA UMMA 指令。只有 leader CTA 的 MMA warp 发射指令，两个 CTA 参与数据加载、计算及 epilogue。相邻 CTA 对应的 N tile 会被配对，以满足协作要求。

其数据流水线大致如下：

<box border radius="lg" padding={3} gap={2}>
  <row align="center" gap={2}>
    <box background="surface-secondary" radius="md" padding={2} flex="1" align="center">
      <text weight="medium" size="sm">TMA Load A</text>
      <caption>Activation + SFA</caption>
    </box>
    <box background="surface-secondary" radius="md" padding={2} flex="1" align="center">
      <text weight="medium" size="sm">TMA Load B</text>
      <caption>Weight + SFB</caption>
    </box>
  </row>
  <row justify="center" align="center" gap={2}>
    <icon name="arrow-down" color="tertiary"/>
    <text size="xs" color="secondary">Full mbarrier</text>
  </row>
  <box border radius="md" padding={3} gap={2} align="center">
    **Shared Memory Pipeline**
    <row justify="center" gap={1}>
      {#each ["Stage 0","Stage 1","Stage 2"] as s}
        <box background="rgba(70,136,200,0.12)" radius="sm" padding={2} flex="1" align="center">
          <text size="xs" weight="medium">{s}</text>
          <text size="2xs" color="secondary">A / B / SF</text>
        </box>
      {/each}
    </row>
    <caption>示意为三级；真实 stage 数由 SMEM 预算决定。</caption>
  </box>
  <row justify="center" gap={2} align="center">
    <icon name="arrow-down" color="tertiary"/>
    <text size="xs" color="secondary">UMMA + UTCCP</text>
  </row>
  <box background="rgba(70,136,200,0.12)" radius="md" padding={3} align="center" gap={1}>
    **Tensor Memory**
    <text size="sm">FP32 accumulator</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="tertiary"/></row>
  <box background="rgba(218,151,76,0.12)" radius="md" padding={3} align="center">
    **Epilogue Warpgroup**
  </box>
</box>

具体地说，TMA Load A 和 B warp 各自遍历相同的任务序列，使用不同的 TMA descriptor 加载对应的数据。

MMA warp 等待 `full_barrier`，确认 A、B 和 scale factors 已经进入 SMEM，然后通过 UTCCP 将 scale factors 搬入 TMEM，并发出 `tcgen05` 系列的 block-scaled UMMA。

相关的真实源码接口包括：

```cpp
cute::UMMA::make_instr_desc_block_scaled<...>();

shared_storage.full_barriers[stage_idx].wait(phase);

ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(...);
```

MMA 发射后，通过 mbarrier 通知 TMA producer 该 SMEM stage 可以复用；完成整个 tile 的累加后，再通知 epilogue 读取 TMEM。

### 这里为什么需要两套 pipeline？

源码实际涉及两个相互衔接的 producer-consumer pipeline。

**SMEM pipeline** 负责 A/B/SF 数据的加载和消费。其 full/empty barriers 控制 TMA producer 什么时候可以复用 SMEM stage。

**TMEM accumulator pipeline** 负责 MMA producer 和 epilogue 的交接。只有 epilogue 消费完前一个 accumulator，MMA 才能安全复用对应的 TMEM 区域。

源码中的 `tmem_full_barriers` 和 `tmem_empty_barriers` 就在协调这种关系，原始实现采用两个 epilogue accumulator stages。<Cite refs={["turn289378view1","turn289378view2","turn289378view3"]}/>

后面 X-Stage 论文发现的瓶颈，恰恰发生在 TMEM accumulator pipeline 的下游：epilogue 被 remote-store backpressure 阻塞，最终占住 staging slot，使 MMA producer 等待 `tmem_empty`。

## L1 Epilogue：SwiGLU、Router Weight 和 FP8 量化

L1 GEMM 结束之后，得到的是 FP32 accumulator。

普通实现可能把 GEMM 输出写回 Global Memory，再启动单独的 SwiGLU kernel。但 MegaMoE 的 epilogue warp 直接读取 TMEM accumulator。

为了高效执行 SwiGLU，源码还将 gate/up 权重预先按固定粒度交错排列，使 TMEM load 获得的数据更容易配成 `(gate, up)`。

Epilogue 执行：

\[
h=\operatorname{SiLU}(g)\odot u\cdot p_e
\]

这里有一个有意思的优化：**Router Weight 提前乘到了 L1 的激活结果上。**

为什么可以这样做？

因为 L2 是线性变换，忽略量化舍入误差：

\[
(p_eh)W_2=p_e(hW_2)
\]

因此，原本在 Combine 阶段乘的 router weight，可以提前乘到 SwiGLU 输出上。这样 Combine 阶段只需要做 top-k 求和。FP8 再量化会引入舍入误差，因此这是一种数值上近似等价的实现。

执行完 SwiGLU 后，还需要进行 FP8 量化。

Epilogue warp 根据数值求 amax，生成对应的 UE8M0 scale factor，将输出量化为 FP8 E4M3，再通过 SMEM 和 TMA store 写入 L2 activation pool。

最后，它更新 `l2_arrival_mask`，通知 L2 对应的数据已经就绪。

这部分融合了原本单独的 activation、router weight multiplication 和量化步骤。<Cite refs={["turn507185view3","turn351026view1"]}/>

## L2 Epilogue：为什么它是 X-Stage 优化的重点？

L2 GEMM 完成后，epilogue 从 TMEM 读取 FP32 accumulator，并转换为 BF16。

对于每个有效 token，Dispatch 阶段已经保存了：

```cpp
struct TokenSrcMetadata {
    int src_rank;
    int src_token_idx;
    int src_topk_idx;
};
```

因此，L2 epilogue 不需要重新执行 unpermute。它能够直接找到源 token 的地址，并将结果写回：

```cpp
auto meta = token_src_metadata[pool_token_idx];

remote_ptr = map_remote(
    combine_buffer,
    meta.src_rank
);

remote_ptr[meta.src_topk_idx]
          [meta.src_token_idx]
          [hidden_offset] = output;
```

上面是逻辑伪代码，真实实现会按 tile 和 vector 组织内存写入。

Combine buffer 的组织方式很重要。假设一个 token 激活三个专家，那么其原始 rank 上有三个不同的输出槽位：

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium" size="sm">Rank 0：Token x 的 Combine Buffer</text>
  <grid columns={3} gap={2}>
    <grid-item>
      <box background="rgba(94,157,216,0.13)" padding={3} radius="md" align="center" gap={1}>
        <text weight="medium" size="xs">topk slot 0</text>
        <text size="sm">Expert A</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(94,157,216,0.13)" padding={3} radius="md" align="center" gap={1}>
        <text weight="medium" size="xs">topk slot 1</text>
        <text size="sm">Expert B</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(94,157,216,0.13)" padding={3} radius="md" align="center" gap={1}>
        <text weight="medium" size="xs">topk slot 2</text>
        <text size="sm">Expert C</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center"><icon name="arrow-down" color="tertiary"/></row>
  <box background="rgba(215,154,83,0.12)" radius="md" padding={3} align="center" gap={1}>
    **Final Reduce**
    <text size="sm">FP32 累加 → BF16 输出 y</text>
  </box>
</box>

每个 expert 的输出写到不同槽位，所以即使来自多个 rank，也不会发生同一地址的写入冲突。

等所有 rank 的 L2 输出写回完成后，kernel 执行一次跨 rank 的 NVLink barrier，随后在原 token 所在 GPU 上读取各个 top-k 槽位，以 FP32 累加，最后输出 BF16。

这里的跨 rank barrier 仍然很重要。**Remote store 发射返回只意味着写入请求进入发送路径，不能直接认为接收端数据已经完成可见。** Combine reduce 必须遵循正确的完成与可见性同步。<Cite refs={["turn285302view5","turn285302view6","turn295793view1"]}/>

## Persistent Scheduler：所有 SM 如何分配 L1/L2 工作？

前面解释了一个 CTA 怎样计算 tile，现在来看所有 CTA 怎样分配这些 tile。

原始代码的核心是 `MegaMoEScheduler`。

每个计算角色都通过 `scheduler.for_each_block(...)` 遍历同一套逻辑任务序列。调度器给出任务，TMA Load A/B、MMA 和 epilogue 分别执行自己负责的部分。

这种设计减少了跨角色分发任务所需的额外全局队列，但不同角色的推进速度仍然可以由流水线 barrier 解耦。

### Expert-Wave 的调度规则

假设本 GPU 有四个 expert，一个 wave 包含两个 expert。

原始执行顺序是：

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium" size="sm">Expert-Wave 调度示例</text>
  {#each [{a:"Wave 0",b:["E0 · L1","E1 · L1"],c:["E0 · L2","E1 · L2"]},{a:"Wave 1",b:["E2 · L1","E3 · L1"],c:["E2 · L2","E3 · L2"]}] as w}
    <box border radius="md" padding={2} gap={2}>
      <text size="xs" color="secondary" weight="medium">{w.a}</text>
      <row gap={1}>
        {#each w.b as t}
          <box background="#B5D5F0" padding={2} flex="1" radius="sm" align="center"><text color="#19354B" size="xs" weight="medium">{t}</text></box>
        {/each}
      </row>
      <row justify="center"><icon name="arrow-down" color="tertiary" size="sm"/></row>
      <row gap={1}>
        {#each w.c as t}
          <box background="#EAC5A1" padding={2} flex="1" radius="sm" align="center"><text color="#443020" size="xs" weight="medium">{t}</text></box>
        {/each}
      </row>
    </box>
  {/each}
  <caption>每个方块在实际执行时仍会拆成多个 `(m_block, n_block)` GEMM tile。</caption>
</box>

为什么不逐个 expert 完成 L1→L2？

因为在 MoE 的小 M GEMM 场景中，单个 expert 产生的 tile 数量可能太少，无法让全部 SM 都保持忙碌。

将多个 expert 组成一个 wave，有利于获得足够的 tile parallelism，同时限制同时活跃的 expert 权重数量，以维持较好的 L2 cache locality。

Wave 大小由启发式方法决定，核心考虑包括：

- 每个 expert 预计有多少 token。
- 每个 expert 的 L1/L2 总共可以产生多少 tile。
- GPU 有多少 SM。
- Wave 过大时权重工作集与缓存局部性如何变化。

### 任务怎样分配到每个 CTA？

源码采用一个很直接的 strided scheduler。

假设 GPU 有 \(S\) 个计算 CTA。一个 CTA 从自己的 `blockIdx.x` 开始，遍历逻辑任务序列中的：

\[
k=\operatorname{blockIdx.x}+tS
\]

其中 \(t=0,1,2,\ldots\)。

源码中的核心就是：

```cpp
block_idx = blockIdx.x;

// 取出当前逻辑任务
get_next_block();

// 下次领取隔 S 个任务的工作
block_idx += kNumSMs;
```

例如一个 wave 有 400 个逻辑 tile、200 个 CTA，那么 CTA 0 负责 tile 0、200，CTA 1 负责 tile 1、201，以此类推。

这里原始实现的相邻 CTA 会配对参与 2-CTA MMA，所以逻辑 tile 序列还需要保证相邻任务的 M/N 坐标满足协作约束。

这个方案不需要每执行一个 tile 都向全局 work queue 发起 atomic fetch。

相应的代价是，在 routing skewed、不同专家 tile 耗时差别较大时，静态 strided 分配仍可能出现负载不均衡。<Cite refs={["turn643185view2","turn643185view1"]}/>

## X-Stage 论文到底修改了 Scheduler 的哪一部分？

现在就可以很准确地定位前面论文的优化了。

原始 `get_next_block()` 的核心是一个两阶段状态机：

```cpp
if (next_phase == Linear1) {
    if (fetch_next_l1_block()) {
        return Linear1;
    }

    next_phase = Linear2;
}

if (next_phase == Linear2) {
    if (fetch_next_l2_block()) {
        return Linear2;
    }

    next_phase = Linear1;
}
```

它在一个 wave 内先发出 L1 任务，再发出 L2 任务。

X-Stage 论文则改成两个独立的逻辑任务流，并根据 L1 对 L2 的 dependency lead 决定下一次调度哪个阶段：

```cpp
while (has_L1 || has_L2) {
    if (has_L2 && L1_is_far_enough_ahead()) {
        emit(next_L2_tile);
    } else {
        emit(next_ready_L1_tile);
    }
}
```

具体算法见论文 <Link url="https://arxiv.org/html/2607.23264v2#A1" title="Appendix A"/>。

它还必须检查 L1 输入是否已经被 Dispatch 准备好，以及输出 ring slot 是否可以写入。

### 为什么这会影响 MMA？

考虑一个 L2 tile 完成后，epilogue 需要发射很多 remote stores。

原始 Expert-Wave 连续安排下一个 L2：

<box border radius="lg" padding={3} gap={2}>
  <text size="sm" weight="medium">原始调度</text>
  <row gap={1} align="center">
    <box background="#B7D6F1" radius="sm" padding={2} flex="1" align="center"><text size="xs" color="#1B3851">L2 MMA</text></box>
    <box background="#E9BE9B" radius="sm" padding={2} flex="1" align="center"><text size="xs" color="#423020">L2 Epi</text></box>
    <box background="#E8A69B" radius="sm" padding={2} flex="1" align="center"><text size="xs" color="#492A24">Stall</text></box>
    <box background="#B7D6F1" radius="sm" padding={2} flex="1" align="center"><text size="xs" color="#1B3851">L2 MMA</text></box>
  </row>
  <text size="sm" weight="medium">X-Stage 调度</text>
  <row gap={1} align="center">
    <box background="#B7D6F1" radius="sm" padding={2} flex="1" align="center"><text size="xs" color="#1B3851">L2 MMA</text></box>
    <box background="#E9BE9B" radius="sm" padding={2} flex="1" align="center"><text size="xs" color="#423020">L2 Epi</text></box>
    <box background="#9CC7A8" radius="sm" padding={2} flex="1" align="center"><text size="xs" color="#173924">L1 MMA</text></box>
    <box background="#B7D6F1" radius="sm" padding={2} flex="1" align="center"><text size="xs" color="#1B3851">L2 MMA</text></box>
  </row>
  <caption>这是依赖与执行顺序示意，实际 MMA、epilogue 可在不同 warp 中并行。</caption>
</box>

原本较短的 L2 mainloop 可能无法完全覆盖上一个 epilogue 的 remote-store span。若 staging slot 迟迟无法释放，MMA producer 就会在 `tmem_empty` 停顿。

新调度可以在此处安排一个已经 ready 的 L1 tile。由于 L1 没有 L2 那样的 Combine remote stores，并且它的计算仍然是必要工作，便可以利用这段计算时间让远端存储请求继续排空。

最终，它同时改变了两个因素：L2 remote stores 的发射间隔，以及下一次复用本地 staging slot 之前能够执行的有用计算量。<Cite refs={["turn295793view0","turn295793view1"]}/>

## 把整个 Kernel 还原成一个 CUDA 程序

如果把细节收起来，MegaMoE 大致可以写成下面这个结构。**这是根据真实实现整理的教学版伪代码，保留数据流与同步语义，省略了 CTA cluster、mbarrier phase、scale factor layout 等细节。**

<CodeBlock language="cpp" editable={false}>__global__ void mega_moe_kernel(Context ctx) {
    int warp = threadIdx.x / 32;

    // 不同 warp 各自运行长期存在的角色循环
    if (warp < NUM_DISPATCH_WARPS) {
        count_routed_tokens(ctx);
        push_routing_metadata(ctx);

        nvlink_barrier_before_pull(ctx);

        for (auto token : assigned_tokens(ctx)) {
            pull_remote_token_to_l1_pool(ctx, token);
            save_source_metadata(ctx, token);
            publish_l1_arrival(ctx, token);
        }

        cleanup_workspace(ctx);
    }
    else if (warp == LOAD_A_WARP) {
        for (auto task : scheduler(ctx)) {
            wait_activation_ready(ctx, task);

            for (int k = 0; k < task.num_k_tiles; ++k) {
                wait_smem_empty(k);
                tma_load_activation_and_sf(ctx, task, k);
                signal_smem_full(k);
            }
        }
    }
    else if (warp == LOAD_B_WARP) {
        for (auto task : scheduler(ctx)) {
            for (int k = 0; k < task.num_k_tiles; ++k) {
                wait_smem_empty(k);
                tma_load_weight_and_sf(ctx, task, k);
                signal_smem_full(k);
            }
        }
    }
    else if (warp == MMA_WARP) {
        for (auto task : scheduler(ctx)) {
            wait_tmem_empty();

            for (int k = 0; k < task.num_k_tiles; ++k) {
                wait_smem_full(k);
                issue_umma(ctx, task, k);
                release_smem_stage(k);
            }

            signal_tmem_full();
        }
    }
    else if (warp >= EPILOGUE_START) {
        for (auto task : scheduler(ctx)) {
            wait_tmem_full();

            if (task.phase == Linear1) {
                swiglu_and_quantize(ctx, task);
                store_l2_activations(ctx, task);
                publish_l2_arrival(ctx, task);
            } else {
                convert_to_bf16(ctx, task);
                remote_store_combine_partials(ctx, task);
            }

            release_tmem_stage();
        }

        nvlink_barrier_before_reduce(ctx);
        reduce_topk_partials(ctx);
    }
}</CodeBlock>

真正的代码当然更加复杂，因为这里所有的 warp 都在同时推进：

- TMA producer 可能已经在准备第 \(n+1\) 个 tile。
- MMA producer 正在计算第 \(n\) 个 tile。
- Epilogue 正在处理第 \(n-1\) 个 tile。
- Dispatch warp 还在拉取后续 experts 的 tokens。
- Remote-store 请求可能仍在 GPU 的通信路径中排空。

整个 kernel 的吞吐由这些生产者、消费者和异步资源共同决定。

## 我认为最值得深入源码的三个地方

从 CUDA kernel 优化的角度，MegaMoE 最有研究价值的是以下三个接口。

**`MegaMoEScheduler::get_next_block()`** 决定 task 的先后顺序。它影响权重 cache locality、计算连续性、L1/L2 依赖等待和 remote-store burst pattern。X-Stage 的主要改动就在这一层。

**`l1_arrival_count` / `l2_arrival_mask`** 决定 task 何时真正可以被执行。它们是让跨 rank 通信、跨 CTA 生产与消费得以流水化的关键，也给调度器施加了正确性约束。

**`tmem_empty_barriers` 和 L2 epilogue** 决定了 remote-store latency 是否会沿着 staging buffer 反压到 MMA。要理解 X-Stage 的性能收益，仅看 NVLink 吞吐是不够的，必须把这个环节也分析清楚。

把这三个部分连起来，你就能理解 MegaMoE 的一个核心设计问题：**Scheduler 应该以什么顺序提交已经就绪的 L1/L2 tiles，才能同时保持 Tensor Core 忙碌、减少权重缓存失效，并避免 remote-store 背压占用计算关键路径？**

这也是为什么我认为，若要在 MegaMoE 上进一步做硬件感知调度研究，最合适的切入点是 scheduler 与 epilogue 的交界处：它既包含实际执行的自由度，也能观察到通信背压对计算流水线的真实影响。

---

If you want, I can:

- 详细展示Dispatch到L1的异步数据流时间线
- 解释源码中Persistent CTA的角色分工及同步机制
- 描述L1和L2 GEMM计算流程及tile任务切换机制