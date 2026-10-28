对，你的理解基本正确。**MegaMoE 采用「源端 push routing metadata，目标端 pull token payload」的设计，核心目的是让 Expert Owner 自己组织 GEMM 输入，将跨 GPU 的数据搬运与按 Expert 的 permute 融合在一起。**

不过这里有两个值得区分的问题：

- 为什么 metadata 要由源端 push？因为只有源端天然知道每个 token 被 router 分配给了哪些 expert。
- 为什么 token payload 要由目标端 pull？因为目标端负责组织自己的 expert-major token pool，可以在接收数据时直接写入 GEMM 所需的布局。

我核对了 <Link url="https://github.com/deepseek-ai/DeepGEMM/blob/88965b078186ee7510ab9fc4f1d5ebc19adfa8d1/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh" title="MegaMoE 原始 CUDA 源码"/>，下面具体解释。

## 1. Push metadata：具体发送了什么？

假设 EP8，每个 rank 拥有 4 个 expert，总共 32 个 expert，每个 token 的 `topk=2`。

Rank 0 有一个 token：

```text
token_idx = 17

topk_idx[17] = [9, 23]
topk_weight[17] = [0.7, 0.3]
```

意味着它需要由 Expert 9 和 Expert 23 处理。

此时 Rank 0 会分别向这两个 expert 所在的 GPU 发送路由信息。

**但真正发送的不是整个 `topk_idx` 数组，也不是 token vector，而是一个 32-bit 的 flattened index：**

\[
\text{token\_topk\_idx}
=\text{token\_idx}\times K_{\text{top}}+\text{topk\_slot}
\]

在这个例子中：

| 目标 Expert | 目标 Rank | Push 的值 |
|---|---|---|
| Expert 9 | Rank 2 | 34 |
| Expert 23 | Rank 5 | 35 |

由于 `topk=2`，目标端拿到 35 就能够反推出：

```cpp
src_token_idx = 35 / 2;  // 17
src_topk_idx  = 35 % 2;  // 1
```

源 Rank 是由 metadata 所在的分区隐式表示的，目标 Expert 由 metadata 所在的 expert 分区表示。

也就是说，每条 routing metadata 中真正需要传输的只有一个 uint32。

此外，源端还会传输**每个源 Rank 发往各个 Expert 的 token 数量**，用于让目标端知道每个分区有多少有效条目。<Cite refs={["turn464995view0","turn979023view1"]}/>

### 为什么不让目标端主动 pull `topk_idx`？

因为此时目标端只知道自己有哪些 expert，还不知道哪些 token 被分配给了自己。

如果直接从各个源 GPU 拉取原始 routing 表，目标端就需要扫描各源 Rank 的 `topk_idx`，过滤出指向本地 expert 的条目。

例如 Rank 2 要处理 Expert 8–11，但它并不知道 Rank 0 的哪些 token 选择了这四个 expert。

而源 Rank 0 只需要扫描一次自己的 routing 结果，就能把条目发送到对应的 expert。

**这相当于让 Routing Producer 构建一个按 Expert 分类的稀疏反向索引。**

当然，也可以先让源端建立反向索引，再由目标端 pull 这个索引。但依然需要源端执行分类、计数和地址发布。MegaMoE 直接选择了 push metadata。

## 2. Metadata 写入目标端的哪个位置？

源码使用三个索引组织 metadata：

```cpp
get_src_token_topk_idx_ptr(
    local_expert_idx,
    src_rank_idx,
    slot_idx
);
```

因此，目标端的 metadata workspace 逻辑上是：

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium" size="sm">目标 Rank 2 的 metadata workspace</text>
  <box background="surface-secondary" radius="md" padding={2} gap={2}>
    <text weight="medium" size="sm">Expert 9</text>
    {#each [{rank:"Src Rank 0",data:"[34, 61, ...]"},{rank:"Src Rank 1",data:"[8, 19, ...]"},{rank:"Src Rank 2",data:"[12, 27, ...]"}] as r}
      <row align="center" gap={2}>
        <box width="95px"><text size="xs" color="secondary">{r.rank}</text></box>
        <box background="rgba(68,144,213,0.13)" flex="1" padding={2} radius="sm">
          <text size="xs" weight="medium">{r.data}</text>
        </box>
      </row>
    {/each}
    <caption>其他 Rank 的分区同理；括号内是 flattened token-topk index 示例。</caption>
  </box>
  <box background="surface-secondary" radius="md" padding={2} gap={1}>
    <text weight="medium" size="sm">Expert 8、10、11</text>
    <text color="secondary" size="xs">同样按源 Rank 分区存储</text>
  </box>
</box>

源端扫描 `topk_idx` 时，通过 atomic 预先分配每个 expert 的写入偏移量，再把 flattened index 写到目标 GPU 的对应分区。

关键源码是：

```cpp
const auto dst_rank_idx =
    expert_idx / kNumExpertsPerRank;

const auto dst_slot_idx =
    atomicAdd_block(
        shared_storage.expert_token_count + expert_idx, 1
    );

const auto dst_ptr =
    workspace.get_src_token_topk_idx_ptr(
        expert_idx % kNumExpertsPerRank,
        sym_buffer.rank_idx,
        dst_slot_idx
    );

*sym_buffer.map(dst_ptr, dst_rank_idx) =
    token_topk_idx;
```

注意，这里的 `dst_slot_idx` 是通过计数器分配的。这样，多个 SM/warp 产生的 routing entries 可以写入同一个 Expert 的连续 metadata 区域。

每个源 Rank 还有独立分区，所以不需要让来自不同 GPU 的条目争抢同一条 metadata 写入队列。<Cite refs={["turn215103view3","turn979023view1"]}/>

## 3. Pull token：确实是在完成 GEMM 所需要的 permute

目标端收到 metadata 和计数、经过 NVLink barrier 后，便可以开始组织自己的 L1 input pool。

举个例子，Rank 2 有两个 expert：

- Expert 8 需要来自 Rank 0、Rank 1 的 token。
- Expert 9 也需要来自多个源 Rank 的 token。

这些 token 在源端的原始位置可能是完全随机的。

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium" size="sm">源端：按原始 token 顺序存储</text>
  <row gap={1}>
    {#each [{id:"T0",e:8},{id:"T1",e:9},{id:"T2",e:8},{id:"T3",e:9},{id:"T4",e:9},{id:"T5",e:8}] as t}
      <box flex="1" background={t.e===8?"#B5D4F1":"#E9C5A3"} radius="sm" padding={{x:1,y:2}} align="center" gap={1}>
        <text size="xs" color="#202D38" weight="medium">{t.id}</text>
        <text size="3xs" color="#283846">E{t.e}</text>
      </box>
    {/each}
  </row>
  <row align="center" justify="center" gap={2}>
    <icon name="arrow-down" color="secondary"/>
    <text size="xs" color="secondary">Metadata 指导远端 gather + pool packing</text>
  </row>
  <text weight="medium" size="sm">目标端：L1 Token Pool</text>
  <row gap={1}>
    {#each [{id:"T0",e:8},{id:"T2",e:8},{id:"T5",e:8},{id:"T1",e:9},{id:"T3",e:9},{id:"T4",e:9}] as t}
      <box flex="1" background={t.e===8?"#B5D4F1":"#E9C5A3"} radius="sm" padding={{x:1,y:2}} align="center" gap={1}>
        <text size="xs" color="#202D38" weight="medium">{t.id}</text>
        <text size="3xs" color="#283846">E{t.e}</text>
      </box>
    {/each}
  </row>
  <row gap={3} align="center">
    <row gap={1}><box size="12px" radius="2xs" background="#B5D4F1"/><text size="xs">Expert 8</text></row>
    <row gap={1}><box size="12px" radius="2xs" background="#E9C5A3"/><text size="xs">Expert 9</text></row>
  </row>
  <caption>简化示意，实际还会按照 BLOCK_M 对各 Expert 的池空间进行对齐，并使用专门的 scale factor 布局。</caption>
</box>

源码中的关键地址计算是：

```cpp
// 根据 metadata 找源端 token
src_token_idx = src_token_topk_idx / kNumTopk;

// 根据目标 Expert 和当前 token 在 Expert 内的编号
// 计算它应该放在 L1 pool 的哪里
pool_token_idx =
    expert_pool_block_offset * BLOCK_M
    + token_idx_in_expert;

// 通过 NVLink 从源端读取
src_ptr = sym_buffer.map(
    input_token_buffer[src_token_idx],
    src_rank
);

// 直接存入 GEMM 的 L1 pool
dst_ptr = l1_token_buffer[pool_token_idx];
```

实际 payload 由 `tma_load_1d` 从源端搬到 SMEM staging buffer，再由 `tma_store_1d` 写入目标端的 L1 token pool。

它还会从源端读取对应的 FP8 scale factors、router weight，构造 Combine 所需的 source metadata，最后发布 L1 tile 的 arrival count。<Cite refs={["turn464995view3","turn979023view0"]}/>

因此，你所说的「直接 Grouped GEMM，不需要 permute」需要稍微精确一点：

**Permute 仍然发生了，只是融合在 Dispatch 的远端 gather 和本地 pool packing 中，省去了一次独立的 permute pass。**

而且 pool 也并非绝对紧凑。每个 Expert 的实际 token 数量需要按 `BLOCK_M` 向上对齐，最后一个 M tile 可能存在 padding。

## 4. 为什么 Payload 不也 Push 过去？

实际上，源端也可以直接把 token payload 写入目标端预先分配好的 Expert Pool。

这种方案在功能上完全可行，也能够产生同样适合 Grouped GEMM 的布局。

但它需要解决一个额外问题：**源端必须协调好每个 token 的最终目标地址。**

MegaMoE 当前的设计中：

- 源端只负责根据 router 决定哪些 token 发往哪个 expert，并把小型索引发送出去。
- 目标端掌握各专家的接收计数，自己决定 token 在 L1 pool 中的顺序。
- 目标端还可以按源 Rank round-robin 处理 token，分散 NVLink pull 请求。
- FP8 token、scale factors 和 router weight 能够由同一目标端 dispatch 逻辑一起组织，并在写入完成后发布 tile readiness。

这样目标端对 GEMM 输入缓冲区和数据就绪状态的管理更集中。

不过，源码并没有提供一个严格的 `push payload` 与 `pull payload` 性能对照实验，所以不能断言 pull 的 NVLink 带宽必然更高，或者该选择一定优于设计良好的 push-permute 实现。

## 一个进一步的性能取舍

MegaMoE 这个设计也有代价。

因为源端先 push metadata，目标端才能知道要 pull 哪些 token，所以 Dispatch 开始时存在一次元数据同步依赖。源码明确使用全 Rank 的 NVLink barrier，保证 metadata 和计数已经可见，然后才开始 payload pull。

此外，pull 的远端地址可能是不规则的，具体的请求大小和并发度会影响 NVLink 吞吐。

因此，MegaMoE 实际是在权衡 metadata 同步成本、跨 GPU 访问模式、目标端 buffer 管理，以及与 GEMM 的流水线融合程度。

在这个设计中，**最有价值的收益是目标端可以边接收、边整理成 Expert-major tiles，并让已经就绪的 tiles 立即进入 L1 GEMM。** 这使 Dispatch 和后面的计算能够真正重叠。

---

If you want, I can:

- 分析MegaMoE中metadata push和payload pull的性能权衡细节
- 探讨目标端如何管理L1 Token Pool中的token位置和对齐策略
- 详细描述metadata写入目标端workspace时的地址计算和并发写入机制