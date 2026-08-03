# FlashAttention-4 Blackwell Forward 源码简报

> 源码基线：`flash-attention` commit `00756db9d921da0846453283ddfbeb7457abd09b`。正文描述 SM100/SM110 的常规 forward 内核 `FlashAttentionForwardSm100`；不讨论 backward。`head_dim=256` 与 MLA 是独立特化内核，差异见末尾。

## 1. Tile：先确定工作粒度

记 `M=tile_m`、`N=tile_n`、`D=round_up(head_dim, 16)`、`Dv=round_up(head_dim_v, 16)`。常规配置是 `M=N=128`。`q_stage=2` 时一个 CTA 同时维护两个 Query subtile；有效 Q 工作块因此是 `2M=256` 行。有效 Q 长度不超过 `M` 时退化为 `q_stage=1`。

| 对象 / 运算 | 1CTA 的典型 shape | 2CTA cluster 的典型 shape | 存放位置 |
|---|---:|---:|---|
| 每个 `Q_j` | `128 × D` | `256 × D`，每 CTA 128 行 | SMEM |
| `K_i` | `128 × D` | 逻辑 shape 不变，B operand 分布到两个 CTA | SMEM |
| `S_j = Q_j K_i^T` | `128 × 128` | `256 × 128`，每 CTA 128 行 | TMEM，FP32 |
| `P_j = exp(S_j-max)` | `128 × 128` | `256 × 128` | TMEM，输入 dtype |
| `V_i` | `128 × Dv` | 逻辑 shape 不变，B operand 分布到两个 CTA | SMEM；源码按 `Dv × 128` 视图供 UMMA 读取 |
| `O_j += P_j V_i` | `128 × Dv` | `256 × Dv`，每 CTA 最终写 128 行 | TMEM，FP32 |

所以默认两次 UMMA 的 tile 是：

- QK：1CTA 为 `(128,128,D)`，2CTA 为 `(256,128,D)`；
- PV：1CTA 为 `(128,Dv,128)`，2CTA 为 `(256,Dv,128)`；
- 一个 scheduler work tile 覆盖 `q_stage × M` 行/CTA；常见的 2CTA、`q_stage=2` 配置合计覆盖 512 个 Q 行。

2CTA 不是普遍默认：接口只在 dense、non-causal、非 SplitKV、非 varlen/block-sparse，且典型 `(D,Dv)=(128,128)` 或 `(192,128)`、有效 Q 长度大于 256 时选择它。其余通常走 1CTA。SplitKV 的异形 head 在长 K 上还可能把 `N` 改成 64；上面的 128 是主路径默认值，不是硬编码的不变量。

## 2. 一条完整流水线：buffer、timeline 与同步

### 2.1 执行角色与 buffer 地图

标准 `q_stage=2` CTA 有 16 个 warp（512 threads）：warp 0–3 做 `Q_0` softmax，4–7 做 `Q_1` softmax，8–11 做 correction，warp 12 单独发射 UMMA，13 做 output epilogue，14 发射 TMA load，15 空闲或运行 CLC scheduler。2CTA 时两个 CTA 都保留这套角色，但只有 leader CTA 的 MMA warp 发射 cluster-wide UMMA。内核用 `setmaxnreg` 把寄存器从 load/MMA/epilogue 侧让给 softmax 与 correction。

SMEM 采用 CUTLASS 为 UMMA operand 生成的硬件 swizzle，而不是普通 row-major：`sQ/sK` 为 K-major，`sV` 为 MN-major，`sO` 按输出布局生成。逻辑 buffer 如下：

```text
sQ[q_stage]       : 每槽 M×D；通常 2 槽
sKV[kv_stage]     : 每槽 max(N×D, N×Dv)
                    同一物理区交替解释成 sK 或 sV，时间序列是 K0,V0,K1,V1,...
sO[q_stage]       : 每槽 M×Dv
sScale            : q_stage×M 个 FP32 临时 scale/row_sum
                    + q_stage×M 个可选 row_max
pipeline mbarrier : 每条环形 pipeline 每槽一对 full/empty barrier
```

`kv_stage` 计数的是单个 K 或 V tile，不是 K/V 对。它按约 224 KiB payload 预算自动取值：

\[
kv\_stage=\min\left(\left\lfloor\frac{224\text{ KiB}-bytes(sQ,sO)}{\max(bytes(K_i),bytes(V_i))/cta\_group}\right\rfloor,32\right).
\]

以 FP16/BF16、`q_stage=2` 为例：1CTA 的 `D=Dv=64/96/128` 分别得到 10/5/3 个 KV slot；2CTA 的 `D=Dv=128` 每 CTA 的 KV footprint 减半，得到 6 个 slot。`(D,Dv)=(192,128)` 会让 `sO` 与 `sQ` 共用物理区；1CTA 还采用 `[48 KiB, 32 KiB, 48 KiB]` 的不等距三槽布局。一般情况下 `sQ` 与 `sO` 分开；K/V 始终共用 `sKV`。SplitKV 且 `Dv>=128` 也会复用 `sQ/sO`，此时关闭 persistent scheduling，避免下一 work tile 的 Q 覆盖尚未写回的 O。

TMEM 保存两个 score/probability 槽和 `q_stage` 个 output accumulator。列偏移为：

```text
S0: 0                 S1: N
P0: N/2               P1: N + N/2
O_j: 2N + j×Dv
总列数: 2N + q_stage×Dv <= 512
```

默认 `N=Dv=128, q_stage=2, FP16/BF16` 时，`S0/S1/O0/O1` 各占 128 列，恰好填满 512 列。softmax 先把 FP32 `S_j` 读到寄存器，再把低精度 `P_j` 写回同一区域的后半段：`P0` 用列 64–127，`P1` 用 192–255；因此 `tP_layout` 虽由 SMEM-layout helper 构造，实际只是 TMEM view，并没有 `sP` 分配。

### 2.2 Pipeline 的握手含义

下表只定义一次各握手；后面的 timeline 直接使用这些名字。

| pipeline | 槽数 | `full` 的含义 | `empty` / 反向到达的含义 |
|---|---:|---|---|
| `Q` | `q_stage` | TMA/cp.async 已把 `Q_j` 放入 `sQ[j]` | MMA 已结束该 work tile 的全部 K 循环，可以覆盖 Q |
| `KV` | `kv_stage` | 当前环槽中的一个 K 或 V tile 已到达 | MMA 已消费该 K/V tile，可以复用该槽 |
| `S/P/O` | `q_stage` | MMA 已把 `S_j` 写入 TMEM，softmax 可读 | softmax 已写好 P 的前段，且 correction 已完成旧 O 的 rescale；MMA 可启动 PV |
| `P-last` | `q_stage` | P 的剩余部分已写完 | 不单独使用 empty；靠 full barrier 的 phase 交替复用 |
| `O-acc` | `q_stage` | 最后一次 PV 已完成，最终 O 可读 | 不单独使用 empty；O 的复用由 `S/P/O` gate 控制 |
| `softmax-stats` | `q_stage` | named barrier 让每对 softmax/correction warp 会合，`sScale` 可读 | pipeline empty credit 保证 correction 读完后 softmax 才能覆盖 |
| `O-epi` | `q_stage` | correction 已把归一化、转型后的 O 放入 `sO[j]` | epilogue 的异步 store 已真正读完 `sO[j]` |

其中 `S/P/O` 是刻意反向使用的 UMMA pipeline：MMA 先以“producer”发布 S，softmax 和 correction 再以“consumer release”共同组成 PV 的就绪条件。默认 `N=128` 时，softmax 写完 P 的前 96 列就到达该 barrier；剩余 32 列写完后再到达 `P-last`。PV 因而可以先启动，执行到后四分之一时才等待第二个 barrier。

### 2.3 Timeline：从预取到稳态再到写回

1. **初始化。** 所有 pipeline mbarrier 先在 SMEM 中建立，2CTA 用 cluster `arrive/wait` 保证两边都完成初始化。MMA warp 分配 512 列 TMEM；softmax、correction 和 MMA warp 通过 `TmemPtr` named barrier 取得同一指针，结束时也必须全部到达后才能释放 TMEM。

2. **Load prologue。** load warp 依次发出 `K_0 → Q_0 → Q_1 → V_0 → K_1 → V_1 → …`。Q 使用自己的 1/2 槽 pipeline；K/V 则沿同一个 `kv_stage` 环前进。每次写槽前 producer 等 `empty`，TMA 的 transaction mbarrier 在数据完整到达后才变成 `full`。dense 路径的 K block 从右向左遍历，以便先处理需要 causal/local mask 的边界块。

3. **第一次 QK。** MMA 等 `Q_0/Q_1 full` 和 `K_0 full`，连续发射 `Q_0K_0^T→S_0`、`Q_1K_0^T→S_1`，分别发布 `S full`；两个 softmax warpgroup 因而可并行处理两个 Q subtile。两个 Q 槽要跨完整 K 循环复用，所以此时只释放 K 槽，不释放 Q。

4. **Softmax 与 correction 并行推进。** 每个 softmax warpgroup 等自己的 `S_j`，将 FP32 score 从 TMEM 搬到寄存器，应用 score/mask，更新 online row-max，并把这一步要求旧 O 乘的 `acc_scale` 写进 `sScale`。对应的 correction warp 用同一 named-barrier 编号与它一对一会合；第一块没有旧 O，直接预先到达，后续块仅在任意线程观察到 `scale<1` 时才执行 TMEM `O→register→scale→TMEM`。这一步结束后 correction 向 `S/P/O` gate 到达。

5. **P 覆盖 S，并提前启动 PV。** softmax 在寄存器中计算 `exp2`，把低精度 P 写回刚读过的 S buffer。写到默认前 96 列后先执行 TMEM store fence，再向 `S/P/O` gate 到达；写完余下 32 列后再次 fence、发布 `P-last`，随后更新 row-sum。MMA 同时等 `V_i full`；当“P 前段完成 + 旧 O 已 correction”都满足时发射 `P_iV_i`，UMMA 内部在越过分割点前等待 `P-last`，因此 softmax 尾部与 PV 头部重叠。

6. **稳态交错。** 单个 MMA warp 的顺序是 `PV_i → QK_{i+1} → PV_{i+1} → …`。完成两个 Q stage 的 `PV_i` 后释放 `V_i` 槽；完成两个 `QK_{i+1}` 后释放 `K_{i+1}` 槽。与此同时，load 在更前方填 KV 环，两个 softmax warpgroup 处理新的 S，correction 修正上一轮 O。full/empty barrier 的环索引与 phase bit 每次复用时翻转，避免旧到达误唤醒新一轮。

7. **尾部与 epilogue。** 最后一个 `PV` 额外发布 `O-acc full`。softmax 的最后一次 stats rendezvous 交付最终 `row_sum/row_max`；correction 等最终 O 后乘 `rcp(row_sum)`（以及 FP8 descale），从 TMEM 分块读入寄存器、转成输出 dtype 并写入 `sO`。TMEM/SMEM store 之后、发布 barrier 之前分别执行 view-async fence，保证 barrier 不会先于数据可见。

8. **写回与复用。** 常规 contiguous 输出由 warp 13 等 `O-epi full` 后发出两个 stage 的 TMA store，并用 `cp_async_bulk_wait_group(..., read=True)` 确认异步操作已经读完 `sO`，才归还 `empty`。Pack-GQA、SplitKV 或 varlen-Q 不能用这条 TMA-O 路径时，warp 8–11 在 named barrier 内完成 `sO→register→global`，不再启用独立 epilogue warp。correction 读走最终 TMEM O 后也向 `S/P/O` gate 到达，使 persistent kernel 的下一 work tile 可以安全复用 O 区。

这条流水线的核心不是简单的“load/compute 双缓冲”，而是三层重叠：KV 多槽环隐藏 HBM→SMEM，两个 Query stage 让两组 softmax 并行，P 的 3/4 提前到达又把 softmax 尾部压到 PV 头部之下；correction 则独立填补 online-softmax 改变 row-max 后的 O rescale 空隙。

## 3. 不应与主 timeline 混写的特化路径

| 路径 | 关键差异 |
|---|---|
| `head_dim=head_dim_v=256` | 使用 `BlackwellFusedMultiHeadAttentionForward`：固定 2CTA、每 CTA `128×128×256` work tile；QK 以 K=128 做两次迭代，`kv_stage=4`、S accumulator 两槽；K 与 V 使用独立 SMEM，不采用正文的 K/V 物理复用。 |
| MLA / `head_dim_v=512` | 使用 `FlashAttentionMLAForwardSm100`：CTA tile `64×128`，另有 QvV 与 PVt 两条 MMA、Qv 两槽、V 四槽以及分裂的 Dv 输出；其 buffer 图和同步图是另一套实现。 |
| SM120 | 调度到 `flash_fwd_sm120.py`，使用 SM80-style MMA；不是本文的 TMEM/UMMA pipeline。 |

## 源码索引

- tile、2CTA 与 dispatch：[`flash_attn/cute/interface.py`](flash-attention/flash_attn/cute/interface.py)
- warp 角色、SMEM/TMEM offsets、pipeline 构造：[`flash_attn/cute/flash_fwd_sm100.py`](flash-attention/flash_attn/cute/flash_fwd_sm100.py)
- 环形 index/phase 与 full/empty 封装：[`flash_attn/cute/pipeline.py`](flash-attention/flash_attn/cute/pipeline.py)
- `head_dim=256` 特化：[`flash_attn/cute/sm100_hd256_2cta_fmha_forward.py`](flash-attention/flash_attn/cute/sm100_hd256_2cta_fmha_forward.py)
- MLA 特化：[`flash_attn/cute/flash_fwd_mla_sm100.py`](flash-attention/flash_attn/cute/flash_fwd_mla_sm100.py)
