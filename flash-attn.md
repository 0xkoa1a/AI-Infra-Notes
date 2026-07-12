# FlashAttention v1

## 1. 问题背景

Scaled Dot-Product Attention：

$$
S=\frac{QK^\top}{\sqrt d},\qquad
P=\operatorname{softmax}(S),\qquad
O=PV
$$

其中：

* $Q,K,V\in\mathbb{R}^{N\times d}$
* $S,P\in\mathbb{R}^{N\times N}$

标准实现会把 $S$ 和 $P$ 写入 GPU HBM。虽然计算复杂度是 $O(N^2d)$，但实际性能往往受 HBM 读写限制，而不是纯计算量限制。

FlashAttention v1 的核心目标是：

> 不改变 Attention 的数学结果，不在 HBM 中物化完整的 $N\times N$ Attention 矩阵。

它通过 tiling、kernel fusion 和 online softmax，将计算拆成可以放入片上 SRAM 的小块，从而减少 HBM 访问。它仍然是 exact attention，而不是稀疏或近似 Attention。

---

## 2. 核心思想

将矩阵划分为块：

$$
Q=[Q_1,Q_2,\ldots],\qquad
K=[K_1,K_2,\ldots],\qquad
V=[V_1,V_2,\ldots]
$$

每次只加载一个 $Q_i$ 和一个 $K_j, V_j$：

$$
S_{ij}=\frac{Q_iK_j^\top}{\sqrt d}
$$

计算完当前块后，立即更新对应的 softmax 统计量和输出，不保存 $S_{ij}$。

每行只需要维护三个状态：

* $m_i$：目前见过的最大 logit；
* $l_i$：以 $m_i$ 为基准的指数和；
* $A_i$：尚未归一化的输出累加器。

---

## 3. Online Softmax

假设已经处理了一部分 key，保存：

$$
m=\max_j s_j
$$

$$
l=\sum_j e^{s_j-m}
$$

$$
A=\sum_j e^{s_j-m}v_j
$$

现在处理一个新的 score block $S_{\text{new}}$。

首先计算新的最大值：

$$
m'=\max\left(m,\max S_{\text{new}}\right)
$$

旧结果需要重新缩放：

$$
\alpha=e^{m-m'}
$$

新块的指数权重：

$$
P_{\text{new}}=e^{S_{\text{new}}-m'}
$$

随后更新：

$$
l'=\alpha l+\sum P_{\text{new}}
$$

$$
A'=\alpha A+P_{\text{new}}V_{\text{new}}
$$

最终：

$$
O=\frac{A}{l}
$$

这使 softmax 可以逐块计算，同时保持数值稳定。

---

## 4. 前向计算流程

对于每个 Query block：

1. 将 $Q_i$ 加载到 SRAM。
2. 初始化：

$$
m_i=-\infty,\qquad l_i=0,\qquad A_i=0
$$

3. 依次遍历所有 $K_j, V_j$ block。
4. 计算 $Q_iK_j^\top$。
5. 使用 online softmax 更新 $m_i,l_i,A_i$。
6. 所有 KV block 处理完成后计算：

$$
O_i=A_i/l_i
$$

7. 只将最终 $O_i$ 写回 HBM。

因此，完整的 $S$ 和 $P$ 从未出现在 HBM 中。

反向传播同样不保存完整的 Attention 概率矩阵，而是保存少量 softmax 归一化统计量，在 backward 中重新计算需要的 score block，以计算换显存。

---

## 5. Triton 教学实现

下面只实现：

* FP16；
* self-attention；
* forward；
* 输入形状 `[batch, heads, sequence, head_dim]`；
* 可选 causal mask。

一个 Triton program 负责一个 Query block，并遍历所有 KV block。它完整体现了 FlashAttention v1 的 tiling、fusion 和 online softmax，但其 program 划分方式更接近后来 FlashAttention-2 的 split-Q 调度。严格复刻 v1 的外层 KV 调度并不适合作为简洁的 Triton 单 kernel 示例。FlashAttention-2 的重要改进之一正是更好的并行划分。

```python
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def flash_attention_fwd_kernel(
    Q,
    K,
    V,
    O,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    # 当前 program 负责哪个 Query block
    pid_m = tl.program_id(0)

    # 当前 batch-head，例如：
    # pid_bh = batch_id * num_heads + head_id
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # 假设 Q/K/V/O 连续存储，形状均为 [B, H, N, D]
    base = pid_bh * N_CTX * HEAD_DIM

    q_ptrs = (
        Q
        + base
        + offs_m[:, None] * HEAD_DIM
        + offs_d[None, :]
    )

    q = tl.load(
        q_ptrs,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )

    # Online softmax 状态
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    # 未归一化输出累加器
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # exp2(x) = exp(x * log2(e))
    qk_scale = sm_scale * 1.4426950408889634

    # 流式遍历所有 K/V block
    for start_n in range(0, N_CTX, BLOCK_N):
        curr_n = start_n + offs_n

        k_ptrs = (
            K
            + base
            + curr_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )

        v_ptrs = (
            V
            + base
            + curr_n[:, None] * HEAD_DIM
            + offs_d[None, :]
        )

        k = tl.load(
            k_ptrs,
            mask=curr_n[:, None] < N_CTX,
            other=0.0,
        )

        v = tl.load(
            v_ptrs,
            mask=curr_n[:, None] < N_CTX,
            other=0.0,
        )

        # [BLOCK_M, D] @ [D, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale

        valid_mask = curr_n[None, :] < N_CTX

        if CAUSAL:
            valid_mask = valid_mask & (
                offs_m[:, None] >= curr_n[None, :]
            )

        qk = tl.where(valid_mask, qk, -float("inf"))

        # 当前 block 与历史数据的统一最大值
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))

        # 对旧累加结果重新缩放
        alpha = tl.exp2(m_i - m_ij)

        # 当前 block 的 softmax 分子
        p = tl.exp2(qk - m_ij[:, None])

        # 更新 softmax 分母
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # 更新未归一化输出
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.float16), v)

        m_i = m_ij

    # 最终 softmax 归一化
    output = acc / l_i[:, None]

    o_ptrs = (
        O
        + base
        + offs_m[:, None] * HEAD_DIM
        + offs_d[None, :]
    )

    tl.store(
        o_ptrs,
        output,
        mask=offs_m[:, None] < N_CTX,
    )


def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """FlashAttention 核心机制的教学版 Triton forward。"""

    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.shape == k.shape == v.shape
    assert q.ndim == 4
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()

    batch, heads, n_ctx, head_dim = q.shape

    # 为了简化 tl.dot 和 block shape 的处理
    assert head_dim in (16, 32, 64, 128)

    output = torch.empty_like(q)

    block_m = 64
    block_n = 64

    grid = (
        triton.cdiv(n_ctx, block_m),
        batch * heads,
    )

    flash_attention_fwd_kernel[grid](
        q,
        k,
        v,
        output,
        1.0 / math.sqrt(head_dim),
        N_CTX=n_ctx,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    return output
```

简单正确性测试：

```python
torch.manual_seed(0)

q = torch.randn(
    2, 8, 512, 64,
    device="cuda",
    dtype=torch.float16,
)
k = torch.randn_like(q)
v = torch.randn_like(q)

actual = flash_attention_v1(q, k, v, causal=True)

expected = F.scaled_dot_product_attention(
    q,
    k,
    v,
    is_causal=True,
)

torch.testing.assert_close(
    actual,
    expected,
    atol=2e-2,
    rtol=2e-2,
)
```

---

## 6. 代码与算法的对应关系

| Triton 变量 | 数学含义                     |
| --------- | ------------------------ |
| `q`       | 当前 $Q_i$ block           |
| `qk`      | 当前 $Q_iK_j^\top/\sqrt d$ |
| `m_i`     | 每个 Query row 当前最大 logit  |
| `l_i`     | softmax 指数和              |
| `acc`     | 未归一化的 $PV$ 累加结果          |
| `alpha`   | 最大值变化后对旧结果的修正系数          |
| `p`       | 当前 KV block 的 softmax 分子 |
| `output`  | $acc/l_i$                |

---

## 7. 需要记住的结论

FlashAttention v1 并没有降低 Attention 的 $O(N^2d)$ 计算复杂度。

它优化的是 GPU 内存层次结构中的 IO：

$$
\mathrm{HBM}\leftrightarrow\mathrm{SRAM/Register}
$$

其关键技术可以概括为：

$$
\boxed{
\mathrm{Tiling}
+\mathrm{Kernel Fusion}
+\mathrm{Online Softmax}
+\mathrm{Backward Recompute}
}
$$

最核心的一句话是：

> FlashAttention 不保存完整 Attention 矩阵，而是在 SRAM 中逐块计算并在线维护 softmax 状态。

# FlashAttention v2

你看到的 `\(...\)` 和 `\[...\]` 是 **LaTeX/MathJax 的数学公式定界符**：

* `\( x+y \)`：行内公式，效果类似 `$x+y$`
* `\[ x+y \]`：独立公式，效果类似 `$$x+y$$`

其中 `$...$` 和 `$$...$$` 源自 TeX，在 Markdown、MathJax、KaTeX 中非常常见；`\(...\)` 和 `\[...\]` 则是 LaTeX 更规范的写法。后续我按你的要求统一使用 `$...$` 和 `$$...$$`。

另外先纠正上一份笔记中的一个重要问题：上一份所谓的 v1 Triton 实现，其 **online softmax 更新方式和并行调度其实已经是 FlashAttention-2 风格**：

1. 使用未归一化的 `acc`，最后只除一次 `l_i`，属于 v2 减少非矩阵乘 FLOPs 的优化。
2. 一个 Triton program 负责一个 Query block，并遍历所有 KV blocks，属于 v2 的 sequence-length parallelism。

因此，上一份代码更准确的名称应当是：

> FlashAttention 核心原理的 FA2-style Triton forward，而不是严格复刻的 FlashAttention v1 kernel。

# FlashAttention-2 学习笔记

## 1. 核心定位

FlashAttention v1 主要解决 Attention 的 **HBM IO 问题**：通过 tiling、online softmax 和 recomputation，避免在 HBM 中保存完整的 Attention 矩阵。

FlashAttention-2 保留这些基本思想，不改变 Attention 的数学结果，也不改变其计算复杂度，而是进一步解决三个 GPU 执行效率问题：

1. 非矩阵乘操作过多；
2. thread block 数量不足，GPU occupancy 较低；
3. warp 之间存在不必要的共享内存通信。

论文报告中，FlashAttention-2 在 A100 上比 v1 快约两倍，并达到理论矩阵运算吞吐的约 $50%\sim73%$。这些数字是论文特定硬件和测试条件下的结果。([arXiv][1])

---

## 2. 改进一：减少 non-matmul FLOPs

GPU 的 Tensor Core 可以极快地执行矩阵乘法，但 `exp`、`max`、除法、标量乘法、mask 和边界判断等 non-matmul 操作通常运行在普通 CUDA Core 上。

以论文使用的 A100 为例：

* FP16/BF16 矩阵乘理论吞吐：$312$ TFLOPS；
* FP32 非矩阵乘理论吞吐：$19.5$ TFLOPS。

因此，一个 non-matmul FLOP 在吞吐意义上可能比一个 Tensor Core matmul FLOP 昂贵约 $16$ 倍。FlashAttention-2 的重点不是减少主要矩阵乘，而是减少 softmax 周围的标量操作。

### v1：每处理一个 KV block 都维护归一化输出

假设之前已经处理了一部分 KV block，保存归一化输出 $O$、最大值 $m$ 和指数和 $l$。

加入新的 KV block 后，需要执行：

$$
O'
==

\frac{
e^{m-m'}lO
+
e^{S_{\text{new}}-m'}V_{\text{new}}
}{
l'
}
$$

这意味着每轮循环都需要对整个输出块执行归一化和重新缩放。

### v2：维护未归一化输出

FlashAttention-2 改为维护未归一化累加器 $\widetilde O$：

$$
\widetilde O
============

\sum_j e^{s_j-m}v_j
$$

处理新的 KV block 时，先计算：

$$
m'
==

\max\left(m,\operatorname{rowmax}(S_{\text{new}})\right)
$$

定义旧结果的缩放系数：

$$
\alpha=e^{m-m'}
$$

当前 block 的未归一化概率为：

$$
\widetilde P_{\text{new}}
=========================

e^{S_{\text{new}}-m'}
$$

然后更新：

$$
l'
==

\alpha l+
\operatorname{rowsum}(\widetilde P_{\text{new}})
$$

$$
\widetilde O'
=============

\alpha\widetilde O+
\widetilde P_{\text{new}}V_{\text{new}}
$$

只有遍历完所有 KV block 后，才进行一次归一化：

$$
O=\frac{\widetilde O}{l}
$$

也就是说，v2 将原本每轮都进行的输出归一化推迟到了循环结束。

此外，forward 不再为 backward 分别保存 $m$ 和 $l$，而只保存 log-sum-exp：

$$
L=m+\log l
$$

在 backward 中可以直接重构概率：

$$
P=e^{S-L}
$$

这减少了保存的中间统计量和相关操作。

---

## 3. 改进二：沿序列维度增加 thread block 并行

### v1 的问题

FlashAttention v1 主要沿 batch 和 attention head 维度并行。可以粗略理解为 thread block 数量约为：

$$
N_{\text{CTA}}^{\text{v1}}\approx B\times H
$$

当 batch 较小、head 数较少时，thread block 的数量可能不足以占满 GPU 的所有 SM。

这在长序列场景尤其常见：序列越长，显存压力越大，通常能使用的 batch size 越小。

### v2 的做法：每个 Query block 一个 thread block

将 $Q$ 按序列维度划分为：

$$
Q=[Q_1,Q_2,\ldots,Q_{T_r}]
$$

其中：

$$
T_r=\left\lceil\frac{N}{B_r}\right\rceil
$$

每个 thread block 独立负责一个 $Q_i$，并遍历所有 $K_j,V_j$：

```text
parallel for each batch, head, query_block i:
    load Q_i
    initialize m_i, l_i, acc_i

    for each KV block j:
        load K_j, V_j
        compute Q_i K_j^T
        update online softmax
        update acc_i

    normalize and store O_i
```

因此，forward 的 thread block 数量约为：

$$
N_{\text{CTA}}^{\text{v2}}
==========================

B\times H\times
\left\lceil\frac{N}{B_r}\right\rceil
$$

不同 Query block 写入不同的输出行，不存在写冲突，也不需要 thread block 之间通信。

这正是上一份 Triton 代码中的调度方式：

```python
grid = (
    triton.cdiv(n_ctx, block_m),
    batch * heads,
)
```

其中每个 program 负责一个 Query block。

---

## 4. 改进三：从 sliced-K 改为 sliced-Q

这一改进发生在 **同一个 thread block 内部的多个 warp 之间**，不要和上一节的 thread block 级序列并行混淆。

### v1：sliced-K

假设一个 thread block 使用四个 warp。

FlashAttention v1 的划分方式大致为：

* 所有 warp 共享同一个 $Q$；
* 不同 warp 分别处理不同的 $K/V$ 切片。

每个 warp 得到相同 Query 行对应的一部分输出：

$$
O^{(w)}
=======

P^{(w)}V^{(w)}
$$

最后必须将各 warp 的部分结果相加：

$$
O=\sum_w O^{(w)}
$$

因此需要：

1. 每个 warp 将部分结果写入 shared memory；
2. warp 同步；
3. 从 shared memory 读取结果；
4. 对部分结果进行归约。

这会产生额外的 shared-memory 访问和同步。

### v2：sliced-Q

FlashAttention-2 改为：

* 所有 warp 共享 $K$ 和 $V$；
* 不同 warp 负责 $Q$ 的不同行。

例如：

```text
Warp 0 -> Q 的第 0 组行
Warp 1 -> Q 的第 1 组行
Warp 2 -> Q 的第 2 组行
Warp 3 -> Q 的第 3 组行
```

每个 warp 计算：

$$
S^{(w)}=Q^{(w)}K^\top
$$

随后得到自己负责的输出：

$$
O^{(w)}
=======

\operatorname{softmax}(S^{(w)})V
$$

由于不同 warp 负责不同的输出行，它们的结果互不重叠：

$$
O=
\begin{bmatrix}
O^{(0)}\
O^{(1)}\
O^{(2)}\
O^{(3)}
\end{bmatrix}
$$

因此 forward 中不再需要跨 warp 对输出进行求和，显著减少了 shared-memory 读写和 warp synchronization。

---

## 5. Causal mask 优化

对于 causal attention：

$$
S_{ij}=-\infty,\qquad j>i
$$

由于 Attention 已经被划分成 block：

* 完全位于主对角线右上方的 block 可以直接跳过；
* 完全位于主对角线左下方的 block 不需要逐元素 mask；
* 只有穿过主对角线的 block 需要执行逐元素 causal mask。

因此，对于一个 Query block，只需遍历其之前的 KV blocks：

$$
j\leq i
$$

而逐元素比较通常只发生在对角线 block 上。这同时减少了矩阵乘计算和 mask 对应的 non-matmul 操作。

---

## 6. Backward

FlashAttention-2 的 backward 基本沿用 v1 的 recomputation 思想：

1. 不保存完整的 $S$ 和 $P$；
2. 保存输出 $O$ 和每行的 log-sum-exp $L$；
3. backward 时重新计算：

$$
S=QK^\top
$$

$$
P=e^{S-L}
$$

4. 再计算 $dQ$、$dK$ 和 $dV$。

其主要变化仍然是更好的 sequence parallelism 和 warp work partitioning，而不是重新设计梯度公式。

forward 中不同 Query block 可以完全独立；backward 中多个工作单元可能共同更新 $dQ$、$dK$ 或 $dV$，因此实现上需要调整遍历方向，部分场景还需要 atomic add 或额外归约。

---

## 7. v1 与 v2 对比

| 方面              | FlashAttention v1 | FlashAttention-2                      |
| --------------- | ----------------- | ------------------------------------- |
| 核心目标            | 减少 HBM IO         | 提高 GPU 执行效率                           |
| Attention 结果    | Exact             | Exact                                 |
| 时间复杂度           | $O(N^2d)$         | $O(N^2d)$                             |
| 显存复杂度           | $O(Nd)$           | $O(Nd)$                               |
| 输出累加            | 循环中反复归一化          | 保存未归一化 $\widetilde O$，最后归一化           |
| backward 状态     | 保存 $m$ 和 $l$      | 只保存 $L=m+\log l$                      |
| thread-block 并行 | 主要沿 batch、head    | 增加 Query sequence block 维度            |
| warp 划分         | sliced-K          | sliced-Q                              |
| warp 通信         | 需要部分输出归约          | forward 基本不需要输出归约                     |
| 主要瓶颈            | HBM IO            | occupancy、non-matmul、shared-memory 通信 |

最需要记住的一句话是：

> FlashAttention v1 主要解决“数据怎么少搬”，FlashAttention-2 进一步解决“工作怎么分配，才能让 GPU 更忙、warp 之间少通信”。

[1]: https://arxiv.org/abs/2307.08691 "[2307.08691] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
