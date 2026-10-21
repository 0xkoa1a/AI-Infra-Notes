---
title: "Ulysses Parallelism"
order: 3
---

# Ulysses Parallelism

**Ulysses Parallelism（UP）在 Attention 前按 head 重新分工，计算后再将输出按 token 交回原卡。** 两次 AllToAll 完成这种转换。

下文使用每卡完整复制权重的 UP，LayerNorm 和 MLP 在本地 token 上执行。[Ulysses 原论文](https://arxiv.org/html/2309.14509v2#S3.SS1)

## 动机：本地序列还不足以完成 Attention

线性投影与 MLP 可以独立处理不同 token，因此把序列分给多张 GPU 后，它们仍能本地执行。Attention 则依赖跨 token 的上下文：本地 Query 需要访问其他卡上由 mask 允许的 K、V。

不同 heads 的 Attention 可以独立计算。只要让每卡收齐自己负责的 heads 的全部 token，所需上下文就已齐全，可以直接调用 FlashAttention 等本地 Attention kernel。

## 方法：沿完整前向看激活与权重

下文以标准多头注意力（MHA）为例，假设序列长度与 head 数都可被并行度均分。

| 符号 | 含义 |
|---|---|
| $B,S$ | batch size 与完整序列长度 |
| $N_h,D_h$ | head 数与每个 head 的维度 |
| $H=N_hD_h$ | 隐藏维度 |
| $p$ | UP 组中的 Rank 数；图中每个 Rank 对应一张 GPU |

<UlyssesLayout />

> 图取 $B=1$、$S=4$、$N_h=4$、$D_h=1$、$p=2$，因此 $H=4$。QKV 的叠片代表三个同布局张量；X、Y 的灰色列表示隐藏特征。

**第二次 AllToAll 放在 O Projection 前，因为这个投影要混合一个 token 的全部 heads。** 先恢复该 token 的全部 head 特征，再用完整 $W_O\in\mathbb{R}^{H\times H}$ 计算 $Y_r=\operatorname{concatHeads}(O_r)W_O$，就能在本卡完成投影。

一般并行度 $p$ 下，各阶段的激活形状如下。QKV AllToAll 中，每个源卡发往一个目标卡的 Q、K、V 块各为 $[B,S/p,N_h/p,D_h]$。

| 数据位置 | 每卡形状 |
|---|---|
| QKV Projection 前的 X | $[B,S/p,H]$ |
| 第一次 AllToAll 前的 Q、K、V | $[B,S/p,N_h,D_h]$ |
| 第一次 AllToAll 后的 Q′、K′、V′ | $[B,S,N_h/p,D_h]$ |
| 本地 Attention 输出 O′ | $[B,S,N_h/p,D_h]$ |
| 第二次 AllToAll 后的 O | $[B,S/p,N_h,D_h]$ |
| Output Projection 后的 Y | $[B,S/p,H]$ |

## 通信量：只交换分片中的非本地部分

设每个元素占 $b$ 字节，一份完整隐藏状态大小为 $N=BSHb$。以下采用均匀 AllToAll 模型，统计**一次前向中，每 Rank 发往其他 Rank 的激活字节数**。

标准 MHA 的完整 Q、K、V、O 各约为 $N$ 字节。每卡只持有各自的分片，其中 $(p-1)/p$ 需要发给其他卡：

| 通信 | 每卡参与交换的数据大小 | 发给其他卡的字节数 |
|---|---|---|
| QKV AllToAll | $3N/p$ | $3\frac{p-1}{p^2}N$ |
| O AllToAll | $N/p$ | $\frac{p-1}{p^2}N$ |

两次相加，一组 Ulysses Attention 的每卡发送量为：

$$
V_{\mathrm{UP}}=4\frac{p-1}{p^2}N.
$$

### 与 TP+SP 的通信量比较

令两种方案的并行度都为 $p>1$，TP+SP 采用 Ring 模型。[TP+SP](./TP_SP.md) 的 Attention 与 MLP 子层各需要一组 AllGather + ReduceScatter；UP 的激活通信集中在 Attention 的两次 AllToAll。

| 前向统计范围 | TP+SP 每 Rank 发送量 | UP 每 Rank 发送量 | UP / TP+SP |
|---|---|---|---|
| Attention 子层 | $2\frac{p-1}{p}N$ | $4\frac{p-1}{p^2}N$ | $2/p$ |
| Attention + MLP 整层 | $4\frac{p-1}{p}N$ | $4\frac{p-1}{p^2}N$ | $1/p$ |

UP 的单个张量在 AllToAll 前后都只包含全局元素的 $1/p$，且 MLP 直接处理本地 token。这两点共同降低了整层的激活通信量。[论文通信分析](https://arxiv.org/html/2309.14509v2#S3.SS2)

> TP+SP 同时切分权重，UP 在每卡保存完整权重。训练时，UP 各卡从不同 token 得到权重梯度贡献，需要跨卡求和；组合 ZeRO 等参数分片方案后，还会增加相应的参数与梯度通信。

## 通信耗时与 head 数限制

两次 AllToAll 分别位于 Attention 的输入与输出端，通信等待会延长这条计算路径。主要成本包括：

- **链路传输与尾延迟**：每卡同时面向多个目标交换数据，拥塞链路或较慢 Rank 会延后交换完成。
- **Pack / Unpack**：如果通信前后需要重排并写出数据，就会增加 HBM 读写、临时缓冲和 kernel 启动。

按 head 均分还要求 $p$ 整除 $N_h$，因此 head 数限制了并行度。GQA/MQA 的 KV heads 更少，分发时需要处理 Query heads 与 KV heads 的共享关系。
