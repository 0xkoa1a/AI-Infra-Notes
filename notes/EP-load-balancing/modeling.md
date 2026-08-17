---
title: "EP 负载均衡：数学建模"
order: 1
---

**给定一个随时间变化的 Expert 负载过程，判断一个负载均衡方案需要具备多大的“布局自由度”、多强的“实时性”和多好的“分流能力”，才能值得它付出的额外成本。**

# 问题 Setup

先固定**一个 layer、一个 step**。设：

- 有 $E$ 个 logical experts。
- 有 $R$ 个 ranks。
- Expert $e$ 当前收到 $x_e$ 个 expert-token assignments。为简洁起见，下面仍称为 token。
- 总 token 数为 $N=\sum_e x_e$，归一化 Expert 负载为 $p_e=x_e/N$。
- Expert $e$ 当前可以在一组 ranks 上执行，记作 $A_e$。
  - 如果没有 replica，$|A_e|=1$。
  - 如果有两个物理实例，$|A_e|=2$。
- reroute 决定这 $x_e$ 个 token 怎么在 $A_e$ 中分。

先采用一个理想化的 compute 模型：所有 token 的计算代价相同、所有 ranks 同构，并且暂时忽略通信拓扑和小 batch 对 kernel efficiency 的影响。

## Expert compute 的负载代理

如果分给 Expert $e$ 在 rank $r$ 上的实例 $y_{er}$ 个 token，那么 rank $r$ 的总负载为：

$$
L_r=\sum_e y_{er}.
$$

这一层 expert compute 的完成时间由收到 token 最多的 rank 决定：

$$
L_{\max}=\max_r L_r
$$

如果完全理想，所有 rank 一样忙，那么每个 rank 应该处理：

$$
\bar L=\frac{N}{R}.
$$

定义**Rank 负载不均衡度**：**最忙 Rank 的工作量是理想平均值的多少倍。**

$$
\rho=\frac{L_{\max}}{N/R}.
$$

- $\rho=1$：完美均衡。
- $\rho=1.2$：最慢 Rank 比理想情况多做 20%。
- $\rho=2$：整个 expert compute 至少浪费接近一半并行能力。

它直接对应当前理想模型中的 expert compute critical path，但还不是端到端时间。

---

# 负载分布的特征

## Expert 负载不均衡度

**最热 Expert 是平均 Expert 负载的多少倍。**

$$
S(t)=E\max_e p_e(t).
$$

---

## Expert 负载集中度

把 Experts 按负载从高到低排好，负载集中度衡量**最热的 $m$ 个 Experts 一共吃掉多少负载。**

$$
C_m(t)=\sum_{i=1}^{m}p_{(i)}(t).
$$

例如两种分布：

- A：一个 Expert 50%，其他很均匀。
- B：四个 Experts 各 12.5%，其他很均匀。

它们可能 peak load 类似，但对 placement 的要求完全不同。

$C_m$ 曲线说明额外 capacity 是集中在一个 Expert，还是分散在一组 Experts 上。它可以提示 placement 需要覆盖多少个热点，但不能单独决定 replica 数量或具体 layout。

## 精确负载变化

定义归一化负载向量：

$$
p(t)=(p_1(t),\ldots,p_E(t)).
$$

可以看一个非常简单的 lag-$\Delta$ drift：**隔了 $\Delta$ 个 step 后，有多少比例的负载质量“搬到了别的 Experts”。**

$$
D(\Delta)=\mathbb E\left[\frac12|p(t)-p(t-\Delta)|_1\right].
$$

- $D(\Delta)\approx0$：分布非常稳定。
- $D(\Delta)$ 很大：历史精确负载已经没有多少参考意义。

对于 historical placement，应该在它实际使用的 layout age 上观察 $D(\Delta)$：$\Delta$ 包括负载统计、placement 计算和 layout 生效之间的延迟，也会随 placement 的更新周期变化。Realtime placement 则直接使用当前 step 的负载重新决定当前 step 的 layout，不依赖历史 lag。

---

## 粗粒度负载变化

因为 placement 并不一定需要预测“Expert 17 下一步到底是 5372 个还是 6141 个 token”。

它可能只需要预测“**Expert 17 下一段时间是不是仍然属于热点、值得多准备一个 replica**”。

所以应该再定义一个**更粗粒度的 temporal locality**：**隔 $\Delta$ 个 step 后，Top-$m$ 热点 Expert 还有多少仍然是热点。**

$$
H_m(\Delta)=\frac{|\operatorname{Top}_m(t)\cap\operatorname{Top}_m(t-\Delta)|}{m}.
$$

于是可能出现一种非常重要的情况：

- $D(\Delta)$ 很高：
  - 精确 load 数字预测不准。
- 但 $H_m(\Delta)$ 很高：
  - “谁是热点”却相当稳定。

那么**历史 placement + 当前实时 reroute 恰好非常合适。**

---


# 解耦布局和分流

给定当前负载 $x$ 和一个候选方案已经产生的 placement $G=\{A_e\}$，先定义这个 layout 上的合法 reroute：

$$
\begin{aligned}
&y_{er}\ge 0,\\
&y_{er}=0, \qquad r\notin A_e,\\
&\sum_{r\in A_e}y_{er}=x_e,\\
&L_r=\sum_e y_{er}.
\end{aligned}
$$

这里先允许连续分流。若 $x_e$ 是整数，也可以要求 $y_{er}$ 为整数；在当前网络流模型中，整数最优解仍然可以精确求出。评价实际 rerouter 时，oracle 必须采用与实际方案相同的分流粒度，否则 reroute gap 会混入 granularity gap。

定义固定 layout 下的最优 bottleneck load：**placement 固定以后，哪怕给我世界上最好的分流算法，最少还能剩多少负载不均衡。**

$$
B^*(x,G)=\min_{\text{合法 reroute}}\max_r L_r,
\qquad
\rho^*(x,G)=\frac{B^*(x,G)}{N/R}.
$$

## 给定 layout 的最优 reroute 可以精确求出

对于一个候选 bottleneck load $B$，构造下面的 flow network：

```text
source --x_e--> Expert e --∞--> r ∈ A_e --B--> sink
```

如果最大流能够送出全部 $N$ 个 token，那么 $B$ 可行。对 $B$ 做搜索，就能得到精确的 $B^*(x,G)$。当 token 数和容量都是整数时，网络流的整数性保证结果也是整数 reroute。

在当前“同构 ranks、token 等价、无额外边容量”的模型下，还可以写成：

$$
B^*(x,G)
=
\max_{\varnothing\ne S\subseteq\{1,\ldots,E\}}
\frac{\sum_{e\in S}x_e}{|N_G(S)|},
$$

其中 $S$ 是任意 Expert 集合，$N_G(S)=\bigcup_{e\in S}A_e$。整数 token 情况对右侧整体取 ceiling。实际计算不需要枚举所有 $S$，max-flow 会隐式找到造成瓶颈的 Expert 集合。

## 评价一个候选方案

设候选方案 $s$ 在当前 step 产生 placement $G^s$ 和实际 reroute，最终不均衡度为 $\rho_s$。固定 layout 下的最优 bottleneck load 为 $\rho^*(x,G^s)$。同样分流粒度下，不受 layout 限制的理想下界记作 $\rho_{\rm ideal}$：连续分流时 $\rho_{\rm ideal}=1$；整数分流时：

$$
\rho_{\rm ideal}
=
\frac{\lceil N/R\rceil}{N/R}.
$$

于是候选方案 $s$ 与理想方案的差距可以分解为 layout residual 和 reroute gap 两部分：

$$
\rho_s-\rho_{\rm ideal}
=
\underbrace{\rho^*(x,G^s)-\rho_{\rm ideal}}_{\text{layout residual}}
+
\underbrace{\rho_s-\rho^*(x,G^s)}_{\text{reroute gap}}.
$$

- **layout residual**：在方案实际产生的 layout 上，即使使用最优 reroute 仍然无法消除的不均衡。
- **reroute gap**：这个 layout 本来能够做到，但方案的实际 rerouter 没有做到的部分。

这个分解不需要构造或求解全局最优 placement。它只评价每个候选方案实际给出的 $G^s$，因此可以直接用来比较不同方案：

- layout residual 大：问题首先在方案产生的 placement。
- layout residual 小、reroute gap 大：问题首先在 rerouter。
- 两者都小：继续看为此付出的系统成本是否值得。

---

可以定义一个简单的算法质量：

$$
\alpha=\frac{\rho_{\text{algorithm}}}{\rho^*}.
$$

表示：**实际 rerouter 比当前 placement 下的理论最优差多少。**

例如一个具体 rerouter 最后做到 $\rho_{\text{algorithm}}=1.08$，而这个 placement 下理论最优是 $\rho^*=1.05$。


# 理解给定 Placement 的结构性残差

下面不是构造新的 placement，而是解释：**为什么一个候选方案已经给出的 layout 会出现非零 layout residual。**

## 单个 Expert 导致的下界

如果希望每个 rank 最多承担目标负载 $B$，那么 Expert $e$ 的 token 必须至少能够到达：

$$
|A_e|
\ge
\left\lceil\frac{x_e}{B}\right\rceil
$$

个 ranks。完美均衡时 $B=N/R$，于是：

$$
|A_e|\ge \left\lceil Rp_e\right\rceil.
$$

> 例如，**一个 Expert 如果吃掉了全局 25% 的 token，而系统有 8 张 GPU，那么它至少需要能在约 $8\times25\%=2$ 张 GPU 上执行。**

如果候选 layout 不满足这个条件，那么无论 rerouter 多好，$\rho^*(x,G)$ 都不可能达到目标。

---

单个 Expert 还不是全部。考虑一组热点 Experts $S$。

这些 Expert 能去的所有 ranks 的并集记作 $N(S)$。

如果希望每个 rank 最多承担目标负载 $B$，必须有：**任何一组 Experts 的总流量，都不能超过“它们所有可访问 GPU”的总处理能力。**

$$
\sum_{e\in S}x_e
\le
B\cdot |N(S)|.
$$

例如：

- Expert 1 和 Expert 2 都很热；
- 它们虽然每个都有两个 replica；
- 但恰好都放在 rank 0 和 rank 1。

那么：

- 单独看 replica 数，好像不错；
- 但两个 Expert 的流量竞争的是同一组 GPU；
- 依然可能无法 balance。

因此评价候选 placement 时，不能只数 replica 数，还要看造成 $B^*(x,G)$ 的瓶颈集合 $S$ 是否共享了过小的 Rank 邻域。这个集合是解释 layout residual 的诊断结果，而不是新的 placement 构造规则。

---

# 选择不同的负载均衡方案

## 负载比较均匀

如果：

- 专家负载不均衡度 $S$ 小；
- GPU 负载不均衡度 $\rho$ 本身已经接近 1；

那么任何复杂 LB 都没有意义。

## 偏斜明显，而且热点具有 temporal locality

如果：

- $S$ / $C_m$ 显示少量热点需要额外 capacity；
- 在 historical placement 的实际 layout age 上，$H_m(\Delta)$ 高；
- 候选方案产生的 layout residual 足够小；
- 但精确 $D(\Delta)$ 不一定很低；

那么比较适合：

> **慢速历史 placement + 当前精确 reroute。**

placement 只需要稳定识别：

- 哪些 experts 值得 replica；
- replica 大概应该在哪些 ranks。

reroute 再解决当前 step 的精确比例。

    
## 热点本身快速变化

如果：

- historical placement 在实际 layout age 上的 $H_m(\Delta)$ 很低；

那么 stale placement 就会出现根本问题：

> replica 做在昨天的热点上，而当前热点没有 replica。

此时有三个可能方向：

- 使用当前 step 的负载重新计算当前 step 的 placement，并实时创建 / 准备 replica，例如 UltraEP / MoonEP 一类；
- 用更多冗余显存复制更多 Experts，让 historical placement 对负载变化更 robust；
- 干脆接受部分 imbalance，因为每个 step 实时 placement 的成本比收益还高。

这里的 **realtime placement** 是一种 per-step 决策：每一步都根据当前负载重新得到 $G_t$，并让新 layout 服务于当前 step。它不依赖热点在后续 steps 中继续存在，因此不需要引入“热点寿命”。评价它时，只需比较：

- 它相对 historical placement 在当前 step 降低了多少 layout residual；
- 当前 step 暴露了多少 planning 和 weight preparation 成本。

所以问题实际上变成一个非常清晰的 trade-off：

> **用显存换 robustness，还是用关键路径时间换 adaptivity。**

---

# 端到端收益分析

假设当前 step 的总 Expert 工作量在完美均衡时需要时间 $C$。如果原始 imbalance 是 $\rho_0$，负载均衡后是 $\rho_1$，那么仅考虑 expert compute：有**理想 expert compute 时间乘以负载不均衡倍率，就是实际 bottleneck compute 时间。**

$$
T_0\approx \rho_0 C,
\qquad
T_1\approx \rho_1 C.
$$

因此理想的 compute speedup 直接约等于负载不均衡度的比值

$$
\text{Speedup}\approx\frac{\rho_0}{\rho_1}.
$$

但系统真正需要判断的是**值不值得做**。

设：

- $\Delta T_{\rm comm}$：reroute 引起的通信时间变化。
- $H_{\rm plan}^{\rm exposed}$：负载均衡计算暴露在关键路径上的开销。
- $H_{\rm move}^{\rm exposed}$：weight preparation / movement 暴露在关键路径上的开销。
- $\Delta T_{\rm kernel}$：token 被切到更多 replicas 后，kernel 粒度变化引起的时间差。

那么负载均衡值得做的条件是：**因为减少长尾而省下来的 compute 时间，必须大于它在同一条关键路径上新增的系统时间。**

$$
(\rho_0-\rho_1)C
>
\Delta T_{\rm comm}
+H_{\rm plan}^{\rm exposed}
+H_{\rm move}^{\rm exposed}
+\Delta T_{\rm kernel}.
$$

这里只应该加入 exposed cost：如果 planning、movement 或通信已经与其他阶段重叠，就不能再把原始 duration 完整相加。$\Delta T_{\rm comm}$ 也不一定为正；改变目的 Rank 后，通信长尾可能变好，也可能变差。

这个式子可以解释为什么：

- training / prefill 更值得做；decode 未必值得做；
- historical placement 可以接受一定滞后，因为它把 weight movement 移出了当前 critical path；
- realtime placement 每一步都必须带来明显更低的 layout residual，才能补偿当前 step 的 planning / moving cost。

## 在负载 trace 上评价候选方案

对于每个候选方案 $s$，应该按照真实时间顺序 replay 整段负载 trace：

- historical placement 只能使用当时已经观察到的历史负载，并施加真实的统计、更新和生效延迟；
- realtime placement 每一步使用当前负载产生当前 $G_t^s$，同时计入当步的 planning 和 weight preparation cost；
- 对每个实际 $G_t^s$ 求固定-layout oracle $\rho^*(x_t,G_t^s)$，分别报告 layout residual 和 reroute gap；
- 报告实际端到端时间的 mean 和高分位数，而不只报告平均 $\rho$。

$\rho$ 只使用 Expert 总负载 $x_e$，因此只能评价 compute balance。若要计算 token dispatch / combine 的通信影响，还需要记录源 Rank 到 Expert 的负载 $x_{s,e}$，并结合实际 NVLink / RDMA 拓扑统计跨域 bytes 和最大链路负载。

此外，以下资源更适合作为硬约束或 Pareto 维度，而不是全部强行折算成时间：

- peak replica memory；
- 每 step weight movement bytes；
- 跨节点 token traffic；
- planning deadline miss rate。

最终选择的是：在相同资源约束下，端到端性能最好的候选方案。负载特征 $S,C_m,D,H_m$ 用来解释结果；固定-layout oracle 用来诊断差距来自 placement 还是 reroute；它们本身不替代实际方案比较。
