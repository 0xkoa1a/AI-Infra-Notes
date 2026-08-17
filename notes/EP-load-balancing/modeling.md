---
title: "EP 负载均衡：数学建模"
order: 1
---

**给定一条随时间变化的 Expert 负载 trace 和目标 Rank imbalance，先求 workload 所需的 layout resource 与 placement freshness，再诊断候选方案距离这些能力边界还有多远，最后判断端到端收益是否值得系统成本。**

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

这里先允许连续分流。若 $x_e$ 是整数，也可以要求 $y_{er}$ 为整数；在当前网络流模型中，整数最优解仍然可以精确求出。评价候选方案时，oracle 与实际结果必须使用同一种连续或整数 token 模型。

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

---

# Layout capability：达到目标 Balance 需要多少实例

$\rho^*(x,G)$ 评价一个已经给出的 layout。现在反过来问：**为了让当前负载达到目标 $\rho_{\rm target}$，workload 理论上至少需要多少额外 Expert instances？**

## Capability 参数与合法 Placement

用：

$$
\Theta=
\left(
\{U_e\},
\{M_r\},
\text{remap permission}
\right)
$$

描述系统允许的 capability：

- $U_e$：Expert $e$ 允许出现的 Rank 集合。
- $M_r$：rank $r$ 最多容纳的 Expert instances 数。
- remap permission：基础实例能否离开当前 Rank。

设当前基础 placement 为 $z^0$。如果不允许 remap，则要求：

$$
z_{er}\ge z_{er}^0,
$$

即当前实例必须保留，只能在其上增加 replicas；如果允许 remap，则基础实例可以在 $U_e$ 内重新选择。Fixed current layout 是更严格的特例 $z=z^0$。

设：

- $z_{er}\in\{0,1\}$：Expert $e$ 是否在 rank $r$ 上存在实例。
- $y_{er}$：Expert $e$ 分给 rank $r$ 的 token 数。
- 目标 bottleneck load：

$$
B_{\rm target}
=
\rho_{\rm target}\frac NR.
$$

合法 placement 和 reroute 满足：

$$
\begin{aligned}
&z_{er}=0, &&r\notin U_e,\\
&\sum_r z_{er}\ge1, &&\forall e,\\
&\sum_e z_{er}\le M_r, &&\forall r,\\
&\sum_r y_{er}=x_e, &&\forall e,\\
&0\le y_{er}\le Nz_{er}, &&\forall e,r,\\
&\sum_e y_{er}\le B_{\rm target}, &&\forall r.
\end{aligned}
$$

$\sum_r z_{er}\ge1$ 强制每个 logical Expert 始终保留一份基础实例，即使当前 step 的 $x_e=0$ 也不能从系统中消失。

正文首先假设同 layer Experts 权重大小相同，因此用 instance 数表达 memory。如果权重大小为 $w_e$，则把 Rank memory 约束替换为：

$$
\sum_e w_ez_{er}\le M_r^{\rm bytes},
$$

并把 objective 改成相对单份基础权重的额外 bytes。

如果 token 必须整数分流，那么 Rank load 也是整数，约束 $L_r\le B_{\rm target}$ 实际等价于 $L_r\le\lfloor B_{\rm target}\rfloor$。若 $\rho_{\rm target}<\rho_{\rm ideal}$，目标本身不可行。

## Extra-instance Requirement

定义：

$$
K_{\rm req}(x,\rho_{\rm target};\Theta)
=
\min_{z,y}
\left(
\sum_{e,r}z_{er}-E
\right).
$$

它表示：**在 capability $\Theta$ 已经固定的前提下，当前负载达到目标 balance 至少需要多少额外 Expert instances。** 因此 $K_{\rm req}$ 衡量的是 extra-instance requirement，而不是脱离 placement domain 和 Rank memory 的抽象“layout freedom”。

同一问题也可以反过来写成：

$$
\rho_{\rm layout}^*(x,K;\Theta)
=
\min_{G\in\mathcal G(K;\Theta)}
\rho^*(x,G),
$$

其中 $\mathcal G(K;\Theta)$ 表示满足 $\Theta$ 且额外实例数不超过 $K$ 的 layouts。两者满足广义逆关系：

$$
K_{\rm req}(x,\rho;\Theta)\le K
\quad\Longleftrightarrow\quad
\rho_{\rm layout}^*(x,K;\Theta)\le\rho.
$$

正文以“给定目标 $\rho$，求 $K_{\rm req}$”为主；扫描 $\rho_{\rm layout}^*(K)$ 用来观察增加 replica memory 后的边际收益和 elbow。

## 三个 Layout 参考点

- **fixed current layout**：$z$ 固定为当前 $G_0$，只求 $\rho^*(x,G_0)$。
- **remap-only**：要求 $\sum_{e,r}z_{er}=E$，允许在 $U_e$ 内重新放置基础实例，但不增加 replica。
- **remap + replication**：允许 $\sum_{e,r}z_{er}\le E+K$，同时决定基础实例和 replicas 的位置。

因此 $K=0$ 只表示“不增加 replica”，不表示 layout 完全不动。即使 $K_{\rm req}=0$，remap 仍然可能比 current fixed layout 明显更好。


# Layout Requirement 的两个来源

下面给出 $K_{\rm req}$ 的解析下界，并解释为什么真实 extra-instance requirement 可能高于“逐个 Expert 数 replica”的结果。相同分析也可以用来解释候选 layout 的 residual。

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

逐个 Expert 相加得到：

$$
K_{\rm indiv}(x,B)
=
\sum_e
\max\left(
0,
\left\lceil\frac{x_e}{B}\right\rceil-1
\right).
$$

它是 $K_{\rm req}$ 的解析 lower bound：

$$
K_{\rm indiv}(x,B)
\le
K_{\rm req}(x,\rho;\Theta).
$$

如果候选 layout 连这个条件都不满足，那么无论 rerouter 多好，$\rho^*(x,G)$ 都不可能达到目标。

---

## Packing / Fragmentation Gap

单 Expert 下界并不考虑多个 Experts 如何共同装入有限的 ranks。一般地先定义：

$$
K_{\rm interaction}
=
K_{\rm req}-K_{\rm indiv}.
$$

在 $U_e$ 覆盖全部 ranks、Rank memory 不成为额外瓶颈的基准模型中，这个差值只来自多个 Experts 共享 ranks 时的 packing / fragmentation，记作：

$$
K_{\rm packing}=K_{\rm interaction}.
$$

在一般 $\Theta$ 下，$K_{\rm interaction}$ 还可能包含 placement domain 和 Rank memory 的影响，不能全部解释成 packing。

例如在上述全 placement domain、充足 Rank memory、连续自由分流的基准模型中，有 $R=3$ 个 ranks、5 个 Experts，每个 Expert 的负载都是 $N/5$，目标是 $B=N/3$。因为：

$$
\frac N5<\frac N3,
$$

每个 Expert 单独看都只需要一个实例，所以 $K_{\rm indiv}=0$。但一个 rank 放不下两个完整 Experts，因为 $2N/5>N/3$；3 个 ranks 最多容纳 3 个不拆分的 Experts，另外两个 Experts 都必须至少跨两个 ranks，因此：

$$
K_{\rm req}\ge2.
$$

两个额外实例也确实足够。例如把前三个 Experts 分别以 $N/5$ 放到三个 ranks，再令：

$$
\begin{aligned}
e_4 &: \left(\frac{2N}{15},\frac N{15},0\right),\\
e_5 &: \left(0,\frac N{15},\frac{2N}{15}\right).
\end{aligned}
$$

三个 ranks 的总负载都恰好是 $N/3$，所以这个例子中：

$$
K_{\rm indiv}=0,
\qquad
K_{\rm req}=K_{\rm packing}=2.
$$

## Expert 子集与 Rank 邻域

进一步考虑一组 Experts $S$。

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

# 从单 Step 到 Trace：Memory--Freshness Frontier

单 step 的 requirement 是：

$$
K_{\rm inst}(t,\rho;\Theta)
=
K_{\rm req}(x_t,\rho;\Theta).
$$

它假设当前负载已经知道，layout 可以为当前 step 单独决定。整条 trace 上真正需要回答的是：**如果一套 layout 必须复用 $L$ 个 steps，在目标 coverage 下至少需要多少额外实例？**

## Placement Epoch 与 Trace SLO

设 trace 包含 $T$ 个 steps。对于给定 layout reuse length $L$ 和 phase offset $o\in\{0,\ldots,L-1\}$，把 trace 划分成 placement epochs：

$$
\mathcal W_{j,o,L}.
$$

每个 epoch $j$ 使用一套固定 placement $z^{(j)}$；epoch 内每个 step 都可以根据当前真实负载独立 reroute。定义：

- $h_t\in\{0,1\}$：step $t$ 是否达到目标 $\rho_{\rm target}$。
- $K$：任意一个 epoch 允许使用的最大额外实例数。
- $B_t=\rho_{\rm target}N_t/R$：step $t$ 的目标 bottleneck load。

对 $t\in\mathcal W_{j,o,L}$，要求：

$$
\begin{aligned}
&z_{er}^{(j)}=0, &&r\notin U_e,\\
&\sum_r z_{er}^{(j)}\ge1, &&\forall e,\\
&\sum_e z_{er}^{(j)}\le M_r, &&\forall r,\\
&\sum_{e,r}z_{er}^{(j)}-E\le K,\\
&\sum_r y_{er,t}=x_e(t), &&\forall e,\\
&0\le y_{er,t}\le N_t z_{er}^{(j)},\\
&\sum_e y_{er,t}
\le
B_t+N_t(1-h_t), &&\forall r.
\end{aligned}
$$

最后一行的含义是：$h_t=1$ 时必须满足目标 bottleneck；$h_t=0$ 时只取消当前 step 的目标上界，token 仍然必须由前面的流守恒约束全部执行，并不表示允许 drop token。

默认 trace SLO 为：

$$
\frac1T\sum_{t=1}^T h_t\ge q,
\qquad
q=0.95.
$$

同时报告 $q=1$ 的 strict 结果，避免 P95 掩盖极端热点。

## Trace Capability Oracle

定义：

$$
K_{\rm trace}(L,\rho,q;\Theta,o)
=
\min_{K,\{z^{(j)}\},\{y_t\},\{h_t\}} K.
$$

它回答：**layout 每 $L$ 个 steps 才能更新一次时，为让整条 trace 至少 $q$ 比例的 steps 达到目标 $\rho$，离线最优至少需要多少额外 instances？**

- $L=1$：每个 step 都可以根据当前负载产生当前-step layout，对应 realtime placement 的纯 capability boundary。
- $L>1$：一套 layout 必须覆盖多个 steps，量化降低 placement freshness 后需要增加多少 memory。

这里没有“热点寿命”假设。$L$ 描述的是 layout reuse length，不是热点持续时间。

这个 oracle 能看到每个 epoch 内的完整负载，因此是 clairvoyant workload boundary，不是 historical placement 算法。Historical candidate 是否能仅凭过去负载接近该边界，必须通过 causal replay 单独评价。

不同 phase offset 可能改变 epoch 边界。通用 workload characterization 报告：

$$
K_{\rm trace}^{\rm median}(L,\rho,q)
=
\operatorname{median}_o
K_{\rm trace}(L,\rho,q;\Theta,o),
$$

以及：

$$
K_{\rm trace}^{\rm worst}(L,\rho,q)
=
\max_o
K_{\rm trace}(L,\rho,q;\Theta,o).
$$

评价具体候选方案时，则使用它真实的 placement update 边界，不对 offset 取 oracle。

给定 memory budget $K$，还可以反向定义允许的最大 layout reuse length：

$$
L_{\max}(K,\rho,q;\Theta)
=
\max\left\{
L:
K_{\rm trace}^{\rm worst}(L,\rho,q;\Theta)\le K
\right\}.
$$

于是 memory--freshness trade-off 变成可计算的 frontier：增大 $K$ 是用显存覆盖更多彼此不兼容的 step-level requirements；减小 $L$ 是让 layout 更频繁地按当前负载重算。

$S,C_m,D,H_m$ 继续作为解释变量：它们帮助解释 $K_{\rm inst}$ 为什么大、$K_{\rm trace}$ 为什么随 $L$ 变化，但不再直接推出 replica 数或最终方案。

## 只刻画给定 Trace

$K_{\rm trace}$ 是 empirical capability characterization，不声称预测未来 workload。除不同 phase offsets 外，还应对 trace 做 contiguous block bootstrap，报告 $K_{\rm inst}$、coverage 和 memory--freshness frontier 的置信区间；block sampling 保留局部时间相关性，不引入额外预测模型。

---

# Candidate Scheme Diagnosis

设候选方案 $s$ 在 step $t$ 实际产生 placement $G_t^s$、reroute $y_t^s$ 和不均衡度 $\rho_s(t)$。

不受 layout 限制、允许每个 Expert 到达所有 ranks 的理想下界记作 $\rho_{\rm ideal}(t)$。连续分流时它为 1；整数 token 分流时：

$$
\rho_{\rm ideal}(t)
=
\frac{\lceil N_t/R\rceil}{N_t/R}.
$$

候选方案与理想值的差距分成两部分：

$$
\rho_s(t)-\rho_{\rm ideal}(t)
=
\underbrace{
\rho^*(x_t,G_t^s)
-\rho_{\rm ideal}(t)
}_{\text{layout residual}}
+
\underbrace{
\rho_s(t)
-\rho^*(x_t,G_t^s)
}_{\text{reroute gap}}.
$$

- **layout residual**：candidate 实际产生的 layout 即使使用最优 reroute 仍然无法消除的部分。
- **reroute gap**：这个 layout 本来能够做到，但 candidate 的实际 rerouter 没有做到的部分。

连续 token 和整数 token 是两套不同的评价模型；计算 residual / gap 时，$\rho_{\rm ideal}$、$\rho^*$ 与 candidate result 必须使用同一套模型。

Candidate replay 必须严格因果：

- historical placement 只能使用当时已经观察到的负载，并施加真实的统计、更新和生效延迟；
- realtime placement 每一步使用当前负载产生当前 $G_t^s$，并让新 layout 服务当前 step；
- capability oracle 与 candidate 结果分栏报告，不能用 clairvoyant $K_{\rm trace}$ 冒充 historical placement 的实际表现。

除 residual / gap 外，还要比较 candidate 的 instance budget 和真实 update interval 是否超过 workload boundary：如果资源明显高于 $K_{\rm trace}$ 但 layout residual 仍大，说明 placement 没有有效利用已有 freedom；如果 layout residual 小但 reroute gap 大，问题在实际 rerouter。

---

# 选择不同的负载均衡方案

方案选择不再从 $S$ 或 $H_m$ 直接跳到某个算法，而是依次检查 capability boundary、candidate gap 和系统成本。

## 第一步：当前 Layout 与 Remap-only

- 如果 current fixed layout 的 $\rho^*$ 已接近 $\rho_{\rm ideal}$，实际 reroute gap 也很小，那么 layout 和 reroute 都没有明显改进空间。
- 如果 fixed layout residual 大，但 remap-only 的 $\rho_{\rm layout}^*(x,0;\Theta)$ 已达到目标，说明不需要额外 replicas，重新摆放基础实例就足够。
- $K_{\rm req}=0$ 不能单独推出“无需复杂 LB”，因为 remap 本身仍可能带来很大收益。

## 第二步：需要多少 Replication

扫描 $\rho_{\rm layout}^*(K)$：

- 少量 $K$ 就让 $\rho_{\rm layout}^*(K)$ 快速下降并进入平台，说明 workload 主要缺少少量热点 capacity。
- $K_{\rm indiv}$ 已经很大，说明少数 Expert 的单体 peak load 就要求高 replication degree。
- 在基准模型中 $K_{\rm packing}$ 占主要部分，说明问题来自多个 Experts 在 ranks 上的 packing / fragmentation，而不只是单热点。一般 $\Theta$ 下应先分解 $K_{\rm interaction}$ 中的 domain 和 memory 约束，不能直接归因给 packing。

## 第三步：需要多高的 Placement Freshness

比较 $K_{\rm trace}(L,\rho,q)$：

- $L$ 增大后 requirement 基本不变：一套 layout 可以覆盖较长区间，historical placement 具有足够的 workload capability；它是否真的可用，还要看 causal candidate replay。
- $L=1$ requirement 不高，但 $L$ 增大后 requirement 明显上升：每个 step 自身不需要很多 replicas，但不同 steps 的 layout requirements 彼此不兼容。此时若不增加 memory，就需要 per-step realtime placement。
- 所有 $L$ 下 requirement 都很高：更实时的 placement 不能消除纯空间 capacity 不足，仍需要更多 replicas 或接受较高 $\rho$。

Realtime placement 在这里始终表示：**每个 step 使用当前负载重新计算当前-step layout。** 它的价值由 $L=1$ capability 相对低 freshness capability 的改善决定，不依赖热点在后续 steps 中持续存在。

## 第四步：端到端 Worth-it Gate

上述三步回答“什么 layout 能力有用”；最终仍要检查它节省的 compute critical path 是否超过 planning、weight preparation、通信和 kernel fragmentation 的 exposed cost。因而最后的决策不是一个脱离资源的排名，而是：

> **在达到目标 coverage 的候选方案中，选择满足 memory、movement 和通信约束的 E2E Pareto 点。**

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
