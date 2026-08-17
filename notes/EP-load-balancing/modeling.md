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

本节提供几个简单易计算的指标，可以用于快速把握当前负载分布的特征。它们不依赖于 layout 或 reroute，仅作为初步诊断。后续会使用更加精确的 layout-aware oracle 来评价 candidate。

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


# 解耦 Layout 和 Reroute

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

# Layout 分析

固定 layout 的 $\rho^*(x,G)$ 只回答“这套 layout 能做到多好”。要评价 workload 需要何种 placement 方案，还要依次回答三个问题：给定资源能做到多好、达到目标至少需要多少资源、一套 layout 复用多个 steps 后资源需求如何变化。
用：

$$
\Theta=
\left(
\{U_e\},
\{M_r\},
\text{remap permission}
\right)
$$

描述系统允许的 layout 空间。其中 $U_e$ 是 Expert $e$ 的合法 Rank 域，$M_r$ 是 rank $r$ 的 instance 上限，remap permission 表示基础实例能否离开当前位置。令 $z_{er}\in\{0,1\}$ 表示 Expert $e$ 是否在 rank $r$ 上存在实例，则合法 placement 满足：

$$
\begin{aligned}
&z_{er}=0, &&r\notin U_e,\\
&\sum_r z_{er}\ge1, &&\forall e,\\
&\sum_e z_{er}\le M_r, &&\forall r.
\end{aligned}
$$

设当前基础 placement 为 $z^0$：不允许 remap 时再要求 $z_{er}\ge z_{er}^0$；fixed current layout 则直接要求 $z=z^0$。

## 给定资源能做到多好：$\rho_{\rm layout}^*(K)$

令 $\mathcal G(K;\Theta)$ 表示满足上述约束、且额外实例数不超过 $K$ 的所有 layouts：

$$
\sum_{e,r}z_{er}-E\le K.
$$

在每个合法 layout 上使用前一章的 exact reroute oracle，定义：

$$
\rho_{\rm layout}^*(x,K;\Theta)
=
\min_{G\in\mathcal G(K;\Theta)}
\rho^*(x,G).
$$

> **核心输出：$\rho_{\rm layout}^*(x,K;\Theta)$。** 它表示给定 $K$ 个额外 instances 时，合法 placement 理论上能够达到的最低 Rank imbalance。

扫描 $K$ 得到 layout resource frontier。少量 $K$ 就使曲线快速下降，说明 workload 只缺少少量热点 capacity；曲线很早进入平台，则继续增加 replicas 的边际收益有限。

这条曲线需要配合三个参考点阅读：

- **fixed current layout**：固定 $z=z^0$，输出 $\rho^*(x,G_0)$。
- **remap-only**：令 $K=0$，但允许在 $U_e$ 内重新选择每个基础实例的位置。
- **remap + replication**：令 $K>0$，同时决定基础实例和 replicas 的位置。

因此 $K=0$ 只表示“不增加 replica”，不表示 layout 不动。Fixed 与 remap-only 的差距衡量重新摆放基础实例的价值；remap-only 与 $K>0$ 曲线的差距衡量 replication 的价值。

## 达到目标需要多少资源：$K_{\rm req}$

给定目标 $\rho_{\rm target}$，令：

$$
B_{\rm target}=\rho_{\rm target}\frac NR.
$$

在合法 placement 上加入当前 step 的流守恒和 Rank load 约束：

$$
\begin{aligned}
&\sum_r y_{er}=x_e, &&\forall e,\\
&0\le y_{er}\le Nz_{er}, &&\forall e,r,\\
&\sum_e y_{er}\le B_{\rm target}, &&\forall r.
\end{aligned}
$$

定义：

$$
K_{\rm req}(x,\rho_{\rm target};\Theta)
=
\min_{z,y}
\left(
\sum_{e,r}z_{er}-E
\right).
$$

> **核心输出：$K_{\rm req}(x,\rho_{\rm target};\Theta)$。** 它表示当前负载达到目标 imbalance 至少需要多少额外 Expert instances。

它不是新的独立模型，而是上一节 resource frontier 的反问题：

$$
K_{\rm req}(x,\rho;\Theta)\le K
\quad\Longleftrightarrow\quad
\rho_{\rm layout}^*(x,K;\Theta)\le\rho.
$$

如果 token 必须整数分流，$y_{er}$ 取整数，Rank 上界等价于 $\sum_e y_{er}\le\lfloor B_{\rm target}\rfloor$。当 $\rho_{\rm target}<\rho_{\rm ideal}$ 时，目标本身不可行。

**单 Expert 下界。** Expert $e$ 至少需要能够到达 $\lceil x_e/B\rceil$ 个 ranks，因此：

$$
K_{\rm indiv}(x,B)
=
\sum_e
\max\left(
0,
\left\lceil\frac{x_e}{B}\right\rceil-1
\right)
\le K_{\rm req}.
$$

完美均衡时等价于 $|A_e|\ge\lceil Rp_e\rceil$。例如一个 Expert 占全局 25% token、系统有 8 张 GPU，则它至少需要能在 2 张 GPU 上执行。

**Interaction / packing 解释。** 定义辅助诊断量：

$$
K_{\rm interaction}=K_{\rm req}-K_{\rm indiv}.
$$

一般 $\Theta$ 下，它还可能包含 placement domain 和 Rank memory 的影响；只有在全 placement domain、Rank memory 不绑定的基准模型中，才可将其称为 $K_{\rm packing}$。

例如 $R=3$、5 个 Experts 各占 $N/5$、目标 $B=N/3$。每个 Expert 单独看都不需要 replica，所以 $K_{\rm indiv}=0$；但一个 rank 无法容纳两个完整 Experts，因为 $2N/5>N/3$，因此至少两个 Experts 必须拆分，得到 $K_{\rm req}\ge2$。两个额外实例也确实足够：

$$
\begin{aligned}
e_4 &: \left(\frac{2N}{15},\frac N{15},0\right),\\
e_5 &: \left(0,\frac N{15},\frac{2N}{15}\right).
\end{aligned}
$$

配合前三个 Experts 各自占据一个 rank，三个 ranks 的总负载都恰好为 $N/3$，所以该例中 $K_{\rm req}=K_{\rm packing}=2$。

**Expert 子集诊断。** 对任意 Expert 集合 $S$，达到 bottleneck $B$ 必须满足：

$$
\sum_{e\in S}x_e\le B\,|N(S)|.
$$

如果多个热点 Experts 共享过小的 Rank 邻域，单独看每个 Expert 的 replica 数可能都够，联合起来仍然不可行。Max-flow 找到的最紧集合 $S$ 可以解释 $K_{\rm req}$ 或候选 layout residual 的来源，但它不是另一个核心输出。

## Layout 复用多个 Steps 需要多少资源：$K_{\rm trace}$

单 step 的 requirement 只是：

$$
K_{\rm inst}(t,\rho;\Theta)
=
K_{\rm req}(x_t,\rho;\Theta).
$$

现在要求一套 layout 固定复用 $L$ 个 steps。设 trace 有 $T$ 个 steps，phase offset 为 $o\in\{0,\ldots,L-1\}$，由此得到 placement epochs $\mathcal W_{j,o,L}$。Epoch $j$ 使用固定 placement $z^{(j)}$，但每个 step 仍可依据当前负载独立 reroute。

令 $h_t\in\{0,1\}$ 表示 step $t$ 是否达到目标，$B_t=\rho_{\rm target}N_t/R$。对 $t\in\mathcal W_{j,o,L}$，要求：

$$
\begin{aligned}
&z_{er}^{(j)}=0, &&r\notin U_e,\\
&\sum_r z_{er}^{(j)}\ge1, &&\forall e,\\
&\sum_e z_{er}^{(j)}\le M_r, &&\forall r,\\
&\sum_{e,r}z_{er}^{(j)}-E\le K,\\
&\sum_r y_{er,t}=x_e(t), &&\forall e,\\
&0\le y_{er,t}\le N_tz_{er}^{(j)},\\
&\sum_e y_{er,t}\le B_t+N_t(1-h_t), &&\forall r.
\end{aligned}
$$

$h_t=0$ 只取消当前 step 的目标 bottleneck 上界，token 仍必须由流守恒约束全部执行。Trace coverage 要求：

$$
\frac1T\sum_{t=1}^T h_t\ge q,
\qquad q=0.95,
$$

并同时报告 strict $q=1$ 结果。最终定义：

$$
K_{\rm trace}(L,\rho,q;\Theta,o)
=
\min_{K,\{z^{(j)}\},\{y_t\},\{h_t\}}K.
$$

> **核心输出：$K_{\rm trace}(L,\rho,q;\Theta,o)$。** 它表示 layout 每 $L$ 个 steps 更新一次时，为让至少 $q$ 比例的 steps 达到目标 $\rho$，离线最优至少需要多少额外 instances。

$L=1$ 允许每步根据当前负载生成当前-step layout，是 realtime placement 的 workload boundary；$L>1$ 要求同一 layout 覆盖多个不同负载。这里的 $L$ 是 layout reuse length，不是热点寿命。

该 oracle 可以看到对应 epoch 的完整负载，因此是 clairvoyant capability boundary，不是 historical policy。Historical candidate 只能使用当时已有历史，必须在后文通过 causal replay 单独评价。

不同 phase offset 会改变 epoch 边界。通用 workload characterization 对同一个核心指标报告：

$$
K_{\rm trace}^{\rm median}
=
\operatorname{median}_o K_{\rm trace}(L,\rho,q;\Theta,o),
\qquad
K_{\rm trace}^{\rm worst}
=
\max_o K_{\rm trace}(L,\rho,q;\Theta,o).
$$

具体 candidate 则使用其真实 update boundary，不对 offset 取 oracle。给定 memory budget $K$ 时，也可以从同一条 frontier 反读：

$$
L_{\max}(K,\rho,q;\Theta)
=
\max\left\{
L:K_{\rm trace}^{\rm worst}(L,\rho,q;\Theta)\le K
\right\}.
$$

$L_{\max}$ 只是 $K_{\rm trace}$ 的反向读法，不是第四个独立指标。$S,C_m,D,H_m$ 也只用于解释 frontier 的形状，不能直接推出方案。

最后，$K_{\rm trace}$ 只刻画给定 trace，不声称预测未来 workload。应使用 contiguous block bootstrap 报告该 frontier 的置信区间，以保留局部时间相关性；这同样不引入热点轮转或热点寿命假设。

---

# Candidate 方案评价

前一章给出了 workload 的 layout resource 和 freshness boundary。本章把这些离线边界用于评价实际 candidate：先判断方案的损失来自 layout 还是 rerouter，再由 workload boundary 判断方案需要具备何种 placement 性质，最后检查这些收益是否足以覆盖系统成本。三步必须按顺序进行；只看最终 latency，无法解释方案为什么有效或无效，只看 oracle，也无法判断它是否值得落地。

## 第一步：定位 Candidate 的 Balance 缺口

设候选方案 $s$ 在 step $t$ 实际产生 placement $G_t^s$、reroute $y_t^s$ 和不均衡度 $\rho_s(t)$。不受 layout 限制、允许每个 Expert 到达所有 ranks 的理想下界记作 $\rho_{\rm ideal}(t)$。连续分流时它为 1；整数 token 分流时：

$$
\rho_{\rm ideal}(t)
=
\frac{\lceil N_t/R\rceil}{N_t/R}.
$$

在 candidate 实际产生的 layout 上运行 exact reroute oracle，就可以把总差距严格拆成：

$$
\rho_s(t)-\rho_{\rm ideal}(t)
=
\underbrace{
\rho^*(x_t,G_t^s)-\rho_{\rm ideal}(t)
}_{\text{layout residual}}
+
\underbrace{
\rho_s(t)-\rho^*(x_t,G_t^s)
}_{\text{reroute gap}}.
$$

- **layout residual**：candidate 的 layout 即使使用最优 reroute 仍然无法消除的部分。
- **reroute gap**：这套 layout 本来能够做到，但 candidate 的实际 rerouter 没有做到的部分。

因此两项的含义不同：layout residual 大，应检查 placement 使用的资源和 freshness；layout residual 小而 reroute gap 大，才说明主要问题在实际分流。连续 token 和整数 token 是两套不同的评价模型，$\rho_{\rm ideal}$、$\rho^*$ 与 candidate result 必须始终使用同一套模型。

这项诊断只有在 replay 因果时才有意义：

- historical placement 只能使用决策时已经观察到的负载，并施加真实的统计、planning、更新和生效延迟；
- realtime placement 每一步使用当前负载产生当前 $G_t^s$，并让它服务当前 step；
- clairvoyant $K_{\rm trace}$ 只作为 workload boundary，与实际 candidate 分栏报告，不能冒充 historical policy 的结果。

## 第二步：判断 Workload 需要何种 Placement 性质

Gap decomposition 告诉我们 candidate 哪里损失了 balance；前一章的三个核心指标进一步回答应该补充哪种 layout 能力。判断顺序是 fixed / remap、replication、freshness，后一个判断只在前一个仍不足时才有意义。

**先判断是否需要改变 current layout。** 比较 fixed current layout 与 remap-only：

- 如果 fixed layout 的 $\rho^*$ 已接近 $\rho_{\rm ideal}$，且实际 reroute gap 也很小，那么 layout 和 reroute 都没有明显改进空间。
- 如果 fixed layout residual 大，但 $\rho_{\rm layout}^*(x,0;\Theta)$ 已达到目标，说明无需额外 replicas，重新摆放基础实例即可。
- 因而 $K_{\rm req}=0$ 不能推出“无需 placement LB”；它只表示 remap-only 已经足够。

**Remap-only 仍不足时，再判断需要多少 replication。** 查看 $\rho_{\rm layout}^*(K)$ 和 $K_{\rm req}$：

- 少量 $K$ 就使 $\rho_{\rm layout}^*(K)$ 快速下降并进入平台，说明 workload 只缺少少量热点 capacity。
- $K_{\rm indiv}$ 已经很大，说明少数 Expert 的单体 peak load 要求较高 replication degree。
- 在基准模型中 $K_{\rm packing}$ 占主要部分，说明多个 Experts 的 packing / fragmentation 才是主要来源；一般 $\Theta$ 下还要排除 placement domain 和 Rank memory 的影响。

**空间资源确定后，最后判断需要多高 freshness。** 比较 $K_{\rm trace}(L,\rho,q)$：

- $L$ 增大后 requirement 基本不变：一套 layout 可以覆盖较长区间，workload 不要求高 placement freshness；historical policy 能否接近该边界仍由 causal replay 决定。
- $L=1$ requirement 较低，但随 $L$ 增大明显上升：不同 steps 的 layout requirements 彼此不兼容。在既定 memory 下，需要 per-step realtime placement，或者接受更低 coverage / 更高 $\rho$。
- 所有 $L$ 下 requirement 都很高：问题是空间 capacity 不足，提高 placement freshness 无法替代 replication。

这里 realtime placement 始终表示：**每个 step 使用当前负载重新计算当前-step layout。** 它的价值来自 $L=1$ 相对较大 $L$ 的 capability 改善，不依赖热点在未来继续存在。

Candidate 的资源使用也要与 workload boundary 对齐：如果 instance budget 和 update frequency 已经优于边界要求，layout residual 却仍然很大，说明不是 workload 要求过高，而是 placement policy 没有有效利用已有 freedom。

## 第三步：判断 Balance 收益是否值得系统成本

前两步回答 candidate 改善了什么，以及 workload 是否真的需要这种能力。最后一步才把 balance 收益转换为端到端收益。

假设当前 step 在完美均衡时的 Expert compute 时间为 $C$，原始 imbalance 为 $\rho_0$，candidate 降低到 $\rho_1$，则：

$$
T_0^{\rm expert}\approx\rho_0C,
\qquad
T_1^{\rm expert}\approx\rho_1C,
$$

仅考虑 Expert compute 时，理想 speedup 为：

$$
\text{Speedup}_{\rm expert}
\approx
\frac{\rho_0}{\rho_1}.
$$

但 placement / reroute 同时会改变通信、planning、weight movement 和 kernel efficiency。定义 candidate 的净关键路径收益：

$$
\Delta T_{\rm net}
=
(\rho_0-\rho_1)C
-\Delta T_{\rm comm}
-H_{\rm plan}^{\rm exposed}
-H_{\rm move}^{\rm exposed}
-\Delta T_{\rm kernel}.
$$

其中：

- $\Delta T_{\rm comm}$ 是 reroute 引起的通信时间变化，可以为正或负。
- $H_{\rm plan}^{\rm exposed}$ 是 placement / reroute planning 真正暴露在关键路径上的部分。
- $H_{\rm move}^{\rm exposed}$ 是 weight preparation / movement 暴露在关键路径上的部分。
- $\Delta T_{\rm kernel}$ 是 token 被切到更多 replicas 后，小 batch 或 kernel fragmentation 引起的时间变化。

Candidate 只有在 $\Delta T_{\rm net}>0$ 时才产生正的端到端收益。这里只计算 exposed cost：已经与其他阶段重叠的 duration 不能再次完整相加。

这也解释了不同方案的适用边界：training / prefill 的 $C$ 较大，更容易覆盖 LB 成本；decode 的 compute saving 较小，未必值得复杂 placement。Historical placement 可以把部分 movement 移出当前 critical path，但承担 stale-layout residual；realtime placement 没有历史滞后，却必须在当前 step 内支付 planning / movement 成本，因此需要带来足够大的即时 balance 改善。

最终不应生成一个脱离资源约束的 candidate 总分，而应在达到目标 coverage 的方案中，选择 memory、movement、communication 与 E2E latency 的 Pareto 点。
