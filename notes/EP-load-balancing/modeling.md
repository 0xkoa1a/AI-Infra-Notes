---
title: "EP 负载均衡：数学建模"
order: 1
---

**给定一个随时间变化的 Expert 负载过程，判断一个负载均衡方案需要具备多大的“布局自由度”、多强的“实时性”和多好的“分流能力”，才能值得它付出的额外成本。**

# 问题 Setup

先固定**一个 layer、一个 step**。设：
* 有 $E$ 个 logical experts。
* 有 $R$ 个 ranks。
* Expert $e$ 当前收到 $x_e$ 个 token。
* 总 token 数为 $N=\sum_e x_e$。
* Expert $e$ 当前可以在一组 ranks 上执行，记作 $A_e$。

  * 如果没有 replica，$|A_e|=1$。
  * 如果有两个物理实例，$|A_e|=2$。
* reroute 决定这 $x_e$ 个 token 怎么在 $A_e$ 中分。

## 端到端性能

如果最终 rank $r$ 收到 $L_r$ 个 token，那么这一层 expert compute 的完成时间由收到 token 最多的 rank 决定：

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

* $\rho=1$：完美均衡。
* $\rho=1.2$：最慢 Rank 比理想情况多做 20%。
* $\rho=2$：整个 expert compute 至少浪费接近一半并行能力。

它直接对应 critical path.

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

* A：一个 Expert 50%，其他很均匀。
* B：四个 Experts 各 12.5%，其他很均匀。

它们可能 peak load 类似，但对 placement 的要求完全不同。

$C_m$ 曲线实际上回答：**需要给多少个 Experts 提供额外副本。**

## 精确负载变化

定义归一化负载向量：

$$
p(t)=(p_1(t),\ldots,p_E(t)).
$$

可以看一个非常简单的 lag-$\Delta$ drift：**隔了 $\Delta$ 个 step 后，有多少比例的负载质量“搬到了别的 Experts”。**

$$
D(\Delta)=\mathbb E\left[\frac12|p(t)-p(t-\Delta)|_1\right].
$$

* $D(\Delta)\approx0$：分布非常稳定。
* $D(\Delta)$ 很大：历史精确负载已经没有多少参考意义。

尤其应该看：$\Delta=$ **你的 placement 从观察负载到真正生效的时间**。例如一个 placement 每 100 step 才能更新一次，那么真正相关的是 $D(100)$，不是 $D(1)$。

---

## 粗粒度负载变化

因为 placement 并不一定需要预测“Expert 17 下一步到底是 5372 个还是 6141 个 token”。

它可能只需要预测“**Expert 17 下一段时间是不是仍然属于热点、值得多准备一个 replica**”。

所以应该再定义一个**更粗粒度的 temporal locality**：**隔 $\Delta$ 个 step 后，Top-$m$ 热点 Expert 还有多少仍然是热点。**

$$
H_m(\Delta)=\frac{|\operatorname{Top}_m(t)\cap\operatorname{Top}_m(t-\Delta)|}{m}.
$$

于是可能出现一种非常重要的情况：

* $D(\Delta)$ 很高：
  * 精确 load 数字预测不准。
* 但 $H_m(\Delta)$ 很高：
  * “谁是热点”却相当稳定。

那么**历史 placement + 当前实时 reroute 恰好非常合适。**

---


# 解耦布局和分流

给定当前负载 $x$ 和 placement $G={A_e}$，假设存在一个**完美的 oracle rerouter**。

定义：

$$
\rho^*(x,G)=\min_{\text{合法 reroute}}\frac{\max_r L_r}{N/R}.
$$

它表示：**placement 固定以后，哪怕给我世界上最好的分流算法，最少还能剩多少负载不均衡。**

于是 placement 和 reroute 彻底解耦了：

* placement 好不好：看 $\rho^*(x,G)$。
* reroute 算法好不好：看它距离 $\rho^*(x,G)$ 有多远。

---

可以定义一个简单的算法质量：

$$
\alpha=\frac{\rho_{\text{algorithm}}}{\rho^*}.
$$

表示：**实际 rerouter 比当前 placement 下的理论最优差多少。**

例如一个具体 rerouter 最后做到 $\rho_{\text{algorithm}}=1.08$，而这个 placement 下理论最优是 $\rho^*=1.05$。


# 方案侧：Placement 侧分析

## 每个专家到底需要多少 replica？

假设 Expert $e$ 负责的 token 占当前所有 token 的比例是：

$$
p_e=\frac{x_e}{N}.
$$

理想状态下，一个 rank 只能承担总工作量的 $1/R$。

为了达到完美均衡，Expert $e$ 至少得能够分布到大约：

$$
|A_e|\ge Rp_e
$$

个 ranks。

> 例如，**一个 Expert 如果吃掉了全局 25% 的 token，而系统有 8 张 GPU，那么它至少需要能在约 $8\times25\%=2$ 张 GPU 上执行。**

可以发现，**负载越集中在少数 Expert 上，热点 Expert 所需的 replication degree 越高。**

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

* Expert 1 和 Expert 2 都很热；
* 它们虽然每个都有两个 replica；
* 但恰好都放在 rank 0 和 rank 1。

那么：

* 单独看 replica 数，好像不错；
* 但两个 Expert 的流量竞争的是同一组 GPU；
* 依然可能无法 balance。

所以一个好的 placement 并不只是热 Expert 多复制几个，应该是**热点 Expert 需要获得足够多、而且彼此适当错开的 Rank 邻域。**

---

# 选择不同的负载均衡方案

## 负载比较均匀

如果：

* 专家负载不均衡度 $S$ 小；
* GPU 负载不均衡度 $\rho$ 本身已经接近 1；

那么任何复杂 LB 都没有意义。

## 偏斜明显，而且热点具有 temporal locality

如果：

* $S$ / $C_m$ 显示少量热点需要额外 capacity；
* $H_m(\Delta)$ 高；
* 但精确 $D(\Delta)$ 不一定很低；

那么比较适合：

> **慢速历史 placement + 当前精确 reroute。**

placement 只需要稳定识别：

* 哪些 experts 值得 replica；
* replica 大概应该在哪些 ranks。

reroute 再解决当前 step 的精确比例。

    
## 热点本身快速变化

如果：

* $H_m(\Delta)$ 也很低；

那么 stale placement 就会出现根本问题：

> replica 做在昨天的热点上，而当前热点没有 replica。

此时有三个可能方向：

* 使用当前负载实时创建 / 准备 replica，例如 UltraEP / MoonEP 一类；
* 用更多冗余显存复制更多 Experts，让 placement 对未来更 robust；
* 干脆接受部分 imbalance，因为实时搬权重的成本比收益还高。

所以问题实际上变成一个非常清晰的 trade-off：

> **用显存换 robustness，还是用关键路径时间换 adaptivity。**

---

# 性能收益分析

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

* $\Delta T_{\rm comm}$：reroute 后额外通信时间。
* $H_{\rm plan}$：负载均衡计算本身的关键路径开销。
* $H_{\rm move}$：实时 weight movement 的关键路径开销。

那么负载均衡值得做的条件是：**因为减少长尾而省下来的 compute 时间，必须大于它新增的通信、求解和权重准备时间。**

$$
(\rho_0-\rho_1)C > \Delta T_{\rm comm} + H_{\rm plan} + H_{\rm move}.
$$


这个式子可以解释为什么：
* training / prefill 更值得做；decode 未必值得做；
* historical placement 可以接受一定预测误差，因为它把 weight movement 移出了当前 critical path；
* realtime placement 必须带来明显更低的 $\rho_1$，才能补偿当前 step 的 planning / moving cost。
