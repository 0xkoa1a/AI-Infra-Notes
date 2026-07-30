# 场景建模

三级流水线之间有两个 buffer：
- MMU（包括 L1 和 L2，L2 的结果会进入 RS 和 NVLink）
    - epilogue buffer
- Remote Store
    - 硬件的发送 buffer
- NVLink

如何指导 MMU 发送 L2 的模式，使得硬件发送 buffer 尽量不会被打爆？

希望让 L2 请求在时间维度尽量散开，从而限制硬件发送 buffer 的瞬时压力。
- 衡量请求的分散程度：在一个时间窗口内，请求数量关于时间的方差。方差越大说明请求越分散。

如何对此场景做一个数学建模？


---


# 场景建模

流水线中存在如下数据路径：

```text
MMU
  ├─ L1
  └─ L2 → Remote Store 硬件发送 buffer → NVLink
```

其中，L2 会产生 Remote Store 请求并进入硬件发送 buffer，而 L1 不会对该 buffer 产生同样的压力。

对于整个算子，L1 和 L2 的总数量都是确定的。设：

* L1 的总数量为 (N_1)；
* L2 的总数量为 (N_2)；
* 总请求数为 (N=N_1+N_2)。

因此，无论如何排列，L2 的总数据量以及平均发送压力都不会改变。可以优化的只有 L1 和 L2 在时间维度上的排列方式。

例如，下面两种序列具有完全相同的 L1、L2 数量：

```text
L2 L2 L2 L2 L1 L1 L1 L1
```

```text
L2 L1 L2 L1 L2 L1 L2 L1
```

第一种排列中，多个 L2 请求连续到达，形成较大的 L2 burst。Remote Store 请求会在短时间内集中进入硬件发送 buffer，而 NVLink 无法瞬间将这些数据全部发送出去，因此 buffer 占用会快速上升。

第二种排列通过在 L2 之间插入 L1，将同样数量的 L2 请求分散到了更长的时间范围内。L1 执行期间，硬件发送 buffer 可以继续通过 NVLink 排出此前积累的数据，因此 buffer 峰值会更低。

因此，这里的目标不是减少 L2 的总数量，而是：

> 在 L1、L2 总数量固定的条件下，通过将更多 L1 interleave 到连续的 L2 序列中，降低任意短时间窗口内出现的 L2 数量。

可以将请求序列表示为：

$$
x_i=
\begin{cases}
1,&\text{第 }i\text{ 个请求是 L2}\\
0,&\text{第 }i\text{ 个请求是 L1}
\end{cases}
$$

并且：

$$
\sum_{i=1}^{N}x_i=N_2
$$

由于 (N_2) 固定，整个序列中的平均 L2 比例始终为：

$$
\frac{N_2}{N}
$$

因此，排列顺序不会改变平均流量，只会改变局部 burst。

为了衡量 L2 burst，可以观察长度为 (W) 的任意连续窗口中，最多包含多少个 L2：

$$
B(W)=
\max_j
\sum_{i=j}^{j+W-1}x_i
$$

其中 (B(W)) 越大，说明存在越集中的 L2 burst；(B(W)) 越小，说明 L2 在时间轴上分布得越均匀。

硬件发送 buffer 的压力，本质上由某个时间窗口内：

```text
L2 新进入 buffer 的数据量
-
NVLink 在该窗口内排出的数据量
```

决定。

因此，只要通过 L1/L2 interleaving，使任意相关时间窗口中的 L2 数量足够少，NVLink 就能在 L1 间隔期间及时排出已有数据，硬件发送 buffer 就不会持续积累并最终溢出。

最直接的优化方向是减少：

* 连续 L2 的最大长度；
* 短时间窗口中的最大 L2 数量；
* L2 请求之间间隔的不均匀程度。

需要注意，单独计算整个二值序列的方差没有意义。由于 L1 和 L2 的总数量固定，(x_i) 的均值和全局方差也都是固定的，与排列顺序无关。真正需要关注的是滑动窗口中的局部 L2 密度。

理想情况下，应当让 L1 尽可能均匀地插入 L2 序列，使 L2 请求的间隔尽量接近，从而将原本集中的 L2 burst 平摊到更长的执行时间中：

```text
集中排列：
L2 L2 L2 L2 L2 L2 L1 L1 L1

均匀交错：
L2 L2 L1 L2 L2 L1 L2 L1 L2
```

两种排列产生的 L2 总流量相同，但后者具有更小的局部 L2 峰值，因此对硬件发送 buffer 的瞬时压力更低。

综上，可以将该问题概括为：

> 在固定的 L1、L2 数量下，重新排列 MMU 请求序列，使 L2 在所有局部时间窗口中的最大密度尽可能低。通过在 L2 burst 中 interleave 更多 L1，为 NVLink 提供排空发送 buffer 的时间，从而避免硬件发送 buffer 被打爆。

---

对，这里应该把“方差”定义在**高压力请求的发射时间**上，而不是定义在每个时间窗口的请求数量上：

* 请求数量的方差越大，通常意味着越 bursty；
* 请求时间的方差越大，则表示同样一批请求在时间上分布得越开。

结合 microbenchmark，可以构造一个既直观、又能直接指导发射模式的指标。

## Burst–Gap 流水线的时间分散指标

考虑一组会对同一个硬件发送队列产生压力的请求。它们可以来自不同指令、不同 tile，甚至不同 producer SM。

设第 (i) 个请求：

* 发射时间为 (t_i)；
* 数据量为 (b_i)。

这一组请求的总数据量为：

[
B=\sum_i b_i
]

对于同一个算子或同一个 pipeline iteration，(B) 通常是给定的。调度改变的不是总数据量，而是这些请求的发射时间 (t_i)。

首先计算按数据量加权的平均发射时间：

[
\bar t=\frac{\sum_i b_i t_i}{B}
]

然后定义时间分布方差：

[
V_t=\frac{\sum_i b_i(t_i-\bar t)^2}{B}
]

它表达的是：

> 同样的 Remote Store 数据量，在时间轴上被摊开到了多大范围。

请求越集中：

```text
████████
```

(V_t) 越小。

请求之间 interleave 了其他工作：

```text
██ ··· ██ ··· ██ ··· ██
```

(V_t) 越大。

这里最好按数据量加权，因为一个 128 KB 请求对队列的影响显然比一个 16 KB 请求更大。

## 把方差转换成直观的“等效分散宽度”

方差本身的单位是 cycles²，不太容易直接解释。可以将它转换成一个时间宽度：

[
W_{\mathrm{var}}=\sqrt{12V_t}
]

这样定义的原因很简单：如果请求近似均匀地分布在长度为 (W) 的时间区间中，那么它的时间方差约为：

[
V_t\approx\frac{W^2}{12}
]

因此：

[
W_{\mathrm{var}}\approx W
]

可以把 (W_{\mathrm{var}}) 理解为：

> 这组请求相当于被均匀摊开了多少 cycles。

例如，同样四个请求：

```text
集中发射：
0, 0, 0, 0

分散发射：
0, 100, 200, 300
```

前者的 (W_{\mathrm{var}}) 接近 0，后者的 (W_{\mathrm{var}}) 明显更大。

---

# 结合 microbenchmark 得到最终 Burst 指标

microbenchmark 给出了两个关键硬件参数：

* (R)：硬件队列通过 NVLink 排出数据的速率；
* (C_{\mathrm{eff}})：硬件队列能够吸收的有效 burst 容量。

于是可以定义：

[
\boxed{
\mathrm{BDS}
============

\frac{C_{\mathrm{eff}}+R,W_{\mathrm{var}}}{B}
}
]

称为 **Burst Dispersion Score，BDS**。

其中：

* (B)：这一批请求的总数据量；
* (C_{\mathrm{eff}})：硬件队列可以立即容纳的数据量；
* (R W_{\mathrm{var}})：请求被摊开的这段时间内，NVLink 可以排出的数据量。

这个指标的直觉是：

```text
队列可以立即吸收的数据
+
请求发射期间能够排出的数据
--------------------------------
这一批请求的总数据量
```

因此：

* (\mathrm{BDS}>1)：请求大致可以被 queue capacity 和同步 drain 吸收；
* (\mathrm{BDS}\approx1)：处于 capacity knee 附近；
* (\mathrm{BDS}<1)：局部 burst 超过了 buffer 的吸收能力，容易产生 backpressure；
* BDS 越大，请求在时间上越分散，队列压力越低。

对于固定的 (B)、(R) 和 (C_{\mathrm{eff}})，提高 BDS 就等价于提高：

[
V_t
]

也就是让请求的发射时间方差更大。

---

# 它如何对应图中的四个实验

## 图 A：足够大的 gap 会消除 sender-visible backpressure

图 A 显示，Remote Store 路径在较小 gap 下存在明显 backpressure；当 producer-side gap (G) 足够大以后，issue cost 回到约 997 cycles 的 idle floor。

图中给出的关键时间是：

[
D=\frac{B}{R}
]

即一整个 burst 需要的队列排空时间。

当：

[
G\gtrsim D
]

前一个 burst 基本已经排空，下一个 burst 到达时就不会受到明显 backpressure。

在通用调度中，通常不存在一个明确的统一 (G)。一些请求之间可能间隔很短，另一些间隔很长。因此，可以用：

[
W_{\mathrm{var}}
]

替代单一的 gap，描述整个 burst 在时间上被摊开的程度。

当暂时忽略 queue capacity 时：

[
\mathrm{BDS}
\approx
\frac{R W_{\mathrm{var}}}{B}
============================

\frac{W_{\mathrm{var}}}{D}
]

于是：

[
\mathrm{BDS}\gtrsim1
]

对应：

[
W_{\mathrm{var}}\gtrsim D
]

这正是图 A 中 (G\gtrsim D) 的推广形式。

---

## 图 B：系统周期由 gap 和 drain time 中较大者决定

图 B 表明实验结果符合：

[
\text{cycle time}
\approx
\max(G+\text{issue cost},D)
]

而不是“发射之后始终同步等待整个传输完成”。

这意味着 Remote Store 本质上是异步 enqueue：

* producer 可以先把请求放入硬件队列；
* NVLink 在后台以速率 (R) 排出请求；
* 只有 producer 产生请求过于集中时，队列才会反过来阻塞 producer。

因此优化目标不是让每个请求都等待传输完成，而是：

> 让请求在整体时间分布上足够分散，使后台 drain 能够追上 producer 的 enqueue。

BDS 正是在衡量 enqueue 的时间分散程度相对于 drain time 是否足够。

---

## 图 C：孤立 burst 存在明确的容量 knee

图 C 表明硬件中确实存在有限的有效发送 buffer，而且可以通过 isolated burst 测出 capacity knee。

图中显示：

* (K=32/64) 时，effective capacity 约为 2.2–2.7 MB；
* (K=148) 时，effective capacity 约为 4.3–4.6 MB；
* 图中的解释是一个共享容量，再加上每个 producer SM 的一部分 in-flight window。

这正好对应 BDS 中的：

[
C_{\mathrm{eff}}
]

如果所有请求几乎同时发射，那么：

[
W_{\mathrm{var}}\approx0
]

于是：

[
\mathrm{BDS}\approx\frac{C_{\mathrm{eff}}}{B}
]

此时：

* (B<C_{\mathrm{eff}})：整个 burst 可以被硬件队列直接吞下；
* (B>C_{\mathrm{eff}})：开始越过 capacity knee，producer issue time 明显增加。

因此，在完全不分散请求的情况下，BDS 恰好退化为“burst 大小相对于 queue capacity”的指标。

---

## 图 D：所有 producer 共享同一个 drain rate

图 D 显示，稳态 drain time 可以由一个共享的：

[
R\approx717\ \mathrm{GB/s}
]

很好地解释。

这说明 drain rate 并不是每个 producer SM 独立拥有一份，而是多个 producer 共同竞争同一个有效 egress rate。

因此，计算 BDS 时不能只看单个 SM：

```text
SM0：请求很分散
SM1：请求很分散
SM2：请求很分散
```

如果这些 SM 恰好同时发射，聚合之后仍然可能形成一个大 burst。

正确做法是：

> 将所有会进入同一个共享发送队列或同一个 NVLink egress 的请求放在一起，计算聚合后的 (t_i)、(B) 和 (V_t)。

这也是 microbenchmark 的一个重要结论：burst 指标必须是 **queue-global** 的，而不能只是 producer-local 的。

---

# 这个指标如何指导 interleaving

假设高压力工作会产生 Remote Store 请求，而低压力工作不会，或者产生得很少。

原始排列：

```text
H H H H H H L L L L L L
```

高压力请求的发射时间集中，(V_t) 和 (W_{\mathrm{var}}) 较小。

经过 interleaving：

```text
H L H L H L H L H L H L
```

请求总量 (B) 完全不变，但高压力请求的 (t_i) 被拉开：

[
V_t\uparrow
]

[
W_{\mathrm{var}}\uparrow
]

[
\mathrm{BDS}\uparrow
]

因此，调度设计可以直接以最大化 BDS 为目标：

> 在不增加总工作量、不过度延长算子执行时间的前提下，增加高压力请求发射时间的方差。

这比优化“瞬时请求速率”更适合静态流水线排布，因为它只需要比较不同 instruction/tile ordering 所对应的请求时间分布。

---

# 实际使用时应计算局部而不是全局方差

不应该对整个 kernel 的所有 Remote Store 请求只计算一次全局方差。

例如：

```text
████████ ···················· ████████
```

两组 burst 相隔很远，全局时间方差可能很大，但每一组内部仍然非常集中，局部 queue pressure 仍然很高。

因此，应当以一个自然的 scheduling region 为单位，例如：

* 一个 pipeline iteration；
* 一个 tile group；
* 一个 producer wave；
* 一轮 collective 或 communication phase。

分别计算每个区域的 BDS，然后取最差值：

[
\mathrm{BDS}_{\mathrm{kernel}}
==============================

\min_m \mathrm{BDS}_m
]

含义是：

> 整个算子的 burst 安全程度，由最集中、最容易打爆队列的那个局部 burst 决定。

如果没有自然的 region，也可以使用滑动窗口，在不同时间区间内计算 BDS，并关注最小值。

---

# 最终建议的模型表述

可以把模型概括为：

> 对于一组总请求量固定的 burst-producing work，调度无法减少其总通信量，但可以通过 interleaving 其他工作，将请求发射时间摊开。使用请求发射时间的加权方差衡量这种分散程度，并将其转换为等效分散宽度 (W_{\mathrm{var}})。结合 microbenchmark 测得的共享 drain rate (R) 和有效 queue capacity (C_{\mathrm{eff}})，定义
>
> [
> \mathrm{BDS}
> ============
>
> \frac{C_{\mathrm{eff}}+R\sqrt{12V_t}}{B}
> ]
>
> BDS 越大，说明同样的数据量在时间上越分散，硬件队列在请求到达期间能够排出更多数据，产生 backpressure 或 buffer overflow 的风险越低。图 A、B 对应 (W_{\mathrm{var}}) 与 drain time (B/R) 的关系，图 C 给出 (C_{\mathrm{eff}})，图 D 给出共享的 (R)。

这个指标完整统一了 microbenchmark 中的三个核心现象：

[
\boxed{
\text{burst 总量}
+
\text{queue capacity}
+
\text{时间分散程度}
+
\text{共享 drain rate}
}
]

同时仍然保留了最重要的调度直觉：**总请求量不能改变，就让它们在时间上尽量分散。**
