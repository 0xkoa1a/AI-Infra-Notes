---
title: "Mixture-of-Experts"
order: 1
---
# 动机

Mixture-of-Experts（MoE）的核心目标是：**在保持每个 token 的计算量较小的情况下，继续扩大模型参数量。**

普通 Dense Transformer 的 FFN 对所有 token 使用同一套参数：

$$
y=f(x)
$$

无论输入是什么，每个 token 都经过同一个 $f$。

如果希望增加模型参数量，就要加深 FFN 的深度。这会导致每 token 计算量也同比增长。

MoE 解耦了每 token 计算量和模型总参数量。它准备多个不同的 FFN（称为 Expert）：

$$
f_1,f_2,\ldots,f_E
$$

对每个 token，它只选择其中少数几个 Expert 执行。也就是说，它引入了“条件计算”。

MoE 同时追求实现三个目标：
- **Sparsity**：每个 token 只激活少量 Expert。
- **Specialization**：不同 Expert 学到不同的输入模式。
- **Balance**：不能让所有 token 最终都集中到少数 Expert。

这三个目标之间存在天然冲突。Router 越自由，越容易产生 specialization，但也越容易发生 collapse；balance 约束越强，Expert 使用越均匀，但 Router 的 specialization 自由度也越小。因此，现代 MoE 的主要算法问题通常是**如何设计和训练 Router**。

# 基本结构

对于 token $x$，一层 MoE 可以抽象为：
- **Router** 根据 $x$ 判断应该使用哪些 Expert。
- **Expert** 对被分配给自己的 token 执行 FFN 计算。

假设共有 $E$ 个 Expert：

$$
f_1(x),f_2(x),\ldots,f_E(x)
$$

Dense mixture 要求所有 Expert 都参与计算，最终结果按照 Router 权重加权混合：

$$
y=\sum_{i=1}^{E}p_i(x)f_i(x)
$$

其中 $p_i(x)$ 表示 Router 给 Expert $i$ 的权重。

现代 LLM 中的 MoE 通常采用 sparse mixture，只保留 Router 得分最高的 $k$ 个 Expert：

$$
S(x)=\operatorname{TopK}(p(x),k)
$$

最终只计算：

$$
y=\sum_{i\in S(x)}g_i(x)f_i(x)
$$

这里：

* $S(x)$ 是被选择的 Expert 集合。
* $g_i(x)$ 是被选择 Expert 的组合权重。
* $k$ 通常远小于 Expert 总数 $E$。

# Router

## 基本结构

Router 的输入通常就是当前 token：

$$
x\in\mathbb{R}^{d}
$$

Router 可以看成一个拥有 $E$ 个类别的分类器：给定 token representation，它为每个 Expert 计算一个匹配分数。

$$
z=W_rx
$$

其中：

* $W_r\in\mathbb{R}^{E\times d}$。
* $E$ 是 Expert 数量。
* $z_i$ 是 token 对 Expert $i$ 的 routing logit。

一种经典做法是进一步使用 softmax：

$$
p_i(x)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

这里 $p_i(x)$ 表示 Expert $i$ 对当前 token 的 router 权重。

注意，对于 Router 来说，并不存在人工标注的“正确 Expert”。Router 和 Expert 都通过最终的语言模型目标共同训练。

## Top-k Routing

现代 sparse MoE 最常见的是 Top-$k$ routing。

假设 Router 对一个 token 给出的概率为：

* Expert 0：0.03。
* Expert 1：0.10。
* **Expert 2：0.61**。
* Expert 3：0.04。
* **Expert 4：0.22**。

如果使用 Top-2，则选择 Expert 2 和 Expert 4。

之后可以只对被选中的 Expert 权重重新归一化：

$$
g_i=
\frac{p_i}
{\sum_{j\in S(x)}p_j}
$$

$k$ 控制 **稀疏程度和 mixture flexibility 之间的 trade-off**。

---

Top-$k$ selection 本身是一个离散操作。

例如某个 Expert 的 score 从 0.1999 变化到 0.2001，可能只是一个很小的连续变化，但如果它刚好越过第 $k$ 名的阈值，它会从完全不参与计算变成参与计算。

因此，MoE routing 并不是一个完全光滑的优化问题。

实践中通常采用较直接的方法：

* Forward 时执行 hard Top-$k$ selection。
* 被选中的 Expert 正常参与 forward 和 backward。
* 被选中的 gating weight 可以正常接收梯度。
* 未被选中的 Expert 不执行当前 token 对应的 Expert computation。

因此，Router 的参数变化不仅改变 gating weight，还可能直接改变某个 token 下一步训练哪个 Expert。

## 专家分工

专家分工来自 Router 和 Expert 之间的正反馈。

假设训练初期，一类 token 因为随机差异稍微更倾向 Expert 1：

* Expert 1 因而更多看到这一类 token。
* Expert 1 在这一类 token 上获得更多训练。
* Expert 1 对这一类 token 的处理逐渐变得更好。
* Router 进一步倾向把这类 token 分给 Expert 1。

因此，MoE 不需要人为规定：

* Expert 0 负责数学。
* Expert 1 负责代码。
* Expert 2 负责中文。

专家分工由训练过程自己形成，而且可能并没有清晰明确的语义边界。

## Load Balance

Specialization 的正反馈也会带来一个直接的问题：**routing collapse**。

假设某个 Expert 在训练初期因为随机因素稍微领先，则会：
* Router 更喜欢它。
* 它收到更多 token。
* 它获得更多梯度。
* 它进一步学得更快。
* Router 因此更加喜欢它。

最终可能出现少数 Expert 垄断绝大多数 token，而其他 Expert 几乎不被使用。

这意味着：
* 大量 Expert 参数没有得到有效训练。
* 模型实际可利用的容量远低于名义容量。
* Expert specialization 退化成少数 Expert 垄断。

因此，MoE 通常会加入 load-balancing regularization。

假设一个 batch 中共有 $T$ 个 token。对于 Expert $i$，可以统计它实际接收到的 token 比例：

$$
f_i=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf{1}[i\in S(x_t)]
$$

同时可以统计 Router 对 Expert $i$ 的平均概率：

$$
P_i=
\frac{1}{T}
\sum_{t=1}^{T}p_i(x_t)
$$

一种经典 auxiliary loss 为：

$$
L_{\mathrm{balance}}
=
E\sum_{i=1}^{E}f_iP_i
$$

其目标是避免 $f_i$ 和 $P_i$ 同时集中在少数 Expert 上。理想状态下，各 Expert 的整体 usage 大致接近：

$$
f_i\approx\frac{1}{E}
$$

最终训练目标可以写成：

$$
L=L_{\mathrm{LM}}+\lambda L_{\mathrm{balance}}
$$

当然，**load balance 并不意味着每个局部输入都应该平均使用所有 Expert**。每个 token 可以有自己最喜欢的专家，但是整体上所有专家之间的负载应该大致均衡。

## Capacity 与 Token Dropping

早期 MoE 经常显式限制每个 Expert 能够接收的 token 数量。

假设：
* 一个 batch 有 $T$ 个 token。
* 一共有 $E$ 个 Expert。
* 使用 Top-1 routing。

如果 routing 完全均匀，则每个 Expert 平均应该收到的 token 数量：

$$
\frac{T}{E}
$$

进一步定义 capacity：每个 Expert 最多允许接收平均负载的 $\alpha$ 倍 token。$\alpha$ 被称为 capacity factor。

$$
C=\alpha\frac{T}{E}
$$

如果 Router 给某个 Expert 分配了超过 capacity 的 token，则需要决定哪些 token 被保留。早期方法可能根据 routing score：
* 保留 score 较高的 token。
* Drop 超出 capacity 的 token。
* 被 drop 的 token 可能只走 residual path，或者采用其他 fallback 规则。

从算法角度，capacity 提供了一种 hard constraint：即使 Router 强烈偏向某个 Expert，它也不能无限地把所有 token 分给同一个 Expert。

现代方法越来越倾向于通过更好的 balance 机制避免 token dropping，因为 dropping 会改变正常的模型计算路径。

## Token-choice Routing 与 Expert-choice Routing

经典 MoE 通常采用 token-choice routing：每个 token 在所有 Expert 中选择最适合自己的 Expert。Expert-choice routing 则反过来：每个 Expert 在 token 序列中选择最适合自己的 token。

需要注意的是，这两种 routing 机制的本质区别在于 Top-K 沿哪个维度做选择，Router 打分网络可以完全不动。

假设一个 batch 中有 $T$ 个 token、$E$ 个 Expert。Router 对所有 token 打分：

$$
S=XW_r,\qquad S\in\mathbb{R}^{T\times E}
$$

其中 $S_{t,e}$ 表示 token $t$ 与 Expert $e$ 的匹配分数。

---

Token-choice Routing 对于每个 token，沿 Expert 维度选择 score 最高的 $k$ 个 Expert：

$$
A_t=\operatorname{TopK}*{e}(S*{t,e},k)
$$

这种方式直接保证：
* 每个 token 都会选择固定数量的 $k$ 个 Expert。
* Routing decision 围绕当前 token 的需求展开。
* 不同 token 可以独立选择不同 Expert。

但它没有约束每个 Expert 最终会收到多少 token。因此 Expert load 可能高度不均衡。

---

Expert-choice 使用同样的 token–expert score，但把选择方向反过来。对于每个 Expert，沿 token 维度选择 score 最高的 $K$ 个 token：

$$
A_e=\operatorname{TopK}*{t}(S*{t,e},K)
$$

因此，每个 Expert 的 load 可以天然固定为：

$$
|A_e|=K
$$

但它同时放弃了 token-choice 中“每个 token 固定激活 $k$ 个 Expert”的保证：

* 一个 token 可能同时被多个 Expert 选中。
* 一个 token 也可能没有被任何 Expert 选中。
* 不同 token 实际激活的 Expert 数量可以不同。

因此，Expert-choice 的主要问题从 **Expert load balance** 转变成了 **token coverage**。


## Softmax 与 Sigmoid Routing

Softmax Router 强制所有 Expert 的概率归一化：

$$
\sum_i p_i(x)=1
$$

因此不同 Expert 的概率存在直接竞争关系。一个 Expert 的概率增加，其他 Expert 的概率总和就必须下降。

另一种选择是对每个 Expert 独立使用 sigmoid：

$$
p_i(x)=\sigma(z_i)
$$

此时各 Expert 的 score 不需要和为 1。

之后仍然可以根据这些 score 做 Top-$k$ selection。

也就是说，softmax 和 sigmoid 隐含了不同的 Expert competition mechanism。

## Noisy Routing

训练早期，Expert 尚未形成稳定 specialization。如果 Router 太早变得 deterministic，可能发生：

* 一个随机的早期偏好让某些 token 进入某个 Expert。
* Expert 因为接收了这些 token 而开始沿这个方向 specialization。
* Router 又因为 Expert 已经 specialization 而继续选择它。
* 最终 routing pattern 被早期随机性过早锁定。

一种经典方法是在 routing logits 中加入噪声：

$$
z_i=W_i x+\epsilon_i
$$

自然语言来说，即使某个 Expert 当前 score 最高，随机扰动也会让 token 偶尔尝试其他 Expert。

这种机制提供了一定程度的 exploration，使训练早期不同 Expert 都有机会看到更多类型的数据。

## Router z-loss

Router 训练还可能出现另一个问题：routing logits 的绝对值不断增大。

例如原本：

$$
[2,1,0]
$$

可能逐渐变成：

$$
[20,10,0]
$$

甚至：

$$
[200,100,0]
$$

虽然 Top-1 的选择结果完全没有变化，但 softmax 会越来越饱和，并可能带来训练稳定性问题。

因此可以加入 router z-loss，例如：

$$
L_z=
\left(
\log\sum_i e^{z_i}
\right)^2
$$

z-loss 不直接要求 Expert 负载均匀，而是限制 Router logits 不要无意义地不断增大。

# Expert

## 基本结构

最简单的 Expert 可以写成：

$$
f_i(x)=W_{2,i}\sigma(W_{1,i}x)
$$

其中 $W_{1,i}$ 将 $x$ 升维，经过非线性后，再由 $W_{2,i}$ 降维投影回原来的维度。

## SwiGLU Expert

现代 LLM 的 MoE Expert 经常采用 SwiGLU FFN。

对于输入 $x\in\mathbb{R}^{d}$，一个 SwiGLU Expert 可以写成：

$$
f(x)=W_2\left[\operatorname{SiLU}(W_1x)\odot W_3x\right]
$$

## Shared Expert

如果所有 Expert 都是 routed expert，那么一个问题是：不同 Expert 可能都需要重复学习一些所有 token 普遍需要的能力。

例如：
* 基本语言表示转换。
* 普遍存在的语法模式。
* 所有领域共享的底层特征。

因此，一些 MoE 架构会额外加入 **Shared Expert**。此时 Expert 被分成两类：
* **Shared Expert** 对所有 token 都执行。
* **Routed Expert** 只处理 Router 选择的 token。

输出可以抽象为：

$$
y=f_{\mathrm{shared}}(x)
+
\sum_{i\in S(x)}g_i(x)f_i(x)
$$

每个 token 都先获得 shared expert 提供的公共能力，再利用 routed experts 提供输入相关的 specialization。

## Fine-grained Expert

另一个设计自由度是 **Expert granularity**。

假设有固定的总激活 FFN 参数量，可以选择：
* 使用较少但较大的 Expert，并让每个 token 激活少数几个。
* 使用更多但更小的 Expert，并让每个 token 激活更多个。

在每个 token 激活的总参数量接近的情况下，细粒度专家可以有更强的表达能力：
- 粗粒度：从 16 个大 Expert 中选 1 个，只能形成 16 种 Expert choice。
- 细粒度：从 64 个小 Expert 中选 4 个，则可以形成 $C_{64}^4$ 种不同的 Expert choice。
