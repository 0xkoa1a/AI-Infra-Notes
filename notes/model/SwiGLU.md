---
title: "SwiGLU"
---

# SwiGLU

SwiGLU 是一种常用于 Transformer FFN 的激活结构。它可以理解为：**让 FFN 的一条支路产生内容，另一条支路产生一个数据依赖的门，然后逐元素相乘。**

相比传统的 `Linear → GELU/ReLU → Linear`，SwiGLU 多了一条并行的线性投影，用 **SiLU/Swish** 生成门控信号。

## 从普通 FFN 开始

最基本的 Transformer FFN 可以写成：

$$
\operatorname{FFN}(x)=W_2\sigma(W_1x)
$$

其中：

* $x\in\mathbb{R}^{d_{\text{model}}}$：输入 hidden state。
* $W_1$：先把维度从 $d_{\text{model}}$ 升到 $d_{\text{ff}}$。
* $\sigma$：ReLU、GELU 等非线性函数。
* $W_2$：再从 $d_{\text{ff}}$ 降回 $d_{\text{model}}$。

这里的非线性完全由一个固定函数 $\sigma$ 提供，每个 hidden dimension 的非线性都是一样的。

## GLU：把激活函数变成一个“门”

GLU，全称 **Gated Linear Unit**，核心思想是把输入分成两条支路。

先分别计算：

$$
a=W_ax,\qquad b=W_bx
$$

其中：
* $a$ 是真正准备传递的内容。
* $b$ 用来决定哪些维度应该通过。

然后将 $b$ 经过一个非线性函数 $\sigma$，再和 $a$ 逐元素相乘：

$$
h=a\odot \sigma(b)
$$

每个 hidden dimension 都有自己的 gate：

$$
h_j=a_j\cdot \sigma(b_j)
$$

不再是：“算出一个值，然后经过固定激活函数”，而是“算出一个值，同时根据当前输入算出一个门，再决定这个值应该保留多少”。这使得非线性更灵活。

## Swish / SiLU

SwiGLU 中的 `Swi` 来自 **Swish**。

现代 Transformer 里通常使用的其实就是 SiLU（Sigmoid Linear Unit）：

$$
\operatorname{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}}
$$

这里的 $\sigma(x)$ 是 sigmoid：

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

SiLU 和 ReLU 的区别：
* ReLU 对 $x<0$ 直接变成 0。SiLU 对负数仍然保留一部分信息。
* 整个函数是平滑的。

它有点像一个**由输入自身控制强弱的平滑 ReLU**。

::: echarts SiLU 与 ReLU 的函数曲线

```js
const xs = Array.from({ length: 241 }, (_, i) => -6 + i * 0.05);

const silu = xs.map((x) => [
  Number(x.toFixed(2)),
  x / (1 + Math.exp(-x)),
]);

const relu = xs.map((x) => [
  Number(x.toFixed(2)),
  Math.max(0, x),
]);

option = {
  tooltip: {
    trigger: "axis",
    valueFormatter: (value) => Number(value).toFixed(3),
  },
  legend: {
    data: ["SiLU", "ReLU"],
  },
  grid: {
    left: 56,
    right: 24,
    top: 48,
    bottom: 48,
  },
  xAxis: {
    type: "value",
    name: "x",
    min: -6,
    max: 6,
    axisLine: { onZero: true },
  },
  yAxis: {
    type: "value",
    name: "y",
    min: -1,
    max: 6,
    axisLine: { onZero: true },
  },
  series: [
    {
      name: "SiLU",
      type: "line",
      data: silu,
      showSymbol: false,
      smooth: true,
      lineStyle: { width: 3 },
    },
    {
      name: "ReLU",
      type: "line",
      data: relu,
      showSymbol: false,
      lineStyle: { width: 2, type: "dashed" },
    },
  ],
};
```

:::

## SwiGLU

把 GLU 中的门控激活换成 SiLU，就得到 SwiGLU。

$$
\boxed{\operatorname{SwiGLU}(x)=W_{\text{down}}\left[\operatorname{SiLU}(W_{\text{gate}}x)\odot(W_{\text{up}}x)\right]}
$$

SwiGLU 的数据流：

$$
x
\rightarrow
\begin{cases}
W_{\text{gate}}x\rightarrow \operatorname{SiLU}\\
W_{\text{up}}x
\end{cases}
\rightarrow \odot
\rightarrow W_{\text{down}}
$$

> 注记：作为对比，以下是普通 FFN 的数据流：
> $$
> x
> \rightarrow W_{\text{up}}x
> \rightarrow \operatorname{GELU}
> \rightarrow W_{\text{down}}
> $$

GLU 比普通 FFN 多了一个线性投影矩阵，因为它必须同时决定产生什么内容（$W_{\text{up}}$）和这些内容应该通过多少（$W_{\text{gate}}$），然后再降维回输入维度（$W_{\text{down}}$）。


### Linear-1 与 Linear-2

有些工作会把 SwiGLU 中的两级线性计算称为 Linear-1 和 Linear-2
* **Linear-1**：同时计算 $W_1x$ 和 $W_3x$。
* **Linear-2**：计算最后的 $W_2h$。

因为 $W_1$ 和 $W_3$：

* 输入都是同一个 $x$。
* 没有相互依赖。
* 都属于从 hidden dimension 到 intermediate dimension 的 projection。

所以通常会把它们合并为一个 Linear-1：

$$
W_{13}=
\begin{bmatrix}
W_1\
W_3
\end{bmatrix}
$$

一次性计算出来：

$$
\begin{bmatrix}
g\
u
\end{bmatrix}
=W_{13}x
$$

## SwiGLU 的参数量和中间维度

传统 Transformer FFN 的中间维度常取：

$$
d_{\text{ff}}=4d_{\text{model}}
$$

它包含两个形状为 $d_{\text{model}}\times d_{\text{ff}}$ 和 $d_{\text{ff}}\times d_{\text{model}}$ 的矩阵，总参数量约为：

$$
d\times4d+4d\times d=8d^2
$$

而 SwiGLU 有三个矩阵。如果仍然使用 $4d$，参数量会直接增加 50%。

$$
d\times4d+d\times4d+4d\times d=12d^2
$$

因此通常会把 SwiGLU 的中间维度缩小，使总参数量与普通 FFN 接近。

设 SwiGLU 的中间维度为 $m$，三个矩阵总参数量约为 $3dm$。为了和普通 FFN 的 $8d^2$ 接近，需要：

$$
m\approx\frac{8}{3}d
$$

所以经常有：

$$
d_{\text{ff}}\approx2.67d_{\text{model}}
$$
