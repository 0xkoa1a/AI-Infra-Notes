# DiT：使用 Transformer 的扩散模型

## 1. DiT 是什么

DiT，全称 **Diffusion Transformer**，是一类使用 Transformer 作为扩散模型去噪网络的生成模型。

传统扩散模型通常采用 U-Net：

$$
\epsilon_\theta(x_t,t,c)=\operatorname{UNet}(x_t,t,c)
$$

DiT 则将 U-Net 替换为 Transformer：

$$
\epsilon_\theta(x_t,t,c)=\operatorname{Transformer}(x_t,t,c)
$$

其中：

* $x_t$：第 $t$ 个扩散时间步的带噪样本；
* $t$：扩散时间步；
* $c$：类别、文本等条件；
* $\epsilon_\theta$：模型预测的噪声；
* $\theta$：模型参数。

DiT 最核心的思想是：

> 将扩散模型中的图像特征切分为 token，然后使用 Vision Transformer 风格的网络完成去噪。

因此，DiT 可以理解为：

$$
\text{DiT}=\text{Latent Diffusion}+\text{Vision Transformer}
$$

DiT 并没有修改扩散模型的概率建模方式，主要改变的是负责预测噪声或速度的神经网络架构。

---

## 2. 扩散模型回顾

### 2.1 前向扩散过程

扩散模型首先逐渐向真实样本 $x_0$ 中加入高斯噪声：

$$
q(x_t\mid x_{t-1})
==================

\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_t I
\right)
$$

经过变量合并，可以直接从 $x_0$ 得到任意时间步的 $x_t$：

$$
x_t
===

\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

其中：

$$
\epsilon\sim\mathcal N(0,I)
$$

$$
\alpha_t=1-\beta_t
$$

$$
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s
$$

随着 $t$ 增大，$x_t$ 中的原始信号逐渐减少，噪声逐渐增加。最终 $x_T$ 接近标准高斯噪声。

---

### 2.2 反向去噪过程

生成时，模型从随机噪声 $x_T$ 出发，逐步预测并去除噪声：

$$
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0
$$

常见训练目标是让模型预测加入的噪声：

$$
\mathcal L_{\text{simple}}
==========================

\mathbb E_{x_0,t,\epsilon}
\left[
\left|
\epsilon-\epsilon_\theta(x_t,t,c)
\right|_2^2
\right]
$$

传统扩散模型中的 $\epsilon_\theta$ 通常是 U-Net；DiT 中的 $\epsilon_\theta$ 则是 Transformer。

---

## 3. 为什么使用 Transformer 替代 U-Net

U-Net 天然适合图像任务，因为它具有：

* 卷积带来的局部归纳偏置；
* 多尺度特征结构；
* 下采样和上采样路径；
* skip connection；
* 较成熟的扩散模型实践。

但 U-Net 也存在一些问题。

第一，U-Net 结构复杂。模型通常由多个不同分辨率的卷积模块、残差模块和注意力模块组成，不容易像标准 Transformer 一样统一扩展。

第二，U-Net 的扩展规律相对不清晰。增加网络宽度、深度或注意力层数时，整体计算结构会发生较复杂的变化。

第三，Transformer 在语言模型和视觉模型中已经证明具有良好的可扩展性。模型规模和训练计算量增大时，性能通常能够持续提升。

DiT 的核心问题因此是：

> 扩散模型是否也可以使用结构统一、容易扩展的 Transformer？

实验结果表明，答案是肯定的。随着 DiT 的参数量和计算量增加，生成质量能够稳定提升。

---

## 4. DiT 的整体结构

原始 DiT 通常工作在 VAE 的 latent space，而不是直接处理 RGB 图像。

完整流程可以写成：

$$
x
\xrightarrow{\text{VAE Encoder}}
z_0
\xrightarrow{\text{加噪}}
z_t
\xrightarrow{\text{DiT}}
\hat\epsilon
$$

生成时：

$$
z_T
\xrightarrow{\text{多步 DiT 去噪}}
z_0
\xrightarrow{\text{VAE Decoder}}
\hat x
$$

其中：

* $x$：原始图像；
* $z_0$：图像对应的干净 latent；
* $z_t$：加噪后的 latent；
* $\hat\epsilon$：DiT 预测的噪声；
* $\hat x$：最终生成图像。

DiT 的主要结构包括：

1. VAE latent 输入；
2. patchify；
3. token embedding；
4. positional embedding；
5. 多层 DiT Block；
6. Final Layer；
7. unpatchify；
8. 输出噪声或其他扩散预测目标。

---

## 5. Latent Diffusion

假设原始图像尺寸为：

$$
x\in\mathbb R^{B\times 3\times H\times W}
$$

经过 VAE Encoder 后，得到 latent：

$$
z\in\mathbb R^{B\times C\times H'\times W'}
$$

通常：

$$
H'=\frac{H}{f},
\qquad
W'=\frac{W}{f}
$$

其中 $f$ 是 VAE 的空间压缩倍数。

例如，原始图像大小为 $256\times256$，VAE 压缩倍数为 $8$，则 latent 空间大小为：

$$
32\times32
$$

如果 latent channel 数为 $4$，则输入 DiT 的张量形状为：

$$
z_t\in\mathbb R^{B\times4\times32\times32}
$$

在 latent space 中训练扩散模型能够显著减少计算量。相比直接处理 $256\times256$ 的 RGB 图像，DiT 实际处理的空间分辨率只有 $32\times32$。

需要注意：

> DiT 本身主要是去噪网络；VAE 负责在像素空间和 latent space 之间进行转换。

---

## 6. Patchify：将 latent 转换为 token

Transformer 接收的是 token 序列，而 latent 是二维特征图，因此需要执行 patchify。

假设 latent 的形状为：

$$
z_t\in\mathbb R^{B\times C\times H'\times W'}
$$

patch 大小为 $p\times p$。

每个 patch 包含：

$$
p^2C
$$

个数值。

patch 的数量为：

$$
N=\frac{H'}{p}\cdot\frac{W'}{p}
$$

经过线性投影后，每个 patch 被映射到 $D$ 维 embedding：

$$
X\in\mathbb R^{B\times N\times D}
$$

例如：

* latent 分辨率：$32\times32$；
* patch size：$p=2$；
* latent channel：$C=4$。

则 token 数为：

$$
N=\frac{32}{2}\times\frac{32}{2}=256
$$

每个原始 patch 的维度为：

$$
2\times2\times4=16
$$

再通过线性层映射到 Transformer hidden dimension $D$。

在代码中，patchify 通常可以使用 stride 等于 kernel size 的卷积完成：

```python
self.x_embedder = nn.Conv2d(
    in_channels=C,
    out_channels=D,
    kernel_size=p,
    stride=p,
)
```

输入：

```text
[B, C, H', W']
```

输出：

```text
[B, D, H'/p, W'/p]
```

再进行 flatten 和 transpose：

```text
[B, D, H'/p, W'/p]
→ [B, D, N]
→ [B, N, D]
```

---

## 7. 位置编码

Self-Attention 本身对 token 顺序不敏感，因此需要加入位置编码：

$$
X_0=X_{\text{patch}}+X_{\text{pos}}
$$

其中：

$$
X_{\text{patch}}\in\mathbb R^{B\times N\times D}
$$

$$
X_{\text{pos}}\in\mathbb R^{1\times N\times D}
$$

原始 DiT 使用二维 sinusoidal positional embedding。

位置编码描述每个 patch 在 latent feature map 中的位置，使模型能够区分左上角、中心和右下角的 token。

---

## 8. DiT Block

标准 Transformer Block 通常写成：

$$
X'=X+\operatorname{MSA}(\operatorname{LN}(X))
$$

$$
X_{\text{out}}
==============

X'
+
\operatorname{MLP}(\operatorname{LN}(X'))
$$

其中：

* MSA：Multi-Head Self-Attention；
* LN：Layer Normalization；
* MLP：前馈网络。

DiT 还必须接收扩散时间步 $t$ 和条件信息 $c$。因此，DiT Block 的关键问题不是 Attention 本身，而是：

> 如何将时间步和类别等条件注入 Transformer？

原始 DiT 对比了多种条件注入方法，其中效果最好的是 **adaLN-Zero**。

---

## 9. 时间步编码

扩散模型在不同时间步执行的任务不同。

当 $t$ 较大时，输入几乎是纯噪声，模型需要恢复图像的全局结构；当 $t$ 较小时，输入已经较清晰，模型主要恢复纹理和细节。

因此，模型必须知道当前时间步 $t$。

首先，将标量时间步转换为 sinusoidal embedding：

$$
e_t=\operatorname{SinusoidalEmbedding}(t)
$$

再通过 MLP：

$$
c_t=\operatorname{MLP}(e_t)
$$

得到：

$$
c_t\in\mathbb R^{B\times D}
$$

时间步 embedding 的思想与 Transformer 的位置编码类似：使用不同频率的正弦和余弦函数表示时间。

---

## 10. 类别条件编码

在 class-conditional DiT 中，每个类别 $y$ 对应一个可学习 embedding：

$$
c_y=\operatorname{Embedding}(y)
$$

其中：

$$
c_y\in\mathbb R^{B\times D}
$$

时间条件和类别条件相加：

$$
c=c_t+c_y
$$

然后使用 $c$ 调制每一个 DiT Block。

在文本到图像模型中，类别 embedding 可以被替换为文本编码器输出，例如 T5 或 CLIP text encoder 产生的文本特征。

不过原始 DiT 论文主要研究类别条件的 ImageNet 生成。

---

## 11. 条件注入方法

原始 DiT 对比了四种 Transformer 条件注入方法。

### 11.1 In-context Conditioning

将条件 embedding 作为额外 token 拼接到输入序列中：

$$
X'=[c_t;c_y;X]
$$

Self-Attention 可以让图像 token 与条件 token 交互。

优点是形式简单，接近语言模型处理特殊 token 的方式。

缺点是条件信息需要通过 Attention 间接传播到所有 token，而且会增加 token 数量。

---

### 11.2 Cross-Attention

图像 token 通过 Cross-Attention 读取条件特征：

$$
Q=XW_Q
$$

$$
K=cW_K,\qquad V=cW_V
$$

$$
\operatorname{CrossAttn}(X,c)
=============================

\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V
$$

Cross-Attention 特别适合文本条件，因为文本本身是一个 token 序列。

但 Cross-Attention 会增加额外参数和计算量。

---

### 11.3 Adaptive LayerNorm

LayerNorm 对 hidden state 进行标准化：

$$
\operatorname{LN}(X)
====================

\frac{X-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

Adaptive LayerNorm 根据条件 $c$ 生成缩放和偏置参数：

$$
\operatorname{adaLN}(X,c)
=========================

\gamma(c)\odot\operatorname{LN}(X)+\beta(c)
$$

其中：

$$
\gamma(c),\beta(c)\in\mathbb R^{B\times D}
$$

将它们 broadcast 到所有 token：

$$
\gamma,\beta:
[B,D]\rightarrow[B,1,D]
$$

因此，同一个样本中的所有 spatial token 使用相同的条件调制参数。

这种方法可以理解为：

> 条件信息不作为额外 token，而是直接改变每一层 Transformer 的特征分布。

---

## 12. adaLN-Zero

DiT 最重要的结构设计是 **Adaptive LayerNorm-Zero**，简称 adaLN-Zero。

对于每一个 DiT Block，条件 embedding $c$ 经过一个 MLP，生成六组参数：

$$
(\beta_1,\gamma_1,\alpha_1,\beta_2,\gamma_2,\alpha_2)
=====================================================

\operatorname{MLP}(c)
$$

其中：

* $\beta_1,\gamma_1$：调制 Attention 前的 LayerNorm；
* $\alpha_1$：控制 Attention 分支的输出；
* $\beta_2,\gamma_2$：调制 MLP 前的 LayerNorm；
* $\alpha_2$：控制 MLP 分支的输出。

DiT Block 可以写成：

$$
X'
==

X+
\alpha_1
\odot
\operatorname{MSA}
\left(
\gamma_1\odot\operatorname{LN}(X)+\beta_1
\right)
$$

$$
X_{\text{out}}
==============

X'
+
\alpha_2
\odot
\operatorname{MLP}
\left(
\gamma_2\odot\operatorname{LN}(X')+\beta_2
\right)
$$

这里的 $\alpha_1$ 和 $\alpha_2$ 也称为 gate。

---

### 12.1 Zero 的含义

adaLN-Zero 中的 Zero 指条件调制 MLP 的最后一层采用零初始化。

初始化时：

$$
\alpha_1=0,\qquad\alpha_2=0
$$

因此初始状态下：

$$
X'=X
$$

$$
X_{\text{out}}=X'
$$

每个 DiT Block 在初始化时都近似恒等映射。

这和残差网络中的 zero-initialized residual branch 思想类似，可以使深层网络的训练更加稳定。

随着训练进行，模型逐渐学习非零的 gate，让 Attention 和 MLP 分支参与计算。

---

### 12.2 adaLN-Zero 的直觉

adaLN-Zero 同时解决了三个问题。

第一，注入条件。时间步和类别信息通过 $\gamma$、$\beta$ 调制每一层。

第二，控制残差分支强度。$\alpha$ 决定 Attention 和 MLP 输出对主干特征的影响程度。

第三，稳定深层网络训练。由于初始时残差分支接近零，整个网络一开始近似恒等映射。

因此，adaLN-Zero 不只是普通的条件 LayerNorm，而是：

$$
\text{Condition Modulation}
+
\text{Residual Gating}
+
\text{Zero Initialization}
$$

---

## 13. Self-Attention 在 DiT 中做什么

对于输入：

$$
X\in\mathbb R^{B\times N\times D}
$$

分别计算：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

每个注意力头的维度为：

$$
d_h=\frac{D}{h}
$$

Attention 输出为：

$$
\operatorname{Attention}(Q,K,V)
===============================

\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)V
$$

Self-Attention 让每一个 latent patch 都能直接关注其他 patch。

例如，一个位于图像左上角的 token 可以直接读取右下角 token 的信息。因此，DiT 很适合建模：

* 物体整体轮廓；
* 图像长距离依赖；
* 多个物体之间的空间关系；
* 全局语义一致性。

相比之下，卷积主要通过堆叠多层逐渐扩大感受野。

---

## 14. MLP 模块

DiT Block 中的 MLP 通常为：

$$
\operatorname{MLP}(X)
=====================

W_2\phi(W_1X+b_1)+b_2
$$

其中 $\phi$ 常用 GELU。

隐藏层维度通常是 Transformer hidden size 的若干倍：

$$
D_{\text{mlp}}=rD
$$

例如 $r=4$。

Attention 主要负责 token 之间的信息交互，MLP 则对每个 token 的 channel 特征执行非线性变换。

---

## 15. Final Layer

经过多个 DiT Block 后，得到：

$$
X_L\in\mathbb R^{B\times N\times D}
$$

Final Layer 通常也包含条件化 LayerNorm：

$$
X_{\text{final}}
================

\gamma_f(c)\odot\operatorname{LN}(X_L)+\beta_f(c)
$$

然后通过线性层将每个 token 投影回 patch：

$$
Y=X_{\text{final}}W_o+b_o
$$

如果每个 patch 的空间大小为 $p\times p$，输出 channel 数为 $C_{\text{out}}$，则：

$$
Y\in
\mathbb R^{B\times N\times(p^2C_{\text{out}})}
$$

最后执行 unpatchify，恢复空间结构：

$$
Y
\rightarrow
\hat\epsilon
\in
\mathbb R^{B\times C_{\text{out}}\times H'\times W'}
$$

---

## 16. 为什么输出 channel 可能是输入的两倍

扩散模型有时不仅预测噪声，还预测反向过程的方差。

如果 latent channel 数为 $C$，模型可能输出：

$$
C_{\text{out}}=2C
$$

输出可以分成两部分：

$$
[\hat\epsilon,\hat\Sigma]
$$

其中：

* $\hat\epsilon$：预测噪声；
* $\hat\Sigma$：用于表示反向扩散方差的参数。

例如输入 latent channel 为 $4$，模型输出 channel 可能为 $8$。

如果扩散过程使用固定方差，则模型只需要输出 $C$ 个 channel。

---

## 17. DiT 的数据形状示例

假设：

* 图像大小：$256\times256$；
* VAE 下采样倍数：$8$；
* latent channel：$4$；
* patch size：$2$；
* hidden size：$1152$；
* batch size：$B$。

原始图像：

$$
[B,3,256,256]
$$

VAE latent：

$$
[B,4,32,32]
$$

patch embedding：

$$
[B,1152,16,16]
$$

flatten 后：

$$
[B,256,1152]
$$

因为：

$$
N=16\times16=256
$$

经过所有 DiT Block 后，形状保持：

$$
[B,256,1152]
$$

假设模型输出 $8$ 个 channel，每个 token 输出：

$$
p^2C_{\text{out}}
=================

# 2^2\times8

32
$$

所以线性层输出：

$$
[B,256,32]
$$

unpatchify 后：

$$
[B,8,32,32]
$$

再将其切分为噪声预测和方差预测：

$$
\hat\epsilon\in\mathbb R^{B\times4\times32\times32}
$$

$$
\hat\Sigma\in\mathbb R^{B\times4\times32\times32}
$$

---

## 18. 训练流程

一次 DiT 训练迭代可以概括为以下过程。

首先，从数据集中采样图像及其类别：

$$
(x_0,y)\sim p_{\text{data}}
$$

使用 VAE Encoder 得到 latent：

$$
z_0=\operatorname{Encoder}(x_0)
$$

随机采样扩散时间步：

$$
t\sim\operatorname{Uniform}{1,\ldots,T}
$$

随机采样高斯噪声：

$$
\epsilon\sim\mathcal N(0,I)
$$

构造带噪 latent：

$$
z_t
===

\sqrt{\bar\alpha_t}z_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

使用 DiT 预测噪声：

$$
\hat\epsilon
============

\epsilon_\theta(z_t,t,y)
$$

计算损失：

$$
\mathcal L
==========

\left|
\epsilon-\hat\epsilon
\right|_2^2
$$

最后反向传播更新参数。

伪代码如下：

```python
image, label = next(data_loader)

with torch.no_grad():
    z0 = vae.encode(image)

t = torch.randint(0, num_timesteps, (batch_size,))
noise = torch.randn_like(z0)

zt = scheduler.add_noise(z0, noise, t)

noise_pred = dit(zt, t, label)

loss = F.mse_loss(noise_pred, noise)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

VAE 通常是预训练的，并在训练 DiT 时冻结。

---

## 19. Classifier-Free Guidance

为了增强条件生成质量，DiT 通常使用 **Classifier-Free Guidance，CFG**。

### 19.1 训练阶段

训练过程中，以一定概率丢弃类别条件：

$$
c=
\begin{cases}
c_y,&\text{保留条件}\
c_{\varnothing},&\text{丢弃条件}
\end{cases}
$$

这样，同一个模型既能执行条件预测，也能执行无条件预测。

---

### 19.2 推理阶段

分别计算无条件和有条件预测：

$$
\epsilon_{\text{uncond}}
========================

\epsilon_\theta(x_t,t,\varnothing)
$$

$$
\epsilon_{\text{cond}}
======================

\epsilon_\theta(x_t,t,c)
$$

然后组合为：

$$
\epsilon_{\text{cfg}}
=====================

\epsilon_{\text{uncond}}
+
s
\left(
\epsilon_{\text{cond}}
----------------------

\epsilon_{\text{uncond}}
\right)
$$

其中 $s$ 是 guidance scale。

当 $s>1$ 时，会增强条件方向：

$$
\epsilon_{\text{cond}}-\epsilon_{\text{uncond}}
$$

较大的 $s$ 通常会提高图像与条件的一致性，但也可能：

* 降低样本多样性；
* 导致颜色过饱和；
* 产生边缘伪影；
* 放大模型预测误差。

CFG 通常需要有条件和无条件两次前向计算，因此会显著增加推理开销。实际实现中常将两组输入沿 batch 维拼接，一次送入模型。

---

## 20. 采样流程

生成时从高斯噪声开始：

$$
z_T\sim\mathcal N(0,I)
$$

在每个时间步执行：

$$
\hat\epsilon_t
==============

\epsilon_\theta(z_t,t,c)
$$

采样器根据预测值更新 latent：

$$
z_{t-1}
=======

\operatorname{SchedulerStep}
(z_t,\hat\epsilon_t,t)
$$

重复执行：

$$
z_T\rightarrow z_{T-1}\rightarrow\cdots\rightarrow z_0
$$

最后由 VAE Decoder 解码：

$$
\hat x=\operatorname{Decoder}(z_0)
$$

伪代码如下：

```python
z = torch.randn(batch_size, latent_channels, latent_h, latent_w)

for t in scheduler.timesteps:
    noise_uncond = dit(z, t, null_label)
    noise_cond = dit(z, t, label)

    noise_pred = noise_uncond + guidance_scale * (
        noise_cond - noise_uncond
    )

    z = scheduler.step(noise_pred, t, z)

image = vae.decode(z)
```

DiT 可以和多种采样器组合，例如：

* DDPM；
* DDIM；
* DPM-Solver；
* Euler；
* Heun；
* flow matching 对应的 ODE sampler。

DiT 是去噪网络架构，而 sampler 决定如何利用模型输出完成数值积分或反向扩散。

---

## 21. DiT 的模型规模

原始 DiT 使用类似 ViT 的命名方法：

* DiT-S：Small；
* DiT-B：Base；
* DiT-L：Large；
* DiT-XL：Extra Large。

模型规模主要由三个因素决定：

* Transformer 深度 $L$；
* hidden dimension $D$；
* Attention head 数 $h$。

命名中的 `/2`、`/4`、`/8` 表示 patch size。

例如：

```text
DiT-XL/2
```

表示：

* XL 规模的 Transformer；
* latent patch size 为 $2\times2$。

---

## 22. Patch Size 对性能和计算量的影响

token 数量为：

$$
N=\frac{H'W'}{p^2}
$$

Self-Attention 的核心复杂度约为：

$$
O(N^2D)
$$

将 $N$ 代入：

$$
O
\left(
\frac{H'^2W'^2}{p^4}D
\right)
$$

因此，patch size 对计算量影响很大。

假设 latent 为 $32\times32$：

当 $p=2$ 时：

$$
N=16\times16=256
$$

当 $p=4$ 时：

$$
N=8\times8=64
$$

当 $p=8$ 时：

$$
N=4\times4=16
$$

从 $p=4$ 改为 $p=2$，token 数增加 $4$ 倍，Attention matrix 大小增加约 $16$ 倍。

小 patch 的优点是保留更多空间细节，模型具有更高的 token 分辨率；缺点是计算量和显存开销显著增加。

因此，DiT-XL/2 通常比 DiT-XL/4 计算更昂贵，但生成质量也更好。

---

## 23. DiT 的计算量

Transformer Block 的主要计算包括：

### 23.1 QKV 投影

$$
O(ND^2)
$$

### 23.2 Attention score

$$
O(N^2D)
$$

### 23.3 Attention 与 V 相乘

$$
O(N^2D)
$$

### 23.4 输出投影

$$
O(ND^2)
$$

### 23.5 MLP

如果 MLP expansion ratio 为 $r$：

$$
O(rND^2)
$$

因此，总体复杂度可以粗略写为：

$$
O(ND^2+N^2D)
$$

在 latent 分辨率不高、hidden dimension 很大的 DiT 中，线性层和 MLP 的计算往往占据很大比例；当图像分辨率或 token 数增加时，$N^2$ Attention 成本会迅速上升。

---

## 24. Gflops 与生成质量

DiT 论文中的重要结论不是单纯证明 Transformer 能生成图像，而是展示了一种扩散模型的 scaling behavior：

> 当模型的训练计算量增大时，生成质量稳定提高。

模型计算量可以通过以下方式增加：

* 增加网络深度；
* 增大 hidden dimension；
* 增加 Attention head 数；
* 减小 patch size；
* 增加训练步数；
* 增加数据量。

论文观察到，模型的 Gflops 与 FID 等生成指标之间存在较稳定的相关关系。

这表明扩散模型也能像语言模型一样，从统一架构和规模扩展中持续获益。

---

## 25. DiT 与 U-Net 的对比

### 25.1 网络结构

U-Net：

* 多尺度层级结构；
* 卷积、残差块和 Attention 混合；
* 下采样和上采样；
* 大量 skip connection。

DiT：

* 单一 token 分辨率；
* 重复堆叠相同的 Transformer Block；
* 通常没有 U-Net 式 encoder-decoder；
* 结构更加规则。

---

### 25.2 归纳偏置

U-Net 具有较强的图像归纳偏置：

* 局部连接；
* 平移等变性；
* 多尺度特征融合。

DiT 的图像归纳偏置较弱，主要依赖：

* patch embedding；
* 位置编码；
* 数据和计算规模；
* Self-Attention 学习空间关系。

归纳偏置较弱意味着 DiT 可能需要更多数据和训练计算，但也意味着其架构更通用、扩展上限更高。

---

### 25.3 全局建模能力

U-Net 中的卷积主要进行局部计算，需要通过多层传播建立远距离关系。

DiT 中 Self-Attention 可以直接连接任意两个 token：

$$
\text{任意 token 间最短信息路径长度}\approx1
$$

因此 DiT 更容易建模全局构图和物体之间的长距离依赖。

---

### 25.4 可扩展性

DiT 的每层结构几乎完全相同，模型可以通过增加 $L$ 和 $D$ 直接扩展。

这使其更加适合：

* 大规模分布式训练；
* Tensor Parallelism；
* Sequence Parallelism；
* FlashAttention；
* fused MLP；
* 标准 Transformer 训练基础设施。

---

## 26. DiT 与 ViT 的区别

DiT 在结构上类似 Vision Transformer，但任务不同。

ViT 的输入是干净图像 patch：

$$
x\rightarrow\text{patch tokens}
$$

输出通常是类别概率：

$$
p(y\mid x)
$$

DiT 的输入是带噪 latent patch：

$$
z_t\rightarrow\text{latent patch tokens}
$$

输出是与输入空间大小相同的噪声、速度或数据预测：

$$
\hat\epsilon,\quad \hat v,\quad \hat x_0
$$

此外，DiT 还需要处理扩散时间步，并在每个 Transformer Block 中注入时间条件。

因此：

$$
\text{DiT}
\neq
\text{直接用于分类的 ViT}
$$

它更接近一个条件化的 dense prediction Transformer。

---

## 27. 预测目标：$\epsilon$、$x_0$ 和 $v$

扩散模型不一定必须预测噪声 $\epsilon$。

### 27.1 Noise Prediction

$$
\hat\epsilon
============

\epsilon_\theta(x_t,t,c)
$$

损失为：

$$
\mathcal L_\epsilon
===================

\left|
\epsilon-\hat\epsilon
\right|^2
$$

---

### 27.2 Data Prediction

直接预测干净样本：

$$
\hat x_0=x_\theta(x_t,t,c)
$$

损失为：

$$
\mathcal L_{x_0}
================

\left|
x_0-\hat x_0
\right|^2
$$

---

### 27.3 Velocity Prediction

定义速度：

$$
v
=

## \sqrt{\bar\alpha_t}\epsilon

\sqrt{1-\bar\alpha_t}x_0
$$

模型预测：

$$
\hat v=v_\theta(x_t,t,c)
$$

不同参数化之间可以相互转换。现代 DiT 类模型可能使用 $\epsilon$-prediction、$v$-prediction、flow matching velocity 等不同目标。

因此：

> DiT 描述的是网络架构，不限定扩散模型必须采用哪一种预测参数化。

---

## 28. 从 DiT 到文本生成图像模型

原始 DiT 使用类别标签作为条件。文本到图像 DiT 则需要处理文本 token。

文本编码器输出：

$$
C_{\text{text}}
\in
\mathbb R^{B\times L_{\text{text}}\times D_c}
$$

图像 latent token 为：

$$
X
\in
\mathbb R^{B\times N_{\text{image}}\times D}
$$

文本条件可以通过以下方式注入。

第一种是 Cross-Attention：

$$
Q=XW_Q
$$

$$
K=C_{\text{text}}W_K
$$

$$
V=C_{\text{text}}W_V
$$

第二种是 Joint Attention，将文本 token 和图像 token 拼接：

$$
S=[C_{\text{text}};X]
$$

然后统一执行 Self-Attention：

$$
S'=\operatorname{Attention}(S)
$$

第三种是将 pooled text embedding 用于 AdaLN 调制，同时使用 Cross-Attention 注入细粒度文本 token。

现代多模态生成模型通常在原始 DiT 上加入更复杂的文本—图像交互机制。

---

## 29. DiT 与 MMDiT

MMDiT 通常指 **Multimodal Diffusion Transformer**。

普通文本条件 DiT 中，文本通常是图像分支的外部条件；MMDiT 则为文本和图像分别保留表示和投影参数，并在 Joint Attention 中进行交互。

设图像 token 为 $X$，文本 token 为 $C$。

分别计算：

$$
Q_x=XW_{Q,x},\quad K_x=XW_{K,x},\quad V_x=XW_{V,x}
$$

$$
Q_c=CW_{Q,c},\quad K_c=CW_{K,c},\quad V_c=CW_{V,c}
$$

再拼接 K 和 V：

$$
K=[K_c;K_x]
$$

$$
V=[V_c;V_x]
$$

文本和图像 token 都可以读取联合的多模态信息。

相比单纯 Cross-Attention，MMDiT 更强调文本和图像两种模态之间的对称交互。

---

## 30. DiT 与 Flow Matching

DiT 也可以用作 Flow Matching 模型的 backbone。

Flow Matching 构造从数据分布到噪声分布的连续路径：

$$
x_t=\alpha_t x_0+\sigma_t\epsilon
$$

模型学习速度场：

$$
v_\theta(x_t,t,c)
\approx
\frac{dx_t}{dt}
$$

生成时求解常微分方程：

$$
\frac{dx_t}{dt}=v_\theta(x_t,t,c)
$$

因此现代生成模型中常见的“Transformer + Flow Matching”仍然可以视为 DiT 路线的延伸。

需要区分：

* DiT：神经网络架构；
* DDPM：概率扩散建模方式；
* Flow Matching：速度场训练目标；
* ODE Solver：生成时的数值积分方法。

这些概念处于不同层次，可以组合使用。

---

## 31. DiT 的优势

### 31.1 架构统一

网络主要由重复的 Transformer Block 构成，代码和系统结构都较简单。

### 31.2 扩展规律较好

增大参数量和计算量通常能够稳定改善生成质量。

### 31.3 适配 Transformer 基础设施

可以直接利用：

* FlashAttention；
* fused LayerNorm；
* fused MLP；
* Tensor Parallelism；
* Sequence Parallelism；
* activation checkpointing；
* Transformer Engine；
* FP16、BF16 和 FP8。

### 31.4 全局感受野

Self-Attention 天然支持所有 token 之间的信息交互。

### 31.5 易于统一多模态建模

文本、图像、视频和音频都可以表示成 token，并在 Transformer 中统一处理。

---

## 32. DiT 的局限

### 32.1 Attention 的二次复杂度

Self-Attention 复杂度随 token 数平方增长：

$$
O(N^2D)
$$

当图像或视频分辨率增大时，计算和显存开销迅速上升。

### 32.2 缺少卷积的局部归纳偏置

DiT 通常需要更多数据和计算，才能学习卷积网络天然具备的局部模式。

### 32.3 扩散采样本身较慢

即使单次 DiT 前向传播已经高度优化，生成过程仍然可能需要多次网络调用。

总计算量约为：

$$
\text{Sampling Cost}
\approx
\text{Number of Steps}
\times
\text{Cost per DiT Forward}
$$

启用 CFG 时还可能接近再乘以 $2$。

### 32.4 高分辨率和视频场景开销巨大

视频 token 数近似为：

$$
N
=

\frac{T}{p_t}
\cdot
\frac{H}{p_h}
\cdot
\frac{W}{p_w}
$$

Attention matrix 大小为：

$$
N\times N
$$

因此视频 DiT 通常需要：

* 时空分解 Attention；
* Window Attention；
* Sequence Parallelism；
* Context Parallelism；
* 稀疏 Attention；
* token 压缩；
* 多阶段或多尺度生成。

---

## 33. 简化版 DiT Block 代码

下面给出一个突出 adaLN-Zero 逻辑的简化实现：

```python
import torch
import torch.nn as nn


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    # x:     [B, N, D]
    # shift: [B, D]
    # scale: [B, D]
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
        )
        self.norm2 = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
        )

        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # 输出：
        # shift_msa, scale_msa, gate_msa,
        # shift_mlp, scale_mlp, gate_mlp
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

        # adaLN-Zero
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        params = self.ada_ln(condition)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = params.chunk(6, dim=-1)

        attn_input = modulate(
            self.norm1(x),
            shift_msa,
            scale_msa,
        )

        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )

        x = x + gate_msa[:, None, :] * attn_output

        mlp_input = modulate(
            self.norm2(x),
            shift_mlp,
            scale_mlp,
        )

        x = x + gate_mlp[:, None, :] * self.mlp(mlp_input)

        return x
```

实际实现通常会使用更高效的 Attention kernel，并对初始化、mixed precision、位置编码和输出层进行更完整的处理。

---

## 34. 简化版完整 DiT 数据流

可以将 DiT 的一次前向传播总结为：

```text
带噪 latent z_t
    │
    ▼
Patch Embedding
    │
    ▼
[B, N, D] token sequence
    │
    ├── 加入 positional embedding
    │
    ├── time embedding
    │
    └── class/text condition
    │
    ▼
DiT Block × L
    │
    ▼
Conditional Final LayerNorm
    │
    ▼
Linear Projection
    │
    ▼
[B, N, p² × C_out]
    │
    ▼
Unpatchify
    │
    ▼
噪声、速度或 x₀ 预测
```

---

## 35. 从系统角度理解 DiT

对于 AI 推理基础设施而言，DiT 与大语言模型有很多相似之处：

* 主要计算是大规模 GEMM；
* Attention 和 MLP 占据主要计算量；
* 网络由大量规则的 Transformer Block 构成；
* 适合使用 Tensor Parallelism；
* 适合 FlashAttention；
* 适合算子融合和低精度计算。

但 DiT 和自回归 LLM 也有显著区别。

### 35.1 没有 KV Cache

自回归 LLM 每一步只生成一个新 token，历史 K/V 可以缓存。

DiT 在每个扩散时间步中，所有图像 token 都会更新。因此下一步的 $Q$、$K$ 和 $V$ 都会变化，通常无法像 LLM decode 那样复用 KV Cache。

### 35.2 每一步都处理完整 token 序列

DiT 每个采样时间步都对全部图像或视频 token 执行完整前向传播。

因此其推理模式更接近：

$$
\text{Repeated Full-Sequence Prefill}
$$

而不是 LLM 的单 token decode。

### 35.3 Batch 和 CFG

CFG 常将条件和无条件样本沿 batch 维拼接：

$$
[B,N,D]
\rightarrow
[2B,N,D]
$$

这提高了单次 GEMM 的规模和 GPU 利用率，但也几乎成倍增加计算量。

### 35.4 通信模式

使用 Tensor Parallelism 时，每个 DiT Block 通常包含与 LLM 类似的 collective communication：

* All-Reduce；
* Reduce-Scatter；
* All-Gather。

视频 DiT 的 sequence length 很大，因此也经常采用 Sequence Parallelism 或 Context Parallelism，在 token 维度切分计算。

---

## 36. 一句话总结

DiT 的本质可以概括为：

> 将 VAE latent 切分为图像 token，使用带有时间和条件调制的 Transformer 反复预测扩散过程中的噪声或速度。

其关键结构是：

$$
\text{Patchify}
+
\text{Transformer Blocks}
+
\text{adaLN-Zero}
+
\text{Unpatchify}
$$

其关键意义是：

> DiT 证明了扩散模型可以摆脱复杂的 U-Net 架构，转向结构统一、易于规模化、能够复用大语言模型训练和推理基础设施的 Transformer 路线。
