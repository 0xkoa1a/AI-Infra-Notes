---
title: "MMDiT：双流 Diffusion Transformer"
order: 3
---

# MMDiT：双流 Diffusion Transformer

MMDiT 为图像和文本分别设置参数，让两条 token 流在每层 Joint Attention 中交换信息。本篇承接 [DiT：Diffusion Transformer](./DiT.md)，重点解释双流结构及其推理影响。架构依据 SD3 论文，具体算子边界以 Diffusers **v0.35.1** 的标准 SD3 双流实现为参照。[SD3 论文](https://arxiv.org/abs/2403.03206)

## 动机与双流架构

文本生成图像需要将文字中的对象、属性和关系对应到图像内容。全局条件向量可以调制整条图像流；将文本保留为 token 序列，则允许模型按文本位置读取细粒度信息。MMDiT 进一步让文本表示在 Block 内持续更新，使当前图像信息也能参与下一层的文本表达。

两种模态各自完成输入投影、归一化、QKV 投影和 MLP。进入 Attention 后，它们的 Q、K、V 沿序列轴组合，每个 Query 可以读取两种模态的 Key 和 Value；输出随后拆回两路。最终去噪目标仍由图像流产生。

图中 $B$ 为前向批量，$N_i,N_t$ 为图像、文本 token 数，$D$ 为隐藏维度，$L$ 为 Block 数；latent 的通道与空间尺寸沿用 DiT 的 $C,H_l,W_l$。文本输入宽度 $D_c,D_p$ 在下节区分，$C_{\mathrm{out}}$ 表示最终预测通道数。

<MMDiTDiagram view="overview" />

| 方面 | 原始类别条件 DiT | SD3 标准 MMDiT |
| --- | --- | --- |
| 持续更新的 token 流 | 图像流 | 图像流、文本流 |
| Block 的全局条件 | 时间步与类别表示 | 时间步与 pooled text 表示 |
| Attention 的序列范围 | 图像位置 | 图像位置与文本位置 |
| 投影与 MLP 参数 | 图像主干参数 | 两路各自的参数 |

> 文本条件 Transformer 也可以使用图像查询文本的 Cross-Attention。它提供了另一种逐 token 条件接口；本篇以双向、逐层更新的文本流解释 MMDiT 的特点。[架构讨论](https://arxiv.org/html/2403.03206v1)

## 双流输入与条件表示

沿用 DiT 的记号：$B$ 是前向批量，$N_i$ 是图像 token 数，$D$ 是隐藏维度。图像路径在 Patchify 和位置编码之后提供 $X_i^0\in\mathbb{R}^{B\times N_i\times D}$；上标 $0$ 表示进入第一个 Block 之前的状态。这里直接以 token 为入口，空间变换见 [DiT 的输入表示](./DiT.md#从-latent-到-token)。

文本条件有两种表示。Token-level embedding 保留序列位置，用于 Joint Attention；pooled embedding 汇总整段提示词，用于条件调制。令文本 token 数为 $N_t$，文本编码宽度为 $D_c$，pooled 表示宽度为 $D_p$：

| 输入或映射 | 形状变化 | 去向 |
| --- | --- | --- |
| 文本 token 表示 $C_{\mathrm{txt}}$ | $[B,N_t,D_c]$ | 进入文本输入投影 |
| 文本输入投影 | $[B,N_t,D_c]\rightarrow[B,N_t,D]$ | 得到初始文本流 $X_{\mathrm{txt}}^0$ |
| Pooled text $c_p$ | $[B,D_p]\rightarrow[B,D]$ | 进入全局条件路径 |
| 时间步 $t$ | $[B]\rightarrow[B,D]$ | 与映射后的 pooled text 相加 |
| 全局条件 $c(t)$ | $[B,D]$ | 调制所有 Block 及最终输出层 |

这里 $D_c$、$D_p$ 可以与 $D$ 不同；两路 token 进入标准 Block 时统一为隐藏宽度 $D$。多文本编码器的输出如何拼接、补齐或投影，由具体 Pipeline 决定；表中的 $C_{\mathrm{txt}}$ 与 $c_p$ 指进入 Transformer 的已经准备好的条件。[输入投影实现](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/transformers/transformer_sd3.py)

在所述实现中，时间步映射 $E_\tau$ 与 pooled text 映射 $E_p$ 分别产生 $D$ 维结果，再相加形成条件：

$$
c(t)=E_\tau(t)+E_p(c_p)\in\mathbb{R}^{B\times D}.
$$

这个向量通过调制路径影响两路 token，不增加 Joint Attention 的序列长度。[条件编码实现](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/embeddings.py)

## 一个 MMDiT Block 如何更新双流

设当前层输入为 $X_i\in\mathbb{R}^{B\times N_i\times D}$ 和 $X_{\mathrm{txt}}\in\mathbb{R}^{B\times N_t\times D}$。图中展示一个完整双流 Block：从各自的调制与投影开始，在 Attention 中汇合，输出后继续各自更新。

<MMDiTDiagram view="block" />

### 双流 AdaLN

两路沿用 [DiT 的 AdaLN 调制与门控残差](./DiT.md#adaln-与-adaln-zero)机制，共享输入条件 $c(t)$，各层、各模态使用独立条件映射。标准完整 Block 为每路生成六组 $[B,D]$ 系数，用于各自的 Attention 和 MLP；它们分别沿 $N_i$ 和 $N_t$ 轴广播。用于归一化的统计量来自各自当前的 hidden state。

因此，同一个时间步的两路调制系数可以在接触图像 hidden state 之前准备好，但归一化后的 token 仍依赖当前输入。后续层的文本 token 已经吸收图像信息，即使提示词和全局条件固定，文本子层的输入也会变化。[双流调制实现](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/attention.py)

### Joint Attention

**两路先用各自的权重产生 Q、K、V，再共同计算注意力。** 投影输入是完成归一化与条件调制后的特征。令 $n_h$ 为注意力头数，$d_h=D/n_h$ 为每头维度；图像和文本分别投影、拆分多头后，形状为：

| 张量 | 图像流 | 文本流 |
| --- | --- | --- |
| Q、K、V 各自投影后 | $[B,N_i,D]$ | $[B,N_t,D]$ |
| Q、K、V 各自拆分多头后 | $[B,n_h,N_i,d_h]$ | $[B,n_h,N_t,d_h]$ |

为了对应图中的左右排列，以下使用文本在前、图像在后的序列约定。对 Q、K、V 分别沿 token 轴拼接，得到 $N=N_t+N_i$ 个位置；三个联合张量的形状均为 $[B,n_h,N,d_h]$：

$$
Q=[Q_{\mathrm{txt}};Q_i],\qquad K=[K_{\mathrm{txt}};K_i],\qquad V=[V_{\mathrm{txt}};V_i].
$$

每个 Query 对全部文本和图像 Key 计算分数，在联合 Key 轴上归一化，再按权重汇总 Value：

$$
O=\operatorname{softmax}_{\text{Key}}\!\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}}\right)V.
$$

注意力权重的逻辑形状为 $[B,n_h,N,N]$，输出 $O$ 为 $[B,n_h,N,d_h]$。这里执行非因果全注意力，softmax 的一行同时覆盖两种模态。

**四种注意力关系。** 下图将联合矩阵按 Query、Key 所属模态分区。每一行描述“谁读取谁”；某个 Query 读到另一模态的 Value 后，更新的是这个 Query 所在的流。

<MMDiTDiagram view="attention" />

| Query → Key | 每个样本、每个头的区域形状 | 更新的内容 |
| --- | --- | --- |
| 文本 → 文本 | $[N_t,N_t]$ | 文本位置之间的信息 |
| 文本 → 图像 | $[N_t,N_i]$ | 文本流读取当前图像信息 |
| 图像 → 文本 | $[N_i,N_t]$ | 图像流读取文本条件 |
| 图像 → 图像 | $[N_i,N_i]$ | 图像位置之间的信息 |

四个分区属于同一次联合归一化。将它们各自做 softmax，再简单相加，不会得到相同计算。图像查询文本的 Cross-Attention 只对应其中一种关系，归一化范围也有所不同。

> 拼接顺序只要与拆分顺序一致，就能表达同一联合注意力。Diffusers v0.35.1 的标准处理器使用图像在前、文本在后的顺序；本文为了图示采用相反顺序。矩阵是逻辑关系图，融合内核不必把完整权重矩阵保存到显存。[Joint Attention 处理器](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/attention_processor.py)

### 双流输出更新

Attention 输出合并多头后为 $[B,N,D]$。按约定的 token 边界拆分，得到文本部分 $[B,N_t,D]$ 和图像部分 $[B,N_i,D]$；每路再经过自己的输出投影、Gate 和残差相加。

两路随后分别对更新后的 hidden state 做 LayerNorm 和条件调制，再进入各自的 MLP。若中间宽度为 $rD$，图像 MLP 的形状为 $[B,N_i,D]\rightarrow[B,N_i,rD]\rightarrow[B,N_i,D]$，文本 MLP 将其中的 $N_i$ 换成 $N_t$；$r$ 为扩展比。各自完成第二次门控残差更新后，再进入下一层。

两条流可以使用同样的算子结构，但它们拥有独立参数。联合 Attention 负责交换信息，模态专属投影与 MLP 则决定各自怎样表示和加工这些信息。

## 最后一层与图像输出

### Context-pre-only

Diffusers v0.35.1 将最后一个 Joint Transformer Block 标记为 `context_pre_only`。该层文本路径仍执行条件归一化和 QKV 投影，参与联合 Attention；随后省略文本输出投影、文本残差更新及文本 MLP，Block 向外返回的文本状态为空。文本入口采用只需 shift、scale 的条件归一化，因此不能按完整文本分支的六组调制系数估算。[最后一层设置](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/transformers/transformer_sd3.py)、[Block 实现](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/models/attention.py)

这一处理服务于最终输出：去噪网络只需要图像预测。所述通用 Attention 处理器仍构造文本 Query，并计算联合输出；“省略文本输出更新”不意味着它自动省掉所有文本 Query 的 Attention 计算。若进一步只计算图像 Query，需要另外实现相应优化。

图像流经过最终条件归一化、输出投影和 [Unpatchify](./DiT.md#从-token-回到-latent)，产生 $[B,C_{\mathrm{out}},H_l,W_l]$ 的预测。$C_{\mathrm{out}}$ 是输出通道数，$H_l,W_l$ 是 latent 空间尺寸；SD3 的 flow 目标在与输入相同的 latent 空间中预测速度场，Scheduler 据此推进当前状态。

> 本文图示对应标准双流路径。SD3.5 的额外图像 Attention、QK Norm，以及 ControlNet、LoRA、IP-Adapter 等扩展会改变具体算子或参数依赖，应按对应实现重新核对形状和缓存条件。

## 推理计算与缓存

### 联合序列与双路算子

联合 Attention 的计算规模由 $N_i+N_t$ 决定。忽略常数，每层核心注意力矩阵乘法为 $O(B(N_i+N_t)^2D)$。展开序列长度的平方，就能对应到前文四个区域：

$$
(N_i+N_t)^2=N_i^2+2N_iN_t+N_t^2.
$$

其中 $N_i^2$ 对应图像内部，$N_t^2$ 对应文本内部，两个 $N_iN_t$ 对应两个方向的跨模态读取。在 $N_i\gg N_t$ 的高分辨率设置下，图像内部项通常占主导；文本长度仍会改变联合序列的尺寸与内核分块。

两路 QKV 与输出投影合计为 $O(B(N_i+N_t)D^2)$，两路 MLP 合计为 $O(Br(N_i+N_t)D^2)$。由于模态权重不同，不能把两路当成一次共享权重的线性映射；实现可以探索分组执行。文本 token 较少时，文本侧矩阵乘法可能更容易受到小算子利用率和调度开销影响，是否成为瓶颈需要实际测量。

### Condition Cache：双流中的复用边界

沿用 [DiT 的 Condition Cache 分析](./DiT.md#condition-cache-静态条件与调制预计算)，先检查结果依赖哪些输入。本节讨论固定提示词、固定权重、确定性推理下的精确复用机会；按时间表预计算属于依赖分析推导出的方案。

<MMDiTDiagram view="cache" />

| 结果 | 跨 step 的行为 | 可行的复用方式 |
| --- | --- | --- |
| 文本编码 $C_{\mathrm{txt}}$、pooled 表示 $c_p$ | 固定提示词下不变 | 编码后复用；文本编码器完成任务后可 offload |
| 初始文本投影 $X_{\mathrm{txt}}^0$、映射后的 $E_p(c_p)$ | 固定输入与权重下不变 | 在各次 Transformer 前向之间复用 |
| 条件 $c(t)$ 与各层、各模态调制系数 | 随时间步变化，但不依赖图像状态 | 按已知时间步、层、模态预计算 |
| 首层文本的调制后特征及 QKV | 依赖初始文本与 $c(t)$，尚未读取图像 | 可按时间步预计算；不同时间步通常对应不同值 |
| 首层 Attention 更新后的文本状态 | 已读取当前图像信息 | 随当前图像变化，继续动态计算 |
| 后续层文本状态及其 QKV | 依赖前面各层的图文交互 | 不能仅凭提示词相同就跨 step 复用 |

首层值得单独看：它的输入文本 $X_{\mathrm{txt}}^0$ 对同一提示词固定，条件 $c(t)$ 也可以在时间步已知时准备，所以首层文本 QKV 可以提前求值。一旦执行 Joint Attention，文本 Query 会读取当前图像的 Key、Value，得到的文本输出就依赖当前图像。后续层的文本输入继承这一依赖，缓存边界由此发生变化。

标准完整 Block 的两路调制参数合计为 $[B,12D]$，逻辑上可组织为 $[B,2,6,D]$，其中长度为二的轴表示模态。所有求值时间、所有完整 Block 都缓存时，空间随 $SLBD$ 增长；$S$ 为模型求值次数，$L$ 为 Block 数。最后一层的文本调制和最终图像输出层应按其实际参数数量单独计入。

缓存首层文本 QKV 则额外保存每个时间步三份 $[B,n_h,N_t,d_h]$ 数据，其存储量随 $SBN_tD$ 增长。它只覆盖首层的一部分计算，准备和读取成本也必须纳入端到端评估。提示词、负面条件、模型权重、LoRA 或时间表变化时，要重新确认相应缓存的有效性。

图像 hidden state 在每次去噪求值中整体变化，深层文本 hidden state 也会随图文交互变化。因此，MMDiT 的复用应围绕这些具体依赖设计。将变化中的 hidden state 跨步复用会引入近似误差；编译图复用则复用执行结构，都应与本节的条件结果缓存分别评估。
