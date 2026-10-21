---
title: "参考资料"
order: 13
---

# 参考资料

## LLM 扩展训练的里程碑论文

**[Megatron-LM](https://arxiv.org/abs/1909.08053)**

介绍用于训练大语言模型的张量并行和高效模型并行技术。

**[Megatron-Turing NLG 530B](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)**

介绍结合 DeepSpeed 和 Megatron-LM 框架训练 530B 参数模型的过程。

**[PaLM](https://arxiv.org/abs/2204.02311)**

介绍 Google 的 Pathways Language Model，展示其在数百项语言任务和推理能力上的出色表现。

**[Gemini](https://arxiv.org/abs/2312.11805)**

介绍 Google 能够处理文本、图像、音频和视频输入的多模态模型架构。

**[Llama 3](https://arxiv.org/abs/2407.21783)**

介绍 Llama 3 模型家族。

**[DeepSeek-V3](https://arxiv.org/abs/2412.19437v1)**

DeepSeek 关于 DeepSeek-V3 模型架构和训练的报告。

## 训练框架

**[Nanotron](https://github.com/huggingface/nanotron)**

我们用于训练大语言模型的框架，具备多种并行策略。

**[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**

NVIDIA 用于训练大语言模型的框架，具备多种并行策略。

**[DeepSpeed](https://www.deepspeed.ai/)**

Microsoft 的深度学习优化库，具备 ZeRO 优化阶段和多种并行策略。

**[FairScale](https://github.com/facebookresearch/fairscale/tree/main)**

面向大规模训练的 PyTorch 扩展库，提供多种并行与优化技术。

**[Colossal-AI](https://colossalai.org/)**

集成多种优化技术的大规模模型训练系统。

**[torchtitan](https://github.com/pytorch/torchtitan)**

用于大模型训练的 PyTorch 原生库。

**[GPT-NeoX](https://github.com/EleutherAI/gpt-neox)**

EleutherAI 用于训练大语言模型的框架，曾用于训练 GPT-NeoX-20B。

**[LitGPT](https://github.com/Lightning-AI/litgpt)**

Lightning AI 对 20 多种先进开源 LLM 的实现，侧重可复现性。

**[OpenDiLoCo](https://github.com/PrimeIntellect-ai/OpenDiLoCo)**

使用 DiLoCo 跨计算集群训练语言模型的开源框架。

**[torchgpipe](https://github.com/kakaobrain/torchgpipe)**

PyTorch 中的 GPipe 实现。

**[OSLO](https://github.com/EleutherAI/oslo)**

用于大规模建模的 Open Source for Large-scale Optimization 框架。

## 调试

**[速度剖析](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)**

PyTorch 官方教程，介绍如何用 profiler 分析模型性能和瓶颈。

**[显存剖析](https://pytorch.org/blog/understanding-gpu-memory-1/)**

理解和优化 PyTorch 中 GPU 显存占用的全面指南。

**[通过简单示例讲解显存剖析](https://huggingface.co/blog/train_memory)**

在 PyTorch 中可视化和理解 GPU 显存的指南。

**[TensorBoard profiler 教程](https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)**

使用 TensorBoard 剖析工具分析 PyTorch 模型的指南。

## 分布式技术

**[数据并行](https://siboehm.com/articles/22/data-parallel-training)**

全面解释深度学习中的数据并行训练。

**[ZeRO](https://arxiv.org/abs/1910.02054)**

介绍通过显存优化训练大模型的零冗余优化器。

**[FSDP](https://arxiv.org/abs/2304.11277)**

PyTorch 中的全分片数据并行训练实现。

**[张量与序列并行 + 选择性重计算](https://arxiv.org/abs/2205.05198)**

结合不同并行策略、高效训练大规模模型的高级技术。

**[流水线并行](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/#pipeline_parallelism)**

NVIDIA 关于为大模型训练实现流水线并行的指南。

**[广度优先流水线并行](https://arxiv.org/abs/2211.05953)**

包含对 PP 调度的广泛讨论。

**[Ring all-reduce](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/)**

详细解释分布式训练中使用的 ring all-reduce 算法。

**[Ring Flash Attention](https://github.com/zhuzilin/ring-flash-attention)**

将 Ring Attention 机制与 FlashAttention 结合，以高效训练的实现。

**[Ring Attention 教程](https://coconut-mode.com/posts/ring-attention/)**

解释 Ring Attention 概念和实现的教程。

**[ZeRO 与 3D](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/#understanding-performance-tradeoff-between-zero-and-3d-parallelism)**

DeepSpeed 关于理解 ZeRO 与 3D 并行策略之间权衡的指南。

**[混合精度训练](https://arxiv.org/abs/1710.03740)**

介绍深度学习模型的混合精度训练技术。

**[可视化 6D 网格并行](https://main-horse.github.io/posts/visualizing-6d/)**

解释 6D 并行网格中的集合通信。

## 硬件

**[Fire-Flyer：含 10,000 颗 PCI 芯片的集群](https://www.arxiv.org/abs/2408.14158)**

DeepSeek 关于设计含 10,000 张 PCI GPU 的集群的报告。

**[Meta 的 24,000 张 H100 集群](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/)**

Meta 对其基于 NVIDIA H100 GPU 构建的大规模 AI 基础设施的详细概述。

**[SemiAnalysis 的 100,000 张 H100 集群](https://www.semianalysis.com/p/100000-h100-clusters-power-network)**

分析大规模 H100 GPU 集群及其对 AI 基础设施的影响。

**[Modal GPU 术语表](https://modal.com/gpu-glossary/readme)**

写给人看的 CUDA 文档。

## 其他

**[Stas Bekman 的手册](https://github.com/stas00/ml-engineering)**

涵盖 LLM 训练各个方面的全面手册。

**[BLOOM 训练纪事](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md)**

详细记录 BLOOM 模型的训练过程和挑战。

**[OPT 日志](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf)**

Meta 记录 OPT-175B 模型训练过程的详细日志。

**[更长时间训练小模型的 Harm 定律](https://www.harmdevries.com/post/model-size-vs-compute-overhead/)**

研究模型规模与训练开销之间的关系。

**[Harm 关于长上下文的博客](https://www.harmdevries.com/post/context-length/)**

从数据和训练成本角度研究长上下文训练。

**[GPU Mode](https://www.youtube.com/@GPUMODE/videos)**

GPU 阅读小组和社区。

**[EleutherAI YouTube 频道](https://youtube.com/playlist?list=PLvtrkEledFjqOLuDB_9FWL3dgivYqc6-3&si=fKWPotx8BflLAUkf)**

机器学习可扩展性与性能阅读小组。

**[Google JAX 扩展训练手册](https://jax-ml.github.io/scaling-book/)**

如何扩展你的模型。

**[@fvsmassa & @TimDarcet FSDP](https://github.com/facebookresearch/capi/blob/main/fsdp.py)**

约 500 行代码的独立 FSDP 实现。

**[thonking.ai](https://www.thonking.ai/)**

Horace He 的一些博客文章。

**[Aleksa 的通俗讲解：FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)**

通俗解释 FlashAttention。

**[TunibAI 的 3D 并行教程](https://github.com/tunib-ai/large-scale-lm-tutorials)**

使用 PyTorch 进行大规模语言建模的教程。

[参考文献（BibTeX）](/ultrascale-playbook/bibliography.bib)
