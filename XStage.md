
# 数学建模

建模的目标是清晰、简单、intuitive 地表达 idea
- 目前的叙述区分了两种 Scenarios：
    - **drain dominate scenario**：buffer 中总是非空的，总是处于 drain 的过程中。一个 burst-gap 周期是一个常数 = 通信量 / drain rate，gap 和 epilogue 的长度就是此消彼长的关系。持续有 $T_{epi} > T_{MMA}$。关键路径上始终有 $T_{epi} - T_{MMA}$ 长度的 stall。此时，重排原本就在周期内的计算没有收益，但是可以把未来的计算（L1）提前放进周期里进行，与 epilogue 重叠，获得收益。
    - 非 drain dominate scenario：buffer 中有时空，有时满，并不是总是处于 drain 的过程中。通过重排本来就在周期内的 burst-gap 顺序，平滑化 burst，可以减少或消除反压。
- 这两种 scenario 能不能统一？
    - 纯 L2 连续发射时，只看 L2 周期，确实是 drain dominate 的。
    - 但是看包含所有 L1 和 L2 的整个周期，不是 drain dominate。看整个 MoE kernel，也不是 drain dominate scenario。
    - drain dominate 只会是一个局部的过程（假如整个 kernel 都是 drain dominate 的，那么也没有优化空间了）。
- 对比两种叙事：
    - 只看 L2 周期，原来的调度是一个 drain dominate scenario，收益机制是把原本处于周期外的未来的 L1 计算提前放进周期里进行，从而与关键路径中的 epilogue 重叠。
        - *For fixed communication volume and drain rate, **scheduling cannot shorten this period**. It can, however, determine how much of the period is useful computation and how much is sender-visible issue stall.*
        - 这个链条初见有点绕。
        - L2 周期的 drain dominate 只是一个局部现象。scheduling 是在整个 L1 + L2 的尺度上进行的，所以只看局部 L2 才会有 scheduling cannot shorten this period 的结论。
        - “interleave 的调度不能缩短 L2 周期，however，它可以...”是一个让步的叙述
    - 另一种叙事：看整个 expert wave 周期，buffer 在 L2 连续发射阶段总是非空，但是在 L1 连续发射阶段长期空。收益机制是重排 L1 + L2 周期内的 burst-gap 顺序，平滑化 L2 burst，从而减少反压。
        - 因果链条更加 straightforward：减少反压 -> 减少 epilogue 执行时间 > 减少关键路径上的 stall > 总的执行周期缩短
- 建模部分
    - 重点考察硬件 buffer 本身，对这个硬件 buffer 做建模，分析它的性质。说明硬件 buffer 反压 -> 增加 epilogue 执行时间 -> 造成 tensor core MMA stall 的因果机制，但不对复杂的三级流水线进行完整建模。
        - 硬件 buffer 是论文的核心发现，而 MMA-epilogue 流水线是大家已经熟知的。重点描述论文发现的新 buffer，少着墨于大家熟知的 MMA-epilogue 的部分。
        - 仍然需要说明硬件 buffer 的反压导致 tensor core MMA stall 的机制，只是可以仅定性描述，而不是完整地建一个包罗整个三级流水线的模型（写者难建模、读者也难理解、费篇幅，而且可能吃力不讨好）。
    - 给出一个指标，作为最优化的目标函数。这个指标的构造需要追求：
        - 最好能够**具有因果性**（而不仅仅是相关性）地衡量 buffer 会不会反压
            - 是反压的原因。不是反压的结果/性质（如 $T_{iss}$）。
        - 构成这个指标的变量能够反映我们采用的 kernel 设计方法
            - 我们的 kernel 设计方法是重排 burst-gap 顺序，从而平滑化 burst。它是一个排列问题。
        - Simple, **Intuitive**：读者一看就知道：这个指标高了/低了会显然地导致反压更严重、采用论文的 kernel 设计方法能显然地改善这个指标。
    - [我尝试从方差的角度定义了一个指标](./XStage-modeling.md)，但它有一些问题（更多是描述性指标）。从**瞬时速率**的角度构造指标或许更优。
    - 完整地看，kernel 设计包含三级流水线
        - 但是完整地对三级流水线的所有细节建模很复杂
        - 简化问题


## Setup

首先定义 drain rate dominate scenario：
- 假如一段时间内硬件发送 buffer 收到数据的速率大于 drain rate，使得 buffer 被打满
- buffer 会反压到 epilogue，增加 epilogue 的执行时间 $T_{epi}$ 
- 如果 $T_{epi} > T_{MMA}$，则反压被进一步传导至 tensor core MMA，关键路径上暴露 $T_{epi} - T_{MMA}$ 长度的 stall


推导: minimize T_{epi}
GT：drain rate, T_epi > T_{MMA}
当 buffer 总是满的时候，一个 burst-gap 周期是一个常数 = 通信量 / drain rate，gap 和 epilogue 的长度就是此消彼长的关系
通过增加 setting，简化问题
