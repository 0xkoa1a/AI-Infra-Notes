
# 数学建模

在数学建模的章节里就引入流水线。但是只考虑 MMA-epilogue 流水线，不考虑 epilogue-硬件发送 buffer 流水线（为了简化讨论）。

建模的目标是清晰、简单、intuitive 地表达 idea
- 不对整个三级流水线的所有细节进行建模
- 方法是 burst-gap 排列问题，目标是最小化 epilogue 的执行时间 $T_{epi}$。
- 目前的叙述区分了两种 Scenarios：
    - **drain dominate scenario**：buffer 中总是非空的，总是处于 drain 的过程中。一个 burst-gap 周期是一个常数 = 通信量 / drain rate，gap 和 epilogue 的长度就是此消彼长的关系。持续有 $T_{epi} > T_{MMA}$。关键路径上始终有 $T_{epi} - T_{MMA}$ 长度的 stall。此时，重排原本就在周期内的计算没有收益，但是可以把未来的计算（L1）提前放进周期里进行，与 epilogue 重叠，获得免费收益。
    - 非 drain dominate scenario：buffer 中有时空，有时满，并不是总是处于 drain 的过程中。通过重排本来就在周期内的 burst-gap 顺序，平滑化 burst，可以减少或消除反压。
- 这两种 scenario 能不能统一？
    - 纯 L2 连续发射时，只看 L2 周期，确实是 drain dominate 的。
    - 但是看包含所有 L1 和 L2 的整个周期，不是 drain dominate。看整个 MoE kernel，也不是 drain dominate scenario。
    - drain dominate 只会是一个局部的过程（假如整个 kernel 都是 drain dominate 的，那么也没有优化空间了）。
- 对比两种叙事：
    - 只看 L2 周期，原来的调度是一个 drain dominate scenario，收益机制是把原本处于周期外的未来的 L1 计算提前放进周期里进行，从而与关键路径中的 epilogue 重叠。
        - *For fixed communication volume and drain rate, **scheduling cannot shorten this period**. It can, however, determine how much of the period is useful computation and how much is sender-visible issue stall.*
        - 这个链条初见有点绕。
        - L2 周期的 drain dominate 只是一个局部现象。scheduling 是在整个 L1 + L2 的 expert wave 尺度上进行的，所以只看局部 L2 才会有 scheduling cannot shorten this period 的结论。
        - “interleave 的调度不能缩短 L2 周期，however，它可以...”听起来像是在承认 scheduling 的弱点，但这个弱点其实并不存在。这种叙述弱化了调度方法的贡献。
    - 看整个 expert wave 周期，buffer 在 L2 连续发射阶段总是非空，但是在 L1 连续发射阶段长期空。收益机制是重排 L1 + L2 周期内的 burst-gap 顺序，平滑化 L2 burst，从而减少反压。
        - 因果链条更加 straightforward，减少反压 -> 减少 epilogue 执行时间 > 减少关键路径上的 stall > 总的执行周期缩短


## Setup

首先定义 drain rate dominate scenario：
- 假如一段时间内硬件发送 buffer 收到数据的速率大于 drain rate，使得 buffer 被打满
- buffer 会反压到 epilogue，增加 epilogue 的执行时间 $T_{epi}$ 
- 如果 $T_{epi} > T_{MMA}$，则反压被进一步传导至 tensor core MMA，关键路径上暴露 $T_{epi} - T_{MMA}$ 长度的 stall


推导: minimize T_{epi}
GT：drain rate, T_epi > T_{MMA}
当 buffer 总是满的时候，一个 burst-gap 周期是一个常数 = 通信量 / drain rate，gap 和 epilogue 的长度就是此消彼长的关系
通过增加 setting，简化问题
