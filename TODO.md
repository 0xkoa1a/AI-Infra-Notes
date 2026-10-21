
Repo 技术栈：
- 不迁移到 VitePress。原因：丑 + 默认不带动画
- 支持全文模糊搜索
- 文章 Outline
- 如何让 Agent 使用 Vue 前端组件画架构图
- 让 Agent 画交互式小组件
- Vue 组件调用 Plotly 等可视化库
- **Single HTML packaging**：将单个页面打包成一个可分享 HTML 文件，保留样式、交互式组件、图表。
    - 打包时采用与知识库不同的，专门用于对外展示的样式

CUDA 中的流水线建模：不同 tile 调度方式会影响 L2 cache hit rate，进而影响数据搬运的带宽。考虑一下这部分的建模。

TileSight

---

可灵占据大部分资源 可图是流量入口。专业创作者：用 mid journey, nano banana 生成好图片 再调用 sd, kling。另一种用 API 调用。成本会更吸引用户。

可图双日会 每周一三五 
存量 可图v3 小卡 成本打低
存量 其他模型的优化

业务做一两个月 后续会有一些专项
---

可图：4 卡 4090 跑通 t2i i2i 两个模型 做 profiling
- i2i 跟一下，然后是 t2i。
- DiT SR 4090 MFU 能不能到 60%

i2i 4090 显存优化。
成本降低 2-3 倍？

VAE 5kpro xjw。VAE 可以问一下 xzy. 先把 i2i 的通信做得足够好，然后 t2i 的优化就可以自然迁移过去。出一个**可图 on 4090 用了哪些通信优化技术的分析报告**（后续也可以作为新人入门指南）。

512 expert, topk 32, EP 4

两种方法，一种是做 local reduce，通信量少一点；一种是不做 local reduce，但是通信量大。

目前是 GEMM 全算完再开始 local reduce，但是会暴露到关键路径。

算子自动融合，一种是编译器自动生成，一种是让 Agent 来做。或许可以结合一下（Agent 生成 IR）

会给我一个 case 让我试试能不能用 ComFusion 做算子融合。



---



moe local reduction

B 卡上试试

先算一下 GEMM HBM 带宽 + reduction 带宽 假如寄存器够 流水编排 MFU MBU
正常 GEMM 70% MFU， MBU 低。reduction MBU 高，MFU 低。

寄存器能否打满 reduction。

先读到 smem，l2 cache。

流水编排：reduction 的访存不能阻塞 TMA，不能干扰 GEMM。


---

compiler：

KDA: agent + kernel，许愿. skill? 

解决 launch bound, sparse attn 前面有一堆小算子, cuda graph 解决，不同请求 shape 重新编译, 

**MPK 的 task 怎么分最好**？产出指导意见.

mega kernel 黑盒，profile 看不出东西。需要 profile 它的工具。各种 task 也做 timeline。作为调优 MPK 的依据。

先做主线，然后用具体例子验证 练手: sparse attn 准备的部分，语言模型的优化（要求低延迟, batchsize 小 = 1, 20% load weight, 7, 80% launch, 即使 cuda graph 也没有充分并行 有的算子只用 1SM）. MPK 初中也是低延迟。

大算子，已经成熟了，用库写好的，还需要融吗？我们不希望改大算子。

MPK 通信怎么融进去。

---

