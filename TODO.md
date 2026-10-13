
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