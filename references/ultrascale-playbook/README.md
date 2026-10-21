# The Ultra-Scale Playbook 本地存档

来源：[Hugging Face 原文](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism)；下载日期：2026-10-21。

保存整篇文档，原链接定位的 Tensor Parallelism 也包含在内。英文原文保留。中文译文统一存放于 `notes/ultrascale-playbook/`，并接入知识库导航。

- [开篇介绍中文译文](../../notes/ultrascale-playbook/introduction.md)。
- [总体概览中文译文](../../notes/ultrascale-playbook/high-level-overview.md)。
- [单 GPU 训练中文译文](../../notes/ultrascale-playbook/first-steps.md)：包含显存占用、激活重计算、梯度累积和性能剖析。
- [数据并行中文译文](../../notes/ultrascale-playbook/data-parallelism.md)：包含 DP 和 ZeRO，按网页原文逐句翻译。
- [张量并行中文译文](../../notes/ultrascale-playbook/tensor-parallelism.md)：包含 TP 和 SP，按网页原文逐句翻译。
- [上下文并行中文译文](../../notes/ultrascale-playbook/context-parallelism.md)：按网页原文逐句翻译。
- [流水线并行中文译文](../../notes/ultrascale-playbook/pipeline-parallelism.md)：包含 AFAB、1F1B、交错阶段、Zero Bubble 和 DualPipe。
- [专家并行中文译文](../../notes/ultrascale-playbook/expert-parallelism.md)。
- [5D 并行概览中文译文](../../notes/ultrascale-playbook/5d-parallelism.md)。
- [寻找最佳训练配置中文译文](../../notes/ultrascale-playbook/training-configuration.md)。
- [GPU 底层优化中文译文](../../notes/ultrascale-playbook/gpu-optimization.md)：包含 kernel 优化、FlashAttention 和混合精度训练。
- [结语中文译文](../../notes/ultrascale-playbook/conclusion.md)。
- [参考资料中文译文](../../notes/ultrascale-playbook/references.md)。
- [附录中文译文](../../notes/ultrascale-playbook/appendix.md)：包含 A0–A3、原文代码与输出，以及本书引用格式。
- [完整 PDF](playbook.pdf)：作者发布的 PDF，直接打开即可离线阅读文字、公式与图片。
- [网页版本](index.html)：保留网页正文、侧注、动态图、公式与交互图表；Tensor Parallelism 对应 `#tensor-parallelism`。
- [Markdown 源文件](upstream.md)：作者提供的补充版本，已改为本地图片路径。该上游版本与网页措辞存在差别，并缺少部分交互图表和内嵌代码，请以网页和 PDF 为主要阅读版本。
- `assets/images/`：130 张原始图片，含 SVG、PNG、JPEG 和 GIF。
- `fragments/`、`assets/data/`：原网页图表数据与片段。
- `code/`：网页内嵌代码，按原文指定的 Git commit 下载。
- `vendor/`：离线公式渲染、字体和 Plotly 脚本。
- `source.html`：未经本地化改动的网页源码。
- `source-tree.json`：下载时上游文件目录；`manifest.json` 记录本地文件大小和 SHA-256。

网页通过 JavaScript 加载图表和参考文献，浏览器直接打开 `file://` 文件时可能阻止这些请求。完整网页建议用本地静态 HTTP 服务阅读，在仓库根目录执行：

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory references/ultrascale-playbook
```

然后访问 [本地网页的 Tensor Parallelism](http://127.0.0.1:8765/index.html#tensor-parallelism)。此服务仅用于读取已下载文件，不需要联网。

外部参考链接仍指向原站；未下载播客音频、外链论文和外部工具。PDF 为作者提供的静态版本，交互图表的操作需使用网页。

归档时保留原始资源，修正网页及 Markdown 图片路径，补齐公式/图表依赖，并将远程代码嵌入替换为本地文字。未运行构建、测试或浏览器验证，未验证渲染后的页面效果。

内容著作权归原作者，本目录用于保存原文和本地阅读。
