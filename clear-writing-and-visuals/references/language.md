# 语言与叙述

本文件中的风格要求必须落实到实际文本，贯穿起草、改写、图表说明、Review 和交付。每次使用本 Skill 均读取本文件，并在完成前检查遵循情况。用户对当前任务明确指定的体裁或风格优先；其余部分继续遵循这些要求。

## 完整语言指引

以下为用户提供的原文。它们约束语言决策，不能只作为泛泛的“简洁写作”建议。

> Default to using clear, concise paragraphs, each developing one main idea. Use lists only when the information is genuinely parallel, sequential, or easier to compare, and avoid nested lists unless the hierarchy cannot be expressed clearly in prose. Use plain, simple language: familiar words, concrete examples, and precise verbs. Prefer active voice and direct statements.
>
> Make sure to state the main point clearly and early, then develop it with the explanation and detail the reader needs. Let each sentence build on what came before. Develop the points that matter and provide enough support to be useful.
>
> Use plain language over jargon, and reference technical details only to the degree that it helps illustrate an idea or your work to the user. Communicate complex concepts in a clear and cohesive manner, and calibrate your writing to the level of background knowledge assumed from the user's prompt and context.
>
> Avoid using slop words or phrases like "Bottom Line:" in conclusions, "delve," "foster," "leverage," "it's worth noting," "importantly," "Question? Answer." or "This isn't about X. It's about Y.", "genuinely" or hyphenated compound descriptions and adjectives. Do not use concluding summary statements such as "In short:..", "The simplest mental model is:...".
>
> State the intended action directly. Avoid adding what you won't do, what will remain unchanged, or how you'll separate or categorize results. Do not use contrastive framing such as "X, not Y" or "X—not Y" that introduces an unprompted alternative that the user didn't ask about. Avoid invented compound labels like "exact-head checks" and "editorial-row layouts", vague qualifiers, and canned transitions; use plain verbs and prepositions to state the actual relationship directly.

## 段落与展开

默认使用清楚简洁的段落，每段发展一个主要观点。尽早陈述判断，再给出理解它所需的机制、例子或证据。让后一条信息建立在前一条之上，避免连续抛出名词、结论或彼此独立的短句。

重要观点需要充分展开。不要将“简洁”执行成省去必要前提、把因果压成口号，或要求读者自行补出中间推理。次要细节按它对当前观点的贡献决定位置。

只有信息确实并列、有先后步骤或需要比较时才使用列表。段落能够清楚表达的层次，优先用段落承接。列表项保持相近的逻辑层级，避免把一个完整推理拆成多层项目符号。

## 用词与句式

使用熟悉的词、具体的对象和准确的动词。能说清“谁做了什么、改变了什么、为什么”时，不用抽象名词替代动作。优先主动语态和直接陈述。

技术术语用于提高准确性；首次出现的陌生术语按读者背景给予就近解释。避免堆积缩写，避免在已具有相应知识的读者面前反复解释基础概念。标准术语保留通用名称，不为显得专业而创造新标签。

禁用原文列出的套话及其同类中文表达，例如“值得注意的是”“重要的是”式无实质过渡、“一句话总结”“最简单的心智模型是”等收尾。直接说出要强调的内容。没有证据或明确尺度支撑的“显著”“极大”“全面”等限定词应改为具体效果或删去。

避免临时拼造复合词和连字符修饰语。用普通名词、动词和介词说清关系；标准技术术语、产品名和准确引用保留原本写法。

## 行动、对比与结束

直接说明准备采取的行动及其用途。不要附加用户没有提出的替代方案，不主动列出不会做什么、什么保持不变或打算如何分类结果。确实影响用户判断的范围限制、风险或阻碍需要简洁说明。

涉及真实比较时，把比较对象和差异直接说清。例如解释序列拼接和特征拼接，可以分别说明各自改变哪个轴。避免用“不是 X，而是 Y”反复引入假想误解。纠正用户明确提出的误解时，直接回应实际问题。

正文表达完整即可结束。若体裁明确需要结论或建议，写出由前文支持的判断和下一步，不追加重复全文的口号式总结。

## 改写示例

这些示例说明如何改善表达，示例中的效果须由具体任务证据支持后才可用于真实文档。

| 待改表达 | 改写方向 |
|---|---|
| “值得注意的是，该机制能够有效提升整体处理效率。” | “合并小请求减少了调度次数，缩短了每批请求的固定开销。” |
| “这不是普通缓存，而是一套真正高效的复用机制。” | “结果写入缓存后，后续请求可以复用相同输入的计算结果。” |
| “通信优化显著降低了通信量。”但证据只显示重叠增加 | “输入收集与本地计算重叠，使部分通信等待被计算覆盖。” |
| “我们将深入探索全链路多维度协同策略。” | “先定位等待发生的位置，再检查能否提前启动独立工作。” |
| “我会修改结构，不会动数据，也不会重新做实验。” | “我会把整体流程前移，并将重复解释合并到对应章节。” |

较弱的论述：“模块 A 输出特征。模块 B 做融合。模型支持缓存。吞吐提高。”

更连贯的论述：“模块 A 的输出在相同输入下保持不变，因此可以缓存。模块 B 每次读取缓存结果，再与当前输入融合。这减少了重复计算；吞吐能提高多少，还取决于模块 A 原先占用的时间和缓存读取成本。”

## 最后一遍语言检查

逐段检查主要观点、句间关系和解释深度。将不必要的列表还原为段落，把抽象动作替换成具体动作，删去重复前缀和结尾。检查术语、限定词、无关对比和禁用表达，确认图注与交付说明也遵循相同标准。

不要仅靠词语匹配完成风格检查：同一问题可能换一种说法出现；出现在待改示例或准确引用中的词，也需要结合上下文判断。
