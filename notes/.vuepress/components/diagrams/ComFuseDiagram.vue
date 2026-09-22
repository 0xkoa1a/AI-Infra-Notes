<script setup lang="ts">
import { computed, useId } from "vue"

type Tone = "data" | "stat" | "flow" | "compile"
type Node = { x: number; y: number; w: number; h: number; tone: Tone; lines: string[] }
type Wire = { d: string; tone: Tone; dashed?: boolean }
type Label = { x: number; y: number; text: string; tone?: Tone; start?: boolean; halo?: boolean }
type Panel = { x: number; y: number; w: number; h: number; boundary?: boolean; dashed?: boolean }
type Diagram = { title: string; description: string; caption: string; height: number; panels: Panel[]; labels: Label[]; nodes: Node[]; wires: Wire[] }

const props = defineProps<{ view: "stages" | "compiler" | "b2b" }>()
const id = useId()
const tones: Tone[] = ["data", "stat", "flow", "compile"]
const diagrams: Record<typeof props.view, Diagram> = {
  stages: {
    title: "一行 8 个值，两个 CTA 协作完成 LayerNorm",
    description: "同一融合 kernel 内，一个 cluster 的两个 CTA 各保留四个值。CTA 0 的 Z 为 1、2、3、4，CTA 1 为 5、6、7、8。局部和 10 与 26 汇总得到均值 4.5，分发后在本地中心化。局部平方和均为 21，汇总得到方差 5.25，再分发给两侧完成归一化，按原列位置写回。只有局部统计量参与跨 CTA 汇总。",
    caption: "数值与分工仅为教学示例。中央框表示逻辑上的汇总与分发，不指定额外 CTA 或固定 leader；实际交换策略见下文。",
    height: 880,
    panels: [
      { x: 12, y: 12, w: 896, h: 850, boundary: true },
      { x: 28, y: 92, w: 864, h: 642, boundary: true, dashed: true },
      { x: 44, y: 150, w: 264, h: 556 },
      { x: 612, y: 150, w: 264, h: 556 },
    ],
    labels: [
      { x: 36, y: 42, text: "一次融合 kernel 启动：MatMul → Bias → LayerNorm", start: true },
      { x: 36, y: 70, text: "M = 1，N = 8；MatMul + Bias 得到这一行 Z = [1, 2, 3, 4, 5, 6, 7, 8]", start: true },
      { x: 48, y: 125, text: "一个 cluster · 同一次归约的 CTA 协作组", start: true },
      { x: 176, y: 176, text: "CTA 0 · 第 1–4 列" },
      { x: 744, y: 176, text: "CTA 1 · 第 5–8 列" },
      { x: 460, y: 176, text: "共享统计量", tone: "stat" },
      { x: 176, y: 330, text: "Z⁽⁰⁾ 留在本地", tone: "data", halo: true },
      { x: 744, y: 330, text: "Z⁽¹⁾ 留在本地", tone: "data", halo: true },
      { x: 328, y: 329, text: "μ", tone: "stat", halo: true },
      { x: 592, y: 329, text: "μ", tone: "stat", halo: true },
      { x: 176, y: 532, text: "D⁽⁰⁾ 留在本地", tone: "data", halo: true },
      { x: 744, y: 532, text: "D⁽¹⁾ 留在本地", tone: "data", halo: true },
      { x: 328, y: 532, text: "v", tone: "stat", halo: true },
      { x: 592, y: 532, text: "v", tone: "stat", halo: true },
      { x: 460, y: 690, text: "两侧共用 μ 和 v，各自保留 4 个元素", tone: "flow" },
      { x: 244, y: 775, text: "写入第 1–4 列" },
      { x: 676, y: 775, text: "写入第 5–8 列" },
    ],
    nodes: [
      { x: 56, y: 190, w: 240, h: 110, tone: "data", lines: ["MatMul + Bias", "Z⁽⁰⁾ = [1, 2, 3, 4] · [1,4]", "局部和 = 10"] },
      { x: 624, y: 190, w: 240, h: 110, tone: "data", lines: ["MatMul + Bias", "Z⁽¹⁾ = [5, 6, 7, 8] · [1,4]", "局部和 = 26"] },
      { x: 348, y: 198, w: 224, h: 88, tone: "stat", lines: ["阶段 1 · 合并局部和", "μ = (10 + 26) / 8 = 4.5", "统计量形状 [1,1]"] },
      { x: 56, y: 358, w: 240, h: 126, tone: "data", lines: ["中心化 D⁽⁰⁾ = Z⁽⁰⁾ − μ", "[−3.5, −2.5, −1.5, −0.5]", "D⁽⁰⁾ 形状 [1,4]", "局部平方和 = 21"] },
      { x: 624, y: 358, w: 240, h: 126, tone: "data", lines: ["中心化 D⁽¹⁾ = Z⁽¹⁾ − μ", "[0.5, 1.5, 2.5, 3.5]", "D⁽¹⁾ 形状 [1,4]", "局部平方和 = 21"] },
      { x: 348, y: 410, w: 224, h: 88, tone: "stat", lines: ["阶段 2 · 合并平方和", "v = (21 + 21) / 8 = 5.25", "统计量形状 [1,1]"] },
      { x: 56, y: 568, w: 240, h: 90, tone: "data", lines: ["阶段 3 · 本地归一化", "Y⁽⁰⁾ = D⁽⁰⁾ / √(5.25 + ε)", "局部输出 [1,4]"] },
      { x: 624, y: 568, w: 240, h: 90, tone: "data", lines: ["阶段 3 · 本地归一化", "Y⁽¹⁾ = D⁽¹⁾ / √(5.25 + ε)", "局部输出 [1,4]"] },
      { x: 348, y: 764, w: 224, h: 68, tone: "flow", lines: ["按原列位置写回显存", "完整输出 Y [1,8]"] },
    ],
    wires: [
      { d: "M296 266 H322 V224 H348", tone: "stat" },
      { d: "M624 266 H598 V224 H572", tone: "stat" },
      { d: "M348 262 H328 V394 H296", tone: "stat" },
      { d: "M572 262 H592 V394 H624", tone: "stat" },
      { d: "M176 300 V358", tone: "data", dashed: true },
      { d: "M744 300 V358", tone: "data", dashed: true },
      { d: "M296 454 H348", tone: "stat" },
      { d: "M624 454 H572", tone: "stat" },
      { d: "M348 480 H328 V607 H296", tone: "stat" },
      { d: "M572 480 H592 V607 H624", tone: "stat" },
      { d: "M176 484 V568", tone: "data", dashed: true },
      { d: "M744 484 V568", tone: "data", dashed: true },
      { d: "M176 658 V798 H348", tone: "flow" },
      { d: "M744 658 V798 H572", tone: "flow" },
    ],
  },
  compiler: {
    title: "追踪一次求均值：表达式 → IR → 后端节点",
    description: "输入图已展开为基础操作：Z 沿 N 求均值得到 μ，后续减法同时使用 μ 与 Z。前端在 IR 中记录归约输入、均值语义、旁路 Z 以及接收 μ 和 Z 的后继阶段。后端映射为 StreamReduction 节点，关联 PostOp 后继阶段，装入 CUTLASS 的 MatMul 实现，再由 NVCC 编译和执行器调用。",
    caption: "蓝色线传递逐元素值，橙色线传递统计量；紫色箭头表示编译转换。这里只展开 LayerNorm 的第一次归约。",
    height: 762,
    panels: [
      { x: 16, y: 16, w: 888, h: 190 },
      { x: 16, y: 274, w: 888, h: 160 },
      { x: 16, y: 500, w: 888, h: 116 },
    ],
    labels: [
      { x: 40, y: 46, text: "输入表达式 · 基础操作和依赖已经可见", start: true },
      { x: 614, y: 96, text: "μ", tone: "stat" },
      { x: 460, y: 188, text: "后续减法还需要 Z", tone: "data", halo: true },
      { x: 460, y: 248, text: "前端：识别归约，建立旁路和后继阶段", tone: "compile", halo: true },
      { x: 40, y: 304, text: "Fusion Spec IR · 记录这次归约的关系", tone: "compile", start: true },
      { x: 460, y: 474, text: "后端：将 IR 中的关系映射到具体节点", tone: "compile", halo: true },
      { x: 40, y: 530, text: "CUTLASS 后处理节点 · StreamReduction", tone: "compile", start: true },
      { x: 460, y: 650, text: "将节点及后继装入 MatMul 模板", tone: "compile", halo: true },
    ],
    nodes: [
      { x: 40, y: 76, w: 220, h: 68, tone: "data", lines: ["Z = P + b", "输入形状 [M,N]"] },
      { x: 348, y: 76, w: 224, h: 68, tone: "stat", lines: ["μ = 沿 N 求均值 (Z)", "输出形状 [M,1]"] },
      { x: 660, y: 76, w: 220, h: 68, tone: "data", lines: ["D = Z − μ", "μ 广播回 [M,N]"] },
      { x: 40, y: 324, w: 240, h: 86, tone: "data", lines: ["输入与旁路", "读取 Z [M,N]", "保留 Z，供后继阶段使用"] },
      { x: 340, y: 324, w: 240, h: 86, tone: "stat", lines: ["归约记录", "对 Z 沿 N 求均值", "得到 μ [M,1]"] },
      { x: 640, y: 324, w: 240, h: 86, tone: "data", lines: ["后继阶段入口", "接收统计量 μ 与旁路 Z", "接下来计算 D = Z − μ"] },
      { x: 40, y: 548, w: 390, h: 52, tone: "stat", lines: ["归约操作：求和；模式：均值"] },
      { x: 490, y: 548, w: 390, h: 52, tone: "compile", lines: ["PostOp：关联计算 D 的后继阶段"] },
      { x: 40, y: 680, w: 248, h: 66, tone: "compile", lines: ["MatMul + EVT 实现", "包含归约节点及后继阶段"] },
      { x: 336, y: 680, w: 248, h: 66, tone: "compile", lines: ["NVCC 编译", "得到可加载的程序模块"] },
      { x: 632, y: 680, w: 248, h: 66, tone: "compile", lines: ["图执行器调用", "传入实际张量与参数"] },
    ],
    wires: [
      { d: "M260 106 H348", tone: "data" },
      { d: "M572 106 H660", tone: "stat" },
      { d: "M150 144 V181 H770 V144", tone: "data", dashed: true },
      { d: "M460 206 V274", tone: "compile" },
      { d: "M280 366 H340", tone: "data" },
      { d: "M580 366 H640", tone: "stat" },
      { d: "M160 410 V421 H760 V410", tone: "data", dashed: true },
      { d: "M460 434 V500", tone: "compile" },
      { d: "M460 616 V650 H164 V680", tone: "compile" },
      { d: "M288 713 H336", tone: "compile" },
      { d: "M584 713 H632", tone: "compile" },
    ],
  },
  b2b: {
    title: "第二次 MatMul：两份局部乘积，相加得到同一个输出",
    description: "教学示例中 S 的形状为 2 乘 8，W2 为 8 乘 3。两个 CTA 各持有 S 的四列，并使用 W2 对应的四行。它们各自得到一个 2 乘 3 的局部乘积，按元素相加形成完整的 2 乘 3 输出。",
    caption: "教学示例：S [2,8] × W₂ [8,3]。沿相乘求和的长度 8 分工，所以两个局部输出都覆盖同样的 2 行、3 列，最终按元素相加。",
    height: 452,
    panels: [{ x: 16, y: 16, w: 424, h: 262 }, { x: 480, y: 16, w: 424, h: 262 }],
    labels: [{ x: 228, y: 46, text: "CTA 0 · 中间维度第 1–4 项" }, { x: 692, y: 46, text: "CTA 1 · 中间维度第 5–8 项" }],
    nodes: [
      { x: 36, y: 72, w: 182, h: 68, tone: "data", lines: ["S⁽⁰⁾ [2,4]", "S 的第 1–4 列"] },
      { x: 238, y: 72, w: 182, h: 68, tone: "data", lines: ["W₂⁽⁰⁾ [4,3]", "W₂ 的第 1–4 行"] },
      { x: 500, y: 72, w: 182, h: 68, tone: "data", lines: ["S⁽¹⁾ [2,4]", "S 的第 5–8 列"] },
      { x: 702, y: 72, w: 182, h: 68, tone: "data", lines: ["W₂⁽¹⁾ [4,3]", "W₂ 的第 5–8 行"] },
      { x: 96, y: 194, w: 264, h: 68, tone: "data", lines: ["O⁽⁰⁾ = S⁽⁰⁾ W₂⁽⁰⁾", "局部乘积 [2,3]"] },
      { x: 560, y: 194, w: 264, h: 68, tone: "data", lines: ["O⁽¹⁾ = S⁽¹⁾ W₂⁽¹⁾", "局部乘积 [2,3]"] },
      { x: 320, y: 358, w: 280, h: 74, tone: "stat", lines: ["O = O⁽⁰⁾ + O⁽¹⁾", "按元素相加 · 完整输出 [2,3]"] },
    ],
    wires: [
      { d: "M127 140 V168 H180 V194", tone: "data" },
      { d: "M329 140 V168 H276 V194", tone: "data" },
      { d: "M591 140 V168 H644 V194", tone: "data" },
      { d: "M793 140 V168 H740 V194", tone: "data" },
      { d: "M228 262 V395 H320", tone: "stat" },
      { d: "M692 262 V395 H600", tone: "stat" },
    ],
  },
}
const diagram = computed(() => diagrams[props.view])
</script>

<template>
  <figure class="comfuse-figure" :aria-label="diagram.title">
    <div class="comfuse-title"><strong>{{ diagram.title }}</strong></div>
    <div v-if="view === 'stages'" class="comfuse-legend">
      <span class="data"><i class="bypass" />本地保留的逐元素值</span>
      <span class="stat"><i />参与交换的统计量</span>
    </div>
    <div class="comfuse-scroll" tabindex="0" role="region" :aria-label="`${diagram.title}，可横向滚动`">
      <svg :viewBox="`0 0 920 ${diagram.height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ diagram.title }}</title>
        <desc :id="`${id}-desc`">{{ diagram.description }}</desc>
        <defs>
          <marker v-for="tone in tones" :id="`${id}-${tone}`" :key="tone" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" :class="tone" fill="currentColor" />
          </marker>
        </defs>
        <rect v-for="(panel, index) in diagram.panels" :key="`panel-${index}`" :x="panel.x" :y="panel.y" :width="panel.w" :height="panel.h" rx="12" :class="panel.boundary ? 'boundary' : 'panel'" :stroke-dasharray="panel.dashed ? '7 5' : undefined" />
        <path v-for="(wire, index) in diagram.wires" :key="`wire-${index}`" :d="wire.d" class="wire" :class="wire.tone" :stroke-dasharray="wire.dashed ? '6 4' : undefined" :marker-end="`url(#${id}-${wire.tone})`" />
        <g v-for="(node, index) in diagram.nodes" :key="`node-${index}`" class="node" :class="node.tone" :transform="`translate(${node.x} ${node.y})`">
          <rect :width="node.w" :height="node.h" rx="9" />
          <text v-for="(line, lineIndex) in node.lines" :key="lineIndex" :x="node.w / 2" :y="28 + lineIndex * 24" :class="lineIndex === 0 ? 'node-title' : 'small'">{{ line }}</text>
        </g>
        <text v-for="(label, index) in diagram.labels" :key="`label-${index}`" :x="label.x" :y="label.y" class="label" :class="[label.tone ?? 'flow', { 'label-halo': label.halo }]" :text-anchor="label.start ? 'start' : 'middle'">{{ label.text }}</text>
      </svg>
    </div>
    <figcaption>{{ diagram.caption }}</figcaption>
  </figure>
</template>

<style scoped>
.comfuse-figure { margin: 1.6rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.comfuse-title { padding: 1.05rem 1.15rem 0.65rem; line-height: 1.6; }
.comfuse-legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1.4rem; padding: 0 1.15rem 0.5rem; font-size: 0.85rem; }
.comfuse-legend span { display: inline-flex; gap: 0.4rem; align-items: center; }
.comfuse-legend i { width: 1.5rem; border-top: 2px solid currentColor; }
.comfuse-legend .bypass { border-top-style: dashed; }
.comfuse-scroll { overflow-x: auto; padding: 0.35rem; }
.comfuse-scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 700px; font-family: inherit; }
.data { color: var(--infra-token-flow); }
.stat { color: var(--infra-weight-flow); }
.flow { color: var(--infra-diagram-muted); }
.compile { color: var(--infra-planning); }
.panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.boundary { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.2; }
.node { text-anchor: middle; }
.node rect { fill: var(--vp-c-bg); stroke: currentColor; stroke-width: 1.6; }
.node-title { font-size: 17px; font-weight: 650; fill: currentColor; }
.small { font-size: 15px; fill: var(--vp-c-text); }
.label { font-size: 16px; fill: currentColor; }
.wire { fill: none; stroke: currentColor; stroke-width: 2; stroke-linejoin: round; }
.label-halo { paint-order: stroke; stroke: var(--vp-c-bg); stroke-width: 8; stroke-linejoin: round; }
figcaption { padding: 0.85rem 1.15rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.65; }
@media print { .comfuse-figure { break-inside: avoid; } .comfuse-scroll { overflow: visible; } svg { min-width: 0; } }
</style>
