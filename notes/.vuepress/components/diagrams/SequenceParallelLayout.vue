<script setup lang="ts">
import { ref, useId } from "vue"

const id = useId()
const view = ref<"compare" | "flow">("compare")
const panels = [
  { x: 0, name: "TP", operation: "AllReduce", scatter: false },
  { x: 410, name: "TP + SP", operation: "ReduceScatter", scatter: true },
]
const stages = [
  { x: 52, name: "SP", cols: 4, rows: 2, partial: false },
  { x: 212, name: "完整输入", cols: 4, rows: 4, partial: false },
  { x: 372, name: "Attention", cols: 2, rows: 4, partial: false },
  { x: 532, name: "局部贡献", cols: 4, rows: 4, partial: true },
  { x: 712, name: "SP", cols: 4, rows: 2, partial: false },
]
const flowEdges = [
  "M110 159 L208 173", "M110 324 L208 338",
  "M110 159 L208 338", "M110 324 L208 173",
  "M270 173 H368", "M270 338 H368",
  "M402 173 H528", "M402 338 H528",
  "M590 173 L708 159", "M590 338 L708 324",
  "M590 173 L708 324", "M590 338 L708 159",
]
</script>

<template>
  <figure class="sp-layout" aria-label="TP 与 Sequence Parallelism 的激活分布">
    <div class="views" role="group" aria-label="选择图示内容">
      <button type="button" :aria-pressed="view === 'compare'" @click="view = 'compare'">归约后的分布</button>
      <button type="button" :aria-pressed="view === 'flow'" @click="view = 'flow'">子层衔接</button>
    </div>
    <div class="scroll" role="region" tabindex="0" aria-label="激活分布图，可横向滚动">
      <svg v-if="view === 'compare'" viewBox="0 0 820 370" role="img" :aria-labelledby="`${id}-compare-title ${id}-compare-desc`">
        <title :id="`${id}-compare-title`">相同的局部贡献，经不同归约得到复制或分片的输出</title>
        <desc :id="`${id}-compare-desc`">两卡各产生四个 token、每个 token 四个特征的局部贡献。普通 TP 的 AllReduce 后，每卡保存四个 token 的完整结果；TP 加 SP 使用 ReduceScatter，两卡分别保存 token 零到一和二到三的完整结果。蓝色表示 token 零到一，橙色表示 token 二到三；斜线表示尚未归约的局部贡献。图不表示先物化完整结果再切分的实现。</desc>
        <defs>
          <pattern :id="`${id}-compare-hatch`" width="7" height="7" patternUnits="userSpaceOnUse"><path d="M0 7 L7 0" class="hatch" /></pattern>
          <marker :id="`${id}-compare-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker>
        </defs>
        <g v-for="p in panels" :key="p.name" :transform="`translate(${p.x},0)`">
          <rect x="8" y="8" width="394" height="350" rx="8" class="panel" />
          <text x="205" y="34" class="strong label">{{ p.name }}</text>
          <g v-for="rank in [0, 1]" :key="rank" :transform="`translate(${80 + rank * 180},70)`">
            <text x="32" y="-12" class="label">GPU {{ rank }}</text>
            <template v-for="row in 4" :key="row">
              <rect v-for="col in 4" :key="col" :x="(col - 1) * 16" :y="(row - 1) * 16" width="16" height="16" :class="row <= 2 ? 'tokens0' : 'tokens1'" />
              <text x="-10" :y="(row - 1) * 16 + 12" class="token">t{{ row - 1 }}</text>
            </template>
            <rect width="64" height="64" :fill="`url(#${id}-compare-hatch)`" />
            <text x="32" y="83" class="muted">P{{ rank === 0 ? '⁽⁰⁾' : '⁽¹⁾' }}</text>
          </g>
          <path d="M112 159 V169 H205 V178 M292 159 V169 H205" class="arrow" />
          <rect x="133" y="178" width="144" height="34" rx="6" class="operation" />
          <text x="205" y="201" class="label">{{ p.operation }}</text>
          <path d="M205 212 V226 H112 V244" class="arrow" :marker-end="`url(#${id}-compare-arrow)`" />
          <path d="M205 226 H292 V244" class="arrow" :marker-end="`url(#${id}-compare-arrow)`" />
          <g v-for="rank in [0, 1]" :key="`out-${rank}`" :transform="`translate(${80 + rank * 180},246)`">
            <template v-for="row in p.scatter ? 2 : 4" :key="row">
              <rect v-for="col in 4" :key="col" :x="(col - 1) * 16" :y="(row - 1) * 16" width="16" height="16" :class="(p.scatter ? rank === 0 : row <= 2) ? 'tokens0' : 'tokens1'" />
              <text x="-10" :y="(row - 1) * 16 + 12" class="token">t{{ (p.scatter ? rank * 2 : 0) + row - 1 }}</text>
            </template>
            <text x="32" :y="(p.scatter ? 32 : 64) + 24" class="label">{{ p.scatter ? 'Y' + rank : 'Y' }}</text>
          </g>
        </g>
      </svg>
      <svg v-else viewBox="0 0 820 410" role="img" :aria-labelledby="`${id}-flow-title ${id}-flow-desc`">
        <title :id="`${id}-flow-title`">SP 与 TP Attention 子层之间的衔接</title>
        <desc :id="`${id}-flow-desc`">两 GPU 起初各持有两个 token 的完整特征。AllGather 后每卡得到四个 token，QKV 投影按头分片，各卡独立执行其头的 Attention。输出投影生成四个 token 的局部贡献，ReduceScatter 合并两卡贡献并按 token 分发。跨卡连线表示通信的输入依赖，不表示具体网络算法。</desc>
        <defs>
          <pattern :id="`${id}-flow-hatch`" width="7" height="7" patternUnits="userSpaceOnUse"><path d="M0 7 L7 0" class="hatch" /></pattern>
          <marker :id="`${id}-flow-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker>
        </defs>
        <rect x="8" y="111" width="804" height="116" rx="8" class="panel" />
        <rect x="8" y="276" width="804" height="116" rx="8" class="panel" />
        <text x="38" y="134" class="small">GPU 0</text>
        <text x="38" y="299" class="small">GPU 1</text>
        <text x="160" y="76" class="label operation-label">AllGather</text>
        <text x="480" y="76" class="label">W<tspan baseline-shift="sub">O</tspan></text>
        <text x="650" y="76" class="label operation-label">ReduceScatter</text>
        <text x="320" y="76" class="label">QKV</text>
        <g v-for="s in stages" :key="s.name + s.x">
          <text :x="s.x + s.cols * 7" y="34" class="label strong">{{ s.name }}</text>
          <g v-for="rank in [0, 1]" :key="rank" :transform="`translate(${s.x},${145 + rank * 165})`">
            <template v-for="row in s.rows" :key="row">
              <rect v-for="col in s.cols" :key="col" :x="(col - 1) * 14" :y="(row - 1) * 14" width="14" height="14" :class="(s.rows === 2 ? rank === 0 : row <= 2) ? 'tokens0' : 'tokens1'" />
              <text v-if="s.rows === 2" x="-7" :y="(row - 1) * 14 + 11" class="token">t{{ rank * 2 + row - 1 }}</text>
            </template>
            <rect v-if="s.partial" width="56" height="56" :fill="`url(#${id}-flow-hatch)`" />
            <text v-if="s.x === 372" x="14" y="76" class="muted">heads {{ rank === 0 ? '0–1' : '2–3' }}</text>
          </g>
        </g>
        <path v-for="edge in flowEdges" :key="edge" :d="edge" class="arrow" :marker-end="`url(#${id}-flow-arrow)`" />
      </svg>
    </div>
    <figcaption>斜线：未归约的局部贡献。<template v-if="view === 'flow'">连线示意依赖，不代表具体网络算法。</template></figcaption>
  </figure>
</template>

<style scoped>
.sp-layout { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.views { display: flex; gap: 0.5rem; padding: 0.8rem 0.8rem 0.25rem; flex-wrap: wrap; }
button { font: inherit; font-size: 0.9rem; border: 1px solid var(--infra-diagram-line); border-radius: 6px; padding: 0.35rem 0.7rem; color: var(--vp-c-text); background: var(--vp-c-bg); cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--infra-token-flow); color: var(--infra-token-flow); background: var(--vp-c-bg-alt); }
button:focus-visible, .scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -2px; }
.scroll { overflow-x: auto; padding: 0.3rem; }
svg { display: block; width: 100%; min-width: 760px; font-family: inherit; }
text { font-size: 16px; fill: var(--vp-c-text); }
.label { text-anchor: middle; }
.strong { font-weight: 650; }
.small { font-size: 13px; }
.muted { font-size: 14px; fill: var(--infra-diagram-muted); text-anchor: middle; }
.token { font-size: 11px; fill: var(--infra-diagram-muted); text-anchor: end; }
.tokens0 { color: var(--infra-token-flow); }
.tokens1 { color: var(--infra-weight-flow); }
rect.tokens0, rect.tokens1 { fill: currentColor; fill-opacity: 0.22; stroke: currentColor; stroke-width: 0.8; }
.panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.operation { fill: var(--vp-c-bg); stroke: var(--infra-diagram-muted); }
.operation-label { fill: var(--infra-execution); }
.arrow { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.3; }
.arrowhead { fill: var(--infra-diagram-muted); }
.hatch { stroke: var(--infra-diagram-muted); stroke-width: 0.8; opacity: 0.6; }
figcaption { padding: 0.15rem 0.9rem 0.8rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
@media print { .sp-layout { break-inside: avoid; } svg { min-width: 0; } .scroll { overflow: visible; } .views { display: none; } }
</style>
