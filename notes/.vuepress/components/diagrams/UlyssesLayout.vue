<script setup lang="ts">
import { useId } from "vue"

const id = useId()
const ranks = [0, 1]
const states = [
  { name: "X", y: 210, sequence: true, heads: false, qkv: false },
  { name: "QKV", y: 350, sequence: true, heads: true, qkv: true },
  { name: "QKV′", y: 490, sequence: false, heads: true, qkv: true },
  { name: "O′", y: 630, sequence: false, heads: true, qkv: false },
  { name: "O", y: 770, sequence: true, heads: true, qkv: false },
  { name: "Y", y: 910, sequence: true, heads: false, qkv: false },
]
const operations = [
  { name: "QKV Proj", y: 280, weight: true },
  { name: "AllToAll", y: 410, weight: false },
  { name: "Attention", y: 560, weight: false },
  { name: "AllToAll", y: 700, weight: false },
  { name: "O Proj", y: 840, weight: true },
]
const weights = [
  { x: 24, cols: 12, name: "QKV", shape: "H×3H" },
  { x: 146, cols: 4, name: "O", shape: "H×H" },
]
const exchanges = [
  { d: "M300 400 V438", head: 0 },
  { d: "M600 400 V438", head: 1 },
  { d: "M300 400 L600 438", head: 1 },
  { d: "M600 400 L300 438", head: 0 },
  { d: "M300 700 V746", head: 0 },
  { d: "M600 700 V746", head: 1 },
  { d: "M300 700 L600 746", head: 0 },
  { d: "M600 700 L300 746", head: 1 },
]
const localSteps = [
  { start: 260, end: 318 },
  { start: 560, end: 586 },
  { start: 820, end: 886 },
]
</script>

<template>
  <figure class="ulysses-layout" aria-label="纯 UP 的完整前向流程与各 Rank 的激活、权重分布">
    <div class="scroll" role="region" tabindex="0" aria-label="按执行顺序从上到下阅读，可横向滚动">
      <svg viewBox="0 0 780 980" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">QKV 投影 → AllToAll → Attention → AllToAll → 输出投影</title>
        <desc :id="`${id}-desc`">两列分别为 Rank 0 和 Rank 1，每列上方的 WQKV 和 WO 都是完整权重副本，贯穿整个流程。例子有四个 token、四个 heads，每个 head 一维。开始时两卡分别持有 token 零到一、二到三的完整隐藏特征。使用完整 WQKV 做本地投影后，两卡各生成本地 token 的全部 Q、K、V heads。第一次 AllToAll 后，Rank 0 持有所有 token 的 head 零到一，Rank 1 持有所有 token 的 head 二到三。各卡完成本地 Attention。第二次 AllToAll 将 O 恢复为各卡原先的 token、全部 heads。最后各卡读取完整 WO 做本地输出投影，得到对应 token 的完整 Y。灰色虚线表示投影读取权重；蓝色和橙色标识 head 分组。各行按前向执行顺序展示主要激活。</desc>
        <defs>
          <marker v-for="head in ranks" :id="`${id}-head-${head}`" :key="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0 L10 5 L0 10 Z" :class="`arrowhead head${head}`" />
          </marker>
          <marker :id="`${id}-local`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0 L10 5 L0 10 Z" class="neutral-head" />
          </marker>
        </defs>

        <text x="450" y="24" class="label note">权重完整复制 · 全程保留</text>
        <text x="80" y="174" class="label note">前向数据流 ↓</text>
        <g v-for="rank in ranks" :key="rank">
          <rect :x="160 + rank * 300" y="40" width="280" height="930" rx="10" class="rank-panel" />
          <text :x="300 + rank * 300" y="66" class="label strong">Rank {{ rank }}</text>
          <g :transform="`translate(${190 + rank * 300},80)`">
            <rect width="220" height="80" rx="6" class="weight-bank" />
            <g v-for="w in weights" :key="w.name" :transform="`translate(${w.x},27)`">
              <text :x="w.cols * 2.5" y="-10" class="label weight-name">W<tspan baseline-shift="sub">{{ w.name }}</tspan></text>
              <template v-for="row in 4" :key="row">
                <rect v-for="col in w.cols" :key="col" :x="(col - 1) * 5" :y="(row - 1) * 5" width="5" height="5" class="weight-cell" />
              </template>
              <text :x="w.cols * 2.5" y="42" class="label note">{{ w.shape }}</text>
            </g>
          </g>
          <path :d="rank === 0 ? 'M190 120 H180 V950' : 'M710 120 H720 V950'" class="weight-rail" />
          <path v-for="y in [280, 840]" :key="`weight-${y}`" :d="rank === 0 ? `M180 ${y} H300` : `M720 ${y} H600`" class="weight-rail" :marker-end="`url(#${id}-local)`" />
        </g>

        <g v-for="s in states" :key="s.name">
          <text x="80" :y="s.y + 6" class="label strong">{{ s.name }}</text>
          <g v-for="rank in ranks" :key="rank" :transform="`translate(${300 + rank * 300 - (s.sequence ? 40 : 20)},${s.y - (s.sequence ? 20 : 40)})`">
            <template v-if="s.qkv">
              <rect v-for="offset in [8, 4]" :key="offset" :x="offset" :y="-offset" :width="s.sequence ? 80 : 40" :height="s.sequence ? 40 : 80" class="tensor-back" />
            </template>
            <template v-for="row in (s.sequence ? 2 : 4)" :key="row">
              <rect v-for="col in (s.sequence ? 4 : 2)" :key="col" :x="(col - 1) * 20" :y="(row - 1) * 20" width="20" height="20" :class="s.heads ? `head${s.sequence ? (col <= 2 ? 0 : 1) : rank}` : 'hidden-cell'" />
              <text x="-10" :y="(row - 1) * 20 + 14" class="token">t{{ (s.sequence ? rank * 2 : 0) + row - 1 }}</text>
            </template>
            <text :x="s.sequence ? 40 : 20" :y="(s.sequence ? 40 : 80) + 20" class="label note">{{ s.heads ? (s.sequence ? 'h0–3' : rank === 0 ? 'h0–1' : 'h2–3') : '完整 H' }}</text>
          </g>
        </g>
        <g v-for="op in operations" :key="op.y">
          <rect x="22" :y="op.y - 15" width="116" height="30" rx="5" :class="op.weight ? 'projection' : 'operation'" />
          <text x="80" :y="op.y + 5" class="label op-label">{{ op.name }}</text>
        </g>
        <g v-for="rank in ranks" :key="`local-${rank}`">
          <path v-for="step in localSteps" :key="step.start" :d="`M${300 + rank * 300} ${step.start} V${step.end}`" class="local-arrow" :marker-end="`url(#${id}-local)`" />
        </g>
        <path v-for="edge in exchanges" :key="edge.d" :d="edge.d" :class="`exchange head${edge.head}`" :marker-end="`url(#${id}-head-${edge.head})`" />
      </svg>
    </div>
    <figcaption>颜色：head 分组；虚线：权重读取。</figcaption>
  </figure>
</template>

<style scoped>
.ulysses-layout { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.scroll { overflow-x: auto; padding: 0.5rem; }
.scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 720px; font-family: inherit; }
text { font-size: 17px; fill: var(--vp-c-text); }
.label { text-anchor: middle; }
.strong { font-weight: 650; }
.note { font-size: 13px; fill: var(--infra-diagram-muted); }
.token { font-size: 12px; fill: var(--infra-diagram-muted); text-anchor: end; }
.weight-name { font-size: 14px; }
.weight-name tspan { font-size: 10px; }
.head0 { color: var(--infra-token-flow); }
.head1 { color: var(--infra-weight-flow); }
rect.head0, rect.head1 { fill: currentColor; fill-opacity: 0.22; stroke: currentColor; stroke-width: 0.8; }
.rank-panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.weight-bank, .tensor-back { fill: var(--vp-c-bg); stroke: var(--infra-diagram-muted); stroke-width: 0.8; }
.weight-cell, .hidden-cell { fill: var(--infra-diagram-muted); fill-opacity: 0.18; stroke: var(--infra-diagram-muted); stroke-width: 0.6; }
.weight-rail { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.1; stroke-dasharray: 4 4; }
.operation, .projection { fill: var(--vp-c-bg); stroke: var(--infra-diagram-line); }
.projection { stroke: var(--infra-diagram-muted); }
.op-label { font-size: 14px; }
.exchange { fill: none; stroke: currentColor; stroke-width: 1.4; }
.arrowhead { fill: currentColor; }
.local-arrow { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.4; }
.neutral-head { fill: var(--infra-diagram-muted); }
figcaption { padding: 0.1rem 0.9rem 0.8rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
@media print { .ulysses-layout { break-inside: avoid; } svg { min-width: 0; } .scroll { overflow: visible; } }
</style>
