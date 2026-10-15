<script setup lang="ts">
import { useId } from "vue"

const id = useId()
const matrices = [
  { x: 102, label: "X", kind: "input" },
  { x: 252, label: "W₁", kind: "weight" },
  { x: 402, label: "Z", kind: "activation" },
  { x: 552, label: "W₂", kind: "weight" },
  { x: 702, label: "P", kind: "partial" },
]
const weights = [
  { x: 228, rows: 4, cols: 8, name: "W₁ · 按列切", split: "column" },
  { x: 552, rows: 8, cols: 4, name: "W₂ · 按行切", split: "row" },
]
const results = [
  { x: 102, name: "P⁽⁰⁾", kind: "rank0" },
  { x: 252, name: "P⁽¹⁾", kind: "rank1" },
  { x: 452, name: "GPU 0 · Y", kind: "result" },
  { x: 602, name: "GPU 1 · Y", kind: "result" },
]
</script>

<template>
  <figure class="tp-matmul" aria-label="两 GPU 的 Column–Row 配对与 AllReduce">
    <div class="scroll" role="region" tabindex="0" aria-label="矩阵切分图，可横向滚动">
      <svg viewBox="0 0 820 540" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">先按列切，再按行切，中间保持分片</title>
        <desc :id="`${id}-desc`">四个 token，隐藏维度四，中间维度八。W1 按列切给两卡，W2 按对应行切分。每卡使用完整 X、本地 W1，经逐元素激活得到 Z 的四列，再与本地 W2 相乘，产生四行四列的局部贡献 P。AllReduce 将两个 P 逐元素求和，两卡各得到一份 Y。蓝色、橙色表示两卡负责的分片，斜线表示未归约的局部贡献。</desc>
        <defs>
          <pattern :id="`${id}-partial`" width="7" height="7" patternUnits="userSpaceOnUse">
            <path d="M-1 1 L1 -1 M0 7 L7 0 M6 8 L8 6" class="hatch" />
          </pattern>
        </defs>
        <g v-for="w in weights" :key="w.name" :transform="`translate(${w.x},48)`">
          <text :x="w.cols * 6" y="-18" class="label strong">{{ w.name }}</text>
          <template v-for="row in w.rows" :key="row">
            <rect v-for="col in w.cols" :key="col" :x="(col - 1) * 12" :y="(row - 1) * 12" width="12" height="12" :class="(w.split === 'column' ? col <= 4 : row <= 4) ? 'rank0' : 'rank1'" />
          </template>
          <path :d="w.split === 'column' ? 'M48 0 V48' : 'M0 48 H48'" class="cut" />
          <text :x="w.cols * 6" :y="w.rows * 12 + 22" class="muted">{{ w.rows }}×{{ w.cols }}</text>
        </g>
        <g v-for="rank in [0, 1]" :key="rank" :transform="`translate(0,${190 + rank * 112})`">
          <rect x="12" y="0" width="796" height="100" rx="8" class="panel" />
          <text x="52" y="50" class="label strong" :class="`rank${rank}`">GPU {{ rank }}</text>
          <g v-for="m in matrices" :key="m.label" :transform="`translate(${m.x},16)`">
            <template v-for="row in 4" :key="row">
              <rect v-for="col in 4" :key="col" :x="(col - 1) * 12" :y="(row - 1) * 12" width="12" height="12" :class="m.kind === 'input' ? 'input' : `rank${rank}`" />
            </template>
            <rect v-if="m.kind === 'partial'" width="48" height="48" :fill="`url(#${id}-partial)`" />
            <text x="24" y="74" class="label">{{ m.label }}{{ m.kind === 'input' ? '' : m.kind === 'activation' ? (rank === 0 ? '₀' : '₁') : (rank === 0 ? '⁽⁰⁾' : '⁽¹⁾') }}</text>
          </g>
          <text x="201" y="47" class="operator">×</text>
          <path d="M326 40 H388 M380 35 L388 40 L380 45" class="arrow" />
          <text x="354" y="27" class="label">σ</text>
          <text x="501" y="47" class="operator">×</text>
          <text x="651" y="47" class="operator">=</text>
        </g>
        <text x="26" y="449" class="strong">AllReduce</text>
        <g v-for="r in results" :key="r.name" :transform="`translate(${r.x},464)`">
          <template v-for="row in 4" :key="row">
            <rect v-for="col in 4" :key="col" :x="(col - 1) * 12" :y="(row - 1) * 12" width="12" height="12" :class="r.kind" />
          </template>
          <rect v-if="r.kind !== 'result'" width="48" height="48" :fill="`url(#${id}-partial)`" />
          <text x="24" y="67" class="label">{{ r.name }}</text>
        </g>
        <text x="201" y="497" class="operator">+</text>
        <text x="376" y="497" class="operator">=</text>
        <path d="M439 458 H663 V516 H439 Z" class="copies" />
        <text x="728" y="492" class="muted">斜线：局部贡献</text>
      </svg>
    </div>
  </figure>
</template>

<style scoped>
.tp-matmul { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.scroll { overflow-x: auto; padding: 0.5rem; }
.scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 760px; font-family: inherit; }
text { fill: var(--vp-c-text); font-size: 16px; }
.label { text-anchor: middle; }
.strong { font-weight: 650; }
.muted { fill: var(--infra-diagram-muted); font-size: 14px; text-anchor: middle; }
.operator { font-size: 24px; text-anchor: middle; }
.rank0 { color: var(--infra-token-flow); }
.rank1 { color: var(--infra-weight-flow); }
.result { color: var(--infra-execution); }
.input { color: var(--infra-diagram-muted); }
rect.rank0, rect.rank1, rect.result, rect.input { fill: currentColor; fill-opacity: 0.2; stroke: currentColor; stroke-width: 0.8; }
text.rank0, text.rank1 { fill: currentColor; }
.panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.cut { stroke: var(--vp-c-text); stroke-width: 2; stroke-dasharray: 4 3; }
.arrow { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.5; }
.hatch { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 0.8; opacity: 0.6; }
.copies { fill: none; stroke: var(--infra-execution); stroke-dasharray: 4 3; }
@media print { .tp-matmul { break-inside: avoid; } svg { min-width: 0; } .scroll { overflow: visible; } }
</style>
