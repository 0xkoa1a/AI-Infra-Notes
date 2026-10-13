<script setup lang="ts">
import { useId } from "vue"

const id = useId()
const inputs = [
  { name: "A · 2×8", x: 128, y: 72, rows: 2, cols: 8, cell: 22, split: "columns" },
  { name: "W · 8×4", x: 460, y: 50, rows: 8, cols: 4, cell: 16, split: "rows" },
]
const local = [
  { name: "A", x: 120, y: 24, rows: 2, cols: 4 },
  { name: "W", x: 290, y: 8, rows: 4, cols: 4 },
  { name: "P", x: 486, y: 24, rows: 2, cols: 4 },
]
const sums = [
  { name: "P⁽⁰⁾", x: 170, owner: "gpu0" },
  { name: "P⁽¹⁾", x: 330, owner: "gpu1" },
  { name: "Y", x: 490, owner: "result" },
]
</script>

<template>
  <figure class="split-figure" aria-label="双 GPU 矩阵乘法：沿求和维度切分，再合并部分结果">
    <div class="scroll" tabindex="0" role="region" aria-label="矩阵切分与合并示意，可横向滚动">
      <svg viewBox="0 0 720 550" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">沿 K 维切分，输出逐元素相加</title>
        <desc :id="`${id}-desc`">A 为 2 行 8 列，沿列切成两份 2 行 4 列；W 为 8 行 4 列，沿行切成两份 4 行 4 列。GPU 0 和 GPU 1 各持有对应分片，分别得到完整输出形状的部分结果 P0 和 P1。AllReduce 逐元素相加得到 Y，两卡各得到一份完整结果。</desc>
        <text x="26" y="28" class="title">沿求和维度 K 切分</text>
        <g v-for="m in inputs" :key="m.name" :transform="`translate(${m.x},${m.y})`">
          <text :x="m.cols * m.cell / 2" y="-12" class="label">{{ m.name }}</text>
          <template v-for="row in m.rows" :key="row">
            <rect v-for="col in m.cols" :key="col" :x="(col - 1) * m.cell" :y="(row - 1) * m.cell" :width="m.cell" :height="m.cell" :class="(m.split === 'columns' ? col <= 4 : row <= 4) ? 'gpu0' : 'gpu1'" />
          </template>
          <path v-if="m.split === 'columns'" :d="`M${4 * m.cell} 0 V${m.rows * m.cell}`" class="cut" />
          <path v-else :d="`M0 ${4 * m.cell} H${m.cols * m.cell}`" class="cut" />
          <text :x="m.cols * m.cell / 2" :y="m.rows * m.cell + 24" class="muted">{{ m.split === 'columns' ? '沿列切' : '沿行切' }}</text>
        </g>
        <text x="176" y="166" class="label gpu0">GPU 0</text>
        <text x="264" y="166" class="label gpu1">GPU 1</text>
        <text x="594" y="85" class="label gpu0">GPU 0</text>
        <text x="594" y="150" class="label gpu1">GPU 1</text>

        <g v-for="gpu in [0, 1]" :key="gpu" :transform="`translate(0,${216 + gpu * 112})`">
          <rect x="22" y="-8" width="676" height="100" rx="8" class="panel" />
          <text x="70" y="40" class="label" :class="gpu === 0 ? 'gpu0' : 'gpu1'">GPU {{ gpu }}</text>
          <g v-for="m in local" :key="m.name" :transform="`translate(${m.x},${m.y})`">
            <template v-for="row in m.rows" :key="row">
              <rect v-for="col in m.cols" :key="col" :x="(col - 1) * 16" :y="(row - 1) * 16" width="16" height="16" :class="gpu === 0 ? 'gpu0' : 'gpu1'" />
            </template>
            <text x="32" :y="m.rows * 16 + 14" class="label">{{ m.name }}{{ gpu === 0 ? '⁽⁰⁾' : '⁽¹⁾' }}</text>
          </g>
          <text x="248" y="40" class="operator">×</text>
          <text x="440" y="40" class="operator">=</text>
          <text x="615" y="40" class="muted">2×4</text>
        </g>

        <text x="26" y="438" class="title">AllReduce · 逐元素相加</text>
        <g v-for="m in sums" :key="m.name" :transform="`translate(${m.x},462)`">
          <template v-for="row in 2" :key="row">
            <rect v-for="col in 4" :key="col" :x="(col - 1) * 18" :y="(row - 1) * 18" width="18" height="18" :class="m.owner" />
          </template>
          <text x="36" y="60" class="label">{{ m.name }}</text>
        </g>
        <text x="286" y="488" class="operator">+</text>
        <text x="446" y="488" class="operator">=</text>
        <text x="630" y="486" class="muted">两卡各得 Y</text>
      </svg>
    </div>
  </figure>
</template>

<style scoped>
.split-figure { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.scroll { overflow-x: auto; padding: 0.4rem; }
.scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 660px; font-family: inherit; }
text { fill: var(--vp-c-text); }
.title { font-size: 18px; font-weight: 650; }
.label { font-size: 17px; text-anchor: middle; }
.muted { font-size: 15px; text-anchor: middle; fill: var(--infra-diagram-muted); }
.operator { font-size: 26px; text-anchor: middle; }
.gpu0 { color: var(--infra-token-flow); }
.gpu1 { color: var(--infra-weight-flow); }
.result { color: var(--infra-execution); }
text.gpu0, text.gpu1 { fill: currentColor; font-weight: 650; }
rect.gpu0, rect.gpu1, rect.result { fill: currentColor; fill-opacity: 0.17; stroke: currentColor; stroke-width: 1; }
.panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.cut { stroke: var(--vp-c-text); stroke-width: 2; stroke-dasharray: 4 3; fill: none; }
@media print { .split-figure { break-inside: avoid; } .scroll { overflow: visible; } svg { min-width: 0; } }
</style>
