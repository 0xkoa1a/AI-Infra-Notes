<script setup lang="ts">
import { computed, ref, useId } from "vue"

const props = defineProps<{ kind: "mlp" | "attention" }>()
const id = useId()
const sequence = ref(false)
const ranks = [0, 1]
const attention = computed(() => props.kind === "attention")
const moduleName = computed(() => attention.value ? "Attention" : "MLP")
type Step = "start" | "gather" | "column" | "middle" | "row" | "reduce"
type StateKind = "hidden" | "features" | "partial" | "result"
type State = { name: string; step: Step; kind: StateKind; rows: number; cols: number; tokensSplit: boolean; stack: boolean; y: number }
const states = computed<State[]>(() => {
  const data: Omit<State, "y">[] = [
    { name: "X", step: "start", kind: "hidden", rows: sequence.value ? 2 : 4, cols: 4, tokensSplit: sequence.value, stack: false },
  ]
  if (sequence.value) data.push({ name: "X", step: "gather", kind: "hidden", rows: 4, cols: 4, tokensSplit: false, stack: false })
  data.push(
    { name: attention.value ? "QKV" : "U", step: "column", kind: "features", rows: 4, cols: attention.value ? 2 : 4, tokensSplit: false, stack: attention.value },
    { name: attention.value ? "O" : "Z", step: "middle", kind: "features", rows: 4, cols: attention.value ? 2 : 4, tokensSplit: false, stack: false },
    { name: "P⁽ʳ⁾", step: "row", kind: "partial", rows: 4, cols: 4, tokensSplit: false, stack: false },
    { name: "Y", step: "reduce", kind: "result", rows: sequence.value ? 2 : 4, cols: 4, tokensSplit: sequence.value, stack: false },
  )
  let y = 250
  return data.map((s, index) => {
    if (index > 0) y += s.step === "reduce" ? 200 : 160
    return { ...s, y }
  })
})
const height = computed(() => states.value[states.value.length - 1].y + 90)
const operations = computed(() => states.value.slice(1).map((next, index) => {
  const previous = states.value[index]
  const names: Record<Exclude<Step, "start">, string> = {
    gather: "AllGather",
    column: attention.value ? "QKV Proj" : "W₁ · Column",
    middle: attention.value ? "Attention" : "Activation σ",
    row: attention.value ? "O Proj · Row" : "W₂ · Row",
    reduce: sequence.value ? "ReduceScatter" : "AllReduce",
  }
  return {
    previous, next, name: names[next.step as Exclude<Step, "start">],
    y: (previous.y + next.y) / 2,
    start: previous.y + previous.rows * 10 + 30,
    end: next.y - next.rows * 10 - (next.stack ? 8 : 0) - 4,
    weight: next.step === "column" || next.step === "row",
  }
}))
const weightGrids = computed(() => attention.value ? [
  { x: 20, y: 38, rows: 4, cols: 4, split: "columns" },
  { x: 52, y: 38, rows: 4, cols: 4, split: "columns" },
  { x: 84, y: 38, rows: 4, cols: 4, split: "columns" },
  { x: 168, y: 38, rows: 4, cols: 4, split: "rows" },
] : [
  { x: 34, y: 38, rows: 4, cols: 8, split: "columns" },
  { x: 168, y: 28, rows: 8, cols: 4, split: "rows" },
])
function ownsCell(rank: number, row: number, col: number, grid: { split: string; rows: number; cols: number }) {
  const coordinate = grid.split === "columns" ? col : row
  const half = (grid.split === "columns" ? grid.cols : grid.rows) / 2
  return coordinate > rank * half && coordinate <= (rank + 1) * half
}
function featureLabel(s: State, rank: number) {
  if (s.kind === "features") return attention.value ? (rank === 0 ? "h0–1" : "h2–3") : (rank === 0 ? "i0–3" : "i4–7")
  return "完整 H"
}
</script>

<template>
  <figure class="parallel-flow" :aria-label="`${moduleName} 的 TP 与 TP+SP 前向、激活和权重分片`">
    <div class="modes" role="group" :aria-label="`${moduleName} 并行方式`">
      <button type="button" :aria-pressed="!sequence" @click="sequence = false">TP</button>
      <button type="button" :aria-pressed="sequence" @click="sequence = true">TP + SP</button>
    </div>
    <div class="scroll" role="region" tabindex="0" aria-label="从上到下追踪两个 Rank 的数据分布，可横向滚动">
      <svg :viewBox="`0 0 780 ${height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ moduleName }} · {{ sequence ? 'TP+SP：先 AllGather，最后 ReduceScatter' : 'TP：完整输入，最后 AllReduce' }}</title>
        <desc :id="`${id}-desc`">两列对应两个 Rank，着色区为本卡保存的权重分片，淡框定位全局矩阵。{{ attention ? 'Q、K、V 权重各沿列切分，WO 沿行切分；每卡负责两个 heads。' : 'W1 沿列切分，W2 沿对应行切分；每卡负责四个中间特征。' }}权重分片全程保留，两种模式相同。{{ sequence ? '开始时每卡持有两个 token，AllGather 后每卡获得全部四个 token。' : '开始时每卡持有全部四个 token。' }}{{ attention ? '分片的 QKV 投影生成所有 token、部分 heads 的 QKV。本地 Attention 后，WO 的行分片产生部分和 P。' : '第一层生成所有 token 的部分中间特征，逐元素激活后，第二层行分片产生部分和 P。' }}P 保存本卡算出的部分和，用斜线标记。{{ sequence ? 'ReduceScatter 将两卡的部分和相加后，分别输出 token 零到一和二到三。' : 'AllReduce 将两卡的部分和相加，每卡得到全部 token 的结果。' }}Σ 表示逐元素相加，连线表示数据依赖，各行按前向执行顺序排列。</desc>
        <defs>
          <marker :id="`${id}-neutral`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="neutral-head" /></marker>
          <marker v-for="rank in ranks" :id="`${id}-rank-${rank}`" :key="rank" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" :class="`arrowhead rank${rank}`" /></marker>
          <marker :id="`${id}-result`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="result-head" /></marker>
          <pattern :id="`${id}-partial`" width="7" height="7" patternUnits="userSpaceOnUse"><path d="M0 7 L7 0" class="hatch" /></pattern>
        </defs>
        <text x="450" y="24" class="label note">权重分片 · 全程保留</text>
        <text x="80" y="204" class="label note">前向数据流 ↓</text>
        <g v-for="rank in ranks" :key="rank">
          <rect :x="160 + rank * 300" y="40" width="280" :height="height - 50" rx="10" class="rank-panel" />
          <text :x="300 + rank * 300" y="66" class="label strong">Rank {{ rank }}</text>
          <g :transform="`translate(${190 + rank * 300},80)`">
            <rect width="220" height="110" rx="6" class="weight-bank" />
            <text x="66" y="18" class="label weight-name">{{ attention ? 'WQ / WK / WV' : 'W₁ · 列切' }}</text>
            <text x="178" y="18" class="label weight-name">{{ attention ? 'WO · 行切' : 'W₂ · 行切' }}</text>
            <g v-for="(w, index) in weightGrids" :key="index" :transform="`translate(${w.x},${w.y})`">
              <template v-for="row in w.rows" :key="row">
                <rect v-for="col in w.cols" :key="col" :x="(col - 1) * 5" :y="(row - 1) * 5" width="5" height="5" :class="ownsCell(rank, row, col, w) ? `rank${rank}` : 'unowned'" />
              </template>
            </g>
            <text x="66" y="95" class="label note">{{ attention ? '各 H×(H/p)' : 'H×(I/p)' }}</text>
            <text x="178" y="95" class="label note">{{ attention ? '(H/p)×H' : '(I/p)×H' }}</text>
          </g>
          <path :d="rank === 0 ? `M190 140 H180 V${height - 28}` : `M710 140 H720 V${height - 28}`" class="weight-rail" />
          <template v-for="op in operations" :key="op.next.step">
            <path v-if="op.weight" :d="rank === 0 ? `M180 ${op.y} H300` : `M720 ${op.y} H600`" class="weight-rail" :marker-end="`url(#${id}-neutral)`" />
          </template>
        </g>
        <g v-for="s in states" :key="s.step">
          <text x="80" :y="s.y + 6" class="label strong">{{ s.name }}</text>
          <g v-for="rank in ranks" :key="rank" :transform="`translate(${300 + rank * 300 - s.cols * 10},${s.y - s.rows * 10})`">
            <template v-if="s.stack">
              <rect v-for="offset in [8, 4]" :key="offset" :x="offset" :y="-offset" :width="s.cols * 20" :height="s.rows * 20" class="tensor-back" />
            </template>
            <template v-for="row in s.rows" :key="row">
              <rect v-for="col in s.cols" :key="col" :x="(col - 1) * 20" :y="(row - 1) * 20" width="20" height="20" :class="s.kind === 'features' || s.kind === 'partial' ? `rank${rank}` : s.kind === 'result' ? 'result-cell' : 'hidden-cell'" />
              <text x="-10" :y="(row - 1) * 20 + 14" class="token">t{{ (s.tokensSplit ? rank * 2 : 0) + row - 1 }}</text>
            </template>
            <rect v-if="s.kind === 'partial'" :width="s.cols * 20" :height="s.rows * 20" :fill="`url(#${id}-partial)`" />
            <text :x="s.cols * 10" :y="s.rows * 20 + 20" class="label note">{{ featureLabel(s, rank) }}</text>
          </g>
        </g>
        <g v-for="op in operations" :key="op.next.step">
          <rect x="10" :y="op.y - 15" width="136" height="30" rx="5" class="operation" />
          <text x="78" :y="op.y + 5" class="label op-label">{{ op.name }}</text>
          <template v-if="op.next.step === 'gather'">
            <template v-for="source in ranks" :key="source">
              <path v-for="destination in ranks" :key="destination" :d="`M${300 + source * 300} ${op.start} L${300 + destination * 300} ${op.end}`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
            </template>
          </template>
          <template v-else-if="op.next.step === 'reduce'">
            <template v-for="source in ranks" :key="source">
              <path v-for="destination in ranks" :key="destination" :d="`M${300 + source * 300} ${op.start} L${300 + destination * 300} ${op.y + 13}`" :class="`exchange rank${source}`" :marker-end="`url(#${id}-rank-${source})`" />
            </template>
            <g v-for="rank in ranks" :key="rank">
              <circle :cx="300 + rank * 300" :cy="op.y + 24" r="10" class="sum" />
              <text :x="300 + rank * 300" :y="op.y + 29" class="label sum-label">Σ</text>
              <path :d="`M${300 + rank * 300} ${op.y + 36} V${op.end}`" class="result-arrow" :marker-end="`url(#${id}-result)`" />
            </g>
          </template>
          <template v-else>
            <path v-for="rank in ranks" :key="rank" :d="`M${300 + rank * 300} ${op.start} V${op.end}`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
          </template>
        </g>
      </svg>
    </div>
    <figcaption>权重淡框定位全局矩阵，着色部分为本卡分片；斜线 P⁽ʳ⁾：本卡的部分和；Σ：逐元素相加；虚线：权重读取。</figcaption>
  </figure>
</template>

<style scoped>
.parallel-flow { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.modes { display: flex; gap: 0.5rem; padding: 0.8rem 0.8rem 0.25rem; }
button { font: inherit; font-size: 0.9rem; border: 1px solid var(--infra-diagram-line); border-radius: 6px; padding: 0.35rem 0.7rem; color: var(--vp-c-text); background: var(--vp-c-bg); cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--infra-token-flow); color: var(--infra-token-flow); background: var(--vp-c-bg-alt); }
button:focus-visible, .scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -2px; }
.scroll { overflow-x: auto; padding: 0.5rem; }
svg { display: block; width: 100%; min-width: 720px; font-family: inherit; }
text { font-size: 17px; fill: var(--vp-c-text); }
.label { text-anchor: middle; }
.strong { font-weight: 650; }
.note { font-size: 12px; fill: var(--infra-diagram-muted); }
.token { font-size: 12px; fill: var(--infra-diagram-muted); text-anchor: end; }
.weight-name { font-size: 12px; }
.rank0 { color: var(--infra-token-flow); }
.rank1 { color: var(--infra-weight-flow); }
rect.rank0, rect.rank1 { fill: currentColor; fill-opacity: 0.22; stroke: currentColor; stroke-width: 0.8; }
.unowned { fill: none; stroke: var(--infra-diagram-muted); stroke-opacity: 0.22; stroke-width: 0.6; stroke-dasharray: 1 1; }
.rank-panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.weight-bank, .tensor-back, .operation { fill: var(--vp-c-bg); stroke: var(--infra-diagram-line); }
.hidden-cell { fill: var(--infra-diagram-muted); fill-opacity: 0.18; stroke: var(--infra-diagram-muted); stroke-width: 0.7; }
.result-cell { fill: var(--infra-execution); fill-opacity: 0.18; stroke: var(--infra-execution); stroke-width: 0.7; }
.weight-rail { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.1; stroke-dasharray: 4 4; }
.op-label { font-size: 13px; }
.local-arrow { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.4; }
.exchange { fill: none; stroke: currentColor; stroke-width: 1.4; }
.arrowhead { fill: currentColor; }
.neutral-head { fill: var(--infra-diagram-muted); }
.hatch { stroke: var(--infra-diagram-muted); stroke-width: 0.8; opacity: 0.6; }
.sum { fill: var(--vp-c-bg); stroke: var(--infra-execution); }
.sum-label { font-size: 14px; }
.result-arrow { fill: none; stroke: var(--infra-execution); stroke-width: 1.4; }
.result-head { fill: var(--infra-execution); }
figcaption { padding: 0.1rem 0.9rem 0.8rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
@media print { .parallel-flow { break-inside: avoid; } svg { min-width: 0; } .scroll { overflow: visible; } .modes { display: none; } }
</style>
