<script setup lang="ts">
import { computed, ref, useId } from "vue"

const id = useId()
const zigzag = ref(false)
const ranks = [0, 1, 2, 3]
const chunks = [0, 1, 2, 3, 4, 5, 6, 7]
const owner = (chunk: number) => zigzag.value ? Math.min(chunk, 7 - chunk) : Math.floor(chunk / 2)
const assignments = computed(() => ranks.map(rank => chunks.filter(chunk => owner(chunk) === rank)))
const work = (chunk: number) => chunk + 0.5
</script>

<template>
  <figure class="cp-balance" aria-label="连续分片与 Zig-Zag 的因果注意力负载对比">
    <div class="modes" role="group" aria-label="token 分片方式">
      <button type="button" :aria-pressed="!zigzag" @click="zigzag = false">连续分片</button>
      <button type="button" :aria-pressed="zigzag" @click="zigzag = true">Zig-Zag</button>
    </div>
    <div class="scroll" role="region" tabindex="0" aria-label="因果注意力矩阵及每 Rank 的计算量，可横向滚动">
      <svg viewBox="0 0 800 390" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ zigzag ? 'Zig-Zag：每卡分得早晚各一块' : '连续分片：靠后 Rank 承担更多计算' }}</title>
        <desc :id="`${id}-desc`">四个 Rank 分配八个等长序列块，矩阵行是 Query 块，列是 KV 块，颜色表示 Query 所属 Rank。未来位置为空，块内下三角用半个方格表示。右侧用相同尺度展示各 Rank 的有效计算面积。{{ zigzag ? '依次分配 C0与C7、C1与C6、C2与C5、C3与C4，每卡均约八个完整块面积。' : '依次分配 C0与C1、C2与C3、C4与C5、C6与C7，各卡约为二、六、十、十四个完整块面积。' }}计算量按大块近似，以一个完整块的计算面积为单位。</desc>
        <text x="204" y="28" class="label">KV 块</text>
        <text x="35" y="205" class="label" transform="rotate(-90 35 205)">Query 块</text>
        <g v-for="q in chunks" :key="q" :transform="`translate(80,${60 + q * 34})`">
          <text x="-12" y="23" class="row-label">C{{ q }}</text>
          <template v-for="k in chunks" :key="k">
            <rect :x="k * 34" width="34" height="34" class="empty" />
            <rect v-if="k < q" :x="k * 34" width="34" height="34" :class="`work rank${owner(q)}`" />
            <path v-else-if="k === q" :d="`M${k * 34} 0 V34 H${(k + 1) * 34} Z`" :class="`work rank${owner(q)}`" />
          </template>
        </g>
        <text v-for="k in chunks" :key="k" :x="97 + k * 34" y="50" class="label small">C{{ k }}</text>
        <text x="530" y="28" class="label">Rank / 持有块</text>
        <text x="705" y="28" class="label">有效计算量</text>
        <g v-for="rank in ranks" :key="rank" :transform="`translate(420,${75 + rank * 70})`">
          <text x="0" y="17" class="rank-label">Rank {{ rank }}</text>
          <g v-for="(chunk, index) in assignments[rank]" :key="chunk" :transform="`translate(${82 + index * 49},0)`">
            <rect width="42" height="26" rx="3" :class="`work rank${rank}`" />
            <text x="21" y="18" class="label small">C{{ chunk }}</text>
          </g>
          <g transform="translate(194,0)">
            <rect width="154" height="26" rx="3" class="empty" />
            <rect v-for="(chunk, index) in assignments[rank]" :key="chunk" :x="index === 0 ? 0 : work(assignments[rank][0]) * 11" :width="work(chunk) * 11" height="26" :class="`work rank${rank}`" />
            <text :x="assignments[rank].reduce((total, chunk) => total + work(chunk), 0) * 11 + 8" y="18" class="small">{{ assignments[rank].reduce((total, chunk) => total + work(chunk), 0) }}</text>
          </g>
        </g>
        <text x="440" y="367" class="label small">同一尺度 · 1 单位 ≈ 一个完整块的计算面积</text>
      </svg>
    </div>
    <figcaption>颜色：Query 所属 Rank；块内下三角按半个块估算。</figcaption>
  </figure>
</template>

<style scoped>
.cp-balance { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.modes { display: flex; gap: 0.5rem; padding: 0.8rem 0.8rem 0.25rem; }
button { font: inherit; font-size: 0.9rem; border: 1px solid var(--infra-diagram-line); border-radius: 6px; padding: 0.35rem 0.7rem; color: var(--vp-c-text); background: var(--vp-c-bg); cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--infra-token-flow); color: var(--infra-token-flow); background: var(--vp-c-bg-alt); }
button:focus-visible, .scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -2px; }
.scroll { overflow-x: auto; padding: 0.5rem; }
svg { display: block; width: 100%; min-width: 720px; font-family: inherit; }
text { fill: var(--vp-c-text); font-size: 16px; }
.label { text-anchor: middle; }
.small { font-size: 13px; }
.row-label { text-anchor: end; font-size: 13px; }
.rank-label { font-size: 14px; }
.rank0 { color: var(--infra-token-flow); }
.rank1 { color: var(--infra-weight-flow); }
.rank2 { color: var(--infra-execution); }
.rank3 { color: var(--infra-placement); }
.empty { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); stroke-width: 0.7; }
.work { fill: currentColor; fill-opacity: 0.3; stroke: currentColor; stroke-width: 0.7; }
figcaption { padding: 0.1rem 0.9rem 0.8rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
@media print { .cp-balance { break-inside: avoid; } svg { min-width: 0; } .scroll { overflow: visible; } .modes { display: none; } }
</style>
