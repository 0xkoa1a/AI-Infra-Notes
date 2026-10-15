<script setup lang="ts">
import { useId } from "vue"

const id = useId()
const stages = [
  { x: 100, name: "QKV", sequence: true },
  { x: 330, name: "QKV", sequence: false },
  { x: 540, name: "O", sequence: false },
  { x: 780, name: "O", sequence: true },
]
const exchanges = [
  { d: "M184 160 L326 140", head: 0 },
  { d: "M184 160 L326 330", head: 1 },
  { d: "M184 350 L326 180", head: 0 },
  { d: "M184 350 L326 370", head: 1 },
  { d: "M584 140 L776 160", head: 0 },
  { d: "M584 180 L776 350", head: 0 },
  { d: "M584 330 L776 160", head: 1 },
  { d: "M584 370 L776 350", head: 1 },
]
</script>

<template>
  <figure class="ulysses-layout" aria-label="Ulysses 的两次 AllToAll 与本地 Attention">
    <div class="scroll" role="region" tabindex="0" aria-label="序列与 head 分片转换图，可横向滚动">
      <svg viewBox="0 0 900 450" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">序列分片 → head 分片 → 本地 Attention → 序列分片</title>
        <desc :id="`${id}-desc`">两张 GPU、四个 token、四个 heads，省略 batch 与每个 head 内部的特征维度。每格表示一个 token 在一个 head 上的数据向量。开始时 GPU 0 持有 token 零到一的所有 heads，GPU 1 持有 token 二到三的所有 heads。第一次 AllToAll 后，GPU 0 持有所有 token 的 head 零到一，GPU 1 持有所有 token 的 head 二到三。各卡在自己的 heads 上执行完整序列的 Attention，得到 O。第二次 AllToAll 将 O 按 token 分回原卡。蓝色始终表示 head 零到一，橙色表示 head 二到三。连线表示分发关系，不表示具体网络路由。</desc>
        <defs>
          <marker v-for="head in [0, 1]" :id="`${id}-arrow-${head}`" :key="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0 L10 5 L0 10 Z" :class="`arrowhead head${head}`" />
          </marker>
          <marker :id="`${id}-compute-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0 0 L10 5 L0 10 Z" class="compute-head" />
          </marker>
        </defs>

        <text x="20" y="76" class="muted">token ↓ · head →</text>
        <text x="250" y="76" class="label strong">AllToAll</text>
        <text x="455" y="76" class="label strong">Attention</text>
        <text x="680" y="76" class="label strong">AllToAll</text>
        <g v-for="rank in [0, 1]" :key="rank">
          <rect x="10" :y="104 + rank * 190" width="880" height="140" rx="8" class="panel" />
          <text x="24" :y="166 + rank * 190" class="strong">GPU {{ rank }}</text>
        </g>

        <g v-for="s in stages" :key="s.x">
          <text :x="s.x + (s.sequence ? 40 : 20)" y="34" class="label strong">{{ s.name }}</text>
          <g v-for="rank in [0, 1]" :key="rank" :transform="`translate(${s.x},${(s.sequence ? 140 : 120) + rank * 190})`">
            <template v-for="row in s.sequence ? 2 : 4" :key="row">
              <rect v-for="col in s.sequence ? 4 : 2" :key="col" :x="(col - 1) * 20" :y="(row - 1) * 20" width="20" height="20" :class="`head${s.sequence ? (col <= 2 ? 0 : 1) : rank}`" />
              <text x="-9" :y="(row - 1) * 20 + 14" class="token">t{{ (s.sequence ? rank * 2 : 0) + row - 1 }}</text>
            </template>
            <path :d="s.sequence ? 'M40 0 V40' : 'M0 40 H40'" class="split" />
            <text :x="s.sequence ? 40 : 20" :y="(s.sequence ? 40 : 80) + 24" class="label heads">{{ s.sequence ? 'h0–3' : rank === 0 ? 'h0–1' : 'h2–3' }}</text>
          </g>
        </g>

        <path v-for="edge in exchanges" :key="edge.d" :d="edge.d" :class="`exchange head${edge.head}`" :marker-end="`url(#${id}-arrow-${edge.head})`" />
        <path v-for="rank in [0, 1]" :key="`compute-${rank}`" :d="`M376 ${160 + rank * 190} H536`" class="compute" :marker-end="`url(#${id}-compute-arrow)`" />
      </svg>
    </div>
    <figcaption>颜色表示 head 分组，不表示数值；连线示意分发关系。</figcaption>
  </figure>
</template>

<style scoped>
.ulysses-layout { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.scroll { overflow-x: auto; padding: 0.5rem; }
.scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 860px; font-family: inherit; }
text { font-size: 16px; fill: var(--vp-c-text); }
.label { text-anchor: middle; }
.strong { font-weight: 650; }
.muted, .heads { font-size: 13px; fill: var(--infra-diagram-muted); }
.token { font-size: 12px; fill: var(--infra-diagram-muted); text-anchor: end; }
.head0 { color: var(--infra-token-flow); }
.head1 { color: var(--infra-weight-flow); }
rect.head0, rect.head1 { fill: currentColor; fill-opacity: 0.22; stroke: currentColor; stroke-width: 0.8; }
.panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.split { fill: none; stroke: var(--infra-diagram-muted); stroke-dasharray: 3 3; stroke-width: 1.2; }
.exchange { fill: none; stroke: currentColor; stroke-width: 1.4; }
.arrowhead { fill: currentColor; }
.compute { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.4; }
.compute-head { fill: var(--infra-diagram-muted); }
figcaption { padding: 0.1rem 0.9rem 0.8rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
@media print { .ulysses-layout { break-inside: avoid; } svg { min-width: 0; } .scroll { overflow: visible; } }
</style>
