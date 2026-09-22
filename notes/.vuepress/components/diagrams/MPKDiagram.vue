<script setup lang="ts">
import { useId } from "vue"

const id = useId()
const states = [
  { title: "A 使用共享内存", detail: "计算第一个 MatMul tile", pages: ["A", "A", "A", "A"] },
  { title: "A 释放不再使用的页", detail: "前两页的最后一次读取已结束", pages: ["空闲", "空闲", "A", "A"] },
  { title: "B 利用空闲页预取", detail: "加载第二个 tile 的输入数据", pages: ["B", "B", "A", "A"] },
]
</script>

<template>
  <figure class="mpk-figure" aria-label="同一 worker 内，共享内存页面如何从任务 A 交给任务 B">
    <div class="mpk-title"><strong>共享内存页面的交接：A 继续计算，B 开始预取</strong></div>
    <div class="mpk-scroll" tabindex="0" role="region" aria-label="四页共享内存的三个状态，可横向滚动">
      <svg viewBox="0 0 920 490" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">同一 worker 的四页共享内存复用</title>
        <desc :id="`${id}-desc`">任务 A 和 B 计算同一 MatMul 的不同输出 tile，输入都已就绪。A 发出全部加载指令后，释放不再访问的前两页，并继续使用后两页。B 将自己的输入预取到空闲的前两页。三行表示先后状态，页面编号固定，不表示执行耗时。</desc>
        <defs>
          <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" class="arrow-head" />
          </marker>
        </defs>
        <text x="318" y="28" class="axis-label">同一个 worker 的共享内存 · 页号固定</text>
        <g v-for="(state, row) in states" :key="state.title" :transform="`translate(0, ${48 + row * 145})`">
          <rect x="16" y="0" width="888" height="100" rx="10" class="row-bg" />
          <text x="34" y="35" class="state-title">{{ state.title }}</text>
          <text x="34" y="65" class="state-detail">{{ state.detail }}</text>
          <g v-for="(owner, page) in state.pages" :key="page" :transform="`translate(${318 + page * 145}, 14)`" class="page" :class="owner === 'A' ? 'current' : owner === 'B' ? 'next' : 'free'">
            <rect width="130" height="72" rx="8" />
            <text x="65" y="26" class="page-index">第 {{ page + 1 }} 页</text>
            <text x="65" y="53" class="page-owner">{{ owner === '空闲' ? '空闲' : `任务 ${owner}` }}</text>
          </g>
          <template v-if="row < states.length - 1">
            <path d="M456 108 V134" class="arrow" :marker-end="`url(#${id}-arrow)`" />
            <text x="482" y="128" class="transition">{{ row === 0 ? '释放前两页' : '将空闲页分配给 B' }}</text>
          </template>
        </g>
        <text x="34" y="474" class="axis-label">示例前提：A 已发出全部加载指令；B 的输入已就绪，预取需要两页。</text>
      </svg>
    </div>
    <figcaption>依据论文 §5.3 绘制。三行展示同一 worker 的页面占用变化，四页为教学配置，长度不代表时间。</figcaption>
  </figure>
</template>

<style scoped>
.mpk-figure { margin: 1.6rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.mpk-title { padding: 1rem 1.15rem 0.5rem; line-height: 1.6; }
.mpk-scroll { padding: 0.35rem; overflow-x: auto; }
.mpk-scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 740px; font-family: inherit; }
.row-bg { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.state-title { fill: var(--vp-c-text); font-size: 18px; font-weight: 650; }
.state-detail, .axis-label, .transition { fill: var(--infra-diagram-muted); font-size: 15px; }
.page { text-anchor: middle; }
.page rect { fill: var(--vp-c-bg); stroke: currentColor; stroke-width: 2; }
.current { color: var(--infra-token-flow); }
.next { color: var(--infra-weight-flow); }
.free { color: var(--infra-diagram-muted); }
.free rect { stroke-dasharray: 5 4; }
.page-index { fill: var(--infra-diagram-muted); font-size: 15px; }
.page-owner { fill: currentColor; font-size: 18px; font-weight: 650; }
.arrow { stroke: var(--infra-diagram-muted); stroke-width: 2; fill: none; }
.arrow-head { fill: var(--infra-diagram-muted); }
figcaption { padding: 0.85rem 1.15rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.65; }
@media print { .mpk-figure { break-inside: avoid; } .mpk-scroll { overflow: visible; } svg { min-width: 0; } }
</style>
