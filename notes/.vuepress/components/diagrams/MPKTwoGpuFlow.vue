<script setup lang="ts">
import { useId } from "vue"

const id = useId()
const lanes = [
  { gpu: 0, x: 200, local: "P⁽⁰⁾", other: "P⁽¹⁾", side: 45 },
  { gpu: 1, x: 560, local: "P⁽¹⁾", other: "P⁽⁰⁾", side: 715 },
]
</script>

<template>
  <figure class="flow-figure" aria-label="双 GPU 第一行结果的就绪、互传与归约流程">
    <div class="scroll" tabindex="0" role="region" aria-label="双 GPU 任务依赖图，可横向滚动">
      <svg viewBox="0 0 760 700" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">第一行就绪后派发通信，两份输入到齐后归约</title>
        <desc :id="`${id}-desc`">两卡各执行第一行 MatMul，写出本地部分结果并通知事件。事件就绪后，scheduler 派发通信任务，将部分结果交给对方。每卡的归约事件等待本地结果和对方已可读取的结果，两份输入到齐后派发本地求和，再通知后继。实线表示张量数据传递，虚线表示就绪通知和任务派发；这是依赖示意，不表示耗时或具体通信算法。</desc>
        <defs>
          <marker :id="`${id}-control`" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="control-head" /></marker>
          <marker v-for="gpu in [0,1]" :id="`${id}-data-${gpu}`" :key="gpu" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" :class="`gpu${gpu} data-head`" /></marker>
        </defs>
        <g v-for="lane in lanes" :key="lane.gpu" :class="`gpu${lane.gpu}`">
          <text :x="lane.x" y="32" class="gpu-label">GPU {{ lane.gpu }}</text>
          <rect :x="lane.x - 120" y="62" width="240" height="70" rx="8" class="task" />
          <text :x="lane.x" y="90" class="label">worker · MatMul 第一行</text>
          <text :x="lane.x" y="115" class="small">写出 {{ lane.local }} 第一行 · 1×4</text>
          <path :d="`M${lane.x} 132 V176`" class="control" :marker-end="`url(#${id}-control)`" />
          <text :x="lane.x + 32" y="158" class="small">通知</text>
          <rect :x="lane.x - 76" y="180" width="152" height="38" rx="19" class="event" />
          <text :x="lane.x" y="204" class="label">本地结果就绪</text>
          <path :d="`M${lane.x} 218 V268`" class="control" :marker-end="`url(#${id}-control)`" />
          <text :x="lane.x" y="248" class="small">scheduler 派发</text>
          <rect :x="lane.x - 120" y="274" width="240" height="52" rx="8" class="task" />
          <text :x="lane.x" y="306" class="label">通信任务 · 发送 {{ lane.local }}</text>
          <path :d="`M${lane.x + (lane.gpu === 0 ? -76 : 76)} 199 H${lane.side} V468 H${lane.x + (lane.gpu === 0 ? -100 : 100)}`" class="control" :marker-end="`url(#${id}-control)`" />
          <text :x="lane.side" y="340" class="small" :transform="`rotate(${lane.gpu === 0 ? -90 : 90},${lane.side},340)`">本地输入就绪</text>
          <rect :x="lane.x - 84" y="378" width="168" height="30" rx="5" class="received" />
          <text :x="lane.x" y="399" class="small">收到 {{ lane.other }} · 已可读</text>
          <path :d="`M${lane.x} 408 V428`" class="control" :marker-end="`url(#${id}-control)`" />
          <rect :x="lane.x - 100" y="432" width="200" height="72" rx="30" class="event" />
          <text :x="lane.x" y="460" class="label">两份输入到齐</text>
          <text :x="lane.x" y="485" class="small">本地 {{ lane.local }} + 收到 {{ lane.other }}</text>
          <path :d="`M${lane.x} 504 V558`" class="control" :marker-end="`url(#${id}-control)`" />
          <text :x="lane.x" y="537" class="small">scheduler 派发</text>
          <rect :x="lane.x - 120" y="564" width="240" height="62" rx="8" class="task" />
          <text :x="lane.x" y="590" class="label">worker · 逐元素相加</text>
          <text :x="lane.x" y="614" class="small">Y 第一行 · 1×4</text>
          <path :d="`M${lane.x} 626 V655`" class="control" :marker-end="`url(#${id}-control)`" />
          <text :x="lane.x" y="681" class="label">通知后继</text>
        </g>
        <path d="M200 326 L560 374" class="data gpu0" :marker-end="`url(#${id}-data-0)`" />
        <path d="M560 326 L200 374" class="cross-gap" />
        <path d="M560 326 L200 374" class="data gpu1" :marker-end="`url(#${id}-data-1)`" />
      </svg>
    </div>
    <figcaption>实线：数据传递；虚线：就绪通知与派发。仅展示第一行的依赖；第二行可独立推进。互传后各自求和为教学示例。</figcaption>
  </figure>
</template>

<style scoped>
.flow-figure { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.scroll { overflow-x: auto; padding: 0.5rem; }
.scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 700px; font-family: inherit; }
.gpu0 { color: var(--infra-token-flow); }
.gpu1 { color: var(--infra-weight-flow); }
text { text-anchor: middle; fill: var(--vp-c-text); }
.gpu-label { fill: currentColor; font-size: 20px; font-weight: 650; }
.label { font-size: 17px; }
.small { font-size: 14px; fill: var(--infra-diagram-muted); }
.task { fill: var(--vp-c-bg-alt); stroke: currentColor; stroke-width: 1.5; }
.received { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.event { fill: var(--vp-c-bg); stroke: var(--infra-execution); stroke-width: 1.5; }
.control { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.7; stroke-dasharray: 5 4; }
.control-head { fill: var(--infra-diagram-muted); }
.data { fill: none; stroke: currentColor; stroke-width: 2.5; }
.data-head { fill: currentColor; }
.cross-gap { fill: none; stroke: var(--vp-c-bg); stroke-width: 8; }
figcaption { padding: 0.8rem 1rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.65; }
@media print { .flow-figure { break-inside: avoid; } .scroll { overflow: visible; } svg { min-width: 0; } }
</style>
