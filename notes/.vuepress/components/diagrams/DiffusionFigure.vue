<script lang="ts">
export type DiffusionTone = "image" | "text" | "joint" | "condition" | "neutral"
export type DiffusionDiagram = {
  title: string
  height: number
  caption: string
  boxes: { x: number; y: number; w: number; h?: number; label: string; detail?: string; tone?: DiffusionTone }[]
  wires: { path: string; tone?: DiffusionTone; dashed?: boolean }[]
  annotations?: { x: number; y: number; label: string; anchor?: "start" | "middle" | "end" }[]
  legend?: { tone: DiffusionTone; label: string }[]
}
</script>

<script setup lang="ts">
import { useId } from "vue"
defineProps<{ diagram: DiffusionDiagram }>()
const id = useId()
const tones: DiffusionTone[] = ["image", "text", "joint", "condition", "neutral"]
</script>

<template>
  <figure class="diffusion-figure" :aria-label="diagram.title">
    <div class="diffusion-figure__title"><strong>{{ diagram.title }}</strong></div>
    <div v-if="diagram.legend" class="diffusion-figure__legend">
      <span v-for="item in diagram.legend" :key="item.label" :class="`diffusion-${item.tone}`">{{ item.label }}</span>
    </div>
    <p class="diffusion-figure__hint">左右滑动查看完整图示；聚焦图示后也可使用左右方向键</p>
    <div class="diffusion-figure__scroll" tabindex="0" role="region" :aria-label="`${diagram.title}，窄屏可横向滚动`">
      <svg :viewBox="`0 0 800 ${diagram.height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ diagram.title }}</title>
        <desc :id="`${id}-desc`">{{ diagram.caption }}</desc>
        <defs>
          <marker v-for="tone in tones" :id="`${id}-${tone}`" :key="tone" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" :class="`diffusion-${tone}`" fill="currentColor" />
          </marker>
        </defs>
        <g fill="none" stroke-width="1.8" stroke-linejoin="round">
          <path v-for="(wire, index) in diagram.wires" :key="index" :d="wire.path" :class="`diffusion-${wire.tone ?? 'neutral'}`" stroke="currentColor" :stroke-dasharray="wire.dashed ? '6 5' : undefined" :marker-end="`url(#${id}-${wire.tone ?? 'neutral'})`" />
        </g>
        <g v-for="box in diagram.boxes" :key="box.label" :transform="`translate(${box.x}, ${box.y})`" :class="`diffusion-${box.tone ?? 'neutral'}`">
          <rect class="diffusion-box" :width="box.w" :height="box.h ?? 72" rx="9" />
          <text class="diffusion-label" :x="box.w / 2" y="29" text-anchor="middle">{{ box.label }}</text>
          <text class="diffusion-detail" :x="box.w / 2" y="52" text-anchor="middle">{{ box.detail }}</text>
        </g>
        <text v-for="(note, index) in diagram.annotations" :key="index" class="diffusion-annotation" :x="note.x" :y="note.y" :text-anchor="note.anchor ?? 'middle'">{{ note.label }}</text>
        <slot />
      </svg>
    </div>
    <figcaption>{{ diagram.caption }}</figcaption>
  </figure>
</template>

<style scoped>
.diffusion-figure {
  margin: 1.6rem 0 2rem;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 12px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text);
  overflow: hidden;
}
.diffusion-figure__title { padding: 1rem 1.2rem 0.6rem; }
.diffusion-figure__legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1.2rem; padding: 0 1.2rem 0.8rem; font-size: 0.82rem; }
.diffusion-figure__hint { display: none; }
.diffusion-figure__scroll { overflow-x: auto; padding: 0.5rem; }
.diffusion-figure__scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 680px; font-family: inherit; }
.diffusion-image { color: var(--infra-token-flow); }
.diffusion-text { color: var(--infra-planning); }
.diffusion-joint { color: var(--infra-execution); }
.diffusion-condition { color: var(--infra-weight-flow); }
.diffusion-neutral { color: var(--infra-diagram-muted); }
.diffusion-box { fill: var(--vp-c-bg); stroke: currentColor; stroke-width: 1.5; }
.diffusion-label { fill: currentColor; font-size: 16px; font-weight: 650; }
.diffusion-detail { fill: var(--vp-c-text); font-size: 13px; }
.diffusion-annotation { fill: var(--infra-diagram-muted); font-size: 14px; }
figcaption { padding: 0.9rem 1.2rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.65; }
@media (max-width: 719px) {
  .diffusion-figure__hint { display: block; margin: 0.3rem 1.2rem; color: var(--infra-diagram-muted); font-size: 0.8rem; }
}
@media print {
  .diffusion-figure { break-inside: avoid; }
  svg { min-width: 0; }
  .diffusion-figure__scroll { overflow: visible; }
  .diffusion-figure__hint { display: none; }
}
</style>
