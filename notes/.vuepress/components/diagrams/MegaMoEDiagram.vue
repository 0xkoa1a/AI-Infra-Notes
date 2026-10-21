<script setup lang="ts">
import { computed } from "vue"
import MegaMoERouting from "./MegaMoERouting.vue"
import MegaMoECompute from "./MegaMoECompute.vue"

const props = defineProps<{ view: "placement" | "symmem" | "dispatch" | "compute" | "partition" | "readiness" | "combine" | "schedule" | "ring" }>()
const routing = computed(() => ["placement", "symmem", "dispatch", "combine"].includes(props.view))
</script>

<template>
  <figure class="mega-figure">
    <MegaMoERouting v-if="routing" :view="view" />
    <MegaMoECompute v-else :view="view" />
  </figure>
</template>

<style scoped>
.mega-figure { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.mega-figure :deep(.controls) { display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem; padding: 0.75rem 0.85rem 0.2rem; }
.mega-figure :deep(.control-label) { font-size: 0.85rem; color: var(--infra-diagram-muted); margin-right: 0.25rem; }
.mega-figure :deep(button) { font: inherit; font-size: 0.85rem; padding: 0.3rem 0.65rem; border: 1px solid var(--infra-diagram-line); border-radius: 6px; color: var(--vp-c-text); background: var(--vp-c-bg); cursor: pointer; }
.mega-figure :deep(button[aria-pressed="true"]) { color: var(--infra-token-flow); border-color: var(--infra-token-flow); background: var(--vp-c-bg-alt); }
.mega-figure :deep(button:focus-visible), .mega-figure :deep(.scroll:focus-visible) { outline: 2px solid var(--infra-token-flow); outline-offset: -2px; }
.mega-figure :deep(.scroll) { overflow-x: auto; padding: 0.5rem; }
.mega-figure :deep(svg) { display: block; width: 100%; min-width: 850px; font-family: inherit; }
.mega-figure :deep(svg.four-gpu) { min-width: 1000px; }
.mega-figure :deep(text) { fill: var(--vp-c-text); font-size: 15px; }
.mega-figure :deep(.center) { text-anchor: middle; }
.mega-figure :deep(.strong) { font-size: 17px; font-weight: 600; }
.mega-figure :deep(.small) { font-size: 12px; }
.mega-figure :deep(.muted) { fill: var(--infra-diagram-muted); }
.mega-figure :deep(.panel) { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.mega-figure :deep(.box) { fill: var(--vp-c-bg); stroke: var(--infra-diagram-line); stroke-width: 1.1; }
.mega-figure :deep(.owner0), .mega-figure :deep(.acts) { color: var(--infra-token-flow); }
.mega-figure :deep(.owner1), .mega-figure :deep(.weight) { color: var(--infra-weight-flow); }
.mega-figure :deep(.owner2), .mega-figure :deep(.ready) { color: var(--infra-execution); }
.mega-figure :deep(.owner3), .mega-figure :deep(.result) { color: var(--infra-placement); }
.mega-figure :deep(.tint) { fill: currentColor; fill-opacity: 0.13; stroke: currentColor; stroke-width: 1; }
.mega-figure :deep(.ink) { fill: currentColor; }
.mega-figure :deep(.dim) { opacity: 0.25; }
.mega-figure :deep(.pending) { fill: var(--vp-c-bg); stroke: var(--infra-diagram-line); stroke-dasharray: 4 3; }
.mega-figure :deep(.edge) { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.4; }
.mega-figure :deep(.data-edge) { fill: none; stroke: currentColor; stroke-width: 1.8; }
.mega-figure :deep(.signal) { fill: none; stroke: var(--infra-execution); stroke-width: 1.5; stroke-dasharray: 4 4; }
.mega-figure :deep(.arrowhead) { fill: var(--infra-diagram-muted); }
.mega-figure :deep(.signal-head) { fill: var(--infra-execution); }
.mega-figure :deep(.weight-cell) { fill: var(--infra-diagram-muted); fill-opacity: 0.18; stroke: var(--infra-diagram-muted); stroke-width: 0.6; }
.mega-figure :deep(figcaption) { padding: 0.15rem 0.85rem 0.8rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
@media print { .mega-figure :deep(svg) { min-width: 0; } .mega-figure :deep(.scroll) { overflow: visible; } }
</style>
