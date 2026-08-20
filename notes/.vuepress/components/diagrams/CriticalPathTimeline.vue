<script setup lang="ts">
withDefaults(
  defineProps<{
    combine?: string
    compute?: string
    note?: string
    planning?: string
    resourceNote?: string
    tokenMovement?: string
    weightMovement?: string
  }>(),
  {
    combine: "Combine",
    compute: "Grouped GEMM",
    note: "Planning → Weight Prefetch → Token Dispatch → Grouped GEMM → Combine。",
    planning: "在线规划",
    resourceNote: "Weight Prefetch 与 Token Dispatch 可以并行，但会竞争同一互连带宽；默认按串行的保守画法表达关键路径。",
    tokenMovement: "Token Dispatch",
    weightMovement: "Weight Prefetch",
  },
)
</script>

<template>
  <figure class="critical-path" aria-label="关键路径及可重叠阶段的时间线">
    <div class="critical-path__canvas">
      <div class="critical-path__axis-label">时间</div>
      <div class="critical-path__axis" aria-hidden="true"></div>

      <div class="critical-path__lane-label critical-path__lane-label--cuda">CUDA Core</div>
      <div class="critical-path__lane critical-path__lane--cuda" aria-hidden="true"></div>
      <div class="critical-path__block critical-path__block--planning">{{ planning }}</div>

      <div class="critical-path__lane-label critical-path__lane-label--tensor">Tensor Core</div>
      <div class="critical-path__lane critical-path__lane--tensor" aria-hidden="true"></div>
      <div class="critical-path__block critical-path__block--compute">{{ compute }}</div>

      <div class="critical-path__lane-label critical-path__lane-label--communication">通信</div>
      <div class="critical-path__lane critical-path__lane--communication" aria-hidden="true"></div>
      <div class="critical-path__block critical-path__block--weight">{{ weightMovement }}</div>
      <div class="critical-path__block critical-path__block--token">{{ tokenMovement }}</div>
      <div class="critical-path__block critical-path__block--combine">{{ combine }}</div>
    </div>

    <figcaption class="critical-path__caption">
      <strong>{{ note }}</strong>
      <span>{{ resourceNote }}</span>
    </figcaption>
  </figure>
</template>

<style scoped>
.critical-path {
  margin: 1.6rem 0 2rem;
  color: var(--vp-c-text-1);
}

.critical-path *,
.critical-path *::before,
.critical-path *::after {
  box-sizing: border-box;
}

.critical-path__canvas {
  display: grid;
  min-width: 42rem;
  grid-template-columns: 6.7rem repeat(12, minmax(0, 1fr));
  grid-template-rows: 1.55rem repeat(3, 4.15rem);
  gap: 0 0.28rem;
  overflow: hidden;
}

.critical-path__axis-label,
.critical-path__lane-label {
  display: flex;
  align-items: center;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.73rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.critical-path__axis-label {
  grid-column: 1;
  grid-row: 1;
  align-items: start;
}

.critical-path__axis {
  position: relative;
  grid-column: 2 / 14;
  grid-row: 1;
  align-self: start;
  height: 1px;
  margin-top: 0.35rem;
  background: var(--infra-diagram-line);
}

.critical-path__axis::after {
  position: absolute;
  top: -0.22rem;
  right: -0.05rem;
  border-top: 0.24rem solid transparent;
  border-bottom: 0.24rem solid transparent;
  border-left: 0.38rem solid var(--infra-diagram-muted);
  content: "";
}

.critical-path__lane {
  grid-column: 2 / 14;
  align-self: end;
  height: 1px;
  background: var(--infra-diagram-line);
}

.critical-path__lane-label--cuda,
.critical-path__lane--cuda {
  grid-row: 2;
}

.critical-path__lane-label--tensor,
.critical-path__lane--tensor {
  grid-row: 3;
}

.critical-path__lane-label--communication,
.critical-path__lane--communication {
  grid-row: 4;
}

.critical-path__block {
  z-index: 1;
  display: flex;
  min-width: 0;
  height: 3rem;
  align-self: center;
  align-items: center;
  justify-content: center;
  padding: 0.35rem 0.5rem;
  border: 1.5px solid currentcolor;
  border-radius: 0.58rem;
  font-family: var(--code-font-family);
  font-size: 0.71rem;
  font-weight: 700;
  line-height: 1.32;
  text-align: center;
}

.critical-path__block--planning {
  grid-column: 2 / 5;
  grid-row: 2;
  color: var(--infra-planning);
  background: color-mix(in srgb, var(--infra-planning) 11%, transparent);
}

.critical-path__block--weight {
  grid-column: 5 / 7;
  grid-row: 4;
  color: var(--infra-weight-flow);
  background: color-mix(in srgb, var(--infra-weight-flow) 11%, transparent);
}

.critical-path__block--token {
  grid-column: 7 / 10;
  grid-row: 4;
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 11%, transparent);
}

.critical-path__block--compute {
  grid-column: 10 / 12;
  grid-row: 3;
  color: var(--infra-execution);
  background: color-mix(in srgb, var(--infra-execution) 11%, transparent);
}

.critical-path__block--combine {
  grid-column: 12 / 14;
  grid-row: 4;
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 8%, transparent);
}

.critical-path__caption {
  display: grid;
  gap: 0.3rem;
  margin-top: 1.1rem;
  color: var(--infra-diagram-muted);
  font-size: 0.79rem;
  line-height: 1.55;
  text-align: center;
}

.critical-path__caption strong {
  color: var(--vp-c-text-1);
  font-weight: 650;
}

@media print {
  .critical-path {
    break-inside: avoid;
  }
}
</style>
