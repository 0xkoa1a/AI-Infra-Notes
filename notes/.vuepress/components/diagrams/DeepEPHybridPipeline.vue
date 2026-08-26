<template>
  <figure class="hybrid-pipeline" aria-labelledby="hybrid-pipeline-caption">
    <div class="hybrid-pipeline__timeline" role="img" aria-label="Hybrid-EP 相邻 chunk 的 warp specialization 流水线">
      <div class="hybrid-pipeline__corner">执行角色</div>
      <div v-for="time in 6" :key="time" class="hybrid-pipeline__tick" :style="{ gridColumn: time + 1 }">
        t{{ time - 1 }}
      </div>

      <div class="hybrid-pipeline__lane-label" style="grid-row: 2">Notify / control</div>
      <div class="hybrid-pipeline__lane" style="grid-row: 2"></div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--control" style="grid-column: 2 / 4; grid-row: 2">
        count / barrier
      </div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--control" style="grid-column: 5 / 7; grid-row: 2">
        tail signal
      </div>

      <div class="hybrid-pipeline__lane-label" style="grid-row: 3">Scale-out warps</div>
      <div class="hybrid-pipeline__lane" style="grid-row: 3"></div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--tma hybrid-pipeline__block--upper" style="grid-column: 2 / 3; grid-row: 3">
        TMA c0
      </div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--rdma hybrid-pipeline__block--lower" style="grid-column: 3 / 5; grid-row: 3">
        RDMA c0
      </div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--tma hybrid-pipeline__block--upper" style="grid-column: 4 / 5; grid-row: 3">
        TMA c1
      </div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--rdma hybrid-pipeline__block--lower" style="grid-column: 5 / 7; grid-row: 3">
        RDMA c1
      </div>

      <div class="hybrid-pipeline__lane-label" style="grid-row: 4">Forward warps</div>
      <div class="hybrid-pipeline__lane" style="grid-row: 4"></div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--forward hybrid-pipeline__block--upper" style="grid-column: 3 / 5; grid-row: 4">
        NVLink c0
      </div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--forward hybrid-pipeline__block--lower" style="grid-column: 5 / 7; grid-row: 4">
        NVLink c1
      </div>

      <div class="hybrid-pipeline__lane-label" style="grid-row: 5">Copy epilogue</div>
      <div class="hybrid-pipeline__lane" style="grid-row: 5"></div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--copy hybrid-pipeline__block--upper" style="grid-column: 4 / 6; grid-row: 5">
        pack c0
      </div>
      <div class="hybrid-pipeline__block hybrid-pipeline__block--copy hybrid-pipeline__block--lower" style="grid-column: 6 / 8; grid-row: 5">
        pack c1
      </div>
    </div>

    <div class="hybrid-pipeline__legend" aria-hidden="true">
      <span class="hybrid-pipeline__legend-tma">TMA staging</span>
      <span class="hybrid-pipeline__legend-rdma">scale-out / RDMA</span>
      <span class="hybrid-pipeline__legend-forward">scale-up / NVLink</span>
      <span class="hybrid-pipeline__legend-copy">layout epilogue</span>
    </div>

    <figcaption id="hybrid-pipeline-caption">
      示意相邻 chunk 在不同 warp 角色间形成流水；横向宽度表达先后与重叠关系，不代表实测时长。
    </figcaption>
  </figure>
</template>

<style scoped>
.hybrid-pipeline {
  margin: 1.6rem 0 2rem;
  color: var(--vp-c-text-1);
}

.hybrid-pipeline *,
.hybrid-pipeline *::before,
.hybrid-pipeline *::after {
  box-sizing: border-box;
}

.hybrid-pipeline__timeline {
  display: grid;
  grid-template-columns: 7.4rem repeat(6, minmax(0, 1fr));
  grid-template-rows: 1.4rem repeat(4, 4.4rem);
  gap: 0 0.25rem;
}

.hybrid-pipeline__corner,
.hybrid-pipeline__tick,
.hybrid-pipeline__lane-label {
  display: flex;
  align-items: center;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.66rem;
  font-weight: 700;
}

.hybrid-pipeline__corner {
  grid-column: 1;
  grid-row: 1;
  align-items: start;
}

.hybrid-pipeline__tick {
  grid-row: 1;
  align-items: start;
  justify-content: center;
  border-top: 1px solid var(--infra-diagram-line);
  padding-top: 0.22rem;
}

.hybrid-pipeline__lane-label {
  grid-column: 1;
  padding-inline-end: 0.45rem;
  line-height: 1.25;
}

.hybrid-pipeline__lane {
  grid-column: 2 / 8;
  align-self: center;
  height: 1px;
  background: var(--infra-diagram-line);
}

.hybrid-pipeline__block {
  z-index: 1;
  display: grid;
  min-width: 0;
  min-height: 1.55rem;
  align-self: center;
  place-items: center;
  padding: 0.2rem 0.25rem;
  border: 1px solid currentcolor;
  border-radius: 0.32rem;
  font-family: var(--code-font-family);
  font-size: 0.58rem;
  font-weight: 700;
  line-height: 1.15;
  text-align: center;
}

.hybrid-pipeline__block--upper {
  align-self: start;
  margin-top: 0.55rem;
}

.hybrid-pipeline__block--lower {
  align-self: end;
  margin-bottom: 0.55rem;
}

.hybrid-pipeline__block--control {
  color: var(--infra-planning);
  background: color-mix(in srgb, var(--infra-planning) 9%, var(--vp-c-bg));
}

.hybrid-pipeline__block--tma {
  color: var(--infra-planning);
  background: color-mix(in srgb, var(--infra-planning) 10%, var(--vp-c-bg));
}

.hybrid-pipeline__block--rdma {
  border-style: dashed;
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 8%, var(--vp-c-bg));
}

.hybrid-pipeline__block--forward {
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 10%, var(--vp-c-bg));
}

.hybrid-pipeline__block--copy {
  color: var(--infra-execution);
  background: color-mix(in srgb, var(--infra-execution) 10%, var(--vp-c-bg));
}

.hybrid-pipeline__legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.45rem 0.9rem;
  margin-top: 0.7rem;
  color: var(--infra-diagram-muted);
  font-size: 0.62rem;
}

.hybrid-pipeline__legend span::before {
  display: inline-block;
  width: 0.8rem;
  height: 0.35rem;
  margin-inline-end: 0.28rem;
  border: 1px solid currentcolor;
  content: "";
}

.hybrid-pipeline__legend-tma {
  color: var(--infra-planning);
}

.hybrid-pipeline__legend-rdma,
.hybrid-pipeline__legend-forward {
  color: var(--infra-token-flow);
}

.hybrid-pipeline__legend-rdma::before {
  border-style: dashed !important;
}

.hybrid-pipeline__legend-copy {
  color: var(--infra-execution);
}

.hybrid-pipeline figcaption {
  margin-top: 0.75rem;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  line-height: 1.5;
  text-align: center;
}

@media (max-width: 600px) {
  .hybrid-pipeline__timeline {
    grid-template-columns: 5.4rem repeat(6, minmax(0, 1fr));
    grid-template-rows: 1.35rem repeat(4, 4.8rem);
    gap: 0 0.12rem;
  }

  .hybrid-pipeline__lane-label,
  .hybrid-pipeline__corner {
    font-size: 0.56rem;
  }

  .hybrid-pipeline__block {
    overflow-wrap: anywhere;
    padding-inline: 0.1rem;
    font-size: 0.5rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .hybrid-pipeline * {
    transition: none !important;
  }
}

@media print {
  .hybrid-pipeline {
    break-inside: avoid;
  }
}
</style>
