<template>
  <figure class="weight-layout" aria-label="MoonEP 在每个 Rank 上统一映射的 E 加 B 行权重布局">
    <div class="weight-layout__legend" aria-hidden="true">
      <span class="weight-layout__swatch weight-layout__swatch--local">本地主实例 E/R</span>
      <span class="weight-layout__swatch weight-layout__swatch--remote">对称映射的远端主实例</span>
      <span class="weight-layout__swatch weight-layout__swatch--prefetch">本地 B 个预取槽位</span>
    </div>

    <div class="weight-layout__rows">
      <div class="weight-layout__row">
        <strong>Rank 0</strong>
        <span class="weight-layout__segment weight-layout__segment--local">本地主实例</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--prefetch">B 预取槽位</span>
      </div>
      <div class="weight-layout__row">
        <strong>Rank 1</strong>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--local">本地主实例</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--prefetch">B 预取槽位</span>
      </div>
      <div class="weight-layout__row">
        <strong>Rank 2</strong>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--local">本地主实例</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--prefetch">B 预取槽位</span>
      </div>
      <div class="weight-layout__row">
        <strong>Rank 3</strong>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--remote">对称映射</span>
        <span class="weight-layout__segment weight-layout__segment--local">本地主实例</span>
        <span class="weight-layout__segment weight-layout__segment--prefetch">B 预取槽位</span>
      </div>
    </div>

    <figcaption>
      所有 Rank 看到同一份连续的 <code>[E+B]</code> 行号视图：行 <code>[0,E)</code> 是全局主实例，行 <code>[E,E+B)</code> 是本地预取槽位；只有本地主实例段和预取槽位占用本地物理显存。
    </figcaption>
  </figure>
</template>

<style scoped>
.weight-layout {
  margin: 1.5rem 0 1.8rem;
  color: var(--vp-c-text-1);
}

.weight-layout__legend {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.35rem 1rem;
  margin-bottom: 0.75rem;
  color: var(--infra-diagram-muted);
  font-size: 0.7rem;
}

.weight-layout__swatch::before {
  display: inline-block;
  width: 0.55rem;
  height: 0.55rem;
  margin-right: 0.3rem;
  border-radius: 0.1rem;
  content: "";
  vertical-align: 0.02rem;
}

.weight-layout__swatch--local::before {
  background: var(--infra-placement);
}

.weight-layout__swatch--remote::before {
  background: var(--infra-diagram-line);
}

.weight-layout__swatch--prefetch::before {
  background: var(--infra-weight-flow);
}

.weight-layout__rows {
  display: grid;
  gap: 0.35rem;
}

.weight-layout__row {
  display: grid;
  grid-template-columns: 4.2rem repeat(4, minmax(0, 1fr)) minmax(5rem, 1.05fr);
  align-items: stretch;
  min-height: 2.45rem;
  border-bottom: 1px solid var(--infra-diagram-line);
}

.weight-layout__row strong {
  display: flex;
  align-items: center;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.72rem;
}

.weight-layout__segment {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
  padding: 0.3rem 0.2rem;
  border: 1px solid var(--vp-c-bg);
  font-size: 0.68rem;
  line-height: 1.25;
  text-align: center;
}

.weight-layout__segment--local {
  color: var(--infra-placement);
  background: color-mix(in srgb, var(--infra-placement) 14%, transparent);
}

.weight-layout__segment--remote {
  color: var(--infra-diagram-muted);
  background: color-mix(in srgb, var(--infra-diagram-line) 32%, transparent);
}

.weight-layout__segment--prefetch {
  color: var(--infra-weight-flow);
  background: color-mix(in srgb, var(--infra-weight-flow) 14%, transparent);
}

.weight-layout figcaption {
  margin-top: 0.85rem;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  line-height: 1.5;
  text-align: center;
}

@media print {
  .weight-layout {
    break-inside: avoid;
  }
}
</style>
