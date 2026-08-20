<template>
  <figure class="ep-map" aria-label="EP 负载均衡的两个基本动作与两种典型范式">
    <figcaption class="ep-map__thesis">
      两个动作不变；分歧在于 placement 使用什么信号、何时生效
    </figcaption>

    <div class="ep-map__actions">
      <div class="ep-map__action ep-map__action--placement">
        <strong>布局 <span>placement</span></strong>
        <p>容量在哪里？</p>
      </div>

      <div class="ep-map__feasible-domain" aria-label="布局划定分流可行域">
        <span>划定分流可行域</span>
        <svg viewBox="0 0 180 20" aria-hidden="true">
          <path d="M1 10h172" />
          <path d="m166 3 7 7-7 7" />
        </svg>
      </div>

      <div class="ep-map__action ep-map__action--reroute">
        <strong>分流 <span>reroute</span></strong>
        <p>流量往哪里走？</p>
      </div>
    </div>

    <div class="ep-map__paradigms">
      <section class="ep-map__paradigm ep-map__paradigm--historical" aria-label="历史布局加实时分流">
        <div class="ep-map__paradigm-title">历史布局 + 实时分流</div>
        <div class="ep-map__lane">
          <span class="ep-map__source">历史 / 已观察负载</span>
          <svg class="ep-map__lane-arrow" viewBox="0 0 28 14" aria-hidden="true">
            <path d="M1 7h22" />
            <path d="m18 2 5 5-5 5" />
          </svg>
          <span class="ep-map__target ep-map__target--placement">低频或异步布局</span>
        </div>
        <div class="ep-map__lane">
          <span class="ep-map__source">当前 Router 输出</span>
          <svg class="ep-map__lane-arrow" viewBox="0 0 28 14" aria-hidden="true">
            <path d="M1 7h22" />
            <path d="m18 2 5 5-5 5" />
          </svg>
          <span class="ep-map__target ep-map__target--reroute">已生效布局内实时分流</span>
        </div>
      </section>

      <section class="ep-map__paradigm ep-map__paradigm--realtime" aria-label="全实时布局加实时分流">
        <div class="ep-map__paradigm-title">全实时布局 + 实时分流</div>
        <div class="ep-map__realtime-flow">
          <div class="ep-map__realtime-step">
            <span>当前 Router 输出</span>
            <b>+</b>
            <span>当前精确负载</span>
          </div>
          <svg class="ep-map__down-arrow" viewBox="0 0 14 28" aria-hidden="true">
            <path d="M7 1v22" />
            <path d="m2 18 5 5 5-5" />
          </svg>
          <div class="ep-map__realtime-step ep-map__realtime-step--joint">当前层联合决策</div>
          <svg class="ep-map__down-arrow" viewBox="0 0 14 28" aria-hidden="true">
            <path d="M7 1v22" />
            <path d="m2 18 5 5 5-5" />
          </svg>
          <div class="ep-map__realtime-step">
            <span class="ep-map__target--placement">当前布局 / 临时副本</span>
            <b>+</b>
            <span class="ep-map__target--reroute">当前 token 分流</span>
          </div>
        </div>
      </section>
    </div>
  </figure>
</template>

<style scoped>
.ep-map {
  display: flex;
  margin: 1.6rem 0 2rem;
  flex-direction: column;
  color: var(--vp-c-text-1);
}

.ep-map *,
.ep-map *::before,
.ep-map *::after {
  box-sizing: border-box;
}

.ep-map__actions {
  order: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(10rem, 1.2fr) minmax(0, 1fr);
  align-items: center;
  gap: 1.25rem;
  padding: 0.4rem 1.25rem 1.65rem;
}

.ep-map__action {
  min-width: 0;
  text-align: center;
}

.ep-map__action strong {
  display: block;
  font-size: 1.14rem;
  font-weight: 750;
  line-height: 1.35;
}

.ep-map__action strong span {
  font-size: 0.92em;
  font-weight: 650;
}

.ep-map__action p {
  margin: 0.42rem 0 0;
  color: var(--vp-c-text-1);
  font-size: 0.92rem;
  line-height: 1.45;
}

.ep-map__action--placement strong {
  color: var(--infra-placement);
}

.ep-map__action--reroute strong {
  color: var(--infra-reroute);
}

.ep-map__feasible-domain {
  display: grid;
  align-items: center;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  font-weight: 600;
  text-align: center;
}

.ep-map__feasible-domain svg,
.ep-map__lane-arrow,
.ep-map__down-arrow {
  overflow: visible;
  fill: none;
  stroke: currentcolor;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 1.5;
}

.ep-map__feasible-domain svg {
  width: 100%;
  height: 1.25rem;
  margin-top: 0.15rem;
}

.ep-map__thesis {
  order: 2;
  margin: 0;
  padding: 1.05rem 1rem 1.15rem;
  border-top: 1px solid var(--infra-diagram-line);
  color: var(--infra-diagram-muted);
  font-size: 0.86rem;
  font-weight: 600;
  line-height: 1.5;
  text-align: center;
}

.ep-map__paradigms {
  order: 3;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  padding-top: 0.35rem;
}

.ep-map__paradigm {
  min-width: 0;
  padding: 0.25rem 1.35rem 0.35rem;
}

.ep-map__paradigm + .ep-map__paradigm {
  border-inline-start: 1px solid var(--infra-diagram-line);
}

.ep-map__paradigm-title {
  margin-bottom: 1.05rem;
  font-size: 1rem;
  font-weight: 750;
  line-height: 1.4;
  text-align: center;
}

.ep-map__paradigm--historical .ep-map__paradigm-title {
  color: var(--infra-history);
}

.ep-map__paradigm--realtime .ep-map__paradigm-title {
  color: var(--infra-reroute);
}

.ep-map__lane {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 1.35rem minmax(0, 1fr);
  align-items: center;
  gap: 0.38rem;
  min-height: 2.75rem;
  font-size: 0.77rem;
  line-height: 1.42;
  text-align: center;
}

.ep-map__lane + .ep-map__lane {
  margin-top: 0.25rem;
}

.ep-map__lane-arrow {
  width: 1.35rem;
  color: var(--infra-diagram-muted);
}

.ep-map__source,
.ep-map__target {
  min-width: 0;
}

.ep-map__target--placement {
  color: var(--infra-placement);
  font-weight: 650;
}

.ep-map__target--reroute {
  color: var(--infra-reroute);
  font-weight: 650;
}

.ep-map__realtime-flow {
  display: grid;
  justify-items: center;
  font-size: 0.77rem;
  line-height: 1.42;
  text-align: center;
}

.ep-map__realtime-step {
  display: flex;
  max-width: 100%;
  align-items: center;
  justify-content: center;
  gap: 0.42rem;
}

.ep-map__realtime-step b {
  color: var(--infra-diagram-muted);
  font-weight: 500;
}

.ep-map__realtime-step--joint {
  color: var(--vp-c-text-1);
  font-weight: 700;
}

.ep-map__down-arrow {
  width: 0.88rem;
  height: 1.35rem;
  margin: 0.08rem 0;
  color: var(--infra-diagram-muted);
}

@media print {
  .ep-map {
    break-inside: avoid;
  }
}
</style>
