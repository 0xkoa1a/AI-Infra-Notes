<template>
  <figure class="balance-iterations" aria-label="MoonEP 主实例组盈亏配对的两轮状态转移">
    <div class="balance-iterations__flow">
      <section class="balance-iterations__state">
        <div class="balance-iterations__state-label">初始</div>
        <div class="balance-iterations__rows">
          <div><span>A</span><b>−20</b></div>
          <div><span>B</span><b>+10</b></div>
          <div><span>C</span><b>+10</b></div>
        </div>
      </section>

      <div class="balance-iterations__transition" aria-hidden="true">
        <svg viewBox="0 0 72 24">
          <path d="M2 12h62" />
          <path d="m56 6 8 6-8 6" />
        </svg>
        <span>B → A 搬 20</span>
      </div>

      <section class="balance-iterations__state">
        <div class="balance-iterations__state-label">第 1 轮</div>
        <div class="balance-iterations__rows">
          <div><span>A</span><b>0</b></div>
          <div><span>B</span><b>−10</b></div>
          <div><span>C</span><b>+10</b></div>
        </div>
      </section>

      <div class="balance-iterations__transition" aria-hidden="true">
        <svg viewBox="0 0 72 24">
          <path d="M2 12h62" />
          <path d="m56 6 8 6-8 6" />
        </svg>
        <span>C → B 搬 10</span>
      </div>

      <section class="balance-iterations__state balance-iterations__state--done">
        <div class="balance-iterations__state-label">第 2 轮 · 退出</div>
        <div class="balance-iterations__rows">
          <div><span>A</span><b>0</b></div>
          <div><span>B</span><b>0</b></div>
          <div><span>C</span><b>0</b></div>
        </div>
      </section>
    </div>

    <figcaption>每轮把接收方的全部亏空归零；发送方允许暂时转负，因此最多经过 <i>R</i> 轮即可结束。</figcaption>
  </figure>
</template>

<style scoped>
.balance-iterations {
  margin: 1.5rem 0 1.8rem;
  color: var(--vp-c-text-1);
}

.balance-iterations__flow {
  display: grid;
  grid-template-columns: minmax(6rem, 0.8fr) minmax(5.4rem, 0.72fr) minmax(6rem, 0.8fr) minmax(5.4rem, 0.72fr) minmax(6rem, 0.8fr);
  align-items: center;
  gap: 0.4rem;
}

.balance-iterations__state {
  min-width: 0;
  padding: 0.45rem 0.65rem 0.55rem;
  border-top: 2px solid var(--infra-diagram-line);
  border-bottom: 1px solid var(--infra-diagram-line);
}

.balance-iterations__state--done {
  border-top-color: var(--infra-execution);
}

.balance-iterations__state-label {
  margin-bottom: 0.4rem;
  color: var(--infra-diagram-muted);
  font-size: 0.77rem;
  font-weight: 700;
  text-align: center;
}

.balance-iterations__rows {
  display: grid;
  gap: 0.22rem;
  font-family: var(--code-font-family);
  font-size: 0.78rem;
}

.balance-iterations__rows div {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
}

.balance-iterations__rows b {
  font-weight: 700;
}

.balance-iterations__transition {
  display: grid;
  justify-items: center;
  gap: 0.15rem;
  color: var(--infra-reroute);
  font-size: 0.68rem;
  line-height: 1.3;
  text-align: center;
}

.balance-iterations__transition svg {
  width: 100%;
  overflow: visible;
  fill: none;
  stroke: currentcolor;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 1.5;
}

.balance-iterations figcaption {
  margin-top: 0.85rem;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  line-height: 1.5;
  text-align: center;
}

@media print {
  .balance-iterations {
    break-inside: avoid;
  }
}
</style>
