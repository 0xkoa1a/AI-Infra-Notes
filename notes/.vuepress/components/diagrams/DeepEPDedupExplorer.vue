<script setup lang="ts">
import { computed, ref } from "vue"

import { deepEpRanks, defaultDedupExperts, nodeForRank, rankForExpert } from "./deepEpExample"

const selectedExperts = ref<number[]>([...defaultDedupExperts])

const toggleExpert = (expert: number): void => {
  selectedExperts.value = selectedExperts.value.includes(expert)
    ? selectedExperts.value.filter((item) => item !== expert)
    : [...selectedExperts.value, expert].sort((left, right) => left - right)
}

const targetRanks = computed(() => [...new Set(selectedExperts.value.map(rankForExpert))])
const targetNodes = computed(() => [...new Set(targetRanks.value.map(nodeForRank))])
const remoteNodes = computed(() => targetNodes.value.filter((node) => node !== 0))

const assignmentLabels = computed(() => selectedExperts.value.map((expert) => `r0:E${expert}`))
const rankLabels = computed(() => targetRanks.value.map((rank) => `r${rank}`))
const remoteNodeLabels = computed(() => remoteNodes.value.map((node) => `Node ${node}`))

const summary = computed(
  () =>
    `${selectedExperts.value.length} 个 Expert assignment，${targetRanks.value.length} 份目标 Rank hidden-state，${remoteNodes.value.length} 份跨节点 RDMA 传输。`,
)
</script>

<template>
  <figure class="dedup-explorer" aria-labelledby="dedup-explorer-caption">
    <div class="dedup-explorer__source">
      <span>source</span>
      <strong>Node 0 · r0 / GPU0</strong>
      <small>选择 routed tokens</small>
    </div>

    <div class="dedup-explorer__nodes">
      <section v-for="node in [0, 1]" :key="node">
        <div class="dedup-explorer__node-label">Node {{ node }}</div>
        <div class="dedup-explorer__ranks">
          <div v-for="rank in deepEpRanks.filter((item) => item.node === node)" :key="rank.rank">
            <span>r{{ rank.rank }} / GPU{{ rank.gpu }}</span>
            <div>
              <label v-for="expert in rank.experts" :key="expert">
                <input
                  type="checkbox"
                  :checked="selectedExperts.includes(expert)"
                  :aria-label="`选择 Expert ${expert}`"
                  @change="toggleExpert(expert)"
                />
                <span>E{{ expert }}</span>
              </label>
            </div>
          </div>
        </div>
      </section>
    </div>

    <div class="dedup-explorer__flow" aria-live="polite">
      <section>
        <div class="dedup-explorer__flow-label">
          <span>Expert 输入</span><strong>{{ selectedExperts.length }}</strong>
        </div>
        <div class="dedup-explorer__chips">
          <span v-for="label in assignmentLabels" :key="label">{{ label }}</span>
          <i v-if="assignmentLabels.length === 0">未选择目标</i>
        </div>
      </section>

      <div class="dedup-explorer__arrow">
        <span>Per-Rank 去重</span>
        <svg viewBox="0 0 64 16" aria-hidden="true"><path d="M1 8h56m-7-6 7 6-7 6" /></svg>
      </div>

      <section>
        <div class="dedup-explorer__flow-label">
          <span>hidden-state 副本</span><strong>{{ targetRanks.length }}</strong>
        </div>
        <div class="dedup-explorer__chips">
          <span v-for="label in rankLabels" :key="label">{{ label }}</span>
          <i v-if="rankLabels.length === 0">0 份</i>
        </div>
      </section>

      <div class="dedup-explorer__arrow">
        <span>RDMA-domain 聚合</span>
        <svg viewBox="0 0 64 16" aria-hidden="true"><path d="M1 8h56m-7-6 7 6-7 6" /></svg>
      </div>

      <section>
        <div class="dedup-explorer__flow-label">
          <span>跨节点传输</span><strong>{{ remoteNodes.length }}</strong>
        </div>
        <div class="dedup-explorer__chips dedup-explorer__chips--rdma">
          <span v-for="label in remoteNodeLabels" :key="label">RDMA → {{ label }}</span>
          <i v-if="remoteNodeLabels.length === 0">仅本节点</i>
        </div>
      </section>
    </div>

    <div class="dedup-explorer__summary">{{ summary }}</div>

    <figcaption id="dedup-explorer-caption">
      Expert assignment、目标 Rank 输入和跨节点网络副本是三种不同计数口径；relay 暂存不算额外 Expert 输入。
    </figcaption>
  </figure>
</template>

<style scoped>
.dedup-explorer {
  margin: 1.6rem 0 2rem;
  color: var(--vp-c-text-1);
}

.dedup-explorer *,
.dedup-explorer *::before,
.dedup-explorer *::after {
  box-sizing: border-box;
}

.dedup-explorer__source {
  display: flex;
  align-items: baseline;
  gap: 0.55rem;
  margin-bottom: 0.75rem;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--infra-diagram-line);
}

.dedup-explorer__source span,
.dedup-explorer__source small {
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.68rem;
}

.dedup-explorer__source strong {
  color: var(--infra-token-flow);
  font-family: var(--code-font-family);
  font-size: 0.8rem;
}

.dedup-explorer__source small {
  margin-inline-start: auto;
}

.dedup-explorer__nodes {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
}

.dedup-explorer__nodes section {
  padding: 0.6rem;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 0.55rem;
}

.dedup-explorer__node-label {
  margin-bottom: 0.45rem;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.68rem;
  font-weight: 700;
}

.dedup-explorer__ranks {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.38rem;
}

.dedup-explorer__ranks > div {
  min-width: 0;
  padding: 0.38rem;
  border-top: 2px solid var(--infra-diagram-line);
  background: color-mix(in srgb, var(--infra-diagram-line) 9%, transparent);
}

.dedup-explorer__ranks > div > span {
  display: block;
  margin-bottom: 0.28rem;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.59rem;
}

.dedup-explorer__ranks > div > div {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.22rem;
}

.dedup-explorer label {
  position: relative;
  cursor: pointer;
}

.dedup-explorer input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
}

.dedup-explorer label span {
  display: grid;
  min-height: 1.75rem;
  place-items: center;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 0.25rem;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.62rem;
}

.dedup-explorer input:checked + span {
  border-color: var(--infra-token-flow);
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 10%, transparent);
  font-weight: 700;
}

.dedup-explorer input:focus-visible + span {
  outline: 2px solid var(--infra-token-flow);
  outline-offset: 2px;
}

.dedup-explorer__flow {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 5.5rem minmax(0, 0.8fr) 5.5rem minmax(0, 0.8fr);
  align-items: stretch;
  gap: 0.45rem;
  margin-top: 1rem;
}

.dedup-explorer__flow section {
  min-width: 0;
  padding-top: 0.4rem;
  border-top: 2px solid var(--infra-token-flow);
}

.dedup-explorer__flow section:last-of-type {
  border-top-style: dashed;
}

.dedup-explorer__flow-label {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.35rem;
  margin-bottom: 0.42rem;
}

.dedup-explorer__flow-label span {
  color: var(--infra-diagram-muted);
  font-size: 0.65rem;
}

.dedup-explorer__flow-label strong {
  color: var(--infra-token-flow);
  font-family: var(--code-font-family);
  font-size: 1rem;
}

.dedup-explorer__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.22rem;
}

.dedup-explorer__chips span,
.dedup-explorer__chips i {
  padding: 0.18rem 0.3rem;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 0.25rem;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.58rem;
  font-style: normal;
}

.dedup-explorer__chips--rdma span {
  border-style: dashed;
  border-color: var(--infra-token-flow);
  color: var(--infra-token-flow);
}

.dedup-explorer__arrow {
  display: grid;
  align-content: center;
  justify-items: center;
  gap: 0.22rem;
  color: var(--infra-diagram-muted);
  font-size: 0.55rem;
  text-align: center;
}

.dedup-explorer__arrow svg {
  width: 100%;
  fill: none;
  stroke: currentcolor;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-width: 1.25;
}

.dedup-explorer__summary {
  margin-top: 0.85rem;
  color: var(--infra-token-flow);
  font-family: var(--code-font-family);
  font-size: 0.72rem;
  font-weight: 700;
  text-align: center;
}

.dedup-explorer figcaption {
  margin-top: 0.75rem;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  line-height: 1.5;
  text-align: center;
}

@media (max-width: 700px) {
  .dedup-explorer__nodes {
    grid-template-columns: 1fr;
  }

  .dedup-explorer__flow {
    grid-template-columns: 1fr;
  }

  .dedup-explorer__arrow {
    min-height: 2.5rem;
  }

  .dedup-explorer__arrow svg {
    width: 3.5rem;
    transform: rotate(90deg);
  }
}

@media (prefers-reduced-motion: reduce) {
  .dedup-explorer * {
    transition: none !important;
  }
}

@media print {
  .dedup-explorer {
    break-inside: avoid;
  }
}
</style>
