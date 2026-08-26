<script setup lang="ts">
import { computed, ref } from "vue"

import {
  compactSlots,
  deepEpAssignments,
  deepEpRanks,
  expertSlotIndices,
  pathSteps,
  r6Counts,
  type DeepEPMode,
} from "./deepEpExample"

const mode = ref<DeepEPMode>("throughput")
const step = ref(0)

const modeName = computed(() => (mode.value === "throughput" ? "High-Throughput" : "Low-Latency"))
const currentDescription = computed(() => {
  const descriptions: Record<DeepEPMode, readonly string[]> = {
    throughput: [
      "r1 上的原始 token 被展开为 routed tokens r1:E3 与 r1:E13。",
      "r6 交换精确 count，并用 prefix-sum 把 10 个 routed tokens 放入紧凑空间。",
      "r1:E13 先经同号 r5 / GPU1 跨节点 RDMA，再由 NVLink 转发到 r6 / GPU2。",
      "通信得到 by-source Buffer；本地 permute 后形成连续的 E12、E13 输入组。",
      "E3 与 E13 分别对自己的 routed token 执行 SwiGLU FFN。",
      "r1:E13 的结果沿分层路径返回 r1，再与本地 r1:E3 做加权 reduction。",
    ],
    latency: [
      "Router 结果不变：仍得到 routed tokens r1:E3 与 r1:E13。",
      "r6 预留 Expert × Source Rank × T_max 的稳定 RDMA 槽位，不等待精确 Shape。",
      "r1:E13 由 r1 / GPU1 通过 Pure RDMA 直接写入 r6 / GPU2。",
      "原始 RDMA 槽位按 Expert 和 Source Rank 分区，有效 token 随后 pack 成 Expert-major 输出。",
      "E3 与 E13 分别执行 SwiGLU FFN；recv_count 指定每个本地 Expert 的有效输入数量。",
      "r1:E13 的结果依据 Source 信息直接 RDMA 返回 r1，并与 r1:E3 加权归约。",
    ],
  }
  return descriptions[mode.value][step.value]
})

const setMode = (nextMode: DeepEPMode): void => {
  mode.value = nextMode
}

const setStep = (nextStep: number): void => {
  step.value = Math.min(pathSteps.length - 1, Math.max(0, nextStep))
}

const rankState = (rank: number): string[] => {
  const states: string[] = []
  if (rank === 1) states.push("source")
  if (rank === 6) states.push("target")
  if (mode.value === "throughput" && step.value === 2 && rank === 5) states.push("relay")
  if (mode.value === "throughput" && step.value === 5 && rank === 2) states.push("relay")
  if (step.value === 4 && (rank === 1 || rank === 6)) states.push("compute")
  return states
}

const isActiveExpert = (expert: number): boolean =>
  deepEpAssignments.some((assignment) => assignment.expert === expert)
</script>

<template>
  <figure class="deep-path" aria-labelledby="deep-path-caption">
    <div class="deep-path__toolbar">
      <div class="deep-path__mode" aria-label="DeepEP 模式">
        <button
          v-for="item in (['throughput', 'latency'] as DeepEPMode[])"
          :key="item"
          type="button"
          :aria-pressed="mode === item"
          @click="setMode(item)"
        >
          {{ item === "throughput" ? "High-Throughput" : "Low-Latency" }}
        </button>
      </div>

      <div class="deep-path__step-nav" aria-label="路径步骤">
        <button
          v-for="(label, index) in pathSteps"
          :key="label"
          type="button"
          :aria-current="step === index ? 'step' : undefined"
          @click="setStep(index)"
        >
          <span>{{ index + 1 }}</span>{{ label }}
        </button>
      </div>
    </div>

    <div class="deep-path__topology" :data-step="step">
      <section v-for="node in [0, 1]" :key="node" class="deep-path__node">
        <div class="deep-path__node-label">Node {{ node }}</div>
        <div class="deep-path__ranks">
          <div
            v-for="rank in deepEpRanks.filter((item) => item.node === node)"
            :key="rank.rank"
            class="deep-path__rank"
            :class="rankState(rank.rank).map((state) => `deep-path__rank--${state}`)"
          >
            <div class="deep-path__rank-name">
              <strong>r{{ rank.rank }}</strong>
              <span>GPU{{ rank.gpu }}</span>
            </div>
            <div class="deep-path__experts">
              <span
                v-for="expert in rank.experts"
                :key="expert"
                :class="{ 'deep-path__expert--active': isActiveExpert(expert) }"
              >
                E{{ expert }}
              </span>
            </div>
            <div v-if="rank.rank === 1" class="deep-path__token">r1:E3 · r1:E13</div>
            <div
              v-if="mode === 'throughput' && ((rank.rank === 5 && step === 2) || (rank.rank === 2 && step === 5))"
              class="deep-path__relay"
            >
              relay
            </div>
          </div>
        </div>
      </section>
    </div>

    <div class="deep-path__stage" aria-live="polite">
      <template v-if="step === 0">
        <div class="deep-path__assignments">
          <div v-for="assignment in deepEpAssignments" :key="assignment.label">
            <span>{{ assignment.remote ? 'remote routed token' : 'local routed token' }}</span>
            <strong>{{ assignment.label }}</strong>
            <small>Top-K slot {{ assignment.slot }} · weight {{ assignment.weight }} · target r{{ assignment.rank }}</small>
          </div>
        </div>
      </template>

      <template v-else-if="step === 1">
        <div v-if="mode === 'throughput'" class="deep-path__space deep-path__space--compact">
          <div class="deep-path__space-title">
            <strong>精确 Shape</strong>
            <span>counts = [{{ r6Counts.join(', ') }}] · total = 10</span>
          </div>
          <div class="deep-path__compact-bar" aria-label="长度为 10 的 compact receive buffer">
            <span
              v-for="slot in compactSlots"
              :key="slot.id"
              :class="{ 'deep-path__compact-token--focus': slot.focus }"
            >
              {{ slot.label }}
            </span>
          </div>
        </div>
        <div v-else class="deep-path__space deep-path__space--reserved">
          <div class="deep-path__space-title">
            <strong>容量上界</strong>
            <span>Expert × Source Rank × T_max</span>
          </div>
          <div v-for="expert in [12, 13]" :key="expert" class="deep-path__capacity-row">
            <b>E{{ expert }}</b>
            <span v-for="rank in 8" :key="rank">r{{ rank - 1 }} × T<sub>max</sub></span>
          </div>
        </div>
      </template>

      <template v-else-if="step === 2 || step === 5">
        <div class="deep-path__transport-stage" :class="{ 'deep-path__transport-stage--combine': step === 5 }">
          <div
            class="deep-path__transport"
            :aria-label="step === 2 ? 'Dispatch 传输路径' : 'Combine 返回路径'"
          >
            <div class="deep-path__hop">
              <strong>{{ step === 2 ? 'r1 / GPU1' : 'r6 / GPU2' }}</strong>
              <span>{{ step === 2 ? 'r1:E13' : 'y[r1:E13]' }}</span>
            </div>
            <template v-if="mode === 'throughput'">
              <div class="deep-path__link deep-path__link--rdma">
                <span>{{ step === 2 ? 'Node 0 → Node 1 · RDMA' : 'Node 1 → Node 0 · RDMA' }}</span>
              </div>
              <div class="deep-path__hop deep-path__hop--relay">
                <strong>{{ step === 2 ? 'r5 / GPU1' : 'r2 / GPU2' }}</strong>
                <span>same-local-rank relay</span>
              </div>
              <div class="deep-path__link deep-path__link--nvlink">
                <span>{{ step === 2 ? 'Node 1 内 · NVLink' : 'Node 0 内 · NVLink' }}</span>
              </div>
            </template>
            <div v-else class="deep-path__link deep-path__link--pure">
              <span>{{ step === 2 ? 'Node 0 → Node 1 · Pure RDMA' : 'Node 1 → Node 0 · Pure RDMA' }}</span>
            </div>
            <div class="deep-path__hop">
              <strong>{{ step === 2 ? 'r6 / GPU2' : 'r1 / GPU1' }}</strong>
              <span>{{ step === 2 ? 'r1:E13 → E13 input' : 'returned y[r1:E13]' }}</span>
            </div>
          </div>

          <div v-if="step === 5" class="deep-path__reduction">
            <div><small>local branch</small><strong>0.35 · y<sub>r1:E3</sub></strong></div>
            <span>+</span>
            <div><small>returned branch</small><strong>0.65 · y<sub>r1:E13</sub></strong></div>
            <span>→</span>
            <div class="deep-path__reduction-result"><small>reduction @ r1</small><strong>output</strong></div>
          </div>
        </div>
      </template>

      <template v-else-if="step === 3">
        <div v-if="mode === 'throughput'" class="deep-path__layout">
          <div class="deep-path__layout-label">by-source</div>
          <div class="deep-path__slot-row">
            <span
              v-for="(slot, index) in compactSlots"
              :key="slot.id"
              :class="{ 'deep-path__slot--focus': slot.focus }"
            >
              <small>{{ index }}</small>{{ slot.label }}
            </span>
          </div>
          <div class="deep-path__permute">local permute ↓</div>
          <div class="deep-path__expert-groups">
            <div v-for="expert in ([12, 13] as const)" :key="expert">
              <strong>E{{ expert }}</strong>
              <span
                v-for="slotIndex in expertSlotIndices[expert]"
                :key="slotIndex"
                :class="{ 'deep-path__slot--focus': compactSlots[slotIndex].focus }"
              >
                {{ compactSlots[slotIndex].label }}
              </span>
            </div>
          </div>
        </div>
        <div v-else class="deep-path__layout deep-path__layout--latency">
          <div class="deep-path__layout-label">RDMA slots → packed Expert-major</div>
          <div class="deep-path__latency-layout">
            <div v-for="expert in ([12, 13] as const)" :key="expert">
              <strong>E{{ expert }} · recv_count = 5</strong>
              <span
                v-for="slotIndex in expertSlotIndices[expert]"
                :key="slotIndex"
                :class="{ 'deep-path__slot--focus': compactSlots[slotIndex].focus }"
              >
                {{ compactSlots[slotIndex].label }}
              </span>
              <i>reserved capacity</i>
            </div>
          </div>
        </div>
      </template>

      <template v-else-if="step === 4">
        <div class="deep-path__compute">
          <div>
            <span>E3 @ r1</span>
            <strong>
              y<sub>r1:E3</sub> = W<sub>2</sub><sup>(3)</sup>[SiLU(W<sub>1</sub><sup>(3)</sup>x<sub>r1:E3</sub>)
              ⊙ W<sub>3</sub><sup>(3)</sup>x<sub>r1:E3</sub>]
            </strong>
            <small>local routed token · r1:E3</small>
          </div>
          <div class="deep-path__compute-parallel">parallel</div>
          <div>
            <span>E13 @ r6</span>
            <strong>
              y<sub>r1:E13</sub> = W<sub>2</sub><sup>(13)</sup>[SiLU(W<sub>1</sub><sup>(13)</sup>x<sub>r1:E13</sub>)
              ⊙ W<sub>3</sub><sup>(13)</sup>x<sub>r1:E13</sub>]
            </strong>
            <small>remote routed token · r1:E13</small>
          </div>
        </div>
      </template>
    </div>

    <div class="deep-path__footer">
      <button type="button" :disabled="step === 0" @click="setStep(step - 1)">← 上一步</button>
      <p><strong>{{ modeName }} · {{ pathSteps[step] }}</strong>{{ currentDescription }}</p>
      <button type="button" :disabled="step === pathSteps.length - 1" @click="setStep(step + 1)">下一步 →</button>
    </div>

    <figcaption id="deep-path-caption">
      同一个原始 token 的两个 routed tokens；模式差异来自接收空间、传输路径与接收布局。
    </figcaption>
  </figure>
</template>

<style scoped>
.deep-path {
  margin: 1.6rem 0 2rem;
  color: var(--vp-c-text-1);
}

.deep-path *,
.deep-path *::before,
.deep-path *::after {
  box-sizing: border-box;
}

.deep-path button {
  color: inherit;
  font: inherit;
}

.deep-path__toolbar {
  display: grid;
  gap: 0.85rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--infra-diagram-line);
}

.deep-path__mode,
.deep-path__step-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.deep-path__mode button,
.deep-path__step-nav button {
  border: 1px solid transparent;
  border-radius: 0.45rem;
  background: transparent;
  cursor: pointer;
}

.deep-path__mode button {
  padding: 0.38rem 0.65rem;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  font-weight: 700;
}

.deep-path__mode button[aria-pressed="true"] {
  border-color: var(--infra-token-flow);
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 9%, transparent);
}

.deep-path__step-nav {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
}

.deep-path__step-nav button {
  display: grid;
  min-width: 0;
  justify-items: center;
  gap: 0.2rem;
  padding: 0.35rem 0.18rem;
  color: var(--infra-diagram-muted);
  font-size: 0.68rem;
  line-height: 1.2;
}

.deep-path__step-nav button span {
  display: grid;
  width: 1.35rem;
  height: 1.35rem;
  place-items: center;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 50%;
  font-family: var(--code-font-family);
  font-size: 0.7rem;
}

.deep-path__step-nav button[aria-current="step"] {
  color: var(--infra-token-flow);
  font-weight: 700;
}

.deep-path__step-nav button[aria-current="step"] span {
  border-color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 11%, transparent);
}

.deep-path__topology {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  padding: 1rem 0;
}

.deep-path__node {
  min-width: 0;
  padding: 0.65rem;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 0.65rem;
}

.deep-path__node-label {
  margin-bottom: 0.55rem;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.72rem;
  font-weight: 700;
}

.deep-path__ranks {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.45rem;
}

.deep-path__rank {
  position: relative;
  min-width: 0;
  padding: 0.45rem;
  border-top: 2px solid var(--infra-diagram-line);
  background: color-mix(in srgb, var(--infra-diagram-line) 10%, transparent);
}

.deep-path__rank--source {
  border-top-color: var(--infra-token-flow);
}

.deep-path__rank--target {
  border-top-color: var(--infra-execution);
}

.deep-path__rank--relay {
  border: 1px dashed var(--infra-token-flow);
  border-top-width: 2px;
}

.deep-path__rank--compute {
  background: color-mix(in srgb, var(--infra-execution) 9%, transparent);
}

.deep-path__rank-name {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.35rem;
  font-family: var(--code-font-family);
}

.deep-path__rank-name strong {
  font-size: 0.76rem;
}

.deep-path__rank-name span {
  color: var(--infra-diagram-muted);
  font-size: 0.63rem;
}

.deep-path__experts {
  display: flex;
  gap: 0.28rem;
  margin-top: 0.38rem;
}

.deep-path__experts span {
  flex: 1;
  padding: 0.17rem 0.1rem;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 0.25rem;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.62rem;
  text-align: center;
}

.deep-path__experts .deep-path__expert--active {
  border-color: var(--infra-execution);
  color: var(--infra-execution);
  font-weight: 700;
}

.deep-path__token,
.deep-path__relay {
  margin-top: 0.35rem;
  color: var(--infra-token-flow);
  font-family: var(--code-font-family);
  font-size: 0.63rem;
  font-weight: 700;
}

.deep-path__relay {
  color: var(--infra-diagram-muted);
}

.deep-path__stage {
  display: grid;
  min-height: 12.5rem;
  align-items: center;
  padding: 1rem;
  border-top: 1px solid var(--infra-diagram-line);
  border-bottom: 1px solid var(--infra-diagram-line);
}

.deep-path__assignments,
.deep-path__compute {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.85rem;
}

.deep-path__assignments > div,
.deep-path__compute > div:not(.deep-path__compute-parallel) {
  display: grid;
  gap: 0.35rem;
  padding: 0.75rem;
  border-inline-start: 3px solid var(--infra-token-flow);
}

.deep-path__assignments span,
.deep-path__compute span {
  color: var(--infra-token-flow);
  font-family: var(--code-font-family);
  font-size: 0.7rem;
  font-weight: 700;
}

.deep-path__assignments small,
.deep-path__compute small {
  color: var(--infra-diagram-muted);
  font-size: 0.7rem;
}

.deep-path__space {
  display: grid;
  gap: 0.8rem;
}

.deep-path__space-title {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
}

.deep-path__space-title strong {
  color: var(--infra-planning);
}

.deep-path__space-title span {
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.7rem;
}

.deep-path__compact-bar,
.deep-path__slot-row {
  display: grid;
  grid-template-columns: repeat(10, minmax(0, 1fr));
  gap: 0.22rem;
}

.deep-path__compact-bar span {
  display: grid;
  min-height: 2.3rem;
  place-items: center;
  border: 1px solid var(--infra-token-flow);
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 8%, transparent);
  font-family: var(--code-font-family);
  font-size: 0.58rem;
  line-height: 1.2;
  text-align: center;
}

.deep-path__compact-bar .deep-path__compact-token--focus {
  border-bottom-width: 3px;
  color: var(--vp-c-text-1);
  background: color-mix(in srgb, var(--infra-token-flow) 14%, transparent);
}

.deep-path__capacity-row {
  display: grid;
  grid-template-columns: 2.8rem repeat(8, minmax(0, 1fr));
  gap: 0.2rem;
  align-items: stretch;
}

.deep-path__capacity-row b {
  display: grid;
  place-items: center;
  color: var(--infra-execution);
  font-family: var(--code-font-family);
  font-size: 0.7rem;
}

.deep-path__capacity-row span {
  display: grid;
  min-height: 2.1rem;
  place-items: center;
  border: 1px dashed var(--infra-diagram-line);
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.57rem;
  text-align: center;
}

.deep-path__transport {
  display: grid;
  grid-template-columns: minmax(6rem, 1fr) minmax(4.5rem, 0.7fr) minmax(6rem, 1fr) minmax(4.5rem, 0.7fr) minmax(6rem, 1fr);
  align-items: center;
  gap: 0.35rem;
}

.deep-path__transport-stage {
  display: grid;
  gap: 1.2rem;
}

.deep-path__transport:has(.deep-path__link--pure) {
  grid-template-columns: minmax(6rem, 1fr) minmax(7rem, 1.1fr) minmax(6rem, 1fr);
}

.deep-path__hop {
  display: grid;
  justify-items: center;
  gap: 0.22rem;
  padding: 0.65rem 0.35rem;
  border-top: 2px solid var(--infra-token-flow);
  font-family: var(--code-font-family);
  text-align: center;
}

.deep-path__hop span {
  color: var(--infra-diagram-muted);
  font-size: 0.65rem;
}

.deep-path__hop--relay {
  border: 1px dashed var(--infra-token-flow);
}

.deep-path__link {
  position: relative;
  height: 1px;
  background: var(--infra-token-flow);
}

.deep-path__link::after {
  position: absolute;
  right: -0.05rem;
  top: -0.23rem;
  border-top: 0.25rem solid transparent;
  border-bottom: 0.25rem solid transparent;
  border-left: 0.4rem solid var(--infra-token-flow);
  content: "";
}

.deep-path__link span {
  position: absolute;
  left: 50%;
  bottom: 0.35rem;
  padding: 0 0.2rem;
  color: var(--infra-token-flow);
  background: var(--vp-c-bg);
  font-family: var(--code-font-family);
  font-size: 0.61rem;
  font-weight: 700;
  text-align: center;
  transform: translateX(-50%);
  white-space: nowrap;
}

.deep-path__reduction {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto minmax(0, 0.8fr);
  align-items: center;
  gap: 0.45rem;
  padding-top: 1rem;
  border-top: 1px solid var(--infra-diagram-line);
}

.deep-path__reduction > div {
  display: grid;
  min-width: 0;
  justify-items: center;
  gap: 0.18rem;
  padding: 0.45rem;
  text-align: center;
}

.deep-path__reduction small {
  color: var(--infra-diagram-muted);
  font-size: 0.61rem;
}

.deep-path__reduction strong {
  color: var(--infra-token-flow);
  font-family: var(--code-font-family);
  font-size: 0.69rem;
}

.deep-path__reduction-result {
  border-inline-start: 3px solid var(--infra-execution);
}

.deep-path__link--rdma {
  background: repeating-linear-gradient(90deg, var(--infra-token-flow) 0 0.35rem, transparent 0.35rem 0.55rem);
}

.deep-path__link--nvlink {
  height: 2px;
}

.deep-path__layout {
  display: grid;
  gap: 0.55rem;
}

.deep-path__layout-label,
.deep-path__permute {
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.68rem;
  font-weight: 700;
  text-align: center;
}

.deep-path__slot-row > span,
.deep-path__expert-groups span,
.deep-path__latency-layout span {
  display: grid;
  min-width: 0;
  min-height: 2.5rem;
  place-items: center;
  padding: 0.2rem;
  border: 1px solid var(--infra-diagram-line);
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.58rem;
  line-height: 1.2;
  text-align: center;
}

.deep-path__slot-row small {
  font-size: 0.52rem;
}

.deep-path__slot-row .deep-path__slot--focus,
.deep-path__expert-groups .deep-path__slot--focus,
.deep-path__latency-layout .deep-path__slot--focus {
  border-color: var(--infra-token-flow);
  color: var(--vp-c-text-1);
  background: color-mix(in srgb, var(--infra-token-flow) 11%, transparent);
}

.deep-path__expert-groups,
.deep-path__latency-layout {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
}

.deep-path__expert-groups > div,
.deep-path__latency-layout > div {
  display: grid;
  grid-template-columns: 2.4rem repeat(5, minmax(0, 1fr));
  gap: 0.18rem;
  align-items: stretch;
}

.deep-path__expert-groups strong,
.deep-path__latency-layout strong {
  display: grid;
  place-items: center;
  color: var(--infra-execution);
  font-family: var(--code-font-family);
  font-size: 0.68rem;
}

.deep-path__latency-layout > div {
  grid-template-columns: minmax(6rem, 1.2fr) repeat(5, minmax(0, 1fr)) minmax(4rem, 0.8fr);
}

.deep-path__latency-layout i {
  display: grid;
  place-items: center;
  border: 1px dashed var(--infra-diagram-line);
  color: var(--infra-diagram-muted);
  font-size: 0.57rem;
  text-align: center;
}

.deep-path__compute {
  align-items: center;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
}

.deep-path__compute-parallel {
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.64rem;
  text-transform: uppercase;
}

.deep-path__compute > div:not(.deep-path__compute-parallel) {
  border-inline-start-color: var(--infra-execution);
}

.deep-path__compute > div > strong {
  overflow-wrap: anywhere;
  font-family: var(--code-font-family);
  font-size: 0.66rem;
  font-weight: 600;
  line-height: 1.55;
}

.deep-path__footer {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.7rem;
  padding-top: 0.85rem;
}

.deep-path__footer button {
  padding: 0.28rem 0;
  border: 0;
  color: var(--infra-token-flow);
  background: transparent;
  font-size: 0.72rem;
  font-weight: 700;
  cursor: pointer;
}

.deep-path__footer button:disabled {
  color: var(--infra-diagram-muted);
  cursor: default;
  opacity: 0.45;
}

.deep-path__footer p {
  display: grid;
  gap: 0.18rem;
  margin: 0;
  font-size: 0.72rem;
  line-height: 1.42;
  text-align: center;
}

.deep-path__footer p strong {
  color: var(--infra-token-flow);
}

.deep-path figcaption {
  margin-top: 0.8rem;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  line-height: 1.5;
  text-align: center;
}

@media (max-width: 700px) {
  .deep-path__step-nav {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }

  .deep-path__topology,
  .deep-path__assignments,
  .deep-path__compute,
  .deep-path__expert-groups,
  .deep-path__latency-layout {
    grid-template-columns: 1fr;
  }

  .deep-path__stage {
    padding-inline: 0;
  }

  .deep-path__transport,
  .deep-path__transport:has(.deep-path__link--pure) {
    grid-template-columns: 1fr;
    gap: 0.75rem;
  }

  .deep-path__link {
    width: 1px;
    height: 1.8rem;
    justify-self: center;
    background: var(--infra-token-flow);
  }

  .deep-path__link::after {
    right: auto;
    left: -0.23rem;
    top: auto;
    bottom: -0.05rem;
    border-top: 0.4rem solid var(--infra-token-flow);
    border-right: 0.25rem solid transparent;
    border-bottom: 0;
    border-left: 0.25rem solid transparent;
  }

  .deep-path__link span {
    left: 0.55rem;
    bottom: 50%;
    transform: translateY(50%);
  }

  .deep-path__compute-parallel {
    justify-self: center;
  }

  .deep-path__reduction {
    grid-template-columns: 1fr;
  }

  .deep-path__compact-bar,
  .deep-path__slot-row {
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }

  .deep-path__capacity-row {
    grid-template-columns: 2.8rem repeat(4, minmax(0, 1fr));
  }

  .deep-path__capacity-row span:nth-of-type(n + 5) {
    grid-row: 2;
  }

  .deep-path__capacity-row span:nth-of-type(5) {
    grid-column: 2;
  }

  .deep-path__footer {
    grid-template-columns: auto auto;
  }

  .deep-path__footer p {
    grid-column: 1 / -1;
    grid-row: 1;
  }
}

@media (prefers-reduced-motion: reduce) {
  .deep-path * {
    scroll-behavior: auto !important;
    transition: none !important;
  }
}

@media print {
  .deep-path {
    break-inside: avoid;
  }

  .deep-path__toolbar,
  .deep-path__footer button {
    display: none;
  }
}
</style>
