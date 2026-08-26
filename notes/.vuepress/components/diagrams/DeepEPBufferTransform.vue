<script setup lang="ts">
import { computed, ref } from "vue"

import { compactSlots, expertSlotIndices, r6Counts, r6Offsets } from "./deepEpExample"

type FocusKey = `slot:${string}` | `source:${number}` | `expert:${12 | 13}`

const pinnedFocus = ref<FocusKey>("slot:focus")
const hoverFocus = ref<FocusKey | null>(null)
const activeFocus = computed(() => hoverFocus.value ?? pinnedFocus.value)

const setHover = (focus: FocusKey | null): void => {
  hoverFocus.value = focus
}

const pin = (focus: FocusKey): void => {
  pinnedFocus.value = focus
}

const isSlotActive = (slotIndex: number): boolean => {
  const focus = activeFocus.value
  const slot = compactSlots[slotIndex]
  if (focus.startsWith("slot:")) return focus === `slot:${slot.id}`
  if (focus.startsWith("source:")) return focus === `source:${slot.source}`
  return focus === `expert:${slot.expert}`
}

const isSourceActive = (source: number): boolean => {
  const focus = activeFocus.value
  if (focus.startsWith("source:")) return focus === `source:${source}`
  if (focus.startsWith("slot:")) {
    const slot = compactSlots.find((item) => `slot:${item.id}` === focus)
    return slot?.source === source
  }
  return compactSlots.some((slot) => slot.source === source && focus === `expert:${slot.expert}`)
}

const isExpertActive = (expert: 12 | 13): boolean => {
  const focus = activeFocus.value
  if (focus.startsWith("expert:")) return focus === `expert:${expert}`
  if (focus.startsWith("slot:")) {
    const slot = compactSlots.find((item) => `slot:${item.id}` === focus)
    return slot?.expert === expert
  }
  const source = Number(focus.split(":")[1])
  return compactSlots.some((slot) => slot.source === source && slot.expert === expert)
}

const mappingText = computed(() => {
  const focus = activeFocus.value
  if (focus.startsWith("slot:")) {
    const slotIndex = compactSlots.findIndex((slot) => `slot:${slot.id}` === focus)
    const slot = compactSlots[slotIndex]
    const expertIndex = expertSlotIndices[slot.expert].indexOf(slotIndex as never)
    if (slot.focus) {
      return "compact[3] → E13[1] → r1:E13 / Top-K slot 1"
    }
    return `compact[${slotIndex}] → E${slot.expert}[${expertIndex}] → source r${slot.source}`
  }
  if (focus.startsWith("source:")) {
    const source = Number(focus.split(":")[1])
    const indices = compactSlots.flatMap((slot, index) => (slot.source === source ? [index] : []))
    return `r${source} 的连续区间 [${r6Offsets[source]}, ${r6Offsets[source + 1]}) → compact[${indices.join(", ")}]`
  }
  const expert = Number(focus.split(":")[1]) as 12 | 13
  return `E${expert} ← compact[${expertSlotIndices[expert].join(", ")}] → 连续 Grouped GEMM 输入`
})
</script>

<template>
  <figure class="buffer-transform" aria-labelledby="buffer-transform-caption">
    <div class="buffer-transform__section-label">1 · Count Exchange</div>
    <div class="buffer-transform__counts" aria-label="r6 从各 source rank 收到的 assignment 数量">
      <button
        v-for="(count, source) in r6Counts"
        :key="source"
        type="button"
        :class="{ 'is-active': isSourceActive(source) }"
        @mouseenter="setHover(`source:${source}`)"
        @mouseleave="setHover(null)"
        @focus="setHover(`source:${source}`)"
        @blur="setHover(null)"
        @click="pin(`source:${source}`)"
      >
        <span>r{{ source }}</span><strong>{{ count }}</strong>
      </button>
    </div>

    <div class="buffer-transform__section-label">2 · Prefix-Sum</div>
    <div class="buffer-transform__offsets" aria-label="前缀和 offsets">
      <span v-for="(offset, index) in r6Offsets" :key="index">
        <small>{{ index === r6Offsets.length - 1 ? 'end' : `r${index}` }}</small>{{ offset }}
      </span>
    </div>

    <div class="buffer-transform__section-label">3 · by-source compact buffer</div>
    <div class="buffer-transform__slots">
      <button
        v-for="(slot, index) in compactSlots"
        :key="slot.id"
        type="button"
        :class="{ 'is-active': isSlotActive(index), 'is-focus': slot.focus }"
        @mouseenter="setHover(`slot:${slot.id}`)"
        @mouseleave="setHover(null)"
        @focus="setHover(`slot:${slot.id}`)"
        @blur="setHover(null)"
        @click="pin(`slot:${slot.id}`)"
      >
        <small>[{{ index }}]</small>
        <span>{{ slot.label }}</span>
      </button>
    </div>

    <div class="buffer-transform__mapping" aria-live="polite">
      <span>permute / inverse permute</span>
      <strong>{{ mappingText }}</strong>
    </div>

    <div class="buffer-transform__section-label">4 · by-expert inputs</div>
    <div class="buffer-transform__experts">
      <div v-for="expert in ([12, 13] as const)" :key="expert">
        <button
          type="button"
          class="buffer-transform__expert-label"
          :class="{ 'is-active': isExpertActive(expert) }"
          @mouseenter="setHover(`expert:${expert}`)"
          @mouseleave="setHover(null)"
          @focus="setHover(`expert:${expert}`)"
          @blur="setHover(null)"
          @click="pin(`expert:${expert}`)"
        >
          E{{ expert }}
        </button>
        <div class="buffer-transform__expert-slots">
          <button
            v-for="(slotIndex, expertIndex) in expertSlotIndices[expert]"
            :key="slotIndex"
            type="button"
            :class="{ 'is-active': isSlotActive(slotIndex), 'is-focus': compactSlots[slotIndex].focus }"
            @mouseenter="setHover(`slot:${compactSlots[slotIndex].id}`)"
            @mouseleave="setHover(null)"
            @focus="setHover(`slot:${compactSlots[slotIndex].id}`)"
            @blur="setHover(null)"
            @click="pin(`slot:${compactSlots[slotIndex].id}`)"
          >
            <small>[{{ expertIndex }}]</small>
            <span>{{ compactSlots[slotIndex].label }}</span>
          </button>
        </div>
      </div>
    </div>

    <figcaption id="buffer-transform-caption">
      通信按 Source 连续，计算按 Expert 连续；选择任意 Source、slot 或 Expert 可追踪同一份数据的地址变化。
    </figcaption>
  </figure>
</template>

<style scoped>
.buffer-transform {
  margin: 1.6rem 0 2rem;
  color: var(--vp-c-text-1);
}

.buffer-transform *,
.buffer-transform *::before,
.buffer-transform *::after {
  box-sizing: border-box;
}

.buffer-transform button {
  color: inherit;
  font: inherit;
}

.buffer-transform__section-label {
  margin: 0.85rem 0 0.38rem;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.buffer-transform__counts {
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  gap: 0.25rem;
}

.buffer-transform__counts button,
.buffer-transform__slots button,
.buffer-transform__expert-slots button,
.buffer-transform__expert-label {
  border: 1px solid var(--infra-diagram-line);
  background: transparent;
  cursor: pointer;
}

.buffer-transform__counts button {
  display: flex;
  min-width: 0;
  align-items: center;
  justify-content: space-between;
  gap: 0.25rem;
  padding: 0.38rem 0.42rem;
  font-family: var(--code-font-family);
  font-size: 0.68rem;
}

.buffer-transform__counts span {
  color: var(--infra-diagram-muted);
}

.buffer-transform__counts strong {
  color: var(--infra-planning);
}

.buffer-transform__offsets {
  display: grid;
  grid-template-columns: repeat(9, minmax(0, 1fr));
  border-top: 1px solid var(--infra-planning);
}

.buffer-transform__offsets span {
  position: relative;
  display: grid;
  justify-items: start;
  gap: 0.12rem;
  padding-top: 0.32rem;
  color: var(--infra-planning);
  font-family: var(--code-font-family);
  font-size: 0.7rem;
}

.buffer-transform__offsets span::before {
  position: absolute;
  top: -0.2rem;
  left: 0;
  width: 1px;
  height: 0.4rem;
  background: var(--infra-planning);
  content: "";
}

.buffer-transform__offsets small {
  color: var(--infra-diagram-muted);
  font-size: 0.55rem;
}

.buffer-transform__slots {
  display: grid;
  grid-template-columns: repeat(10, minmax(0, 1fr));
  gap: 0.22rem;
}

.buffer-transform__slots button,
.buffer-transform__expert-slots button {
  display: grid;
  min-width: 0;
  min-height: 3.45rem;
  place-items: center;
  gap: 0.12rem;
  padding: 0.26rem 0.16rem;
  font-family: var(--code-font-family);
  font-size: 0.58rem;
  line-height: 1.2;
  text-align: center;
}

.buffer-transform__slots small,
.buffer-transform__expert-slots small {
  color: var(--infra-diagram-muted);
  font-size: 0.52rem;
}

.buffer-transform__slots strong,
.buffer-transform__expert-slots strong {
  color: var(--infra-token-flow);
  font-size: 0.57rem;
}

.buffer-transform__counts .is-active,
.buffer-transform__slots .is-active,
.buffer-transform__expert-slots .is-active,
.buffer-transform__expert-label.is-active {
  border-color: var(--infra-token-flow);
  color: var(--vp-c-text-1);
  background: color-mix(in srgb, var(--infra-token-flow) 10%, transparent);
}

.buffer-transform__slots .is-focus,
.buffer-transform__expert-slots .is-focus {
  border-bottom-width: 3px;
  border-bottom-color: var(--infra-token-flow);
}

.buffer-transform__mapping {
  display: grid;
  justify-items: center;
  gap: 0.25rem;
  padding: 0.85rem 0.5rem;
  text-align: center;
}

.buffer-transform__mapping::before {
  width: 1px;
  height: 1rem;
  background: var(--infra-token-flow);
  content: "";
}

.buffer-transform__mapping span {
  color: var(--infra-diagram-muted);
  font-size: 0.65rem;
}

.buffer-transform__mapping strong {
  color: var(--infra-token-flow);
  font-family: var(--code-font-family);
  font-size: 0.72rem;
}

.buffer-transform__experts {
  display: grid;
  gap: 0.5rem;
}

.buffer-transform__experts > div {
  display: grid;
  grid-template-columns: 3.6rem minmax(0, 1fr);
  gap: 0.35rem;
}

.buffer-transform__expert-label {
  color: var(--infra-execution);
  font-family: var(--code-font-family);
  font-size: 0.72rem;
  font-weight: 700;
}

.buffer-transform__expert-slots {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 0.22rem;
}

.buffer-transform figcaption {
  margin-top: 0.85rem;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  line-height: 1.5;
  text-align: center;
}

.buffer-transform button:focus-visible {
  outline: 2px solid var(--infra-token-flow);
  outline-offset: 2px;
}

@media (max-width: 640px) {
  .buffer-transform__counts {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .buffer-transform__offsets {
    grid-template-columns: repeat(5, minmax(0, 1fr));
    row-gap: 0.55rem;
  }

  .buffer-transform__offsets span:nth-child(n + 6) {
    border-top: 1px solid var(--infra-planning);
  }

  .buffer-transform__slots {
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }

  .buffer-transform__experts > div {
    grid-template-columns: 1fr;
  }

  .buffer-transform__expert-label {
    padding: 0.35rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .buffer-transform * {
    transition: none !important;
  }
}

@media print {
  .buffer-transform {
    break-inside: avoid;
  }
}
</style>
