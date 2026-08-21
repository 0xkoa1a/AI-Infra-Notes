<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue"

import { BarChart } from "echarts/charts"
import { AriaComponent, GridComponent, LegendComponent, MarkLineComponent, TooltipComponent } from "echarts/components"
import { init, use } from "echarts/core"
import { CanvasRenderer } from "echarts/renderers"

use([AriaComponent, BarChart, GridComponent, LegendComponent, MarkLineComponent, TooltipComponent, CanvasRenderer])

const RANK_COUNT = 8
const EXPERT_COUNT = 16
const EXPERTS_PER_RANK = 2
const STEP_COUNT = 64

const chartElement = ref<HTMLDivElement | null>(null)
const timeIndex = ref(16)
const targetRho = ref(2)
const replicaLag = ref(4)

let chart: ReturnType<typeof init> | undefined
let resizeObserver: ResizeObserver | undefined
let themeObserver: MutationObserver | undefined

const seededNoise = (step: number, expert: number): number => {
  const value = Math.sin((Math.floor(step / 2) + 1) * 17.17 + (expert + 3) * 41.73) * 43758.5453
  return value - Math.floor(value)
}

const trace = Array.from({ length: STEP_COUNT }, (_, step) => {
  const weights = Array.from(
    { length: EXPERT_COUNT },
    (_, expert) =>
      0.35 +
      Math.exp(
        1.45 * Math.sin(step * 0.06 + expert * 1.31) +
          Math.sin(step * 0.023 + expert * 2.17) +
          0.12 * (seededNoise(step, expert) - 0.5),
      ),
  )
  const weightSum = weights.reduce((sum, value) => sum + value, 0)

  return weights.map((weight) => weight / weightSum)
})

const currentExpertLoads = computed(() => trace[timeIndex.value])
const currentRankLoads = computed(() =>
  Array.from({ length: RANK_COUNT }, (_, rank) =>
    currentExpertLoads.value
      .slice(rank * EXPERTS_PER_RANK, (rank + 1) * EXPERTS_PER_RANK)
      .reduce((sum, value) => sum + value, 0),
  ),
)
const meanRankLoad = computed(() => 1 / RANK_COUNT)
const maxRankLoad = computed(() => Math.max(...currentRankLoads.value))
const maxRank = computed(() => currentRankLoads.value.indexOf(maxRankLoad.value))
const rankImbalance = computed(() => maxRankLoad.value / meanRankLoad.value)
const meanExpertLoad = computed(() => 1 / EXPERT_COUNT)
const maxExpertLoad = computed(() => Math.max(...currentExpertLoads.value))
const maxExpert = computed(() => currentExpertLoads.value.indexOf(maxExpertLoad.value))
const expertPeakSkew = computed(() => maxExpertLoad.value / meanExpertLoad.value)
const safeTargetRho = computed(() => {
  const value = Number(targetRho.value)
  return Number.isFinite(value) ? Math.min(RANK_COUNT, Math.max(1, value)) : 2
})
const safeReplicaLag = computed(() => {
  const value = Math.round(Number(replicaLag.value))
  return Number.isFinite(value) ? Math.min(STEP_COUNT - 1, Math.max(1, value)) : 4
})
const targetRankLoad = computed(() => safeTargetRho.value * meanRankLoad.value)
const locationCounts = computed(() =>
  currentExpertLoads.value.map((load) => Math.max(1, Math.ceil(load / targetRankLoad.value - 1e-10))),
)
const extraLocationCounts = computed(() => locationCounts.value.map((count) => Math.max(0, count - 1)))
const maxLocationCount = computed(() => Math.max(...locationCounts.value))
const replicaExpertCount = computed(() => extraLocationCounts.value.filter((count) => count > 0).length)
const replicaLocationPressure = computed(() => extraLocationCounts.value.reduce((sum, count) => sum + count, 0))

const smoothReplicaPressure = (loads: number[]): number[] =>
  loads.map((load) => Math.max(0, load / targetRankLoad.value - 1))

const replicaOverlap = computed<number | null>(() => {
  if (timeIndex.value < safeReplicaLag.value) return null

  const currentPressure = smoothReplicaPressure(currentExpertLoads.value)
  const previousPressure = smoothReplicaPressure(trace[timeIndex.value - safeReplicaLag.value])
  const currentTotal = currentPressure.reduce((sum, value) => sum + value, 0)
  if (currentTotal <= Number.EPSILON) return null

  return (
    currentPressure.reduce((sum, value, expert) => sum + Math.min(value, previousPressure[expert]), 0) /
    currentTotal
  )
})

const formatPercent = (value: number): string => `${(value * 100).toFixed(1)}%`
const formatOverlap = (value: number | null): string => (value === null ? "—" : formatPercent(value))

const withAlpha = (color: string, alpha: number): string => {
  const match = color.match(/^#([0-9a-f]{6})$/i)
  if (!match) return color
  return `${color}${Math.round(alpha * 255)
    .toString(16)
    .padStart(2, "0")}`
}

const readColor = (name: string, fallback: string): string => {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return value || fallback
}

const renderChart = (): void => {
  if (!chart) return

  const colors = {
    accent: readColor("--infra-token-flow", "#2369d8"),
    grid: readColor("--infra-diagram-line", "#c2c2c4"),
    hot: readColor("--infra-weight-flow", "#c45a18"),
    muted: readColor("--infra-diagram-muted", "#60646c"),
    target: readColor("--infra-planning", "#6f42c1"),
    text: readColor("--vp-c-text-1", "#2c2c30"),
  }
  const yMax = Math.min(1, Math.max(0.4, maxRankLoad.value * 1.18, targetRankLoad.value * 1.18))
  const evenExpertLoads = currentRankLoads.value.map((_, rank) => currentExpertLoads.value[rank * 2])
  const oddExpertLoads = currentRankLoads.value.map((_, rank) => currentExpertLoads.value[rank * 2 + 1])
  const segmentColor = (rank: number, alpha: number): string =>
    withAlpha(currentRankLoads.value[rank] > targetRankLoad.value ? colors.hot : colors.accent, alpha)

  chart.setOption(
    {
      animationDurationUpdate: 180,
      color: [withAlpha(colors.accent, 0.92), withAlpha(colors.accent, 0.5)],
      aria: {
        enabled: true,
        description: `Step ${timeIndex.value + 1} 的 16 Expert、8 Rank 负载分布。每个 Rank 包含两个 Expert；最大 Rank 负载 ${formatPercent(maxRankLoad.value)}，Rank 负载不均衡度 ${rankImbalance.value.toFixed(2)}，Expert 负载不均衡度 ${expertPeakSkew.value.toFixed(2)}。`,
      },
      grid: { bottom: 46, containLabel: true, left: 20, right: 18, top: 54 },
      legend: {
        data: ["Expert 2r", "Expert 2r+1"],
        itemHeight: 9,
        itemWidth: 16,
        textStyle: { color: colors.muted, fontSize: 11 },
        top: 10,
      },
      tooltip: {
        backgroundColor: readColor("--vp-c-bg", "#ffffff"),
        borderColor: colors.grid,
        textStyle: { color: colors.text, fontSize: 12 },
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (items: Array<{ dataIndex: number; seriesName: string; value: number }>) => {
          const rank = items[0]?.dataIndex ?? 0
          return [
            `Rank ${rank}：${formatPercent(currentRankLoads.value[rank])}`,
            `Expert ${rank * 2}：${formatPercent(currentExpertLoads.value[rank * 2])}`,
            `Expert ${rank * 2 + 1}：${formatPercent(currentExpertLoads.value[rank * 2 + 1])}`,
          ].join("<br/>")
        },
      },
      xAxis: {
        axisLabel: { color: colors.muted, fontFamily: "monospace", fontSize: 11 },
        axisLine: { lineStyle: { color: colors.grid } },
        axisTick: { show: false },
        data: Array.from({ length: RANK_COUNT }, (_, rank) => `Rank ${rank}`),
        name: "Rank",
        nameGap: 28,
        nameLocation: "middle",
        nameTextStyle: { color: colors.muted, fontSize: 12 },
        type: "category",
      },
      yAxis: {
        axisLabel: {
          color: colors.muted,
          formatter: (value: number) => `${Math.round(value * 100)}%`,
          fontSize: 11,
        },
        axisLine: { show: false },
        axisTick: { show: false },
        max: yMax,
        min: 0,
        name: "Rank 负载（收到 token / 总 token）",
        nameGap: 42,
        nameLocation: "middle",
        nameTextStyle: { color: colors.muted, fontSize: 11 },
        splitLine: { lineStyle: { color: colors.grid, opacity: 0.55 } },
        type: "value",
      },
      series: [
        {
          barMaxWidth: 34,
          data: evenExpertLoads,
          itemStyle: {
            color: (params: { dataIndex: number }) => segmentColor(params.dataIndex, 0.92),
          },
          name: "Expert 2r",
          stack: "rank-load",
          type: "bar",
        },
        {
          barMaxWidth: 34,
          data: oddExpertLoads,
          itemStyle: {
            borderRadius: [4, 4, 0, 0],
            color: (params: { dataIndex: number }) => segmentColor(params.dataIndex, 0.5),
          },
          markLine: {
            animation: false,
            data: [{ yAxis: targetRankLoad.value }],
            label: {
              color: colors.target,
              formatter: `目标 ρ = ${safeTargetRho.value.toFixed(1)}`,
              position: "insideEndTop",
            },
            lineStyle: { color: colors.target, type: "dashed", width: 1.5 },
            symbol: "none",
          },
          name: "Expert 2r+1",
          stack: "rank-load",
          type: "bar",
        },
      ],
    },
    true,
  )
}

const normalizeTarget = (): void => {
  targetRho.value = safeTargetRho.value
}

const normalizeReplicaLag = (): void => {
  replicaLag.value = safeReplicaLag.value
}

onMounted(async () => {
  await nextTick()
  if (!chartElement.value) return

  chart = init(chartElement.value)
  renderChart()

  resizeObserver = new ResizeObserver(() => chart?.resize())
  resizeObserver.observe(chartElement.value)

  themeObserver = new MutationObserver(() => renderChart())
  themeObserver.observe(document.documentElement, { attributeFilter: ["data-theme"], attributes: true })
})

watch([timeIndex, safeTargetRho, safeReplicaLag], renderChart)

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  themeObserver?.disconnect()
  chart?.dispose()
})
</script>

<template>
  <figure class="rank-load-distribution">
    <div class="rank-load-distribution__body">
      <div
        ref="chartElement"
        class="rank-load-distribution__chart"
        :aria-label="`Step ${timeIndex + 1} 的 Rank 负载分布`"
        role="img"
      ></div>

      <aside class="rank-load-distribution__metrics" aria-live="polite">
        <section>
          <h4>Rank</h4>
          <dl>
            <div>
              <dt>max</dt>
              <dd>{{ formatPercent(maxRankLoad) }} <small>Rank {{ maxRank }}</small></dd>
            </div>
            <div>
              <dt>mean</dt>
              <dd>{{ formatPercent(meanRankLoad) }}</dd>
            </div>
            <div>
              <dt>负载不均衡度</dt>
              <dd>{{ rankImbalance.toFixed(2) }}×</dd>
            </div>
          </dl>
        </section>

        <section>
          <h4>Expert</h4>
          <dl>
            <div>
              <dt>max</dt>
              <dd>{{ formatPercent(maxExpertLoad) }} <small>Expert {{ maxExpert }}</small></dd>
            </div>
            <div>
              <dt>PeakSkew</dt>
              <dd>{{ expertPeakSkew.toFixed(2) }}×</dd>
            </div>
          </dl>
        </section>

        <section>
          <h4>Target-aware pressure</h4>
          <dl>
            <div>
              <dt>n<sub>max</sub><sup>loc</sup></dt>
              <dd>{{ maxLocationCount }}</dd>
            </div>
            <div>
              <dt>E<sub>replica</sub></dt>
              <dd>{{ replicaExpertCount }}</dd>
            </div>
            <div>
              <dt>K<sub>replica</sub></dt>
              <dd>{{ replicaLocationPressure }}</dd>
            </div>
            <div>
              <dt>ReplicaOverlap (Δ={{ safeReplicaLag }})</dt>
              <dd>{{ formatOverlap(replicaOverlap) }}</dd>
            </div>
          </dl>
        </section>
      </aside>
    </div>

    <div class="rank-load-distribution__controls">
      <label class="rank-load-distribution__time-control">
        <span>时间</span>
        <input v-model.number="timeIndex" :max="STEP_COUNT - 1" min="0" type="range" />
        <output>Step {{ timeIndex + 1 }} / {{ STEP_COUNT }}</output>
      </label>

      <label class="rank-load-distribution__target-control">
        <span>目标不均衡度</span>
        <input v-model.number="targetRho" :max="RANK_COUNT" min="1" step="0.1" type="number" @change="normalizeTarget" />
      </label>

      <label class="rank-load-distribution__lag-control">
        <span>ReplicaOverlap Δ</span>
        <input
          v-model.number="replicaLag"
          :max="STEP_COUNT - 1"
          min="1"
          step="1"
          type="number"
          @change="normalizeReplicaLag"
        />
      </label>
    </div>

    <figcaption>
      示例轨迹由固定种子在 Expert 粒度生成；Rank r 固定承载 Expert 2r 和 Expert 2r+1。橙色判断、Target-aware pressure
      与 ReplicaOverlap 均读取当前用户输入。
    </figcaption>
  </figure>
</template>

<style scoped>
.rank-load-distribution {
  margin: 1.5rem 0 2rem;
  color: var(--vp-c-text-1);
}

.rank-load-distribution *,
.rank-load-distribution *::before,
.rank-load-distribution *::after {
  box-sizing: border-box;
}

.rank-load-distribution__body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 13rem;
  align-items: stretch;
  border-top: 1px solid var(--infra-diagram-line);
  border-bottom: 1px solid var(--infra-diagram-line);
}

.rank-load-distribution__chart {
  width: 100%;
  height: 26rem;
  min-width: 0;
}

.rank-load-distribution__metrics {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 1rem;
  padding: 1rem 0 1rem 1rem;
  border-inline-start: 1px solid var(--infra-diagram-line);
}

.rank-load-distribution__metrics section + section {
  padding-top: 0.85rem;
  border-top: 1px solid color-mix(in srgb, var(--infra-diagram-line) 65%, transparent);
}

.rank-load-distribution__metrics h4 {
  margin: 0 0 0.48rem;
  color: var(--infra-diagram-muted);
  font-family: var(--code-font-family);
  font-size: 0.65rem;
  font-weight: 750;
  letter-spacing: 0.04em;
  line-height: 1.2;
  text-transform: uppercase;
}

.rank-load-distribution__metrics dl {
  display: grid;
  gap: 0.38rem;
  margin: 0;
}

.rank-load-distribution__metrics dl > div {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.65rem;
}

.rank-load-distribution__metrics dt,
.rank-load-distribution__metrics small {
  color: var(--infra-diagram-muted);
  font-size: 0.67rem;
  line-height: 1.25;
}

.rank-load-distribution__metrics dd {
  margin: 0;
  color: var(--vp-c-text-1);
  font-family: var(--code-font-family);
  font-size: 0.78rem;
  font-weight: 700;
  line-height: 1.25;
  text-align: end;
  white-space: nowrap;
}

.rank-load-distribution__metrics sub,
.rank-load-distribution__metrics sup {
  font-size: 0.72em;
}

.rank-load-distribution__controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 9.5rem 9.5rem;
  gap: 1rem;
  padding-top: 1rem;
}

.rank-load-distribution__controls label {
  color: var(--infra-diagram-muted);
  font-size: 0.72rem;
  font-weight: 650;
}

.rank-load-distribution__time-control {
  display: grid;
  grid-template-columns: 2.5rem minmax(0, 1fr) 7.3rem;
  align-items: center;
  gap: 0.65rem;
}

.rank-load-distribution__time-control input {
  width: 100%;
  accent-color: var(--infra-token-flow);
}

.rank-load-distribution__time-control output {
  color: var(--vp-c-text-1);
  font-family: var(--code-font-family);
  font-size: 0.72rem;
  text-align: end;
}

.rank-load-distribution__target-control,
.rank-load-distribution__lag-control {
  display: grid;
  gap: 0.35rem;
}

.rank-load-distribution__target-control input,
.rank-load-distribution__lag-control input {
  width: 100%;
  height: 2rem;
  padding: 0 0.5rem;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 0.35rem;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  font-family: var(--code-font-family);
  font-size: 0.78rem;
}

.rank-load-distribution__target-control input:focus-visible,
.rank-load-distribution__lag-control input:focus-visible {
  border-color: var(--infra-planning);
  outline: 2px solid color-mix(in srgb, var(--infra-planning) 25%, transparent);
  outline-offset: 1px;
}

.rank-load-distribution figcaption {
  margin-top: 0.75rem;
  color: var(--infra-diagram-muted);
  font-size: 0.74rem;
  line-height: 1.45;
  text-align: center;
}

@media print {
  .rank-load-distribution {
    break-inside: avoid;
  }

  .rank-load-distribution__controls {
    display: none;
  }
}
</style>
