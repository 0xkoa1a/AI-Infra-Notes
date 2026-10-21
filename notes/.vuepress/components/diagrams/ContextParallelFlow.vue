<script setup lang="ts">
import { computed, useId } from "vue"

const props = defineProps<{ kind: "mlp" | "attention" }>()
const id = useId()
const ranks = [0, 1]
const attention = computed(() => props.kind === "attention")
const height = computed(() => attention.value ? 1100 : 790)
const center = (rank: number) => 330 + rank * 350
const side = (rank: number) => rank === 0 ? 1 : -1
const queryX = (rank: number) => center(rank) - side(rank) * 75
const kvX = (rank: number) => center(rank) + side(rank) * 60
type Tensor = { key: string; name: string; x: number; y: number; cols: number; owner: number; stack?: boolean; state?: boolean }
const tensors = computed<Tensor[]>(() => ranks.flatMap(rank => {
  const x = center(rank)
  const tensor = (key: string, name: string, y: number, cols = 4, extra: Partial<Tensor> = {}): Tensor => ({ key: `${rank}-${key}`, name, x, y, cols, owner: rank, ...extra })
  return attention.value ? [
    tensor("x", "X", 230),
    tensor("q", "Q", 355, 4, { x: queryX(rank) }),
    tensor("kv", "K, V", 355, 4, { x: kvX(rank), stack: true }),
    tensor("a0", "A⁽⁰⁾", 510, 4, { state: true }),
    tensor("remote", "K, V", 625, 4, { x: kvX(rank), owner: 1 - rank, stack: true }),
    tensor("a1", "A⁽¹⁾", 790, 4, { state: true }),
    tensor("o", "O", 900),
    tensor("y", "Y", 1035),
  ] : [
    tensor("x", "X", 240),
    tensor("u", "U", 400, 8),
    tensor("z", "Z", 560, 8),
    tensor("y", "Y", 720),
  ]
}))
const mlpSteps = [
  { name: "W₁", y: 320, start: 287, end: 353 },
  { name: "Activation σ", y: 480, start: 447, end: 513 },
  { name: "W₂", y: 640, start: 607, end: 673 },
]
const weightGrids = computed(() => attention.value ? [
  { name: "WQKV", x: 30, y: 31, rows: 4, cols: 12, shape: "H×3H" },
  { name: "WO", x: 175, y: 31, rows: 4, cols: 4, shape: "H×H" },
] : [
  { name: "W₁", x: 40, y: 31, rows: 4, cols: 8, shape: "H×I" },
  { name: "W₂", x: 175, y: 21, rows: 8, cols: 4, shape: "I×H" },
])
</script>

<template>
  <figure class="cp-flow" :aria-label="`纯 CP 的 ${attention ? 'Ring Attention' : 'MLP'} 前向流程`">
    <div class="scroll" role="region" tabindex="0" aria-label="两个 Rank 的激活与完整权重，从上到下阅读，可横向滚动">
      <svg :viewBox="`0 0 880 ${height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ attention ? 'Ring CP：Q 留在本卡，KV 逐轮流入' : 'CP MLP：各自处理本地 token，全程无激活通信' }}</title>
        <desc :id="`${id}-desc`">两卡将四个 token 均分，每卡完整复制权重，激活的列维保持完整。{{ attention ? '图采用无因果 mask 的 MHA。QKV 投影后，先用本地 KV 更新本地 m、l、A 状态。随后两卡交换 KV，用同一份本地 Q 和收到的远端 KV 继续更新状态。A 除以 l 得到本地 O，再用完整 WO 得到本地 Y。每卡负责不同 token 的输出。Q 通过旁路连接两个 Block Attention，累积状态通过竖直箭头延续，KV 跨 Rank 传递。' : 'W1 将本地两个 token 投影到完整中间维度 I，逐元素激活后，完整 W2 将其投影回 H。各卡输出仍对应原来的 token。' }}虚线表示读取权重，各行按前向执行顺序展示主要激活。</desc>
        <defs>
          <marker :id="`${id}-neutral`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="neutral-head" /></marker>
          <marker v-for="rank in ranks" :id="`${id}-rank-${rank}`" :key="rank" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" :class="`arrowhead owner${rank}`" /></marker>
        </defs>
        <text x="505" y="24" class="label note">完整权重 · 每卡复制 · 全程保留</text>
        <text x="80" y="208" class="label note">前向数据流 ↓</text>
        <g v-for="rank in ranks" :key="rank">
          <rect :x="center(rank) - 155" y="40" width="310" :height="height - 50" rx="10" class="rank-panel" />
          <text :x="center(rank)" y="66" class="label strong">Rank {{ rank }}</text>
          <g :transform="`translate(${center(rank) - 125},80)`">
            <rect width="250" height="100" rx="6" class="weight-bank" />
            <g v-for="w in weightGrids" :key="w.name">
              <text :x="w.x + w.cols * 2.5" y="17" class="label weight-name">{{ w.name }}</text>
              <g :transform="`translate(${w.x},${w.y})`">
                <template v-for="row in w.rows" :key="row"><rect v-for="col in w.cols" :key="col" :x="(col - 1) * 5" :y="(row - 1) * 5" width="5" height="5" class="weight-cell" /></template>
              </g>
              <text :x="w.x + w.cols * 2.5" y="85" class="label note">{{ w.shape }}</text>
            </g>
          </g>
          <path :d="`M${center(rank) + (rank === 0 ? -125 : 125)} 130 H${center(rank) + (rank === 0 ? -145 : 145)} V${attention ? 970 : 640}`" class="weight-rail" />
          <path v-for="y in attention ? [290, 970] : [320, 640]" :key="y" :d="`M${center(rank) + (rank === 0 ? -145 : 145)} ${y} H${center(rank)}`" class="weight-rail" :marker-end="`url(#${id}-neutral)`" />
        </g>

        <template v-if="attention">
          <g v-for="rank in ranks" :key="rank">
            <path :d="`M${center(rank)} 277 V295 M${center(rank)} 295 L${queryX(rank)} 304 M${center(rank)} 295 L${kvX(rank)} 300`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
            <path :d="`M${queryX(rank)} 402 V414 H${center(rank)} V420 M${kvX(rank)} 402 V414 H${center(rank)}`" class="local-arrow" />
            <path :d="`M${center(rank)} 450 V463`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
            <path :d="`M${queryX(rank)} 402 H${center(rank) - side(rank) * 115} V710 H${center(rank) - side(rank) * 65}`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
            <text :x="center(rank) - side(rank) * 130" y="570" class="label note">Q</text>
            <path :d="`M${kvX(rank)} 402 H${center(rank) + side(rank) * 112} V552 L${kvX(1 - rank)} 577`" :class="`exchange owner${rank}`" :marker-end="`url(#${id}-rank-${rank})`" />
            <path :d="`M${center(rank)} 557 V694`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
            <path :d="`M${kvX(rank)} 672 V710 H${center(rank) + side(rank) * 65}`" :class="`exchange owner${1 - rank}`" :marker-end="`url(#${id}-rank-${1 - rank})`" />
            <path :d="`M${center(rank)} 726 V743`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
            <path :d="`M${center(rank)} 837 V853`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
            <path :d="`M${center(rank)} 947 V988`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
            <g v-for="y in [435, 710]" :key="y">
              <rect :x="center(rank) - 65" :y="y - 15" width="130" height="30" rx="5" class="operation" />
              <text :x="center(rank)" :y="y + 5" class="label op-label">Block Attention</text>
            </g>
          </g>
          <text x="80" y="295" class="label op-label">QKV Proj</text>
          <text x="80" y="440" class="label op-label">Round 0</text>
          <text x="80" y="577" class="label op-label">KV · P2P</text>
          <text x="80" y="715" class="label op-label">Round 1</text>
          <text x="80" y="850" class="label op-label">O = A / l</text>
          <text x="80" y="975" class="label op-label">O Proj</text>
        </template>
        <template v-else>
          <g v-for="step in mlpSteps" :key="step.name">
            <rect x="12" :y="step.y - 15" width="136" height="30" rx="5" class="operation" />
            <text x="80" :y="step.y + 5" class="label op-label">{{ step.name }}</text>
            <path v-for="rank in ranks" :key="rank" :d="`M${center(rank)} ${step.start} V${step.end}`" class="local-arrow" :marker-end="`url(#${id}-neutral)`" />
          </g>
        </template>

        <g v-for="t in tensors" :key="t.key" :transform="`translate(${t.x - t.cols * 9},${t.y - 18})`">
          <text :x="t.cols * 9" y="-12" class="label tensor-name">{{ t.name }}</text>
          <rect v-if="t.stack" x="5" y="-5" :width="t.cols * 18" height="36" class="tensor-back" />
          <template v-for="row in 2" :key="row">
            <rect v-for="col in t.cols" :key="col" :x="(col - 1) * 18" :y="(row - 1) * 18" width="18" height="18" :class="`cell owner${t.owner}`" />
            <text x="-8" :y="(row - 1) * 18 + 13" class="token">t{{ t.owner * 2 + row - 1 }}</text>
          </template>
          <text :x="t.cols * 9" y="57" class="label note">{{ t.cols === 8 ? '完整 I' : attention && !['X', 'Y'].includes(t.name) ? 'h0–3' : '完整 H' }}</text>
          <g v-if="t.state" :transform="`translate(${t.cols * 18 + 12},0)`">
            <rect width="36" height="36" rx="4" class="state-badge" />
            <text x="18" y="23" class="label note">m, l</text>
          </g>
        </g>
      </svg>
    </div>
    <figcaption>颜色：token 所属 Rank；虚线：权重读取。{{ attention ? '叠片代表 K、V。' : '' }}</figcaption>
  </figure>
</template>

<style scoped>
.cp-flow { margin: 1.5rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.scroll { overflow-x: auto; padding: 0.5rem; }
.scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -2px; }
svg { display: block; width: 100%; min-width: 780px; font-family: inherit; }
text { font-size: 17px; fill: var(--vp-c-text); }
.label { text-anchor: middle; }
.strong { font-weight: 600; }
.note, .weight-name, .op-label { font-size: 13px; }
.tensor-name { font-size: 15px; }
.token { text-anchor: end; font-size: 12px; }
.owner0 { color: var(--infra-token-flow); }
.owner1 { color: var(--infra-weight-flow); }
.cell { fill: currentColor; fill-opacity: 0.22; stroke: currentColor; stroke-width: 0.8; }
.rank-panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.weight-bank, .operation, .tensor-back, .state-badge { fill: var(--vp-c-bg); stroke: var(--infra-diagram-line); }
.weight-cell { fill: var(--infra-diagram-muted); fill-opacity: 0.25; stroke: var(--infra-diagram-muted); stroke-width: 0.7; }
.weight-rail { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.1; stroke-dasharray: 4 4; }
.local-arrow { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.4; }
.exchange { fill: none; stroke: currentColor; stroke-width: 1.7; }
.arrowhead { fill: currentColor; }
.neutral-head { fill: var(--infra-diagram-muted); }
figcaption { padding: 0.1rem 0.9rem 0.8rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
@media print { .cp-flow { break-inside: avoid; } svg { min-width: 0; } .scroll { overflow: visible; } }
</style>
