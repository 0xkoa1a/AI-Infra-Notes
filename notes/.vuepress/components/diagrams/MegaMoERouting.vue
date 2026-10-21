<script setup lang="ts">
import { computed, ref, useId } from "vue"

const props = defineProps<{ view: string }>()
const id = useId()
const gpu = ref(0)
const tokenIndex = ref(0)
const step = ref(0)
const indicesFilled = ref(false)
const ranks = [0, 1, 2, 3]
const cx = (rank: number) => 140 + rank * 240
type Token = { name: string; rank: number; index: number; experts: [number, number] }
type Route = { token: Token; expert: number; slot: number; target: number }
const tokens: Token[] = [
  { name: "a0", rank: 0, index: 0, experts: [0, 3] },
  { name: "a1", rank: 0, index: 1, experts: [1, 4] },
  { name: "b0", rank: 1, index: 0, experts: [2, 5] },
  { name: "b1", rank: 1, index: 1, experts: [3, 6] },
  { name: "c0", rank: 2, index: 0, experts: [4, 7] },
  { name: "c1", rank: 2, index: 1, experts: [5, 0] },
  { name: "d0", rank: 3, index: 0, experts: [6, 1] },
  { name: "d1", rank: 3, index: 1, experts: [7, 2] },
]
const routes: Route[] = tokens.flatMap(token => token.experts.map((expert, slot) => ({ token, expert, slot, target: Math.floor(expert / 2) })))
const localTokens = (rank: number) => tokens.filter(t => t.rank === rank)
const received = (expert: number) => routes.filter(r => r.expert === expert)
const indexGroup = (expert: number, rank: number) => received(expert).filter(r => r.token.rank === rank)
const indexValue = (expert: number, rank: number) => {
  const route = indexGroup(expert, rank)[0]
  return route ? route.token.index * 2 + route.slot : "—"
}
const expertRow = (route: Route) => received(route.expert).indexOf(route)
const active = (route: Route) => (props.view === "combine" || step.value === 0) ? route.token.rank === gpu.value : route.target === gpu.value
const dispatchEdges = computed(() => routes.filter(active))
const combineEdges = computed(() => routes.filter(r => r.token.rank === gpu.value && r.token.index === tokenIndex.value))
const selectedToken = computed(() => localTokens(gpu.value)[tokenIndex.value])
const needs = (token: Token) => [...new Set(token.experts.map(e => Math.floor(e / 2)))]
const labels = computed(() => props.view === "dispatch" ? ["发布索引与计数", "接收卡拉取输入", "发布输入块就绪"] : ["专家结果写回", "等待专家卡通知", "源卡逐元素求和"])
const title = computed(() => {
  const titles: Record<string, string> = { placement: "四张 GPU 的源 token 与专家", symmem: "各卡拥有布局对应的 Symmem 缓冲", dispatch: labels.value[step.value], indices: "GPU 0 · 本地专家 × 来源 GPU", batch: "GPU 0 · E0 的输入行与来源记录", combine: labels.value[step.value] }
  return titles[props.view] || "MegaMoE 数据流"
})
const height = computed(() => {
  const heights: Record<string, number> = { placement: 390, symmem: 670, dispatch: 650, indices: 415, batch: 460, combine: 710 }
  return heights[props.view] || 710
})
const regions = [
  { y: 94, h: 45, name: "状态与计数", sub: "就绪 / 释放 / 回传完成" },
  { y: 139, h: 50, name: "接收索引列表", sub: "本地专家 × 来源 GPU" },
  { y: 189, h: 50, name: "来源记录", sub: "来源 GPU / token / 槽" },
  { y: 239, h: 53, name: "源输入 X + scale", sub: "本卡 token" },
  { y: 292, h: 45, name: "路由编号 + α", sub: "每 token 两个槽" },
  { y: 337, h: 58, name: "L1 输入缓冲", sub: "X + scale + α" },
  { y: 395, h: 53, name: "L2 输入缓冲", sub: "A + scale" },
  { y: 448, h: 83, name: "Combine 回传槽", sub: "槽 0 / 槽 1" },
]
const arrow = `url(#${id}-arrow)`
const signalArrow = `url(#${id}-signal)`
const sx = (rank: number) => 280 + rank * 200
const resultX = (route: Route) => cx(route.target) - 20 + expertRow(route) * 74
const resultY = (route: Route) => 94 + route.expert % 2 * 78
function combinePath(route: Route) {
  const outlet = cx(route.target) + 113
  const inlet = cx(gpu.value) + (route.slot === 0 ? -113 : 113)
  const end = cx(gpu.value) + (route.slot === 0 ? -58 : 95)
  const rowY = 461 + route.token.index * 70
  return `M${resultX(route)} ${resultY(route) + 37} V${resultY(route) + 57} H${outlet} V284 C${outlet} 323,${inlet} 323,${inlet} 365 V${rowY} H${end}`
}
const description = computed(() => {
  if (props.view === "placement") return "四卡各有两个token和两个专家。a0选择E0/E3，a1选择E1/E4，b0选择E2/E5，b1选择E3/E6，c0选择E4/E7，c1选择E5/E0，d0选择E6/E1，d1选择E7/E2。颜色跟踪来源GPU，专家权重保存在所属卡。"
  if (props.view === "symmem") return "四卡的Symmem字段依次对应：状态与计数、接收索引、来源记录、源输入和scale、路由编号和alpha、L1输入缓冲、L2输入缓冲、Combine回传槽。专家权重在所属卡另行存放。"
  if (props.view === "indices") return `GPU0在Dispatch前就为E0、E1与四张来源GPU的全部组合分配八组空间，每组容量Tmax，总容量8乘Tmax。当前${indicesFilled.value ? '已写入本轮索引：E0/G0为0，E0/G2为3，E1/G0为2，E1/G3为1，各有效1条，其余组有效0条。' : '所有组均已预留，有效计数为0。'}切换只改变索引与计数，各组的位置和容量保持一致。计数在状态区单独保存。`
  if (props.view === "batch") return "GPU0将本卡a0和GPU2的c1放成E0输入块的两行，每行512个特征，旁边分别保存G0/a0/s0和G2/c1/s1的来源记录。块容量64行，有效行数2。这两行的输入、scale、alpha写完后，该块就绪，L1可以读取。"
  if (props.view === "dispatch") return `当前为${title.value}，跟踪GPU${gpu.value}。源卡先发布索引和计数，专家卡随后按列表拉取输入并保存来源记录，输入块写完后通知L1。颜色跟踪来源GPU，高亮相关条目。`
  return `跟踪源GPU${gpu.value}的${selectedToken.value.name}。专家卡将结果写回原token的路由槽，源卡等待所选专家卡完成本轮全部L2写回的通知，再将两个槽逐元素相加。`
})
</script>

<template>
  <div v-if="view === 'dispatch' || view === 'combine'" class="controls" role="group" aria-label="执行步骤">
    <button v-for="(label, index) in labels" :key="label" type="button" :aria-pressed="step === index" @click="step = index">{{ label }}</button>
  </div>
  <div v-if="view === 'indices'" class="controls" role="group" aria-label="索引空间分配与本轮写入">
    <button type="button" :aria-pressed="!indicesFilled" @click="indicesFilled = false">Dispatch 前：预留空间</button>
    <button type="button" :aria-pressed="indicesFilled" @click="indicesFilled = true">本轮：写入索引</button>
  </div>
  <div v-if="view === 'dispatch' || view === 'combine'" class="controls" role="group" aria-label="跟踪的 GPU">
    <span class="control-label">{{ view === 'dispatch' ? (step === 0 ? '跟踪发送卡' : '跟踪接收卡') : '跟踪源卡' }}</span>
    <button v-for="rank in ranks" :key="rank" type="button" :aria-pressed="gpu === rank" @click="gpu = rank">GPU {{ rank }}</button>
  </div>
  <div v-if="view === 'combine'" class="controls" role="group" aria-label="跟踪的 token">
    <span class="control-label">token</span>
    <button v-for="token in localTokens(gpu)" :key="token.name" type="button" :aria-pressed="tokenIndex === token.index" @click="tokenIndex = token.index">{{ token.name }}</button>
  </div>
  <div class="scroll" role="region" tabindex="0" :aria-label="`${title}，可横向滚动`">
    <svg class="four-gpu" :viewBox="`0 0 1000 ${height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
      <title :id="`${id}-title`">{{ title }}</title>
      <desc :id="`${id}-desc`">{{ description }}</desc>
      <defs>
        <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker>
        <marker :id="`${id}-signal`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="signal-head" /></marker>
      </defs>

      <g v-if="view === 'placement'">
        <text x="500" y="25" class="center small muted">每个 token 选择两个专家 · 路由槽 0 / 1</text>
        <g v-for="rank in ranks" :key="rank" :transform="`translate(${cx(rank) - 105},42)`">
          <rect width="210" height="330" rx="9" class="panel" />
          <text x="105" y="28" class="center strong">GPU {{ rank }}</text>
          <g v-for="token in localTokens(rank)" :key="token.name" :transform="`translate(12,${65 + token.index * 82})`">
            <path d="M45 19 H64 M22 41 V52 H158 V42" class="edge" :marker-end="arrow" />
            <rect width="42" height="38" rx="5" :class="`tint owner${rank}`" />
            <text x="21" y="25" class="center">{{ token.name }}</text>
            <g v-for="(expert, slot) in token.experts" :key="expert">
              <rect :x="68 + slot * 63" width="55" height="38" rx="4" class="box" />
              <text :x="95 + slot * 63" y="17" class="center">E{{ expert }}</text>
              <text :x="95 + slot * 63" y="31" class="center small muted">GPU {{ Math.floor(expert / 2) }}</text>
              <text :x="95 + slot * 63" y="-8" class="center small muted">槽 {{ slot }}</text>
            </g>
          </g>
          <text x="105" y="247" class="center small muted">本卡专家权重</text>
          <g v-for="expert in [rank * 2, rank * 2 + 1]" :key="expert" :transform="`translate(${15 + expert % 2 * 95},259)`">
            <rect width="85" height="53" rx="5" class="box" />
            <text x="42" y="18" class="center">E{{ expert }}</text>
            <template v-for="row in 3" :key="row"><rect v-for="col in 6" :key="col" :x="24 + (col - 1) * 6" :y="25 + (row - 1) * 6" width="6" height="6" class="weight-cell" /></template>
          </g>
        </g>
      </g>

      <g v-else-if="view === 'symmem'">
        <text x="500" y="24" class="center small muted">字段按相同顺序分配 · 相同字段具有相同相对偏移</text>
        <g v-for="region in regions" :key="region.y">
          <text x="15" :y="region.y + 19">{{ region.name }}</text>
          <text x="15" :y="region.y + 36" class="small muted">{{ region.sub }}</text>
        </g>
        <text x="15" y="587">专家权重</text>
        <text x="15" y="610" class="small muted">另存于所属 GPU</text>
        <g v-for="rank in ranks" :key="rank">
          <rect :x="sx(rank) - 85" y="40" width="170" height="505" rx="9" class="panel" />
          <text :x="sx(rank)" y="65" class="center strong">GPU {{ rank }}</text>
          <text :x="sx(rank)" y="85" class="center small muted">base{{ rank }} ↓</text>
          <g v-for="region in regions" :key="region.y">
            <rect :x="sx(rank) - 72" :y="region.y" width="144" :height="region.h" :class="region.y === 239 ? `tint owner${rank}` : 'box'" />
            <text v-if="region.y === 94" :x="sx(rank)" :y="region.y + 29" class="center muted">○  ○  ○  ○</text>
            <text v-else-if="region.y === 139" :x="sx(rank)" :y="region.y + 31" class="center">E{{ rank * 2 }} / E{{ rank * 2 + 1 }}</text>
            <text v-else-if="region.y === 189" :x="sx(rank)" :y="region.y + 31" class="center small muted">GPU · token · 槽</text>
            <text v-else-if="region.y === 239" :x="sx(rank)" :y="region.y + 32" class="center">{{ localTokens(rank).map(t => t.name).join(' / ') }}</text>
            <text v-else-if="region.y === 292" :x="sx(rank)" :y="region.y + 29" class="center small muted">s0 / s1 · α₀ / α₁</text>
            <g v-else-if="region.y === 337 || region.y === 395">
              <rect v-for="slot in [0, 1]" :key="slot" :x="sx(rank) - 57 + slot * 42" :y="region.y + 14" width="35" height="25" :class="region.y === 337 ? 'tint acts' : 'tint result'" />
              <text :x="sx(rank) + 42" :y="region.y + 32" class="center muted">…</text>
            </g>
            <template v-else>
              <g v-for="slot in [0, 1]" :key="slot">
                <text :x="sx(rank) - 64" :y="region.y + 42 + slot * 22" class="small muted">s{{ slot }}</text>
                <g v-for="token in localTokens(rank)" :key="token.name">
                  <rect :x="sx(rank) - 38 + token.index * 50" :y="region.y + 28 + slot * 22" width="47" height="20" class="box" />
                  <text :x="sx(rank) - 14 + token.index * 50" :y="region.y + 42 + slot * 22" class="center small">{{ token.name }}</text>
                </g>
              </g>
            </template>
          </g>
          <rect :x="sx(rank) - 72" y="562" width="144" height="67" rx="6" class="box" />
          <text :x="sx(rank)" y="587" class="center">E{{ rank * 2 }} / E{{ rank * 2 + 1 }}</text>
          <text :x="sx(rank)" y="612" class="center small muted">两次投影权重</text>
        </g>
      </g>

      <g v-else-if="view === 'dispatch'">
        <text x="500" y="26" class="center strong">{{ step === 0 ? `GPU ${gpu} 写入目标卡的索引列表` : step === 1 ? `GPU ${gpu} 发起读取 · X、scale、α 从源卡流入` : 'L1 输入块写完 → 发布就绪' }}</text>
        <g v-for="rank in ranks" :key="rank">
          <rect :x="cx(rank) - 105" y="45" width="210" height="153" rx="8" class="panel" />
          <text :x="cx(rank)" y="72" class="center strong">GPU {{ rank }} · 源输入</text>
          <g v-for="token in localTokens(rank)" :key="token.name" :transform="`translate(${cx(rank) - 92 + token.index * 96},95)`">
            <rect width="88" height="70" rx="4" :class="`tint owner${rank}`" />
            <text x="44" y="23" class="center">{{ token.name }}</text>
            <text x="44" y="45" class="center small">{{ step === 0 ? `E${token.experts[0]} / E${token.experts[1]}` : 'X + scale' }}</text>
            <text x="44" y="61" class="center small muted">{{ step === 0 ? '槽 0 / 1' : 'α₀ / α₁' }}</text>
          </g>
          <rect :x="cx(rank) - 105" y="332" width="210" height="300" rx="8" class="panel" />
          <text :x="cx(rank)" y="359" class="center strong">GPU {{ rank }} · {{ step === 0 ? '接收索引' : '专家输入' }}</text>
          <g v-for="expert in [rank * 2, rank * 2 + 1]" :key="expert" :transform="`translate(${cx(rank) - 92},${381 + expert % 2 * 122})`">
            <rect width="184" height="108" rx="5" class="box" />
            <text x="12" y="21">E{{ expert }}</text>
            <text v-if="step < 2" x="172" y="21" class="small muted" text-anchor="end">{{ step === 0 ? '2 条有效索引' : '拉取 / 保存来源' }}</text>
            <g v-else>
              <text x="95" y="21" class="small ready ink">✓</text>
              <path d="M108 16 H143" class="signal" :marker-end="signalArrow" />
              <text x="163" y="21" class="center small">L1</text>
            </g>
            <g v-for="(route, row) in received(expert)" :key="route.token.name" :class="{ dim: !active(route) }">
              <rect x="10" :y="31 + row * 31" width="164" height="27" rx="3" :class="`tint owner${route.token.rank}`" />
              <text x="20" :y="49 + row * 31" class="small">{{ step === 0 ? `G${route.token.rank} · ${route.token.name} · s${route.slot}` : route.token.name }}</text>
              <text v-if="step > 0" x="164" :y="49 + row * 31" text-anchor="end" class="small">G{{ route.token.rank }} / s{{ route.slot }}</text>
            </g>
          </g>
        </g>
        <g v-if="step < 2">
          <path v-for="route in dispatchEdges" :key="`${route.token.name}-${route.slot}`" :d="`M${cx(route.token.rank) - 48 + route.token.index * 96 + route.slot * 6} 170 C${cx(route.token.rank) - 48 + route.token.index * 96 + route.slot * 6} 255,${cx(route.target) - 26 + route.slot * 45} 273,${cx(route.target) - 26 + route.slot * 45} 324`" :class="`data-edge owner${route.token.rank}`" :marker-end="arrow" />
          <text x="500" y="304" class="center small muted">{{ step === 0 ? '索引 + 每专家计数' : '数据方向：源输入 → 专家卡' }}</text>
        </g>
        <g v-else>
          <text x="500" y="270" class="center small muted">每个输入块独立通知对应的 L1 任务</text>
        </g>
      </g>

      <g v-else-if="view === 'indices'">
        <text x="500" y="27" class="center strong">{{ title }}</text>
        <rect x="135" y="78" width="835" height="279" rx="9" class="panel" />
        <g v-for="rank in ranks" :key="rank">
          <text :x="242 + rank * 205" y="63" class="center">来源 G{{ rank }}</text>
        </g>
        <g v-for="expert in [0, 1]" :key="expert">
          <text x="75" :y="159 + expert * 135" class="center strong">E{{ expert }}</text>
          <g v-for="rank in ranks" :key="rank" :transform="`translate(${150 + rank * 205},${94 + expert * 135})`">
            <rect width="185" height="110" rx="6" class="box" />
            <text x="92" y="24" class="center small muted">有效 {{ indicesFilled ? indexGroup(expert, rank).length : 0 }} 条</text>
            <g v-for="slot in [0, 1, 2, 3]" :key="slot">
              <rect :x="13 + slot * 41" y="40" width="36" height="31" rx="3" :class="indicesFilled && slot === 0 && indexGroup(expert, rank).length ? `tint owner${rank}` : 'pending'" />
              <text :x="31 + slot * 41" y="61" class="center">{{ slot === 2 ? '…' : indicesFilled && slot === 0 ? indexValue(expert, rank) : '—' }}</text>
            </g>
            <text x="92" y="95" class="center small">容量 Tₘₐₓ</text>
          </g>
        </g>
        <text x="550" y="390" class="center strong">索引区总容量：2 × 4 × Tₘₐₓ 个位置</text>
      </g>

      <g v-else-if="view === 'batch'">
        <text x="500" y="27" class="center strong">{{ title }}</text>
        <text x="130" y="66" class="center">源输入 · H=512</text>
        <text x="480" y="66" class="center">L1 输入缓冲 · 本卡显存</text>
        <text x="840" y="66" class="center">来源记录</text>
        <g v-for="(route, row) in received(0)" :key="route.token.name">
          <text x="20" :y="122 + row * 50" class="small">G{{ route.token.rank }} · {{ route.token.name }}</text>
          <rect v-for="col in [0, 1, 2, 3]" :key="col" :x="100 + col * 32" :y="99 + row * 50" width="32" height="32" :class="`tint owner${route.token.rank}`" />
          <path :d="`M233 ${115 + row * 50} H344`" class="edge" :marker-end="arrow" />
          <rect x="350" :y="90 + row * 50" width="260" height="50" :class="`tint owner${route.token.rank}`" />
          <text x="480" :y="121 + row * 50" class="center">行 {{ row }} · {{ route.token.name }}</text>
          <path :d="`M615 ${115 + row * 50} H704`" class="edge" :marker-end="arrow" />
          <rect x="710" :y="90 + row * 50" width="260" height="50" rx="4" class="box" />
          <text x="840" :y="121 + row * 50" class="center">G{{ route.token.rank }} / {{ route.token.name }} / s{{ route.slot }}</text>
        </g>
        <text x="130" y="215" class="center small muted">各自携带 scale、α</text>
        <rect x="350" y="190" width="260" height="125" class="pending" />
        <text x="480" y="259" class="center muted">其余 62 行容量</text>
        <text x="480" y="342" class="center strong">有效 2 行 / 容量 64 行</text>
        <g v-for="(route, row) in received(0)" :key="route.token.name">
          <rect :x="185 + row * 135" y="382" width="120" height="42" rx="5" class="tint ready" />
          <text :x="245 + row * 135" y="409" class="center">{{ route.token.name }} ✓</text>
        </g>
        <path d="M245 428 V440 H510 V403 H544 M380 428 V440 M770 403 H815" class="signal" :marker-end="signalArrow" />
        <rect x="550" y="382" width="215" height="42" rx="5" class="tint ready" />
        <text x="657" y="409" class="center">输入块就绪</text>
        <text x="870" y="409" class="center">L1 读取</text>
      </g>

      <g v-else>
        <text x="500" y="26" class="center strong">{{ title }}</text>
        <g v-for="rank in ranks" :key="rank">
          <rect :x="cx(rank) - 105" y="45" width="210" height="215" rx="8" class="panel" />
          <text :x="cx(rank)" y="72" class="center strong">GPU {{ rank }} · 专家结果</text>
          <g v-for="expert in [rank * 2, rank * 2 + 1]" :key="expert" :transform="`translate(${cx(rank) - 93},${94 + expert % 2 * 78})`">
            <text x="0" y="20">E{{ expert }}</text>
            <g v-for="(route, row) in received(expert)" :key="route.token.name" :class="{ dim: route.token.rank !== gpu || route.token.index !== tokenIndex }">
              <rect :x="38 + row * 74" width="70" height="33" rx="4" :class="`tint owner${route.token.rank}`" />
              <text :x="73 + row * 74" y="22" class="center">{{ route.token.name }}</text>
            </g>
          </g>
          <text v-if="step > 0" :x="cx(rank)" y="248" class="center small ready ink" :class="{ dim: !needs(selectedToken).includes(rank) }">本卡全部 L2 写回完成 ✓</text>
          <rect :x="cx(rank) - 105" y="380" width="210" height="310" rx="8" class="panel" :class="{ dim: rank !== gpu }" />
          <text :x="cx(rank)" y="407" class="center strong">源 GPU {{ rank }} · 回传槽</text>
          <text :x="cx(rank) - 21" y="433" class="center small muted">s0</text>
          <text :x="cx(rank) + 54" y="433" class="center small muted">s1</text>
          <g v-for="token in localTokens(rank)" :key="token.name" :transform="`translate(${cx(rank) - 92},${446 + token.index * 70})`" :class="{ dim: token.index !== tokenIndex }">
            <text x="3" y="-5">{{ token.name }}</text>
            <g v-for="(expert, slot) in token.experts" :key="expert">
              <rect :x="38 + slot * 75" width="70" height="30" rx="3" :class="`tint owner${rank}`" />
              <text :x="73 + slot * 75" y="21" class="center">E{{ expert }}</text>
            </g>
            <text x="111" y="48" class="center small muted">{{ needs(token).map(r => `G${r}${step > 0 ? ' ✓' : ' 等待'}`).join('  ') }}</text>
          </g>
          <g v-if="step === 2" :class="{ dim: rank !== gpu }">
            <path :d="`M${cx(rank) - 21} ${480 + tokenIndex * 70} V${486 + tokenIndex * 70} H${cx(rank) - 99} V590 H${cx(rank) - 10} V601 M${cx(rank) + 54} ${480 + tokenIndex * 70} V${486 + tokenIndex * 70} H${cx(rank) + 99} V590 H${cx(rank) + 10} V601`" class="edge" :marker-end="arrow" />
            <circle :cx="cx(rank)" cy="615" r="12" class="box" />
            <text :x="cx(rank)" y="620" class="center">Σ</text>
            <path :d="`M${cx(rank)} 630 V639`" class="edge" :marker-end="arrow" />
            <rect :x="cx(rank) - 72" y="643" width="144" height="30" rx="4" :class="`tint owner${rank}`" />
            <text :x="cx(rank)" y="664" class="center">Y · {{ localTokens(rank)[tokenIndex].name }}</text>
          </g>
        </g>
        <template v-if="step === 0">
          <path v-for="route in combineEdges" :key="`${route.token.name}-${route.slot}`" :d="combinePath(route)" :class="`data-edge owner${gpu}`" :marker-end="arrow" />
        </template>
        <g v-else-if="step === 1">
          <path v-for="rank in needs(selectedToken)" :key="rank" :d="`M${cx(rank)} 267 L${cx(gpu)} 373`" class="signal" :marker-end="signalArrow" />
          <text x="500" y="345" class="center small muted">每个 token 等自己所选专家卡的本轮通知</text>
        </g>
      </g>
    </svg>
  </div>
  <figcaption v-if="view === 'placement'">颜色跟踪 token 的来源 GPU；专家权重保存在所属 GPU。</figcaption>
  <figcaption v-else-if="view === 'symmem'">按字段顺序展示布局；图中区段高度采用示意比例。</figcaption>
  <figcaption v-else-if="view === 'indices'">虚线格已分配空间；数字为本轮有效索引。有效计数另存于状态区，图中标在对应组旁。</figcaption>
  <figcaption v-else-if="view === 'batch'">每行保存一个 token 的全部输入特征；来源记录保留这一行的回传位置。</figcaption>
  <figcaption v-else-if="view === 'combine'">每个回传槽保存一条 H 维专家贡献；实线为数据，虚线为完成通知。</figcaption>
  <figcaption v-else>颜色：token 来源；实线：索引或数据；虚线：就绪通知。高亮所选 GPU 的相关条目。</figcaption>
</template>
