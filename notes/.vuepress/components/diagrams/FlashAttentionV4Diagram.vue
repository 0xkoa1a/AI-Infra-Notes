<script setup lang="ts">
import { computed, useId } from "vue"

const props = withDefaults(defineProps<{
  view?: "ownership" | "operands" | "storage" | "pipeline"
}>(), { view: "ownership" })
const id = `fa4-${useId().replace(/:/g, "")}`
const info = computed(() => ({
  ownership: {
    title: "两个 Query stage，各含两个 CTA 的行分片",
    height: 180,
    caption: "从左到右为连续 Query 行。颜色表示 CTA 归属；每个 CTA 持有两个 stage 的独立状态。",
  },
  operands: {
    title: "QK 与 PV：右操作数分片，结果保留完整列宽",
    height: 590,
    caption: "每个面板只画一个 Query stage、一个 KV block。颜色表示 CTA 归属；右操作数的两半共同参与两份输出。矩阵为逻辑布局，Kᵀ 表示计算方向。",
  },
  storage: {
    title: "一份 Query 状态：从输入到最终输出",
    height: 570,
    caption: "实线表示数据流，虚线传递缩放系数。图中 QK/PV 使用两 CTA 提供的操作数，S/P/O 按本地 Query 行展示；P 复用 S 工作区，O 独立保存。",
  },
  pipeline: {
    title: "stage 0 准备 P 和 O 时，Tensor Core 推进 stage 1",
    height: 400,
    caption: "依赖与重叠示意，无实测时间比例。只展开 stage 0 的本地计算，两 CTA 对称执行；省略 stage 1 的 Softmax 与其他加载。箭头表示所选依赖，空白不表示已测得的硬件空闲。",
  },
}[props.view]))
const panels = [
  { name: "QK", y: 0, left: "Q", right: "Kᵀ", output: "S", axis: "Key position", reduction: "head dimension", leftShape: "128 × 128" },
  { name: "PV", y: 292, left: "P", right: "V", output: "O", axis: "Value dimension", reduction: "Key position", leftShape: "128 × 128" },
]
</script>

<template>
  <figure class="fa4" :data-view="view">
    <div class="fa4__title"><strong>{{ info.title }}</strong></div>
    <div class="fa4__scroll" tabindex="0" role="region" :aria-label="`${info.title}，窄屏可横向滚动`">
      <svg :viewBox="`0 0 780 ${info.height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ info.title }}</title>
        <desc :id="`${id}-desc`">{{ info.caption }}</desc>
        <defs>
          <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" class="arrow-head" />
          </marker>
        </defs>

        <g v-if="view === 'ownership'">
          <g v-for="stage in [0, 1]" :key="stage" :transform="`translate(${25 + stage * 380}, 0)`">
            <path d="M5 47 V35 H350 V47" class="guide" />
            <text x="177" y="24" class="center label">stage {{ stage }} · 联合 QK 的 256 行</text>
            <g v-for="cta in [0, 1]" :key="cta" :transform="`translate(${cta * 180}, 60)`">
              <rect width="170" height="67" rx="5" class="box" :class="cta === 0 ? 'cta0' : 'cta1'" />
              <text x="85" y="27" class="center" :class="cta === 0 ? 'cta0' : 'cta1'">CTA {{ cta }}</text>
              <text x="85" y="51" class="center small">128 行 · 本地 S/P/O</text>
            </g>
          </g>
          <path d="M30 154 H751" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="390" y="176" class="center small">Query 行号增加</text>
        </g>

        <g v-else-if="view === 'operands'">
          <g v-for="panel in panels" :key="panel.name" :transform="`translate(0, ${panel.y})`">
            <text x="24" y="27" class="label">{{ panel.name }}</text>
            <text x="105" y="52" class="center small">{{ panel.left }} · {{ panel.reduction }} →</text>
            <text x="487" y="25" class="center label">{{ panel.right }} · {{ panel.axis }} →</text>
            <g v-for="cta in [0, 1]" :key="`rhs-${cta}`">
              <rect :x="332 + cta * 155" y="42" width="155" height="53" class="box" :class="cta === 0 ? 'cta0' : 'cta1'" />
              <text :x="409 + cta * 155" y="64" class="center small" :class="cta === 0 ? 'cta0' : 'cta1'">CTA {{ cta }} 加载</text>
              <text :x="409 + cta * 155" y="85" class="center small">128 × 64</text>
              <path :d="`M${409 + cta * 155} 96 V122`" class="arrow" :marker-end="`url(#${id}-arrow)`" />
            </g>
            <g v-for="cta in [0, 1]" :key="`lhs-${cta}`" :transform="`translate(0, ${127 + cta * 65})`">
              <rect x="25" width="163" height="65" class="box" :class="cta === 0 ? 'cta0' : 'cta1'" />
              <text x="106" y="26" class="center" :class="cta === 0 ? 'cta0' : 'cta1'">{{ panel.left }} · CTA {{ cta }}</text>
              <text x="106" y="50" class="center small">{{ panel.leftShape }}</text>
              <path d="M190 32 H328" class="arrow" :marker-end="`url(#${id}-arrow)`" />
              <rect x="332" width="310" height="65" class="box" :class="cta === 0 ? 'cta0' : 'cta1'" />
              <path d="M487 0 V65" class="guide" />
              <text x="409" y="39" class="center">{{ panel.output }} 左半</text>
              <text x="564" y="39" class="center">{{ panel.output }} 右半</text>
              <text x="659" y="28" class="small" :class="cta === 0 ? 'cta0' : 'cta1'">CTA {{ cta }}</text>
              <text x="659" y="49" class="small">本地 TMEM</text>
            </g>
            <text x="259" y="115" class="center small">2CTA UMMA</text>
            <text x="487" y="281" class="center small">每 CTA 的 128 行都覆盖完整 128 列</text>
          </g>
        </g>

        <g v-else-if="view === 'storage'">
          <text x="120" y="35" class="center small">GMEM → TMA Load</text>
          <rect x="25" y="60" width="190" height="70" rx="6" class="box input" />
          <text x="120" y="88" class="center label input">Q / K · SMEM</text>
          <text x="120" y="112" class="center small">Q 驻留，K 逐块更换</text>
          <path d="M216 95 H251" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <rect x="255" y="70" width="100" height="50" rx="6" class="box compute" />
          <text x="305" y="101" class="center label compute">QK</text>
          <path d="M356 95 H401" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <rect x="405" y="70" width="110" height="50" rx="6" class="box state" />
          <text x="460" y="91" class="center state">S · FP32</text>
          <text x="460" y="111" class="center small">TMEM</text>
          <path d="M516 95 H561" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <rect x="565" y="60" width="185" height="70" rx="6" class="box state" />
          <text x="657" y="87" class="center label state">Softmax</text>
          <text x="657" y="112" class="center small">寄存器 · 行统计与指数</text>
          <path d="M657 131 V241" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="644" y="198" class="end small">转换、分段写入</text>
          <rect x="565" y="245" width="185" height="50" rx="6" class="box state" />
          <text x="657" y="266" class="center state">P · BF16/FP16</text>
          <text x="657" y="286" class="center small">TMEM · 复用 S 工作区</text>
          <path d="M563 270 H414" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="120" y="209" class="center small">GMEM → TMA Load</text>
          <rect x="25" y="235" width="190" height="70" rx="6" class="box input" />
          <text x="120" y="265" class="center label input">V · SMEM</text>
          <text x="120" y="289" class="center small">与 K 共用 ring</text>
          <path d="M216 270 H276" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <rect x="280" y="245" width="130" height="50" rx="6" class="box compute" />
          <text x="345" y="276" class="center label compute">PV</text>
          <path d="M345 296 V361" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="359" y="333" class="small">累加新贡献</text>
          <rect x="280" y="365" width="130" height="60" rx="6" class="box output" />
          <text x="345" y="390" class="center output">O · FP32</text>
          <text x="345" y="414" class="center small">TMEM · 跨块保留</text>
          <path d="M750 96 H768 V395 H414" class="signal" :marker-end="`url(#${id}-arrow)`" />
          <text x="585" y="380" class="center small">Correction · 按系数缩放旧 O</text>
          <path d="M345 426 V476" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="359" y="456" class="small">全部 KV 完成</text>
          <rect x="280" y="480" width="130" height="50" rx="6" class="box output" />
          <text x="345" y="501" class="center output">归一化、转换</text>
          <text x="345" y="522" class="center small">Correction</text>
          <path d="M411 505 H471" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <rect x="475" y="480" width="110" height="50" rx="6" class="box output" />
          <text x="530" y="501" class="center output">sO</text>
          <text x="530" y="522" class="center small">SMEM</text>
          <path d="M586 505 H651" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="620" y="469" class="center small">TMA Store</text>
          <rect x="655" y="480" width="100" height="50" rx="6" class="box output" />
          <text x="705" y="511" class="center output">GMEM</text>
          <text x="620" y="556" class="center small">Epilogue 发起写回</text>
        </g>

        <g v-else>
          <text x="157" y="26" class="small">时间推进 →（仅示意依赖与可重叠区间）</text>
          <g v-for="lane in [{ y: 55, label: 'TMA Load' }, { y: 140, label: 'Tensor Core' }, { y: 270, label: 'CUDA Core' }]" :key="lane.label">
            <rect x="150" :y="lane.y" width="610" :height="lane.y === 270 ? 83 : 66" rx="6" class="panel" />
            <text x="18" :y="lane.y + 31" class="label">{{ lane.label }}</text>
          </g>
          <text x="18" y="324" class="label">/ MUFU</text>
          <rect x="166" y="72" width="134" height="32" rx="4" class="box input" />
          <text x="233" y="94" class="center small input">当前 Vⱼ</text>
          <path d="M301 88 H550 V173" class="arrow" />
          <text x="437" y="78" class="small">V 可读</text>

          <rect x="160" y="155" width="98" height="36" rx="4" class="box stage0" />
          <text x="209" y="179" class="center stage0">QK₀,ⱼ</text>
          <rect x="278" y="155" width="117" height="36" rx="4" class="box stage1" />
          <text x="336" y="179" class="center stage1">PV₁,ⱼ₋₁</text>
          <rect x="415" y="155" width="112" height="36" rx="4" class="box stage1" />
          <text x="471" y="179" class="center stage1">QK₁,ⱼ</text>
          <rect x="558" y="155" width="108" height="36" rx="4" class="box stage0" />
          <rect x="678" y="155" width="71" height="36" rx="4" class="box stage0" />
          <text x="611" y="178" class="center small stage0">PV₀,ⱼ 前段</text>
          <text x="713" y="178" class="center small stage0">后段</text>
          <text x="611" y="211" class="center small">96 项</text>
          <text x="713" y="211" class="center small">32 项</text>

          <path d="M258 173 H263 V305 H269" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="258" y="248" class="end small">S 可读</text>
          <rect x="273" y="287" width="59" height="49" rx="4" class="box stage0" />
          <text x="302" y="308" class="center small stage0">行最大</text>
          <text x="302" y="328" class="center small">与 α</text>
          <rect x="342" y="287" width="73" height="49" rx="4" class="box stage0" />
          <text x="378" y="308" class="center small stage0">修正 O</text>
          <text x="378" y="328" class="center small">按需</text>
          <rect x="431" y="287" width="117" height="49" rx="4" class="box stage0" />
          <text x="489" y="308" class="center small stage0">指数、转换</text>
          <text x="489" y="328" class="center small">写 P 前段</text>
          <rect x="560" y="287" width="81" height="49" rx="4" class="box stage0" />
          <text x="600" y="308" class="center small stage0">写 P 尾段</text>
          <text x="600" y="328" class="center small">32 列</text>

          <path d="M415 305 H422 V235 H550 V173" class="arrow" />
          <text x="443" y="227" class="small">O 可累加</text>
          <path d="M548 289 H550 V173" class="arrow" />
          <text x="557" y="260" class="small">P 前段可读</text>
          <circle cx="550" cy="173" r="4" class="arrow-head" />
          <path d="M550 173 H556" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M642 310 H672 V173 H676" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="681" y="251" class="small">尾段可读</text>

          <text x="165" y="384" class="small">stage 0：绿色　 stage 1：紫色；下标为 stage、KV 访问次序</text>
        </g>
      </svg>
    </div>
    <figcaption>{{ info.caption }}</figcaption>
  </figure>
</template>

<style scoped>
.fa4 { margin: 1.6rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.fa4__title { padding: 1rem 1.1rem 0.4rem; line-height: 1.6; }
.fa4__scroll { padding: 0.35rem; overflow-x: auto; }
.fa4__scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 700px; font-family: inherit; }
text { fill: var(--vp-c-text); font-size: 16px; }
.label { font-size: 18px; font-weight: 650; }
.small { font-size: 15px; fill: var(--infra-diagram-muted); }
.center { text-anchor: middle; }
.end { text-anchor: end; }
.panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.box { fill: var(--vp-c-bg); stroke: currentColor; stroke-width: 1.8; }
.input, .cta0 { color: var(--infra-token-flow); }
.cta1, .stage1 { color: var(--infra-placement); }
.state { color: var(--infra-planning); }
.compute, .output, .stage0 { color: var(--infra-execution); }
text.input, text.cta0, text.cta1, text.state, text.compute, text.output, text.stage0, text.stage1 { fill: currentColor; }
.arrow, .signal { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.8; }
.signal { stroke-dasharray: 5 4; }
.arrow-head { fill: var(--infra-diagram-muted); }
.guide { fill: none; stroke: var(--infra-diagram-muted); stroke-dasharray: 4 4; }
figcaption { padding: 0.8rem 1.1rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.65; }
@media print { .fa4 { break-inside: avoid; } .fa4__scroll { overflow: visible; } svg { min-width: 0; } }
</style>
