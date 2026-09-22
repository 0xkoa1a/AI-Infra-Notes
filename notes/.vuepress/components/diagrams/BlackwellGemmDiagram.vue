<script setup lang="ts">
import { computed, useId } from "vue"

const props = defineProps<{ view: "cluster" | "storage" | "tmem" | "overlap" }>()
const id = useId()
const info = computed(() => ({
  cluster: { title: "两个 CTA 分担输入，各自得到一个输出 tile", height: 466, caption: "每轮 K block 的分工。B 按原矩阵的 N 行分半，对应输出 D 的前后 128 列；两半仍位于各 CTA 的 SMEM。" },
  storage: { title: "一个 CTA 的数据从哪里来、到哪里去", height: 584, caption: "实线表示张量数据经过的存储位置。容量按每个 CTA 计算；寄存器框只画一次 8-column TMEM load。" },
  tmem: { title: "逻辑上的 256 行，放进 TMEM 的 128 条 lane", height: 372, caption: "图示一个 CTA 的完整输出。两个 wave 使用相同的 lane 范围，结果同时保存在不同 column 区间；另一 CTA 使用自己的 TMEM。" },
  overlap: { title: "旧 tile 归还 TMEM 后，下一 tile 的 UMMA 才能开始", height: 420, caption: "末尾执行关系示意，无实测时间比例。下一 tile 的 A/B 输入已就绪；中间 chunk 合并显示，其跨度包含 CD stage 等待。TMA 操作按保守次序排列，实际重叠受搬运资源争用影响。" },
}[props.view]))
const storage = [
  { name: "Global A / B", shape: "A [M,K]、B [N,K] · BF16", step: "TMA Load", color: "input" },
  { name: "本地 A/B SMEM", shape: "4 ×（A [256,64] + B [128,64]）· 192 KiB", step: "2CTA UMMA", color: "input" },
  { name: "本地 TMEM", shape: "128 lanes × 512 columns · FP32 · 256 KiB", step: "tcgen05.ld + wait::ld", color: "result" },
  { name: "Epilogue 线程寄存器", shape: "每线程：8 FP32 → 8 BF16（打包为 4 个 uint32）", step: "swizzled store", color: "result" },
  { name: "CD SMEM", shape: "2 × [128,64] · BF16 · 共 32 KiB", step: "TMA Store", color: "result" },
  { name: "Global D", shape: "8 个 [128,64] chunk → 本 CTA 的 [256,256] tile", step: "", color: "result" },
]
</script>

<template>
  <figure class="bw-figure" :data-view="view">
    <div class="bw-title"><strong>{{ info.title }}</strong></div>
    <div class="bw-scroll" tabindex="0" role="region" :aria-label="info.title">
      <svg :viewBox="`0 0 780 ${info.height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ info.title }}</title>
        <desc :id="`${id}-desc`">{{ info.caption }}</desc>
        <defs>
          <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" class="arrow-head" />
          </marker>
        </defs>

        <g v-if="view === 'cluster'">
          <g v-for="cta in [0, 1]" :key="cta" :transform="`translate(${24 + cta * 386}, 14)`">
            <rect width="346" height="172" rx="10" class="panel" />
            <text x="18" y="29" class="label">CTA {{ cta }} / SM {{ cta }} · {{ cta === 0 ? 'leader' : 'peer' }}</text>
            <rect x="18" y="46" width="310" height="48" rx="7" class="box input" />
            <text x="173" y="76" class="center input">本地 A [256,64] · 不同 M 行</text>
            <rect x="18" y="106" width="310" height="48" rx="7" class="box weight" />
            <text x="173" y="136" class="center weight">B [128,64] · N {{ cta === 0 ? '[0,128)' : '[128,256)' }}</text>
          </g>
          <path d="M197 186 V213 H583 V186" class="arrow" />
          <path d="M390 213 V231" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <rect x="130" y="234" width="520" height="66" rx="9" class="box result" />
          <text x="390" y="260" class="center label result">leader 发射 cta_group::2 UMMA</text>
          <text x="390" y="283" class="center small">每个 K block：4 个 K-step × 2 个 M-wave</text>
          <path d="M390 300 V322" class="arrow" />
          <path d="M390 322 H197 V341" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M390 322 H583 V341" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <g v-for="cta in [0, 1]" :key="cta" :transform="`translate(${24 + cta * 386}, 346)`">
            <rect width="346" height="94" rx="9" class="box result" />
            <text x="173" y="27" class="center label result">CTA {{ cta }} 的本地结果</text>
            <text x="173" y="53" class="center">TMEM [128 lanes, 512 columns]</text>
            <text x="173" y="77" class="center small">对应 D [256,256] · 各自写回</text>
          </g>
        </g>

        <g v-else-if="view === 'storage'">
          <g v-for="(node, i) in storage" :key="node.name" :transform="`translate(0, ${12 + i * 96})`">
            <rect x="190" width="560" height="66" rx="9" class="box" :class="node.color" />
            <text x="212" y="26" class="label" :class="node.color">{{ node.name }}</text>
            <text x="212" y="50" class="small">{{ node.shape }}</text>
            <template v-if="node.step">
              <path d="M470 68 V92" class="arrow" :marker-end="`url(#${id}-arrow)`" />
              <text x="172" y="87" class="operation">{{ node.step }}</text>
            </template>
          </g>
          <text x="24" y="39" class="small">每 CTA</text>
        </g>

        <g v-else-if="view === 'tmem'">
          <text x="33" y="32" class="label">逻辑输出 [256,256]</text>
          <rect x="34" y="72" width="222" height="102" class="box result" />
          <rect x="34" y="174" width="222" height="102" class="box wave" />
          <text x="145" y="112" class="center label result">wave 0</text>
          <text x="145" y="143" class="center">行 [0,128)</text>
          <text x="145" y="215" class="center label wave">wave 1</text>
          <text x="145" y="246" class="center">行 [128,256)</text>
          <text x="145" y="301" class="center small">每行 256 个 FP32</text>
          <text x="380" y="32" class="label">物理 TMEM [128 lanes,512 columns]</text>
          <rect x="384" y="97" width="170" height="150" class="box result" />
          <rect x="554" y="97" width="170" height="150" class="box wave" />
          <text x="469" y="147" class="center label result">wave 0</text>
          <text x="639" y="147" class="center label wave">wave 1</text>
          <text x="469" y="183" class="center">cols [0,256)</text>
          <text x="639" y="183" class="center">cols [256,512)</text>
          <text x="554" y="272" class="center small">两块都使用 lanes [0,128)</text>
          <path d="M258 122 H319 V74 H469 V94" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M258 226 H302 V315 H746 V200 H727" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="34" y="354" class="small">Epilogue：输出行 = tile 起始行 + w × 128 + local_tid</text>
        </g>

        <g v-else>
          <text x="172" y="28" class="small">先后关系 →（旧 tile 的最后一个 K block 与写回尾部）</text>
          <g v-for="(lane, i) in ['TMA 搬运', 'Tensor Core', 'Epilogue 线程']" :key="lane">
            <rect x="164" :y="58 + i * 88" width="592" height="62" rx="7" class="panel" />
            <text x="18" :y="93 + i * 88" class="label">{{ lane }}</text>
          </g>
          <rect x="180" y="72" width="115" height="34" rx="5" class="box input" />
          <text x="237" y="95" class="center small input">最后一批 A/B</text>
          <rect x="417" y="72" width="103" height="34" rx="5" class="box result" />
          <text x="468" y="95" class="center small result">较早 Store</text>
          <rect x="604" y="72" width="139" height="34" rx="5" class="box result" />
          <text x="674" y="95" class="center small result">最后一笔 Store</text>
          <rect x="310" y="160" width="91" height="34" rx="5" class="box result" />
          <text x="355" y="183" class="center small result">末轮 UMMA</text>
          <rect x="588" y="160" width="155" height="34" rx="5" class="box wave" />
          <text x="665" y="183" class="center small wave">下一 tile UMMA</text>
          <rect x="412" y="247" width="95" height="36" rx="5" class="box result" />
          <text x="459" y="270" class="center small result">chunk 0…6</text>
          <rect x="525" y="247" width="55" height="36" rx="5" class="box result" />
          <text x="552" y="270" class="center small result">末块</text>
          <path d="M295 106 V139 H310 V157" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M401 194 V223 H412 V244" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M580 247 V215 H588 V197" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M580 265 H596 V109 H604" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M580 290 V316" class="guide" />
          <text x="374" y="339" class="small">最后一次 TMEM 读取结束 → 256 次 arrival</text>
          <text x="374" y="362" class="small">tmem_empty 就绪，允许下一 tile 覆盖 TMEM</text>
          <text x="18" y="394" class="small">发射者：Load 由 warp 0；UMMA 由 leader warp 1；Store 由一个 epilogue 线程。</text>
        </g>
      </svg>
    </div>
    <figcaption>{{ info.caption }}</figcaption>
  </figure>
</template>

<style scoped>
.bw-figure { margin: 1.6rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.bw-title { padding: 1rem 1.1rem 0.4rem; line-height: 1.6; }
.bw-scroll { padding: 0.35rem; overflow-x: auto; }
.bw-scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 700px; font-family: inherit; }
text { fill: var(--vp-c-text); font-size: 16px; }
.label { font-size: 18px; font-weight: 650; }
.small { font-size: 15px; fill: var(--infra-diagram-muted); }
.center { text-anchor: middle; }
.operation { text-anchor: end; font-size: 14px; fill: var(--infra-diagram-muted); }
.panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.box { fill: var(--vp-c-bg); stroke: currentColor; stroke-width: 1.8; }
.input { color: var(--infra-token-flow); }
.weight { color: var(--infra-weight-flow); }
.result { color: var(--infra-execution); }
.wave { color: var(--infra-placement); }
text.input, text.weight, text.result, text.wave { fill: currentColor; }
.arrow { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.8; }
.arrow-head { fill: var(--infra-diagram-muted); }
.guide { stroke: var(--infra-diagram-muted); stroke-dasharray: 4 4; }
figcaption { padding: 0.8rem 1.1rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.65; }
@media print { .bw-figure { break-inside: avoid; } .bw-scroll { overflow: visible; } svg { min-width: 0; } }
</style>
