<script setup lang="ts">
import { computed, useId } from "vue"

const props = defineProps<{ view: "cluster" | "storage" | "tmem" | "overlap" }>()
const id = useId()
const info = computed(() => ({
  cluster: { title: "A 按行分工，B 的两半共同参与每份输出", height: 386, caption: "每轮 K block 的逻辑分块；Bᵀ 仅表示计算方向，两半 B 仍分别放在各 CTA 的 SMEM。图中每个输出分块累加本轮贡献，遍历完整 K 后才得到最终结果。" },
  storage: { title: "每个 CTA：四份输入缓冲，一份累加区，两份输出缓冲", height: 488, caption: "方块数量表示独立缓冲份数。输入 stage 轮转使用；完整结果保留在 TMEM，随后按 chunk 搬出，交替经过 CD0、CD1 写回。" },
  tmem: { title: "逻辑上的 256 行，放进 TMEM 的 128 条 lane", height: 334, caption: "一个 CTA 的两个 wave 同时保存在不同 column 区间；另一 CTA 使用自己的 TMEM。" },
  overlap: { title: "读完旧结果，即可让下一 tile 使用 TMEM", height: 340, caption: "依赖示意，无实测时间比例；假定下一 tile 的输入已就绪。绿色为旧 tile，紫色为下一 tile；实际重叠受资源条件影响。" },
}[props.view]))
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
          <text x="104" y="34" class="label">A：按 M 分行</text>
          <text x="451" y="34" class="label">Bᵀ：按 N 分列</text>
          <text x="508" y="65" class="center small input">CTA 0 加载</text>
          <text x="640" y="65" class="center small wave">CTA 1 加载</text>
          <rect x="442" y="78" width="132" height="58" class="box input" />
          <rect x="574" y="78" width="132" height="58" class="box wave" />
          <text x="508" y="104" class="center input">B₀ᵀ</text>
          <text x="640" y="104" class="center wave">B₁ᵀ</text>
          <text x="508" y="125" class="center small">[64,128]</text>
          <text x="640" y="125" class="center small">[64,128]</text>
          <path d="M508 137 V184" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M640 137 V184" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <g v-for="cta in [0, 1]" :key="cta" :transform="`translate(0, ${188 + cta * 78})`">
            <rect x="104" width="150" height="78" class="box" :class="cta === 0 ? 'input' : 'wave'" />
            <text x="179" y="30" class="center label" :class="cta === 0 ? 'input' : 'wave'">A{{ cta === 0 ? '₀' : '₁' }}</text>
            <text x="179" y="56" class="center">[256,64]</text>
            <text x="20" y="44" class="small" :class="cta === 0 ? 'input' : 'wave'">CTA {{ cta }}</text>
            <path d="M255 39 H438" class="arrow" :marker-end="`url(#${id}-arrow)`" />
            <rect x="442" width="264" height="78" class="box" :class="cta === 0 ? 'input' : 'wave'" />
            <path d="M574 0 V78" class="guide" />
            <text x="508" y="44" class="center">A{{ cta === 0 ? '₀' : '₁' }}B₀ᵀ</text>
            <text x="640" y="44" class="center">A{{ cta === 0 ? '₀' : '₁' }}B₁ᵀ</text>
            <text x="719" y="34" class="small">CTA {{ cta }}</text>
            <text x="719" y="56" class="small">累加区</text>
          </g>
          <text x="349" y="173" class="center small">2CTA UMMA</text>
          <text x="179" y="371" class="center small">不同 M 行</text>
          <text x="574" y="371" class="center small">每 CTA 覆盖完整 256 列</text>
        </g>

        <g v-else-if="view === 'storage'">
          <text x="30" y="33" class="label">Global A / B</text>
          <text x="226" y="33" class="label input">输入 SMEM · 四个 stage</text>
          <rect x="210" y="49" width="545" height="90" rx="9" class="panel" />
          <g v-for="stage in [0, 1, 2, 3]" :key="stage" :transform="`translate(${226 + stage * 133}, 63)`">
            <rect width="114" height="62" rx="5" class="box input" />
            <text x="57" y="25" class="center input">stage {{ stage }}</text>
            <text x="57" y="48" class="center small">A + B</text>
          </g>
          <path d="M32 57 V93 H206" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="105" y="81" class="center small">TMA Load</text>
          <path d="M482 140 V192" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="500" y="172" class="small">UMMA · 逐 K block 累加</text>
          <text x="30" y="232" class="label result">TMEM · 一份</text>
          <rect x="354" y="196" width="256" height="64" rx="7" class="box result" />
          <text x="482" y="222" class="center label result">完整输出 tile</text>
          <text x="482" y="247" class="center small">[256,256] · FP32</text>
          <path d="M482 261 V314" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="500" y="286" class="small">按 chunk 读取</text>
          <text x="500" y="307" class="small">寄存器中 FP32 → BF16</text>
          <text x="30" y="361" class="label result">CD SMEM · 两份</text>
          <rect x="295" y="320" width="375" height="76" rx="9" class="panel" />
          <g v-for="stage in [0, 1]" :key="stage" :transform="`translate(${308 + stage * 184}, 332)`">
            <rect width="165" height="52" rx="5" class="box result" />
            <text x="82" y="22" class="center result">CD{{ stage }}</text>
            <text x="82" y="43" class="center small">chunk [128,64]</text>
          </g>
          <path d="M482 398 V447" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="500" y="429" class="small">TMA Store · 交替写回</text>
          <text x="482" y="473" class="center label">Global D</text>
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
        </g>

        <g v-else>
          <text x="173" y="28" class="small">执行进度 →</text>
          <g v-for="(lane, i) in ['TMA 写回', 'Tensor Core', 'Epilogue 线程']" :key="lane">
            <rect x="164" :y="53 + i * 83" width="592" height="59" rx="7" class="panel" />
            <text x="18" :y="88 + i * 83" class="label">{{ lane }}</text>
          </g>
          <rect x="342" y="66" width="112" height="33" rx="5" class="box result" />
          <text x="398" y="88" class="center small result">较早 Store</text>
          <rect x="576" y="66" width="160" height="33" rx="5" class="box result" />
          <text x="656" y="88" class="center small result">最后一笔 Store</text>
          <rect x="180" y="149" width="112" height="33" rx="5" class="box result" />
          <text x="236" y="171" class="center small result">末轮 UMMA</text>
          <rect x="576" y="149" width="160" height="33" rx="5" class="box wave" />
          <text x="656" y="171" class="center small wave">下一 tile UMMA</text>
          <rect x="310" y="232" width="129" height="33" rx="5" class="box result" />
          <text x="374" y="254" class="center small result">chunk 0…6</text>
          <rect x="454" y="232" width="68" height="33" rx="5" class="box result" />
          <text x="488" y="254" class="center small result">chunk 7</text>
          <path d="M292 166 H300 V248 H307" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M326 230 V83 H339" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M440 248 H451" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M523 248 H552 V166 H573" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M552 166 V83 H573" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <circle cx="552" cy="248" r="4" class="arrow-head" />
          <path d="M552 256 V290" class="guide" />
          <text x="552" y="317" class="center label">TMEM 可覆盖</text>
          <text x="656" y="129" class="center small">↕ 可以重叠</text>
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
