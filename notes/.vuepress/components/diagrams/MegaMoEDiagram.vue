<script setup lang="ts">
import { computed, useId } from "vue"

const props = defineProps<{ view: "flow" | "partition" | "readiness" | "ring" | "overlap" }>()
const id = useId()
const info = computed(() => ({
  flow: { title: "一个 token，两个专家，两个结果槽", height: 580, caption: "教学示例：GPU 0 的一个 token 选择 E₀、E₂。专家所在 GPU 拉取输入，分别计算，再把结果写回 GPU 0 的独立槽位。实际 GEMM 会把多个 token 组成 tile。" },
  partition: { title: "同一批 token，由两个 CTA 分别计算不同输出列", height: 370, caption: "示例为一个 K-block：Bₘ = 64、Bₖ = 128，每 CTA 负责 128 个原始输出特征。激活的两半分别加载后 multicast 给两个 CTA；权重按输出列分开加载。遍历完整 K 后才得到最终结果。" },
  readiness: { title: "两份 L1 输出，凑齐一个 L2 的 K-block", height: 360, caption: "教学尺寸 H = 512、F = 256、Bₘ = 64。前两个 L1 列块已写完，L2 可以加载 K ∈ [0,128)；后两个列块仍在准备。L2 的完整结果仍需要全部 K-block。" },
  ring: { title: "输入环与中间结果环，分别等待各自的消费者", height: 340, caption: "q 与 q + R 映射到同一物理槽 r。实线表示数据流，虚线表示允许覆盖的条件；两条释放条件独立，图中距离不表示耗时。" },
  overlap: { title: "旧结果离开 TMEM 后，远端回传仍可以继续", height: 385, caption: "展示 L2 → L1 → L2 交错序列的尾部依赖，省略中间 L1 和先前输出 chunk。任务 j 与 j + 2 使用同一 TMEM stage；假定后续输入已就绪。无实测时间比例，重叠仍受资源约束。" },
}[props.view]))
</script>

<template>
  <figure class="mega-figure" :data-view="view">
    <div class="mega-title"><strong>{{ info.title }}</strong></div>
    <div class="mega-scroll" tabindex="0" role="region" :aria-label="info.title">
      <svg :viewBox="`0 0 780 ${info.height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ info.title }}</title>
        <desc :id="`${id}-desc`">{{ info.caption }}</desc>
        <defs>
          <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" class="arrow-head" />
          </marker>
        </defs>

        <g v-if="view === 'flow'">
          <g v-for="gpu in [0, 1]" :key="gpu" :transform="`translate(${20 + gpu * 390}, 18)`">
            <rect width="350" height="367" rx="10" class="panel" />
            <text x="175" y="30" class="center label">GPU {{ gpu }} · E{{ gpu === 0 ? '₀、E₁' : '₂、E₃' }}</text>
            <rect x="85" y="57" width="180" height="44" rx="6" class="box input" />
            <text x="175" y="85" class="center input">{{ gpu === 0 ? '源 token x' : '拉取 x 的副本' }}</text>
            <path d="M175 103 V134" class="arrow" :marker-end="`url(#${id}-arrow)`" />
            <rect x="75" y="138" width="200" height="47" rx="6" class="box l1" />
            <text x="175" y="168" class="center l1">E{{ gpu === 0 ? '₀' : '₂' }} · Linear 1</text>
            <path d="M175 187 V213" class="arrow" :marker-end="`url(#${id}-arrow)`" />
            <rect x="75" y="217" width="200" height="47" rx="6" class="box l1" />
            <text x="175" y="247" class="center l1">SwiGLU × α{{ gpu === 0 ? '₀' : '₁' }}</text>
            <path d="M175 266 V292" class="arrow" :marker-end="`url(#${id}-arrow)`" />
            <rect x="75" y="296" width="200" height="47" rx="6" class="box l2" />
            <text x="175" y="326" class="center l2">Linear 2</text>
          </g>
          <path d="M286 97 H491" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="389" y="84" class="center small">peer memory</text>
          <rect x="20" y="430" width="350" height="134" rx="10" class="panel" />
          <text x="195" y="452" class="center small">源 GPU 0 · Combine</text>
          <rect x="43" y="468" width="130" height="38" rx="5" class="box l2" />
          <rect x="217" y="468" width="130" height="38" rx="5" class="box l2" />
          <text x="108" y="493" class="center l2">槽 0：z₀</text>
          <text x="282" y="493" class="center l2">槽 1：z₁</text>
          <path d="M195 362 V406 H108 V466" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M585 362 V408 H282 V466" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="541" y="397" class="small">写回源 GPU</text>
          <path d="M108 508 V526 H169" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M282 508 V526 H222" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="195" y="533" class="center label">＋</text>
          <text x="195" y="553" class="center small">y = z₀ + z₁</text>
        </g>

        <g v-else-if="view === 'partition'">
          <text x="108" y="34" class="center label input">激活 X</text>
          <text x="568" y="34" class="center label weight">权重 Wᵀ</text>
          <g v-for="cta in [0, 1]" :key="cta">
            <text :x="503 + cta * 132" y="66" class="center small">CTA {{ cta }} 加载</text>
            <rect :x="437 + cta * 132" y="79" width="132" height="63" class="box weight" />
            <text :x="503 + cta * 132" y="105" class="center weight">128 个输出列</text>
            <text :x="503 + cta * 132" y="129" class="center small">[128,128]</text>
            <path :d="`M${503 + cta * 132} 144 V184`" class="arrow" :marker-end="`url(#${id}-arrow)`" />
            <rect :x="437 + cta * 132" y="189" width="132" height="108" class="box l1" />
            <text :x="503 + cta * 132" y="218" class="center label l1">CTA {{ cta }}</text>
            <text :x="503 + cta * 132" y="248" class="center">[64,128]</text>
            <text :x="503 + cta * 132" y="278" class="center small">不同 N 列</text>
          </g>
          <rect x="35" y="189" width="146" height="54" class="box input" />
          <rect x="35" y="243" width="146" height="54" class="box input" />
          <text x="108" y="222" class="center input">32 × 128</text>
          <text x="108" y="276" class="center input">32 × 128</text>
          <text x="108" y="172" class="center small">各 CTA 加载一半</text>
          <path d="M183 243 H432" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="307" y="221" class="center small">multicast → 2CTA UMMA</text>
          <text x="108" y="327" class="center small">相同 64 行 token</text>
          <text x="569" y="327" class="center small">合计 [64,256] 原始输出</text>
          <text x="390" y="357" class="center small">每 CTA 的 TMEM：128 lanes × 64 columns</text>
        </g>

        <g v-else-if="view === 'readiness'">
          <text x="105" y="32" class="center label">L1 列块</text>
          <text x="370" y="32" class="center label">SwiGLU 后</text>
          <text x="641" y="32" class="center label">L2 输入</text>
          <g v-for="block in [0, 1, 2, 3]" :key="block" :transform="`translate(0, ${54 + block * 62})`">
            <rect x="30" width="150" height="44" rx="5" class="box" :class="block < 2 ? 'l1' : 'pending'" />
            <text x="105" y="28" class="center" :class="block < 2 ? 'l1' : 'muted'">n = {{ block }} · 128 列</text>
            <path d="M182 22 H307" class="arrow" :class="{ dashed: block > 1 }" :marker-end="`url(#${id}-arrow)`" />
            <rect x="312" width="116" height="44" rx="5" class="box" :class="block < 2 ? 'l1' : 'pending'" />
            <text x="370" y="28" class="center" :class="block < 2 ? 'l1' : 'muted'">[{{ block * 64 }},{{ (block + 1) * 64 }})</text>
          </g>
          <g v-for="pair in [0, 1]" :key="pair">
            <path :d="`M430 ${76 + pair * 124} H481 V${107 + pair * 124} H544`" class="arrow" :class="{ dashed: pair === 1 }" :marker-end="`url(#${id}-arrow)`" />
            <path :d="`M430 ${138 + pair * 124} H481 V${107 + pair * 124}`" class="arrow" :class="{ dashed: pair === 1 }" />
            <rect x="549" :y="73 + pair * 124" width="185" height="68" rx="7" class="box" :class="pair === 0 ? 'l2' : 'pending'" />
            <text x="641" :y="99 + pair * 124" class="center" :class="pair === 0 ? 'l2' : 'muted'">K ∈ [{{ pair * 128 }},{{ (pair + 1) * 128 }})</text>
            <text x="641" :y="125 + pair * 124" class="center small">{{ pair === 0 ? '可以加载' : '等待后两块' }}</text>
          </g>
          <text x="105" y="332" class="center small">就绪位：1 1 0 0</text>
          <text x="370" y="332" class="center small">每块留下 64 个特征</text>
        </g>

        <g v-else-if="view === 'ring'">
          <text x="390" y="32" class="center label">物理槽 r = q mod R</text>
          <g v-for="ring in [0, 1]" :key="ring" :transform="`translate(0, ${68 + ring * 142})`">
            <text x="27" y="40" class="label" :class="ring === 0 ? 'input' : 'l1'">{{ ring === 0 ? 'L1 输入环' : 'L2 输入环' }}</text>
            <rect x="176" width="195" height="70" rx="7" class="box" :class="ring === 0 ? 'input' : 'l1'" />
            <text x="273" y="29" class="center" :class="ring === 0 ? 'input' : 'l1'">{{ ring === 0 ? 'X(q)' : 'A(q)' }}</text>
            <text x="273" y="55" class="center small">{{ ring === 0 ? '[Bₘ,H]' : '[Bₘ,F]' }}</text>
            <path d="M373 35 H586" class="arrow dashed" :marker-end="`url(#${id}-arrow)`" />
            <text x="480" y="19" class="center small">{{ ring === 0 ? '所有 L1 N 块用完' : '所有 L2 N 块用完' }}</text>
            <rect x="591" width="153" height="70" rx="7" class="box dashed" :class="ring === 0 ? 'input' : 'l1'" />
            <text x="667" y="29" class="center" :class="ring === 0 ? 'input' : 'l1'">{{ ring === 0 ? 'X(q + R)' : 'A(q + R)' }}</text>
            <text x="667" y="55" class="center small">下一轮覆盖</text>
          </g>
          <path d="M273 140 V206" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="289" y="181" class="small">L1 + SwiGLU</text>
          <path d="M667 140 V206" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <text x="390" y="317" class="center small">统计“用完”的消费者；轮次区分同一槽的新旧内容</text>
        </g>

        <g v-else>
          <text x="172" y="27" class="small">执行进度 →</text>
          <g v-for="(lane, i) in ['TMA Load', 'Tensor Core', 'Epilogue 线程', 'NVLink']" :key="lane">
            <rect x="165" :y="48 + i * 74" width="590" height="54" rx="7" class="panel" />
            <text x="20" :y="81 + i * 74" class="label">{{ lane }}</text>
          </g>
          <rect x="182" y="59" width="83" height="32" rx="5" class="box l2" />
          <text x="223" y="81" class="center small l2">L2 输入</text>
          <rect x="448" y="59" width="119" height="32" rx="5" class="box l2" />
          <text x="507" y="81" class="center small l2">后续 L2 输入</text>
          <rect x="277" y="133" width="108" height="32" rx="5" class="box l2" />
          <text x="331" y="155" class="center small l2">L2 · 任务 j</text>
          <rect x="576" y="133" width="161" height="32" rx="5" class="box l2" />
          <text x="656" y="155" class="center small l2">L2 · 任务 j + 2</text>
          <rect x="401" y="207" width="86" height="32" rx="5" class="box l2" />
          <text x="444" y="229" class="center small l2">末次读取</text>
          <rect x="503" y="207" width="193" height="32" rx="5" class="box l2" />
          <text x="599" y="229" class="center small l2">剩余转换、peer store</text>
          <rect x="546" y="281" width="192" height="32" rx="5" class="box l2 dashed" />
          <text x="642" y="303" class="center small l2">请求继续传输</text>
          <path d="M265 75 H270 V149 H274" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M387 149 H394 V223 H398" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M489 223 H496 V149 H573" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M568 75 H571 V130" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M548 241 V278" class="arrow" :marker-end="`url(#${id}-arrow)`" />
          <path d="M489 225 V345" class="guide" />
          <text x="489" y="370" class="center small">同一 TMEM stage 可覆盖</text>
        </g>
      </svg>
    </div>
    <figcaption>{{ info.caption }}</figcaption>
  </figure>
</template>

<style scoped>
.mega-figure { margin: 1.6rem 0; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); overflow: hidden; }
.mega-title { padding: 1rem 1.1rem 0.4rem; line-height: 1.6; }
.mega-scroll { padding: 0.35rem; overflow-x: auto; }
.mega-scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 700px; font-family: inherit; }
text { fill: var(--vp-c-text); font-size: 16px; }
.label { font-size: 18px; font-weight: 650; }
.small, .muted { font-size: 15px; fill: var(--infra-diagram-muted); }
.center { text-anchor: middle; }
.panel { fill: var(--vp-c-bg-alt); stroke: var(--infra-diagram-line); }
.box { fill: var(--vp-c-bg); stroke: currentColor; stroke-width: 1.8; }
.input { color: var(--infra-token-flow); }
.weight { color: var(--infra-weight-flow); }
.l1 { color: var(--infra-execution); }
.l2 { color: var(--infra-placement); }
text.input, text.weight, text.l1, text.l2 { fill: currentColor; }
.pending { color: var(--infra-diagram-muted); stroke-dasharray: 5 4; }
.arrow { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.8; }
.arrow-head { fill: var(--infra-diagram-muted); }
.dashed { stroke-dasharray: 5 4; }
.guide { stroke: var(--infra-diagram-muted); stroke-dasharray: 4 4; }
figcaption { padding: 0.8rem 1.1rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.65; }
@media print { .mega-figure { break-inside: avoid; } .mega-scroll { overflow: visible; } svg { min-width: 0; } }
</style>
