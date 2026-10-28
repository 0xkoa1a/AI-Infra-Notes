<script setup lang="ts">
import { computed, useId } from "vue"

type View = "moe-flow" | "cta-roles" | "token-pool" | "push-pull" | "dispatch-overlap" | "gemm-pipeline" | "combine-slots" | "expert-wave" | "x-stage" | "metadata-layout" | "token-permute"
const props = defineProps<{ view: View }>()
const id = useId()
const arrow = `url(#${id}-arrow)`
const titles: Record<View, string> = {
  "moe-flow": "普通 MoE 推理流程",
  "cta-roles": "单个 SM 上的 Persistent CTA",
  "token-pool": "L1 Token Pool（按 M 维连续组织）",
  "push-pull": "假设 Rank 0 的 token x 需要送给 Rank 1 的 Expert E",
  "dispatch-overlap": "时间线示意",
  "gemm-pipeline": "GEMM 数据流水线",
  "combine-slots": "Rank 0：Token x 的 Combine Buffer",
  "expert-wave": "Expert-Wave 调度示例",
  "x-stage": "原始调度与 X-Stage 调度",
  "metadata-layout": "目标 Rank 2 的 metadata workspace",
  "token-permute": "源端 token 顺序与目标端 Expert-major 顺序",
}
const title = computed(() => titles[props.view])
const stages = [
  ["Dispatch All-to-All", "把 token 发送到专家所在 GPU"],
  ["Grouped GEMM L1", "Gate / Up projection"],
  ["SwiGLU + Quantization", ""],
  ["Grouped GEMM L2", "Down projection"],
  ["Combine All-to-All + Reduce", ""],
]
const pool = [
  { name: "E0", valid: 27, capacity: 32, tone: "acts" },
  { name: "E1", valid: 8, capacity: 16, tone: "result" },
  { name: "E2", valid: 19, capacity: 32, tone: "ready" },
]
const metadata = [
  { rank: "Src Rank 0", data: "[34, 61, ...]" },
  { rank: "Src Rank 1", data: "[8, 19, ...]" },
  { rank: "Src Rank 2", data: "[12, 27, ...]" },
]
const tokens = [
  { name: "T0", expert: 8 }, { name: "T1", expert: 9 },
  { name: "T2", expert: 8 }, { name: "T3", expert: 9 },
  { name: "T4", expert: 9 }, { name: "T5", expert: 8 },
]
const packed = [0, 2, 5, 1, 3, 4]
</script>

<template>
  <figure class="shared-mega" :aria-labelledby="`${id}-title`">
    <div :id="`${id}-title`" class="figure-title">{{ title }}</div>

    <div v-if="view === 'moe-flow'" class="vertical flow">
      <template v-for="(stage, index) in stages" :key="stage[0]">
        <div v-if="index" class="down" aria-hidden="true">↓</div>
        <div class="card" :class="{ acts: index === 1 || index === 3 }">
          <strong>{{ stage[0] }}</strong><span v-if="stage[1]" class="sub">{{ stage[1] }}</span>
        </div>
      </template>
    </div>

    <div v-else-if="view === 'cta-roles'" class="vertical">
      <div class="card"><div class="between"><strong>Dispatch warps</strong><span class="sub">4 warps</span></div><span>Routing 计数、远端 token 拉取、写入 L1 token pool</span></div>
      <div class="two">
        <div class="card acts"><strong>TMA Load A</strong><span class="sub">Activation / scale</span></div>
        <div class="card acts"><strong>TMA Load B</strong><span class="sub">Weight / scale</span></div>
      </div>
      <div class="card acts"><strong>MMA Issue warp</strong><span class="sub">提交 Blackwell UMMA 指令，结果进入 TMEM</span></div>
      <div class="down" aria-hidden="true">↓</div>
      <div class="card result"><div class="between"><strong>Epilogue warps</strong><span class="sub">多个 warpgroup</span></div><span>L1：SwiGLU + FP8 quantize + L2 pool 写入</span><span>L2：BF16 转换 + NVLink remote store</span></div>
    </div>

    <div v-else-if="view === 'token-pool'" class="vertical">
      <div v-for="expert in pool" :key="expert.name" class="pool-row">
        <strong>{{ expert.name }}</strong>
        <div class="pool-track" :aria-label="`${expert.name}：${expert.valid} valid，${expert.capacity - expert.valid} pad`">
          <div class="valid" :class="expert.tone" :style="{ width: `${expert.valid / expert.capacity * 100}%` }">{{ expert.valid }} valid</div>
          <div class="pad">{{ expert.capacity - expert.valid }} pad</div>
        </div>
      </div>
    </div>

    <div v-else-if="view === 'push-pull'" class="vertical">
      <div class="two">
        <div class="card"><strong>Rank 0</strong><span class="sub">Token owner</span><span>x、topk_idx、weight</span></div>
        <div class="card"><strong>Rank 1</strong><span class="sub">Expert owner</span><span>E 的权重、L1 pool</span></div>
      </div>
      <div class="scroll" role="region" tabindex="0" aria-label="Metadata push 与 token pull">
        <svg viewBox="0 0 900 160" role="img" :aria-labelledby="`${id}-push-title`">
          <title :id="`${id}-push-title`">Rank 0 push 路由元数据到 Rank 1；Rank 1 根据元数据从 Rank 0 pull token 内容。</title>
          <defs><marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker></defs>
          <text x="450" y="22" class="center small">Rank 0 push token 索引与路由元数据到 Rank 1</text>
          <path d="M110 38 H790" class="edge" :marker-end="arrow" />
          <text x="450" y="76" class="center small">Rank 1 根据元数据，从 Rank 0 pull token 内容</text>
          <path d="M790 92 H110" class="edge" :marker-end="arrow" />
          <text x="805" y="97" class="small">请求</text>
          <path d="M110 136 H790" class="edge" :marker-end="arrow" />
          <text x="450" y="128" class="center small">Token payload：Rank 0 → Rank 1</text>
        </svg>
      </div>
      <div class="card acts">Rank 1 的 Dispatch warp 将 x 放进 E 的 L1 token pool</div>
    </div>

    <div v-else-if="view === 'dispatch-overlap'" class="scroll" role="region" tabindex="0" aria-label="Dispatch 与 L1 按块推进的依赖示意">
      <svg viewBox="0 0 980 280" role="img" :aria-labelledby="`${id}-overlap-title`">
        <title :id="`${id}-overlap-title`">Block 0、1、2 各自沿 Dispatch、L1 MMA、L1 Epi 推进；后续块的搬运可与此前块的计算重叠。</title>
        <defs><marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker></defs>
        <g v-for="(lane, index) in [['Dispatch', 'NVLink / TMA'], ['L1 MMA', 'Tensor Core'], ['L1 Epi', 'CUDA Core / TMA']]" :key="lane[0]">
          <text x="8" :y="46 + index * 76">{{ lane[0] }}</text>
          <text x="8" :y="66 + index * 76" class="small muted">{{ lane[1] }}</text>
          <path :d="`M155 ${72 + index * 76} H950`" class="guide" />
        </g>
        <g v-for="block in [0, 1, 2]" :key="block">
          <g v-for="stage in [0, 1, 2]" :key="stage">
            <rect :x="160 + block * 210 + stage * 130" :y="22 + stage * 76" :width="stage === 2 ? 90 : 110" height="40" rx="5" class="tint" :class="stage === 2 ? 'result' : 'acts'" />
            <text :x="160 + block * 210 + stage * 130 + (stage === 2 ? 45 : 55)" :y="47 + stage * 76" class="center">Block {{ block }}</text>
          </g>
          <path :d="`M${270 + block * 210} 62 V86 H${290 + block * 210} V98 M${400 + block * 210} 138 V162 H${420 + block * 210} V174`" class="signal" :marker-end="arrow" />
        </g>
        <path d="M160 259 H950" class="edge" :marker-end="arrow" />
        <text x="945" y="249" text-anchor="end" class="small muted">t · 示意</text>
        <text x="160" y="241" class="small muted">Dispatch 与 L1 Epi 涉及共享的 TMA 资源</text>
      </svg>
    </div>

    <div v-else-if="view === 'gemm-pipeline'" class="vertical">
      <div class="two">
        <div class="card"><strong>TMA Load A</strong><span class="sub">Activation + SFA</span></div>
        <div class="card"><strong>TMA Load B</strong><span class="sub">Weight + SFB</span></div>
      </div>
      <div class="down" aria-hidden="true">↓</div>
      <div class="card"><strong>Shared Memory Pipeline</strong><div class="three"><div v-for="stage in [0, 1, 2]" :key="stage" class="card acts"><strong>Stage {{ stage }}</strong><span class="sub">A / B / SF</span></div></div><span class="sub">示意为三级；真实 stage 数由 SMEM 预算决定。</span></div>
      <div class="connector">↓ <span>Full mbarrier</span></div>
      <div class="connector">↓ <span>UMMA + UTCCP</span></div>
      <div class="card acts"><strong>Tensor Memory</strong><span>FP32 accumulator</span></div>
      <div class="down" aria-hidden="true">↓</div>
      <div class="card result"><strong>Epilogue Warpgroup</strong></div>
    </div>

    <div v-else-if="view === 'combine-slots'" class="vertical">
      <div class="three"><div v-for="(expert, index) in ['A', 'B', 'C']" :key="expert" class="card acts"><strong>topk slot {{ index }}</strong><span>Expert {{ expert }}</span></div></div>
      <div class="down" aria-hidden="true">↓</div>
      <div class="card result"><strong>Final Reduce</strong><span>FP32 累加 → BF16 输出 y</span></div>
    </div>

    <div v-else-if="view === 'expert-wave'" class="vertical">
      <template v-for="wave in [0, 1]" :key="wave">
        <div v-if="wave" class="down" aria-hidden="true">↓</div>
        <div class="card"><strong class="sub">Wave {{ wave }}</strong><div class="two"><div v-for="expert in [wave * 2, wave * 2 + 1]" :key="expert" class="card acts">E{{ expert }} · L1</div></div><div class="down" aria-hidden="true">↓</div><div class="two"><div v-for="expert in [wave * 2, wave * 2 + 1]" :key="expert" class="card result">E{{ expert }} · L2</div></div></div>
      </template>
    </div>

    <div v-else-if="view === 'x-stage'" class="vertical">
      <div v-for="(name, index) in ['原始调度', 'X-Stage 调度']" :key="name" class="schedule-row" role="region" tabindex="0" :aria-label="name"><strong>{{ name }}</strong><div class="sequence"><template v-for="(stage, j) in ['L2 MMA', 'L2 Epi', index === 0 ? 'Stall' : 'L1 MMA', 'L2 MMA']" :key="j"><span v-if="j" class="right" aria-hidden="true">→</span><div class="card" :class="j === 1 ? 'result' : j === 2 ? (index === 0 ? 'stall' : 'ready') : 'acts'">{{ stage }}</div></template></div></div>
    </div>

    <div v-else-if="view === 'metadata-layout'" class="vertical">
      <div class="card"><strong>Expert 9</strong><div v-for="row in metadata" :key="row.rank" class="metadata-row"><span class="sub">{{ row.rank }}</span><div class="card acts">{{ row.data }}</div></div><span class="sub">其他 Rank 的分区同理；括号内是 flattened token-topk index 示例。</span></div>
      <div class="card"><strong>Expert 8、10、11</strong><span class="sub">同样按源 Rank 分区存储</span></div>
    </div>

    <div v-else class="scroll" role="region" tabindex="0" aria-label="六个 token 按 Expert 8、Expert 9 重排">
      <svg viewBox="0 0 900 335" role="img" :aria-labelledby="`${id}-permute-title`">
        <title :id="`${id}-permute-title`">源端按 T0、T1、T2、T3、T4、T5 存储；目标端按 T0、T2、T5、T1、T3、T4 组成 Expert 8 和 Expert 9 的输入。</title>
        <defs><marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker></defs>
        <text x="20" y="25">源端：按原始 token 顺序存储</text>
        <g v-for="(token, index) in tokens" :key="token.name">
          <rect :x="20 + index * 145" y="43" width="130" height="64" rx="5" class="tint" :class="token.expert === 8 ? 'acts' : 'result'" />
          <text :x="85 + index * 145" y="69" class="center">{{ token.name }}</text><text :x="85 + index * 145" y="93" class="center small">E{{ token.expert }}</text>
          <path :d="`M${85 + index * 145} 111 C${85 + index * 145} 158,${85 + packed.indexOf(index) * 145} 159,${85 + packed.indexOf(index) * 145} 208`" class="edge token-edge" :marker-end="arrow" />
        </g>
        <rect x="190" y="139" width="520" height="29" rx="4" class="label-bg" /><text x="450" y="159" class="center small">Metadata 指导远端 gather + pool packing</text>
        <rect x="15" y="179" width="270" height="25" rx="4" class="label-bg" /><text x="20" y="198">目标端：L1 Token Pool</text>
        <g v-for="(source, index) in packed" :key="source">
          <rect :x="20 + index * 145" y="213" width="130" height="64" rx="5" class="tint" :class="tokens[source].expert === 8 ? 'acts' : 'result'" />
          <text :x="85 + index * 145" y="239" class="center">{{ tokens[source].name }}</text><text :x="85 + index * 145" y="263" class="center small">E{{ tokens[source].expert }}</text>
        </g>
        <path d="M20 284 V294 H440 V284 M455 284 V294 H875 V284" class="edge" />
        <text x="230" y="320" class="center small">Expert 8</text><text x="665" y="320" class="center small">Expert 9</text>
      </svg>
    </div>

    <figcaption v-if="view === 'cta-roles'">逻辑分工示意。原始实现还使用两个 CTA 组成 cluster 协同执行 MMA。</figcaption>
    <figcaption v-else-if="view === 'token-pool'">示例中的 5 个 M tiles 总共占据 80 行。不同专家的 tile 不会混在一起，方便直接按 expert 索引读取权重。</figcaption>
    <figcaption v-else-if="view === 'dispatch-overlap'">同一时间，不同角色可以处理不同 tile。这里展示了依赖关系，方块长度不表示实际运行时长。</figcaption>
    <figcaption v-else-if="view === 'expert-wave'">每个方块在实际执行时仍会拆成多个 <code>(m_block, n_block)</code> GEMM tile。</figcaption>
    <figcaption v-else-if="view === 'x-stage'">这是依赖与执行顺序示意，实际 MMA、epilogue 可在不同 warp 中并行。</figcaption>
    <figcaption v-else-if="view === 'token-permute'">简化示意，实际还会按照 BLOCK_M 对各 Expert 的池空间进行对齐，并使用专门的 scale factor 布局。</figcaption>
  </figure>
</template>

<style scoped>
.shared-mega { margin: 1.5rem 0; padding: 1rem; border: 1px solid var(--infra-diagram-line); border-radius: 12px; background: var(--vp-c-bg); }
.figure-title { margin-bottom: 1rem; font-weight: 600; }
.vertical { display: flex; flex-direction: column; gap: 0.65rem; }
.flow { max-width: 34rem; margin: 0 auto; }
.card { display: flex; flex-direction: column; gap: 0.35rem; padding: 0.8rem; border: 1px solid var(--infra-diagram-line); border-radius: 7px; background: var(--vp-c-bg-alt); text-align: center; }
.two, .three { display: grid; gap: 0.65rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.between { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 0.5rem; }
.sub { font-size: 0.85rem; color: var(--infra-diagram-muted); }
.down { text-align: center; color: var(--infra-diagram-muted); line-height: 1; font-size: 1.3rem; }
.connector { display: flex; align-items: center; justify-content: center; gap: 0.5rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
.card.acts, .valid.acts { border-color: var(--infra-token-flow); background: color-mix(in srgb, var(--infra-token-flow) 12%, var(--vp-c-bg)); }
.card.result, .valid.result { border-color: var(--infra-weight-flow); background: color-mix(in srgb, var(--infra-weight-flow) 12%, var(--vp-c-bg)); }
.card.ready, .valid.ready { border-color: var(--infra-execution); background: color-mix(in srgb, var(--infra-execution) 12%, var(--vp-c-bg)); }
.card.stall { border-style: dashed; border-color: var(--infra-diagram-muted); }
.pool-row { display: grid; grid-template-columns: 3rem 1fr; align-items: center; gap: 0.7rem; }
.pool-track { display: flex; min-height: 2.5rem; overflow: hidden; border: 1px solid var(--infra-diagram-line); border-radius: 5px; }
.valid, .pad { display: flex; align-items: center; justify-content: center; white-space: nowrap; font-size: 0.85rem; }
.valid { border-right: 1px solid var(--infra-diagram-line); }
.pad { flex: 1; color: var(--infra-diagram-muted); background: var(--vp-c-bg-alt); }
.schedule-row { display: flex; flex-direction: column; gap: 0.65rem; overflow-x: auto; }
.sequence { display: flex; align-items: center; gap: 0.4rem; min-width: 540px; }
.sequence .card { flex: 1; padding: 0.8rem 0.3rem; font-size: 0.9rem; }
.right { color: var(--infra-diagram-muted); }
.metadata-row { display: grid; grid-template-columns: 7rem 1fr; gap: 0.6rem; align-items: center; text-align: left; }
.scroll { overflow-x: auto; }
.scroll:focus-visible, .schedule-row:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: 3px; }
svg { display: block; width: 100%; min-width: 760px; font-family: inherit; }
text { fill: var(--vp-c-text); font-size: 16px; }
.center { text-anchor: middle; }
.small { font-size: 13px; }
.muted { fill: var(--infra-diagram-muted); }
.tint { fill: currentColor; fill-opacity: 0.14; stroke: currentColor; }
.acts { color: var(--infra-token-flow); }
.result { color: var(--infra-weight-flow); }
.ready { color: var(--infra-execution); }
.edge { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.5; }
.guide { fill: none; stroke: var(--infra-diagram-line); }
.signal { fill: none; stroke: var(--infra-execution); stroke-width: 1.4; stroke-dasharray: 4 3; }
.arrowhead { fill: var(--infra-diagram-muted); }
.token-edge { opacity: 0.55; }
.label-bg { fill: var(--vp-c-bg); }
figcaption { margin-top: 0.8rem; font-size: 0.85rem; color: var(--infra-diagram-muted); }
@media (max-width: 560px) { .shared-mega { padding: 0.75rem; } .card { padding: 0.65rem; } .three { gap: 0.4rem; } }
@media print { .scroll, .schedule-row { overflow: visible; } svg { min-width: 0; } .sequence { min-width: 0; } }
</style>
