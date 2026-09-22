<script setup lang="ts">
import { useId } from "vue"
const id = useId()
const positions = Array.from({ length: 64 }, (_, n) => ({ row: Math.floor(n / 8), col: n % 8 }))
const patches = Array.from({ length: 16 }, (_, n) => ({ index: n + 1, row: Math.floor(n / 4), col: n % 4 }))
</script>

<template>
  <figure class="dit-patchify" aria-label="Patch Embedding：8×8 latent 切成 16 个 patch，逐块投影后得到 16 个 token">
    <div class="patch-title"><strong>Patch Embedding · 从四维 latent 到三维 token 序列</strong></div>
    <div class="example">贯穿示例：B = 1，C = 4，Hₗ = Wₗ = 8，pₕ = p<sub>w</sub> = 2，D = 768</div>

    <div class="stage">
      <div class="stage-heading"><strong>1 · Latent feature map</strong><span class="shape">[B, C, Hₗ, Wₗ] = [1, 4, 8, 8]</span></div>
      <div class="drawing" tabindex="0" role="region" aria-label="latent 包含 8 行 8 列，每个位置有 4 个通道">
        <svg viewBox="0 0 740 245" role="img" :aria-labelledby="`${id}-latent-title`">
          <title :id="`${id}-latent-title`">8×8 空间网格，每个格子包含 C=4 个特征值</title>
          <g class="feature-map">
            <rect v-for="cell in positions" :key="`${cell.row}-${cell.col}`" :x="58 + cell.col * 24" :y="22 + cell.row * 24" width="24" height="24" />
          </g>
          <rect x="58" y="22" width="24" height="24" class="selected" />
          <text x="154" y="237" text-anchor="middle">Wₗ = 8</text><text x="9" y="124">Hₗ = 8</text>
          <path class="leader" d="M82 34 H316" />
          <text x="340" y="42" class="emphasis">一个空间位置 = 一个 C 维特征向量</text>
          <g v-for="channel in 4" :key="channel" class="channel-value">
            <rect :x="340 + (channel - 1) * 65" y="64" width="54" height="42" rx="3" />
            <text :x="367 + (channel - 1) * 65" y="91" text-anchor="middle">c{{ channel }}</text>
          </g>
          <text x="340" y="144">每个小格代表 latent 中一个位置的 4 个特征。</text>
          <text x="340" y="176">本例每个样本共有 4 × 8 × 8 = 256 个数。</text>
          <text x="340" y="208" class="secondary">图中展示一个样本；后续操作保留 B 轴。</text>
        </svg>
      </div>
    </div>
    <div class="flow-arrow">↓ 拆出 patch 内坐标，再重排轴；数值总量不变</div>

    <div class="stage">
      <div class="stage-heading"><strong>2 · 在原网格上划分 patch</strong><span class="shape">[B, H′, W′, C, pₕ, p<sub>w</sub>] = [1, 4, 4, 4, 2, 2]</span></div>
      <div class="drawing" tabindex="0" role="region" aria-label="原 8×8 网格按 2×2 分组得到 16 个 patch，仍保留全部 64 个空间位置">
        <svg viewBox="0 0 740 245" role="img" :aria-labelledby="`${id}-partition-title`">
          <title :id="`${id}-partition-title`">原始 8×8 网格被粗线分成 16 个 2×2 patch</title>
          <g class="feature-map"><rect v-for="cell in positions" :key="`${cell.row}-${cell.col}`" :x="58 + cell.col * 24" :y="22 + cell.row * 24" width="24" height="24" /></g>
          <g v-for="patch in patches" :key="patch.index" class="partition-patch" :class="{ 'is-selected': patch.index === 1 }" :data-patch="patch.index">
            <rect :x="58 + patch.col * 48" :y="22 + patch.row * 48" width="48" height="48" />
            <rect class="index-bg" :x="72 + patch.col * 48" :y="37 + patch.row * 48" width="20" height="19" rx="2" />
            <text :x="82 + patch.col * 48" :y="51 + patch.row * 48" text-anchor="middle">{{ patch.index }}</text>
          </g>
          <text x="154" y="237" text-anchor="middle">粗线：patch 边界；细线：原空间位置</text>
          <text x="340" y="48" class="emphasis">H′ = 8 / 2 = 4，W′ = 8 / 2 = 4</text>
          <text x="340" y="85">Nᵢ = H′W′ = 16 个 patch</text>
          <text x="340" y="122">每个 patch 包含 C × pₕ × p<tspan baseline-shift="sub" font-size="11">w</tspan> = 4 × 2 × 2 个数。</text>
          <text x="340" y="159">C、pₕ、p<tspan baseline-shift="sub" font-size="11">w</tspan> 仍是三个独立的轴。</text>
          <text x="340" y="196" class="secondary">橙色 patch 1 在下一步放大展示。</text>
        </svg>
      </div>
    </div>
    <div class="flow-arrow">↓ 只合并每个 patch 内的 C、pₕ、p<sub>w</sub> 三个轴</div>

    <div class="stage">
      <div class="stage-heading"><strong>3 · Flatten each patch · 每块变成一个向量</strong><span class="shape">[B, H′, W′, Cpₕp<sub>w</sub>] = [1, 4, 4, 16]</span></div>
      <div class="drawing" tabindex="0" role="region" aria-label="将 patch 1 的四个 2×2 通道按顺序展开成 16 维向量">
        <svg viewBox="0 0 740 210" role="img" :aria-labelledby="`${id}-flatten-title`">
          <title :id="`${id}-flatten-title`">单个 patch 的四个通道分别含四个数，展平得到十六个数；操作应用于全部十六个 patch</title>
          <defs><marker :id="`${id}-flatten-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" /></marker></defs>
          <text x="158" y="26" text-anchor="middle" class="selected-text">放大 patch 1 · [C, pₕ, p<tspan baseline-shift="sub" font-size="11">w</tspan>] = [4, 2, 2]</text>
          <g v-for="channel in 4" :key="channel" :transform="`translate(${38 + (channel - 1) * 64}, 61)`" class="channel-plane">
            <text x="20" y="-10" text-anchor="middle">c{{ channel }}</text>
            <rect v-for="cell in 4" :key="cell" :x="((cell - 1) % 2) * 20" :y="Math.floor((cell - 1) / 2) * 20" width="20" height="20" />
          </g>
          <path class="transform-wire" d="M297 81 H361" :marker-end="`url(#${id}-flatten-arrow)`" />
          <text x="330" y="61" text-anchor="middle">展平</text>
          <g v-for="value in 16" :key="value" class="patch-value" :data-value="value"><rect :x="389 + (value - 1) * 18" y="61" width="15" height="40" rx="1" /></g>
          <text x="532" y="135" text-anchor="middle">[x₁, x₂, …, x₁₆] · 长度 Cpₕp<tspan baseline-shift="sub" font-size="11">w</tspan> = 16</text>
          <text x="370" y="181" text-anchor="middle">全部 patch 都执行相同重排；仍保留 4 × 4 的网格位置。</text>
        </svg>
      </div>
    </div>
    <div class="flow-arrow">↓ 每个 patch 共用同一线性投影；Cpₕp<sub>w</sub> → D</div>

    <div class="stage">
      <div class="stage-heading"><strong>4 · Linear projection · 产生 D 维 embedding</strong><span class="shape">[B, H′, W′, D] = [1, 4, 4, 768]</span></div>
      <div class="drawing" tabindex="0" role="region" aria-label="每个 16 维 patch 向量经过同一个 16×768 的投影得到 768 维 token，仍按 4×4 排列">
        <svg viewBox="0 0 740 268" role="img" :aria-labelledby="`${id}-projection-title`">
          <title :id="`${id}-projection-title`">线性映射改变特征宽度，十六个 patch 变成十六个 768 维 token</title>
          <defs><marker :id="`${id}-projection-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" /></marker></defs>
          <rect x="35" y="41" width="270" height="129" rx="6" class="projection" />
          <text x="170" y="71" text-anchor="middle" class="emphasis">每个 patch：e = pW<tspan baseline-shift="sub" font-size="11">patch</tspan> + b<tspan baseline-shift="sub" font-size="11">patch</tspan></text>
          <text x="170" y="103" text-anchor="middle">W<tspan baseline-shift="sub" font-size="11">patch</tspan>：[Cpₕp<tspan baseline-shift="sub" font-size="11">w</tspan>, D] = [16, 768]</text>
          <text x="170" y="132" text-anchor="middle">b<tspan baseline-shift="sub" font-size="11">patch</tspan>：[D] = [768]</text>
          <text x="170" y="157" text-anchor="middle" class="selected-text">patch 1：16 个数 → token 1：768 个数</text>
          <path class="transform-wire" d="M305 105 H457" :marker-end="`url(#${id}-projection-arrow)`" />
          <text x="382" y="85" text-anchor="middle">逐 patch 投影</text>
          <g v-for="patch in patches" :key="patch.index" class="embedding-cell" :class="{ 'is-selected': patch.index === 1 }" :data-token="patch.index">
            <rect :x="478 + patch.col * 47" :y="17 + patch.row * 47" width="44" height="44" rx="3" />
            <text :x="500 + patch.col * 47" :y="35 + patch.row * 47" text-anchor="middle" class="small">{{ patch.index }}</text>
            <text :x="500 + patch.col * 47" :y="53 + patch.row * 47" text-anchor="middle">D</text>
          </g>
          <text x="571" y="231" text-anchor="middle">4 × 4 embedding grid；每格 D = 768 维</text>
          <text x="170" y="213" text-anchor="middle">p、e 分别是一个 patch 的输入、输出向量。</text>
          <text x="170" y="239" text-anchor="middle" class="secondary">这是学习特征的步骤，会改变数值。</text>
        </svg>
      </div>
    </div>
    <div class="flow-arrow">↓ 按行排列位置：n = uW′ + v；只合并 H′、W′ 两个轴</div>

    <div class="stage">
      <div class="stage-heading"><strong>5 · Flatten spatial positions · Transformer 输入</strong><span class="shape">[B, Nᵢ, D] = [1, 16, 768]</span></div>
      <div class="drawing" tabindex="0" role="region" aria-label="按行展开四乘四网格，完整展示按一到十六编号的十六个 token">
        <svg viewBox="0 0 740 138" role="img" :aria-labelledby="`${id}-sequence-title`">
          <title :id="`${id}-sequence-title`">十六个 embedding 按行变成十六个序列位置，每个位置保留全部 768 个特征</title>
          <g v-for="patch in patches" :key="patch.index" class="sequence-token" :class="{ 'is-selected': patch.index === 1 }" :data-token="patch.index">
            <rect :x="23 + (patch.index - 1) * 44" y="18" width="37" height="46" rx="3" />
            <text :x="41.5 + (patch.index - 1) * 44" y="46" text-anchor="middle">D</text>
            <text :x="41.5 + (patch.index - 1) * 44" y="86" text-anchor="middle">{{ patch.index }}</text>
          </g>
          <text x="370" y="121" text-anchor="middle">Nᵢ = 4 × 4 = 16；空间坐标编码在序列顺序中，D 维特征完整保留。</text>
        </svg>
      </div>
    </div>

    <details class="implementation">
      <summary>等价的 Conv2d 实现：查看轴的变化</summary>
      <div class="impl-step"><span>输入</span><span class="shape">[B, C, Hₗ, Wₗ]</span></div>
      <div class="impl-step"><span>↓ Conv2d：kernel = stride = (pₕ, p<sub>w</sub>)，输出通道为 D</span><span class="shape">[B, D, H′, W′]</span></div>
      <div class="impl-step"><span>↓ 只展平空间轴 H′、W′</span><span class="shape">[B, D, Nᵢ]</span></div>
      <div class="impl-step"><span>↓ 交换序列轴与特征轴</span><span class="shape">[B, Nᵢ, D]</span></div>
    </details>
    <figcaption>示例中的 D = 768 用于说明维度变化。拆分、轴重排与展平只改变数值的组织方式；线性投影产生新特征。u、v 是从零开始的 patch 行列坐标，n 是从零开始的序列位置；图中编号 1–16 便于追踪同一个 patch。位置编码在 Patch Embedding 之后相加。</figcaption>
  </figure>
</template>

<style scoped>
.dit-patchify { margin: 1.5rem 0 2rem; border: 1px solid var(--infra-diagram-line); border-radius: 8px; overflow: hidden; background: var(--vp-c-bg); color: var(--vp-c-text); }
.patch-title { padding: 1rem 1rem .5rem; }
.example { padding: 0 1rem .9rem; font-size: .85rem; color: var(--infra-diagram-muted); }
.stage { border-top: 1px solid var(--infra-diagram-line); }
.stage-heading { padding: .6rem 1rem .25rem; display: flex; flex-direction: column; gap: .25rem; font-size: .9rem; }
.shape { font-variant-numeric: tabular-nums; color: var(--infra-token-flow); }
.drawing { overflow-x: auto; }
.drawing:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -2px; }
svg { display: block; width: 100%; min-width: 680px; font-family: inherit; }
svg text { fill: var(--vp-c-text); font-size: 14px; }
svg .emphasis { font-weight: 600; }
svg .secondary { fill: var(--infra-diagram-muted); }
svg .small { font-size: 12px; }
.feature-map rect { fill: color-mix(in srgb, var(--infra-token-flow) 4%, var(--vp-c-bg)); stroke: var(--infra-diagram-line); stroke-width: .8; }
.partition-patch rect { fill: none; stroke: var(--infra-token-flow); stroke-width: 2; }
.partition-patch .index-bg { fill: var(--vp-c-bg); stroke: none; }
.selected, .channel-plane rect, .patch-value rect { fill: color-mix(in srgb, var(--infra-weight-flow) 12%, var(--vp-c-bg)); stroke: var(--infra-weight-flow); }
.channel-value rect, .projection, .embedding-cell rect, .sequence-token rect { fill: color-mix(in srgb, var(--infra-token-flow) 5%, var(--vp-c-bg)); stroke: var(--infra-token-flow); }
.is-selected rect { stroke: var(--infra-weight-flow); fill: color-mix(in srgb, var(--infra-weight-flow) 14%, var(--vp-c-bg)); }
.is-selected .index-bg { stroke: none; fill: var(--vp-c-bg); }
.is-selected text, .selected-text { fill: var(--infra-weight-flow); }
.leader { fill: none; stroke: var(--infra-weight-flow); stroke-width: 1.4; }
.transform-wire { fill: none; stroke: var(--infra-token-flow); stroke-width: 1.8; }
marker path { fill: var(--infra-token-flow); }
.flow-arrow { padding: .35rem 1rem; text-align: center; color: var(--infra-diagram-muted); font-size: .85rem; }
.implementation summary { cursor: pointer; font-weight: 600; }
.implementation summary:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: 3px; }
.implementation { padding: .8rem 1rem; border-top: 1px solid var(--infra-diagram-line); font-size: .85rem; }
.impl-step { display: flex; justify-content: space-between; gap: .75rem; flex-wrap: wrap; margin-top: .7rem; }
figcaption { border-top: 1px solid var(--infra-diagram-line); padding: .8rem 1rem; color: var(--infra-diagram-muted); font-size: .85rem; line-height: 1.65; }
</style>
