<script setup lang="ts">
import { computed, useId } from "vue"

type View = "overview" | "patchify" | "conditioning" | "unpatchify" | "block"
type Tone = "image" | "text" | "joint" | "condition" | "neutral"
type Box = { x: number; y: number; w: number; h?: number; label: string; detail?: string; tone?: Tone }
type Wire = { path: string; tone?: Tone; dashed?: boolean }
type Diagram = { title: string; height: number; caption: string; boxes: Box[]; wires: Wire[] }

const props = withDefaults(defineProps<{ view?: View }>(), { view: "block" })
const id = useId()
const tones: Tone[] = ["image", "text", "joint", "condition", "neutral"]
const diagrams: Record<View, Diagram> = {
  overview: {
    title: "DiT：从带噪 latent 到去噪预测",
    height: 550,
    caption: "上方展示训练时的加噪路径；每个去噪步都重新执行蓝色骨干。预测保持 latent 布局，交给 Scheduler 更新。",
    boxes: [
      { x: 30, y: 20, w: 170, label: "Image", detail: "训练图像" },
      { x: 270, y: 20, w: 200, label: "VAE Encoder", detail: "图像 → latent" },
      { x: 540, y: 20, w: 230, label: "Clean latent z₀", detail: "Scheduler 加噪前" },
      { x: 540, y: 150, w: 230, label: "Noisy latent zₜ", detail: "[B, C, Hₗ, Wₗ]", tone: "image" },
      { x: 260, y: 150, w: 230, label: "Patchify + Position", detail: "切分、投影、加入位置", tone: "image" },
      { x: 30, y: 150, w: 180, label: "Image tokens", detail: "[B, Nᵢ, D]", tone: "image" },
      { x: 30, y: 285, w: 180, label: "DiT Block × L", detail: "逐层更新图像流", tone: "image" },
      { x: 290, y: 285, w: 200, label: "Output tokens", detail: "每个 token 对应一个 patch", tone: "image" },
      { x: 560, y: 285, w: 210, label: "Linear + Unpatchify", detail: "投影并恢复二维布局", tone: "image" },
      { x: 560, y: 425, w: 210, label: "去噪目标预测", detail: "噪声 / velocity / flow", tone: "joint" },
      { x: 260, y: 425, w: 230, label: "Scheduler 更新", detail: "zₜ → zₜ₋₁", tone: "neutral" },
    ],
    wires: [
      { path: "M200 56 H270" }, { path: "M470 56 H540" },
      { path: "M655 92 V150", dashed: true },
      { path: "M540 186 H490", tone: "image" }, { path: "M260 186 H210", tone: "image" },
      { path: "M120 222 V285", tone: "image" }, { path: "M210 321 H290", tone: "image" },
      { path: "M490 321 H560", tone: "image" }, { path: "M665 357 V425", tone: "image" },
      { path: "M560 461 H490", tone: "joint" },
    ],
  },
  patchify: {
    title: "Patchify：二维 latent → token 序列",
    height: 290,
    caption: "2 × 2 的不重叠窗口产生 16 × 16 个位置；每个位置投影为 D 维，因此一共得到 256 个 token。网格仅示意排列方式。",
    boxes: [
      { x: 20, y: 158, w: 220, label: "Latent · 32 × 32", detail: "[B, 4, 32, 32]", tone: "image" },
      { x: 290, y: 158, w: 220, label: "Patch grid · 16 × 16", detail: "[B, D, 16, 16]", tone: "image" },
      { x: 560, y: 158, w: 220, label: "256 tokens", detail: "[B, 256, D]", tone: "image" },
    ],
    wires: [{ path: "M240 194 H290", tone: "image" }, { path: "M510 194 H560", tone: "image" }],
  },
  conditioning: {
    title: "DiT Block：条件调制 + 两次残差更新",
    height: 580,
    caption: "Attention 与 MLP 顺序执行。条件 MLP 为两个子层分别生成 shift、scale、gate；虚线表示条件调制，旁路表示残差。",
    boxes: [
      { x: 40, y: 25, w: 230, label: "Condition c", detail: "时间步向量 + 类别向量", tone: "condition" },
      { x: 40, y: 140, w: 230, label: "Condition MLP", detail: "为两个子层各生成调制参数", tone: "condition" },
      { x: 420, y: 25, w: 270, label: "Image tokens", detail: "当前 Block 输入", tone: "image" },
      { x: 420, y: 140, w: 270, label: "adaLN → Self-Attention", detail: "shift、scale 调制归一化结果", tone: "image" },
      { x: 420, y: 255, w: 270, label: "Gate × Attention output + x", detail: "第一次残差更新", tone: "image" },
      { x: 420, y: 370, w: 270, label: "adaLN → MLP", detail: "使用第一次更新后的 hidden state", tone: "image" },
      { x: 420, y: 485, w: 270, label: "Gate × MLP output + x′", detail: "输出到下一个 Block", tone: "image" },
    ],
    wires: [
      { path: "M155 97 V140", tone: "condition" },
      { path: "M270 176 H420", tone: "condition", dashed: true },
      { path: "M310 176 V521 H420", tone: "condition", dashed: true },
      { path: "M310 291 H420", tone: "condition", dashed: true },
      { path: "M310 406 H420", tone: "condition", dashed: true },
      { path: "M555 97 V140", tone: "image" }, { path: "M555 212 V255", tone: "image" },
      { path: "M555 327 V370", tone: "image" }, { path: "M555 442 V485", tone: "image" },
      { path: "M690 61 H750 V291 H690", tone: "image" },
      { path: "M555 346 H725 V521 H690", tone: "image" },
    ],
  },
  unpatchify: {
    title: "Unpatchify：把输出 patch 放回二维网格",
    height: 350,
    caption: "Final Linear 已将每个 token 投影成 pₕ × p𝓌 × C_out 个值；Unpatchify 只重排这些值，恢复 patch 的空间邻接关系。",
    boxes: [
      { x: 35, y: 22, w: 730, label: "Final Linear 输出 · 每个 token 对应一个展开的 patch", detail: "[B, H′ × W′, pₕ × p𝓌 × C_out]", tone: "image" },
      { x: 35, y: 142, w: 730, label: "Reshape · 显式展开 patch 网格与 patch 内部坐标", detail: "[B, H′, W′, pₕ, p𝓌, C_out]", tone: "image" },
      { x: 35, y: 262, w: 730, label: "Permute + merge · 拼回二维 latent", detail: "[B, C_out, H′ × pₕ, W′ × p𝓌]", tone: "joint" },
    ],
    wires: [{ path: "M400 94 V142", tone: "image" }, { path: "M400 214 V262", tone: "image" }],
  },
  block: {
    title: "MMDiT Block：双流独立参数，一次联合 Attention",
    height: 1100,
    caption: "展示完整双流 Block。两路分别生成 QKV，沿序列维拼接后共同计算，再按 token 边界拆分。常见 SD3 的 context_pre_only 最后一层省略完整文本输出更新与文本 MLP，最终保留图像流。",
    boxes: [
      { x: 160, y: 15, w: 480, label: "e = TimeTextEmbed(t, pooled text)", detail: "调制两路 adaLN 与 Gate · 不加入 token 序列", tone: "condition" },
      { x: 65, y: 130, w: 280, label: "Text hidden Xₜ", detail: "[B, Nₜ, D]", tone: "text" },
      { x: 455, y: 130, w: 280, label: "Image hidden Xᵢ", detail: "[B, Nᵢ, D]", tone: "image" },
      { x: 65, y: 245, w: 280, label: "Text adaLN → Text QKV", detail: "文本专属权重 · Qₜ, Kₜ, Vₜ", tone: "text" },
      { x: 455, y: 245, w: 280, label: "Image adaLN → Image QKV", detail: "图像专属权重 · Qᵢ, Kᵢ, Vᵢ", tone: "image" },
      { x: 155, y: 380, w: 490, label: "Concat · 沿 token 序列维分别拼接", detail: "Q = [Qₜ; Qᵢ]    K = [Kₜ; Kᵢ]    V = [Vₜ; Vᵢ]", tone: "joint" },
      { x: 155, y: 495, w: 490, h: 205, label: "Joint Attention · 非因果全注意力", detail: "同一 Query 对全部 Nₜ + Nᵢ 个 Key 做一次 softmax", tone: "joint" },
      { x: 155, y: 745, w: 490, label: "Split · 按原 token 边界拆分输出", detail: "Oₜ : Nₜ 个 token     /     Oᵢ : Nᵢ 个 token", tone: "joint" },
      { x: 65, y: 875, w: 280, label: "Text output projection", detail: "Gate × 输出 + 文本残差", tone: "text" },
      { x: 455, y: 875, w: 280, label: "Image output projection", detail: "Gate × 输出 + 图像残差", tone: "image" },
      { x: 65, y: 1000, w: 280, label: "Text adaLN → MLP", detail: "Gate + Residual → 下一层 Xₜ", tone: "text" },
      { x: 455, y: 1000, w: 280, label: "Image adaLN → MLP", detail: "Gate + Residual → 下一层 Xᵢ", tone: "image" },
    ],
    wires: [
      { path: "M160 51 H12 V1036 H65", tone: "condition", dashed: true },
      { path: "M12 281 H65", tone: "condition", dashed: true },
      { path: "M12 911 H65", tone: "condition", dashed: true },
      { path: "M640 51 H788 V1036 H735", tone: "condition", dashed: true },
      { path: "M788 281 H735", tone: "condition", dashed: true },
      { path: "M788 911 H735", tone: "condition", dashed: true },
      { path: "M205 202 V245", tone: "text" }, { path: "M595 202 V245", tone: "image" },
      { path: "M205 317 V350 H280 V380", tone: "text" },
      { path: "M595 317 V350 H520 V380", tone: "image" },
      { path: "M400 452 V495", tone: "joint" }, { path: "M400 700 V745", tone: "joint" },
      { path: "M280 817 V846 H205 V875", tone: "text" },
      { path: "M520 817 V846 H595 V875", tone: "image" },
      { path: "M205 947 V1000", tone: "text" }, { path: "M595 947 V1000", tone: "image" },
      { path: "M65 166 H40 V933 H65", tone: "text" },
      { path: "M735 166 H760 V933 H735", tone: "image" },
      { path: "M205 965 H40 V1058 H65", tone: "text" },
      { path: "M595 965 H760 V1058 H735", tone: "image" },
    ],
  },
}
const diagram = computed(() => diagrams[props.view])
</script>

<template>
  <figure class="mmdit-diagram" :aria-label="diagram.title">
    <div class="mmdit-diagram__title"><strong>{{ diagram.title }}</strong></div>
    <div v-if="view === 'block'" class="mmdit-diagram__legend">
      <span class="mmdit-diagram__text">━ 文本流</span>
      <span class="mmdit-diagram__image">━ 图像流</span>
      <span class="mmdit-diagram__joint">━ 联合计算</span>
      <span class="mmdit-diagram__condition">┄ 条件调制</span>
    </div>
    <p class="mmdit-diagram__hint">左右滑动查看完整架构图</p>
    <div class="mmdit-diagram__scroll" tabindex="0" role="region" :aria-label="`${diagram.title}，窄屏可横向滚动`">
      <svg :viewBox="`0 0 800 ${diagram.height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ diagram.title }}</title>
        <desc :id="`${id}-desc`">{{ diagram.caption }}</desc>
        <defs>
          <marker v-for="tone in tones" :id="`${id}-${tone}`" :key="tone" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 Z" :class="`mmdit-diagram__${tone}`" fill="currentColor" />
          </marker>
        </defs>
        <g fill="none" stroke-width="1.8" stroke-linejoin="round">
          <path v-for="(wire, index) in diagram.wires" :key="index" :d="wire.path" :class="`mmdit-diagram__${wire.tone ?? 'neutral'}`" stroke="currentColor" :stroke-dasharray="wire.dashed ? '6 5' : undefined" :marker-end="`url(#${id}-${wire.tone ?? 'neutral'})`" />
        </g>
        <g v-for="box in diagram.boxes" :key="box.label" :transform="`translate(${box.x}, ${box.y})`" :class="`mmdit-diagram__${box.tone ?? 'neutral'}`">
          <rect class="mmdit-diagram__box" :width="box.w" :height="box.h ?? 72" rx="9" />
          <text class="mmdit-diagram__label" :x="box.w / 2" y="29" text-anchor="middle">{{ box.label }}</text>
          <text class="mmdit-diagram__detail" :x="box.w / 2" y="52" text-anchor="middle">{{ box.detail }}</text>
        </g>
        <g v-if="view === 'overview'" class="mmdit-diagram__annotations">
          <text x="671" y="126">加噪</text>
          <text x="240" y="526">相邻去噪步按 Scheduler 结果依次推进</text>
        </g>
        <g v-if="view === 'patchify'">
          <g v-for="col in 4" :key="col" class="mmdit-diagram__image">
            <rect v-for="row in 4" :key="row" class="mmdit-diagram__tile" :x="76 + (col - 1) * 28" :y="20 + (row - 1) * 28" width="25" height="25" rx="2" />
            <rect v-for="row in 4" :key="`patch-${row}`" class="mmdit-diagram__tile" :x="350 + (col - 1) * 25" :y="30 + (row - 1) * 25" width="20" height="20" rx="3" />
          </g>
          <g class="mmdit-diagram__image">
            <rect v-for="token in 8" :key="token" class="mmdit-diagram__tile" :x="574 + (token - 1) * 24" y="53" width="18" height="54" rx="3" />
          </g>
          <g class="mmdit-diagram__annotations" text-anchor="middle">
            <text x="265" y="255">Conv2d · kernel = stride = 2</text>
            <text x="550" y="277">Flatten spatial + transpose</text>
          </g>
        </g>
        <g v-if="view === 'block'" class="mmdit-diagram__matrix">
          <g v-for="(cell, index) in ['Text → Text', 'Text → Image', 'Image → Text', 'Image → Image']" :key="cell" :transform="`translate(${205 + (index % 2) * 200}, ${570 + Math.floor(index / 2) * 52})`" :class="index === 0 ? 'mmdit-diagram__text' : index === 3 ? 'mmdit-diagram__image' : 'mmdit-diagram__joint'">
            <rect class="mmdit-diagram__tile" width="190" height="44" rx="5" />
            <text x="95" y="27" text-anchor="middle">{{ cell }}</text>
          </g>
          <text class="mmdit-diagram__detail" x="400" y="689" text-anchor="middle">四种 Query → Key 关系 · 共同构成一个 Attention 矩阵</text>
        </g>
      </svg>
    </div>
    <figcaption>{{ diagram.caption }}</figcaption>
  </figure>
</template>

<style scoped>
.mmdit-diagram {
  margin: 1.6rem 0 2rem;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 12px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text);
  overflow: hidden;
}
.mmdit-diagram__title { padding: 1rem 1.2rem 0.5rem; }
.mmdit-diagram__legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1.2rem; padding: 0 1.2rem 0.8rem; font-size: 0.82rem; }
.mmdit-diagram__hint { display: none; }
.mmdit-diagram__scroll { overflow-x: auto; padding: 0.5rem; }
.mmdit-diagram__scroll:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -3px; }
svg { display: block; width: 100%; min-width: 660px; font-family: inherit; }
.mmdit-diagram__image { color: var(--infra-token-flow); }
.mmdit-diagram__text { color: var(--infra-planning); }
.mmdit-diagram__joint { color: var(--infra-execution); }
.mmdit-diagram__condition { color: var(--infra-weight-flow); }
.mmdit-diagram__neutral { color: var(--infra-diagram-muted); }
.mmdit-diagram__box { fill: var(--vp-c-bg); stroke: currentColor; stroke-width: 1.5; }
.mmdit-diagram__label { fill: currentColor; font-size: 16px; font-weight: 650; }
.mmdit-diagram__detail { fill: var(--vp-c-text); font-size: 12px; }
.mmdit-diagram__tile { fill: color-mix(in srgb, currentColor 10%, var(--vp-c-bg)); stroke: currentColor; stroke-width: 1.2; }
.mmdit-diagram__matrix text { fill: currentColor; font-size: 14px; }
.mmdit-diagram__matrix .mmdit-diagram__detail { fill: var(--vp-c-text); font-size: 12px; }
.mmdit-diagram__annotations { fill: var(--infra-diagram-muted); font-size: 13px; }
figcaption { padding: 0.9rem 1.2rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.65; }
@media (max-width: 719px) {
  .mmdit-diagram__hint { display: block; margin: 0.3rem 1.2rem; color: var(--infra-diagram-muted); font-size: 0.8rem; }
}
@media print {
  .mmdit-diagram { break-inside: avoid; }
  svg { min-width: 0; }
  .mmdit-diagram__scroll { overflow: visible; }
}
</style>
