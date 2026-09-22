<script setup lang="ts">
import { computed } from "vue"
import DiffusionFigure, { type DiffusionDiagram } from "./DiffusionFigure.vue"
type View = "overview" | "block" | "attention" | "cache"
const props = withDefaults(defineProps<{ view?: View }>(), { view: "block" })
const diagrams: Record<View, DiffusionDiagram> = {
  overview: {
    title: "MMDiT：双流输入与全局条件",
    height: 745,
    caption: "Token-level 文本进入持续更新的文本流；pooled text 与时间步生成全局条件。图像流最终还原为 latent 预测，交给 Scheduler 继续去噪。",
    boxes: [
      {"x": 25, "y": 20, "w": 235, "label": "文本 token 编码", "detail": "C_txt · [B, Nₜ, D_c]", "tone": "text"},
      {"x": 285, "y": 20, "w": 230, "label": "带噪 latent zₜ", "detail": "[B, C, Hₗ, Wₗ]", "tone": "image"},
      {"x": 555, "y": 20, "w": 225, "label": "Pooled text 与 t", "detail": "[B, D_p] 与 [B]", "tone": "condition"},
      {"x": 25, "y": 150, "w": 235, "label": "文本输入投影", "detail": "X_txt⁰ · [B, Nₜ, D]", "tone": "text"},
      {"x": 285, "y": 150, "w": 230, "label": "Patchify + Position", "detail": "Xᵢ⁰ · [B, Nᵢ, D]", "tone": "image"},
      {"x": 555, "y": 150, "w": 225, "label": "各自映射后相加", "detail": "c(t) · [B, D]", "tone": "condition"},
      {"x": 25, "y": 290, "w": 490, "label": "完整 MMDiT Block × (L − 1)", "detail": "双流持续更新 · 每层一次 Joint Attention", "tone": "joint"},
      {"x": 25, "y": 430, "w": 490, "label": "最后一个 Block · context-pre-only", "detail": "文本仍提供上下文；只保留图像更新", "tone": "joint"},
      {"x": 25, "y": 570, "w": 490, "label": "最终图像归一化、投影与 Unpatchify", "detail": "图像 token → [B, C_out, Hₗ, Wₗ]", "tone": "image"},
      {"x": 555, "y": 570, "w": 225, "label": "Scheduler", "detail": "更新 latent，继续去噪", "tone": "neutral"},
    ],
    wires: [
      {"path": "M142 92 V150", "tone": "text"},
      {"path": "M400 92 V150", "tone": "image"},
      {"path": "M667 92 V150", "tone": "condition"},
      {"path": "M142 222 V290", "tone": "text"},
      {"path": "M400 222 V290", "tone": "image"},
      {"path": "M142 362 V430", "tone": "text"},
      {"path": "M400 362 V430", "tone": "image"},
      {"path": "M400 502 V570", "tone": "image"},
      {"path": "M515 606 H555", "tone": "image"},
      {"path": "M667 222 V255 H540 V548 H490 V570", "tone": "condition", "dashed": true},
      {"path": "M540 326 H515", "tone": "condition", "dashed": true},
      {"path": "M540 466 H515", "tone": "condition", "dashed": true},
    ],
    annotations: [
      {"x": 400, "y": 701, "label": "L 为 Block 数；C_out 为输出通道数", "anchor": "middle"},
    ],
    legend: [
      {"tone": "text", "label": "━ 文本流"},
      {"tone": "image", "label": "━ 图像流"},
      {"tone": "joint", "label": "━ 联合计算"},
      {"tone": "condition", "label": "┄ 条件调制"},
    ],
  },
  attention: {
    title: "Joint Attention：同一矩阵中的四种关系",
    height: 455,
    caption: "展示一个样本、一个注意力头的逻辑矩阵。行是 Query，列是 Key；矩形面积仅用于区分关系，不代表 Nᵢ 与 Nₜ 的比例。",
    boxes: [
    ],
    wires: [
    ],
    annotations: [
      {"x": 330, "y": 37, "label": "Key：文本 Nₜ", "anchor": "middle"},
      {"x": 610, "y": 37, "label": "Key：图像 Nᵢ", "anchor": "middle"},
      {"x": 160, "y": 125, "label": "Query：文本 Nₜ", "anchor": "end"},
      {"x": 160, "y": 255, "label": "Query：图像 Nᵢ", "anchor": "end"},
      {"x": 470, "y": 365, "label": "每一行的 softmax 横跨全部 Nₜ + Nᵢ 个 Key", "anchor": "middle"},
      {"x": 470, "y": 411, "label": "输出保留 Query 的位置：文本更新文本流，图像更新图像流", "anchor": "middle"},
    ],
  },
  cache: {
    title: "文本缓存边界：首层交互前与交互后",
    height: 580,
    caption: "固定提示词可复用编码和初始投影。已知时间步下，首层文本 QKV 可提前求值；经过 Joint Attention 后的文本输出依赖当前图像，后续层继承这一依赖。",
    boxes: [
      {"x": 35, "y": 20, "w": 310, "label": "固定提示词 → 文本编码与投影", "detail": "X_txt⁰ · [B, Nₜ, D]", "tone": "text"},
      {"x": 455, "y": 20, "w": 310, "label": "时间步与 pooled text", "detail": "c(t) · [B, D]", "tone": "condition"},
      {"x": 35, "y": 165, "w": 310, "label": "首层文本 AdaLN → QKV", "detail": "依赖初始文本、c(t)、固定权重", "tone": "text"},
      {"x": 455, "y": 165, "w": 310, "label": "当前图像 AdaLN → QKV", "detail": "依赖当前 zₜ", "tone": "image"},
      {"x": 180, "y": 315, "w": 440, "label": "首层 Joint Attention", "detail": "文本 Query 读取当前图像 Key / Value", "tone": "joint"},
      {"x": 180, "y": 475, "w": 440, "label": "更新后的文本 → 后续层", "detail": "动态文本状态 · [B, Nₜ, D]", "tone": "text"},
    ],
    wires: [
      {"path": "M190 92 V165", "tone": "text"},
      {"path": "M455 56 H400 V201 H345", "tone": "condition", "dashed": true},
      {"path": "M610 92 V165", "tone": "condition", "dashed": true},
      {"path": "M190 237 V280 H300 V315", "tone": "text"},
      {"path": "M610 237 V280 H500 V315", "tone": "image"},
      {"path": "M400 387 V475", "tone": "joint"},
    ],
    annotations: [
      {"x": 420, "y": 438, "label": "文本输出依赖当前图像", "anchor": "start"},
    ],
    legend: [
      {"tone": "text", "label": "━ 文本流"},
      {"tone": "image", "label": "━ 图像流"},
      {"tone": "joint", "label": "━ 联合计算"},
      {"tone": "condition", "label": "┄ 条件调制"},
    ],
  },
  block: {
    title: "MMDiT Block：双流独立参数，一次联合 Attention",
    height: 985,
    caption: "完整 Block：实线表示 hidden state 的流动与残差，虚线表示条件系数；各路先独立投影 QKV 并拆分多头。最后一层的文本路径例外见正文。",
    legend: [{"tone": "text", "label": "━ 文本流"}, {"tone": "image", "label": "━ 图像流"}, {"tone": "joint", "label": "━ 联合计算"}, {"tone": "condition", "label": "┄ 条件调制"}],
    boxes: [
      { x: 160, y: 15, w: 480, label: "c(t) · 时间步 + pooled text", detail: "[B, D] · 两路各自产生调制系数", tone: "condition" },
      { x: 65, y: 130, w: 280, label: "Text hidden X_txt", detail: "[B, Nₜ, D]", tone: "text" },
      { x: 455, y: 130, w: 280, label: "Image hidden Xᵢ", detail: "[B, Nᵢ, D]", tone: "image" },
      { x: 65, y: 245, w: 280, label: "Text adaLN → Text QKV", detail: "文本专属权重 · [B, Nₜ, D]", tone: "text" },
      { x: 455, y: 245, w: 280, label: "Image adaLN → Image QKV", detail: "图像专属权重 · [B, Nᵢ, D]", tone: "image" },
      { x: 155, y: 380, w: 490, label: "Concat · 沿 token 序列维分别拼接", detail: "Q / K / V 各为 [B, nₕ, Nₜ + Nᵢ, dₕ]", tone: "joint" },
      { x: 155, y: 495, w: 490, h: 90, label: "Joint Attention · 非因果全注意力", detail: "沿联合 Key 轴归一化 · 输出 [B, nₕ, Nₜ + Nᵢ, dₕ]", tone: "joint" },
      { x: 155, y: 630, w: 490, label: "Split · 合并多头，再拆分序列", detail: "Oₜ : Nₜ 个 token     /     Oᵢ : Nᵢ 个 token", tone: "joint" },
      { x: 65, y: 760, w: 280, label: "Text output projection", detail: "Gate × 输出 + 文本残差", tone: "text" },
      { x: 455, y: 760, w: 280, label: "Image output projection", detail: "Gate × 输出 + 图像残差", tone: "image" },
      { x: 65, y: 885, w: 280, label: "Text adaLN → MLP", detail: "Gate + Residual → 下一层 X_txt", tone: "text" },
      { x: 455, y: 885, w: 280, label: "Image adaLN → MLP", detail: "Gate + Residual → 下一层 Xᵢ", tone: "image" },
    ],
    wires: [
      { path: "M 160 51 H 12 V 921 H 65", tone: "condition", dashed: true },
      { path: "M 12 281 H 65", tone: "condition", dashed: true },
      { path: "M 12 796 H 65", tone: "condition", dashed: true },
      { path: "M 640 51 H 788 V 921 H 735", tone: "condition", dashed: true },
      { path: "M 788 281 H 735", tone: "condition", dashed: true },
      { path: "M 788 796 H 735", tone: "condition", dashed: true },
      { path: "M 205 202 V 245", tone: "text" }, { path: "M 595 202 V 245", tone: "image" },
      { path: "M 205 317 V 350 H 280 V 380", tone: "text" },
      { path: "M 595 317 V 350 H 520 V 380", tone: "image" },
      { path: "M 400 452 V 495", tone: "joint" }, { path: "M 400 585 V 630", tone: "joint" },
      { path: "M 280 702 V 731 H 205 V 760", tone: "text" },
      { path: "M 520 702 V 731 H 595 V 760", tone: "image" },
      { path: "M 205 832 V 885", tone: "text" }, { path: "M 595 832 V 885", tone: "image" },
      { path: "M 65 166 H 40 V 818 H 65", tone: "text" },
      { path: "M 735 166 H 760 V 818 H 735", tone: "image" },
      { path: "M 205 850 H 40 V 943 H 65", tone: "text" },
      { path: "M 595 850 H 760 V 943 H 735", tone: "image" },
    ],
  },
}
const diagram = computed(() => diagrams[props.view])
const cells = [
  { label: "文本 → 文本", shape: "[Nₜ, Nₜ]", tone: "text" },
  { label: "文本 → 图像", shape: "[Nₜ, Nᵢ]", tone: "joint" },
  { label: "图像 → 文本", shape: "[Nᵢ, Nₜ]", tone: "joint" },
  { label: "图像 → 图像", shape: "[Nᵢ, Nᵢ]", tone: "image" },
]
</script>

<template>
  <DiffusionFigure :diagram="diagram" class="mmdit-diagram">
    <g v-if="view === 'attention'" class="attention-matrix">
      <g v-for="(cell, index) in cells" :key="cell.label" :transform="`translate(${195 + (index % 2) * 280}, ${65 + Math.floor(index / 2) * 130})`" :class="`attention-matrix--${cell.tone}`">
        <rect width="270" height="115" rx="8" />
        <text x="135" y="45" text-anchor="middle">{{ cell.label }}</text>
        <text class="attention-matrix__shape" x="135" y="80" text-anchor="middle">{{ cell.shape }}</text>
      </g>
    </g>
  </DiffusionFigure>
</template>

<style scoped>
.attention-matrix--text { color: var(--infra-planning); }
.attention-matrix--image { color: var(--infra-token-flow); }
.attention-matrix--joint { color: var(--infra-execution); }
.attention-matrix rect { fill: color-mix(in srgb, currentColor 10%, var(--vp-c-bg)); stroke: currentColor; stroke-width: 1.5; }
.attention-matrix text { fill: currentColor; font-size: 18px; font-weight: 600; }
.attention-matrix .attention-matrix__shape { fill: var(--vp-c-text); font-size: 16px; font-weight: 400; }
</style>
