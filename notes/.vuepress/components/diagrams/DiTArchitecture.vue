<script setup lang="ts">
import { computed, ref, useId } from "vue"
import katex from "katex"
import DiTPatchify from "./DiTPatchify.vue"

const props = defineProps<{ standaloneBlock?: boolean }>()
const id = useId()
type Detail = "patch" | "condition" | "block" | "output" | "ddpm"
const selected = ref<Detail | null>(null)
const detailOptions: { key: Detail; label: string }[] = [
  { key: "patch", label: "Patch Embedding" },
  { key: "condition", label: "条件编码" },
  { key: "block", label: "DiT Block" },
  { key: "output", label: "输出层与 Unpatchify" },
  { key: "ddpm", label: "DDPM 采样" },
]
const toggle = (key: Detail) => { selected.value = selected.value === key ? null : key }
const tex = String.raw
type Tone = "image" | "condition" | "sampler" | "neutral"
type Node = { x: number; y: number; w: number; h: number; title: string; tone: Tone; formulas: string[]; shapes: string[]; note?: string }
type Wire = { path: string; tone: Tone; dashed?: boolean }
type Panel = { key: string; title: string; height: number; nodes: Node[]; wires: Wire[]; labels?: { x: number; y: number; text: string; tone?: Tone }[] }
// Only author-owned formulas are rendered as HTML; invalid formulas fail at build time.
const math = (value: string) => katex.renderToString(value, { throwOnError: true, output: "htmlAndMathml" })
const node = (x: number, y: number, w: number, h: number, title: string, tone: Tone, formulas: string[] = [], shapes: string[] = [], note?: string): Node => ({ x, y, w, h, title, tone, formulas: formulas.map(math), shapes, note })
const wire = (path: string, tone: Tone = "image", dashed = false): Wire => ({ path, tone, dashed })
const tones: Tone[] = ["image", "condition", "sampler", "neutral"]

const overview: Panel = {
  key: "overview", title: "一次 DiT 前向与外部采样循环", height: 815,
  nodes: [
    node(105, 18, 330, 66, "当前带噪 latent · zₜ", "image", [], ["[B, C, Hₗ, Wₗ]；初次输入为随机噪声"]),
    node(520, 18, 235, 66, "时间步 t · 类别 y", "condition", [], ["各为 [B]；每次更新 t，保持 y"]),
    node(105, 142, 330, 80, "Patch Embedding + 位置编码", "image", [], ["[B, C, Hₗ, Wₗ] → [B, Nᵢ, D]"]),
    node(520, 142, 235, 80, "条件编码 · c(t)", "condition", [], ["时间步向量 + 类别向量", "[B, D]"]),
    node(105, 263, 330, 72, "DiT Block × L", "image", [], ["[B, Nᵢ, D] → [B, Nᵢ, D]"]),
    node(105, 376, 330, 72, "Final Layer · AdaLN + Linear", "image", [], ["每个 token → 一个输出 patch"]),
    node(105, 489, 330, 72, "Unpatchify · 恢复空间排列", "image", [], ["[B, 2C, Hₗ, Wₗ]：噪声与方差参数"]),
    node(105, 617, 330, 82, "DDPM Scheduler · 网络外部", "sampler", [], ["接收 zₜ、网络预测、时间表和新噪声", "计算下一步 latent"]),
    node(105, 740, 330, 62, "下一步 latent · zₜ₋₁", "sampler", [], ["[B, C, Hₗ, Wₗ]"]),
    node(520, 740, 235, 62, "VAE Decoder → 图像", "neutral", [], ["最后一步：解码 z₀"]),
  ],
  wires: [
    wire("M270 84 V142"), wire("M270 222 V263"), wire("M270 335 V376"), wire("M270 448 V489"), wire("M270 561 V617"),
    wire("M637 84 V142", "condition"), wire("M520 182 H480 V412 H435", "condition", true), wire("M480 299 H435", "condition", true),
    wire("M105 51 H76 V658 H105", "sampler"), wire("M270 699 V740", "sampler"),
    wire("M105 771 H28 V51 H105", "sampler"), wire("M435 771 H520", "sampler"),
  ],
  labels: [
    { x: 433, y: 126, text: "DiT 网络 · 一次前向" },
    { x: 620, y: 310, text: "各层独立生成调制参数", tone: "condition" },
    { x: 620, y: 413, text: "最终层生成 scale / shift", tone: "condition" },
    { x: 619, y: 652, text: "新噪声：每步独立采样（末步除外）", tone: "sampler" },
    { x: 479, y: 728, text: "采样结束 →", tone: "sampler" },
  ],
}

const conditioning: Panel = {
  key: "condition", title: "条件编码：时间步与类别分别编码，再逐元素相加", height: 435,
  nodes: [
    node(50, 20, 330, 60, "时间步 t", "condition", [], ["[B]"]),
    node(50, 118, 330, 82, "Sin-Cos 频率编码", "condition", [tex`\phi_t=[\cos(t\omega_k);\sin(t\omega_k)]_k`], ["[B] → [B, Dτ]"]),
    node(50, 240, 330, 110, "时间步 MLP", "condition", [tex`h_t=\operatorname{SiLU}(\phi_tW_1+b_1)`, tex`e_t=h_tW_2+b_2`], ["Dτ → D → D；保留 B 轴"]),
    node(460, 20, 290, 60, "类别索引 y", "condition", [], ["[B]"]),
    node(460, 118, 290, 82, "从可学习的类别表中取向量", "condition", [tex`e_y=E_y[y]`], ["[B, D]"]),
    node(460, 250, 290, 90, "⊕ 全局条件 · 所有层共享", "condition", [tex`c(t)=e_t+e_y`], ["[B, D]"]),
  ],
  wires: [wire("M215 80 V118", "condition"), wire("M215 200 V240", "condition"), wire("M605 80 V118", "condition"), wire("M605 200 V250", "condition"), wire("M380 295 H460", "condition")],
  labels: [{ x: 400, y: 402, text: "频率与各权重的定义见正文“时间步与类别条件”", tone: "condition" }],
}

const block: Panel = {
  key: "block-detail", title: "展开一层：归一化、调制、Attention / MLP、门控残差", height: 1260,
  nodes: [
    node(85, 20, 335, 65, "当前层图像状态", "image", [tex`X^\ell\quad[B,N_i,D]`]),
    node(85, 125, 335, 75, "LayerNorm · 沿 D 轴", "image", [tex`U_0=\operatorname{LN}(X^\ell)`]),
    node(85, 240, 335, 82, "条件调制 · Attention", "image", [tex`U=U_0\odot(1+s_a)+b_a`], ["[B, 1, D] 沿 Nᵢ 广播"]),
    node(85, 362, 335, 140, "多头 Self-Attention", "image", [tex`[Q,K,V]=UW_{qkv}+b_{qkv}`, tex`A=\operatorname{softmax}_{\rm Key}(QK^\top/\sqrt{d_h})`, tex`O=\operatorname{MergeHeads}(AV)W_O+b_O`], ["Q、K、V 各：[B, nₕ, Nᵢ, dₕ]；dₕ = D/nₕ"]),
    node(85, 542, 335, 60, "Gate · 逐特征控制写回幅度", "image", [tex`G_a=g_a\odot O`]),
    node(85, 642, 335, 65, "⊕ Attention 残差", "image", [tex`X'=X^\ell+G_a`]),
    node(85, 747, 335, 68, "LayerNorm · 使用更新后的 X′", "image", [tex`V_0=\operatorname{LN}(X')`]),
    node(85, 855, 335, 90, "条件调制 · MLP", "image", [tex`V_m=V_0\odot(1+s_m)+b_m`], ["[B, 1, D] 沿 Nᵢ 广播"]),
    node(85, 973, 335, 88, "逐 token MLP", "image", [], ["Linear → GELU → Linear", "[B, Nᵢ, D] → [B, Nᵢ, rD] → [B, Nᵢ, D]"]),
    node(85, 1088, 335, 60, "Gate · MLP", "image", [tex`G_m=g_m\odot\operatorname{MLP}(V_m)`]),
    node(85, 1180, 335, 60, "⊕ MLP 残差 → 下一层", "image", [tex`X^{\ell+1}=X'+G_m\quad[B,N_i,D]`]),
    node(520, 20, 250, 65, "全局条件 c(t)", "condition", [], ["[B, D]；本层权重独立"]),
    node(520, 125, 250, 95, "SiLU → Linear → 沿特征轴拆分", "condition", [tex`m_\ell=\operatorname{Linear}_\ell(\operatorname{SiLU}(c))`], ["[B, D] → [B, 6D] → 六份 [B, D]"]),
    node(520, 244, 250, 74, "Attention 调制", "condition", [tex`s_a,b_a\quad[B,D]`]),
    node(520, 539, 250, 66, "Attention Gate", "condition", [tex`g_a\quad[B,D]`]),
    node(520, 857, 250, 74, "MLP 调制", "condition", [tex`s_m,b_m\quad[B,D]`]),
    node(520, 1085, 250, 66, "MLP Gate", "condition", [tex`g_m\quad[B,D]`]),
  ],
  wires: [wire("M252 85 V125"), wire("M252 200 V240"), wire("M252 322 V362"), wire("M252 502 V542"), wire("M252 602 V642"), wire("M252 707 V747"), wire("M252 815 V855"), wire("M252 945 V973"), wire("M252 1061 V1088"), wire("M252 1148 V1180"), wire("M85 52 H36 V674 H85"), wire("M252 727 H55 V1210 H85"), wire("M645 85 V125", "condition"), wire("M770 172 H789 V1118 H770", "condition", true), wire("M789 281 H770", "condition", true), wire("M789 572 H770", "condition", true), wire("M789 894 H770", "condition", true), wire("M520 281 H420", "condition", true), wire("M520 572 H420", "condition", true), wire("M520 894 H420", "condition", true), wire("M520 1118 H420", "condition", true)],
}

const output: Panel = {
  key: "output", title: "输出：条件归一化 → 预测 patch → 恢复空间轴", height: 875,
  nodes: [
    node(60, 20, 355, 70, "最终 LayerNorm", "image", [tex`F_0=\operatorname{LN}(X^L)\quad[B,N_i,D]`]),
    node(60, 130, 355, 82, "条件调制 · Final AdaLN", "image", [tex`F=F_0\odot(1+s_f)+b_f`], ["沿 D 归一化；[B, 1, D] 沿 Nᵢ 广播"]),
    node(60, 252, 355, 98, "Final Linear · 预测输出 patch", "image", [tex`R=FW_{\rm out}+b_{\rm out}`, tex`[B,N_i,D]\to[B,N_i,p_hp_wC_{\rm out}]`]),
    node(60, 390, 355, 72, "Reshape · 恢复网格与 patch 坐标", "image", [tex`[B,H',W',p_h,p_w,C_{\rm out}]`]),
    node(60, 502, 355, 72, "Permute · 通道前置，空间轴相邻", "image", [tex`[B,C_{\rm out},H',p_h,W',p_w]`]),
    node(60, 614, 355, 98, "Merge · 拼回 latent 空间布局", "image", [tex`[B,C_{\rm out},H' p_h,W' p_w]`, tex`=[B,C_{\rm out},H_l,W_l]`]),
    node(505, 20, 265, 94, "最终条件投影 · 独立参数", "condition", [tex`\operatorname{Linear}_f(\operatorname{SiLU}(c))`], ["[B, D] → [B, 2D]"]),
    node(505, 138, 265, 66, "沿特征轴拆成两份", "condition", [tex`b_f,s_f\quad[B,D]`]),
    node(505, 288, 265, 110, "本图：噪声预测 + 学习方差", "neutral", [tex`C_{\rm out}=2C`], [], "仅预测噪声时输出 C 个通道；采样规则需匹配训练目标。"),
    node(60, 738, 710, 110, "按通道轴拆分网络输出 · 分界在 C", "image", [tex`f_\theta(z_t,t,y)=[\hat\epsilon;\hat v]_{\rm channel}`], ["前 C 个通道：噪声预测 ε̂；后 C 个通道：方差参数 v̂", "两路各为 [B, C, Hₗ, Wₗ]"]),
  ],
  wires: [wire("M237 0 V20"), wire("M237 90 V130"), wire("M237 212 V252"), wire("M237 350 V390"), wire("M237 462 V502"), wire("M237 574 V614"), wire("M237 712 V738"), wire("M637 114 V138", "condition"), wire("M505 171 H415", "condition", true)],
}

const sampling: Panel = {
  key: "ddpm", title: "DDPM：均值、方差与新噪声共同产生下一步 latent", height: 580,
  nodes: [
    node(60, 20, 710, 70, "保留本轮输入 zₜ，并接收网络输出 ε̂、v̂", "sampler", [], ["βₜ 来自采样时间表；αₜ = 1 − βₜ，ᾱₜ = ∏ⱼ₌₁ᵗ αⱼ"]),
    node(60, 145, 335, 120, "均值 · 由 zₜ 与噪声预测决定", "sampler", [tex`\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\left(z_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon\right)`], ["[B, C, Hₗ, Wₗ]"]),
    node(435, 145, 335, 120, "方差 · v̂ 指定对数方差参数", "sampler", [tex`a=(\hat v+1)/2`, tex`\log\sigma_\theta^2=a\log\beta_t+(1-a)\log\tilde\beta_t`], ["[B, C, Hₗ, Wₗ]"]),
    node(60, 333, 430, 100, "均值 + 按标准差缩放的新噪声", "sampler", [tex`z_{t-1}=\mu_\theta+\sigma_\theta\odot\xi`], ["σθ = exp(½ log σθ²)；输出 [B, C, Hₗ, Wₗ]"]),
    node(550, 333, 220, 100, "独立采样噪声", "neutral", [tex`\xi\sim\mathcal N(0,I)`], ["[B, C, Hₗ, Wₗ]"], "最后一步不注入噪声"),
    node(60, 493, 335, 72, "继续采样 → 返回顶部入口", "sampler", [], ["以新 latent、新时间步再次调用 DiT"]),
    node(435, 493, 335, 72, "采样完成 → VAE Decoder", "neutral", [], ["按 VAE 约定缩放 latent，再解码成图像"]),
  ],
  wires: [wire("M227 90 V145", "sampler"), wire("M602 90 V145", "sampler"), wire("M227 265 V333", "sampler"), wire("M602 265 V304 H400 V333", "sampler"), wire("M550 383 H490", "neutral"), wire("M227 433 V493", "sampler"), wire("M400 433 V466 H602 V493", "sampler")],
}
const detailPanels: Partial<Record<Detail, Panel>> = { condition: conditioning, block, output, ddpm: sampling }
const panels = computed(() => props.standaloneBlock ? [block] : [overview, ...(selected.value && detailPanels[selected.value] ? [detailPanels[selected.value]!] : [])])
</script>

<template>
  <figure class="dit-architecture" :class="{ 'is-standalone': standaloneBlock }" :aria-label="standaloneBlock ? 'DiT Block 内部计算' : 'DiT 完整数据流与 DDPM 采样'">
    <div class="architecture-title"><strong>{{ standaloneBlock ? 'DiT Block：两次条件调制与残差更新' : 'DiT：一次前向与去噪循环' }}</strong></div>
    <div class="legend"><span class="tone-image">━ 图像主干</span><span class="tone-condition">┄ 条件调制</span><span v-if="!standaloneBlock" class="tone-sampler">━ 外部采样</span></div>
    <template v-for="panel in panels" :key="panel.key">
      <section :id="panel.key !== 'overview' ? `${id}-detail-${selected ?? 'block'}` : undefined" class="panel" :class="{ 'detail-panel': panel.key !== 'overview' && !standaloneBlock }" :aria-label="panel.title">
        <div v-if="panel.key !== 'overview'" class="panel-heading"><strong>{{ panel.title }}</strong></div>
        <div class="panel-scroll" tabindex="0" role="region" :aria-label="panel.title">
          <svg :viewBox="`0 0 800 ${panel.height}`" role="img" :aria-labelledby="`${id}-${panel.key}-title`">
            <title :id="`${id}-${panel.key}-title`">{{ panel.title }}</title>
            <defs><marker v-for="tone in tones" :id="`${id}-${panel.key}-${tone}`" :key="tone" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" :class="`tone-${tone}`" fill="currentColor" /></marker></defs>
            <rect v-if="panel.key === 'overview'" class="network-region" x="90" y="102" width="680" height="482" rx="10" />
            <template v-if="panel.key === 'block-detail'">
              <rect class="sublayer-region" x="66" y="104" width="374" height="617" rx="8" />
              <rect class="sublayer-region" x="66" y="737" width="374" height="511" rx="8" />
            </template>
            <path v-for="(edge, index) in panel.wires" :key="`edge-${index}`" :d="edge.path" :class="`tone-${edge.tone}`" fill="none" stroke="currentColor" stroke-width="1.8" :stroke-dasharray="edge.dashed ? '5 4' : undefined" :marker-end="`url(#${id}-${panel.key}-${edge.tone})`" />
            <text v-if="panel.key === 'overview'" x="19" y="505" transform="rotate(-90 19 505)" class="wire-label tone-sampler">继续采样：更新 t，返回输入</text>
            <text v-if="panel.key === 'overview'" x="68" y="495" transform="rotate(-90 68 495)" class="wire-label tone-sampler">保留本轮 zₜ，交给采样器</text>
            <g v-for="(box, index) in panel.nodes" :key="index" :class="`tone-${box.tone}`">
              <rect class="node-box" :x="box.x" :y="box.y" :width="box.w" :height="box.h" rx="6" />
              <foreignObject :x="box.x + 8" :y="box.y + 5" :width="box.w - 16" :height="box.h - 10">
                <div class="node-content">
                  <strong>{{ box.title }}</strong><div v-for="formula in box.formulas" :key="formula" class="formula" v-html="formula" /><div v-for="shape in box.shapes" :key="shape" class="shape">{{ shape }}</div><div v-if="box.note" class="node-note">{{ box.note }}</div>
                </div>
              </foreignObject>
            </g>
            <text v-for="label in panel.labels" :key="label.text" :x="label.x" :y="label.y" :class="`wire-label tone-${label.tone ?? 'image'}`" text-anchor="middle">{{ label.text }}</text>
          </svg>
        </div>
      </section>
      <template v-if="panel.key === 'overview'">
        <div class="detail-controls" aria-label="展开模块细节">
          <span>按需展开：</span>
          <button v-for="option in detailOptions" :key="option.key" type="button" :aria-expanded="selected === option.key" :aria-controls="`${id}-detail-${option.key}`" @click="toggle(option.key)">{{ option.label }} <span aria-hidden="true">{{ selected === option.key ? '−' : '+' }}</span></button>
        </div>
        <div v-for="option in detailOptions.filter(option => option.key !== selected)" :id="`${id}-detail-${option.key}`" :key="option.key" hidden />
        <div v-if="selected === 'patch'" :id="`${id}-detail-patch`" class="patch-detail"><DiTPatchify /></div>
      </template>
    </template>
    <figcaption v-if="!standaloneBlock">网络输出噪声预测和方差参数，DDPM 再结合本轮输入计算下一步 latent。主图中的条件箭头连接每层各自的映射；反复采样时，时间步与图像状态一起更新。</figcaption>
    <figcaption v-else>浅色分组对应 Attention、MLP 两个子层。左侧旁路分别保留各子层的输入；右侧条件总线分发六组系数。每层输出形状保持为 [B, Nᵢ, D]。</figcaption>
  </figure>
</template>

<style scoped>
.dit-architecture { margin: 1.5rem 0 2rem; color: var(--vp-c-text); border: 1px solid var(--infra-diagram-line); border-radius: 10px; overflow: hidden; background: var(--vp-c-bg); }
.architecture-title { padding: 1rem 1rem .5rem; font-size: 1.05rem; }
.legend { display: flex; flex-wrap: wrap; gap: 1.2rem; padding: 0 1rem .8rem; font-size: .85rem; }
.panel-heading { padding: .8rem 1rem; font-size: .9rem; }
.detail-panel { border-top: 1px solid var(--infra-diagram-line); }
.detail-controls { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; padding: .8rem 1rem; border-top: 1px solid var(--infra-diagram-line); font-size: .85rem; }
.detail-controls button { border: 1px solid var(--infra-diagram-line); border-radius: 5px; padding: .4rem .65rem; background: var(--vp-c-bg); color: var(--infra-token-flow); font: inherit; cursor: pointer; }
.detail-controls button:hover, .detail-controls button[aria-expanded="true"] { border-color: var(--infra-token-flow); background: color-mix(in srgb, var(--infra-token-flow) 7%, var(--vp-c-bg)); }
.panel-scroll { overflow-x: auto; }
.patch-detail { padding: 0 .65rem; }
.panel-scroll:focus-visible, button:focus-visible { outline: 2px solid var(--infra-token-flow); outline-offset: -2px; }
svg { display: block; width: 100%; min-width: 680px; font-family: inherit; }
.tone-image { color: var(--infra-token-flow); }
.tone-condition { color: var(--infra-weight-flow); }
.tone-sampler { color: var(--infra-execution); }
.tone-neutral { color: var(--infra-diagram-muted); }
.network-region, .sublayer-region { fill: color-mix(in srgb, var(--infra-token-flow) 3%, var(--vp-c-bg)); stroke: var(--infra-diagram-line); stroke-dasharray: 5 4; }
.sublayer-region { stroke-dasharray: none; }
.node-box { fill: var(--vp-c-bg); stroke: currentColor; stroke-width: 1.2; }
.node-content { height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 4px; text-align: center; font-size: 16px; line-height: 1.35; }
.node-content strong { font-size: 16px; font-weight: 600; }
.formula { color: var(--vp-c-text); font-size: 15px; }
.formula :deep(.katex) { font-size: 1em; }
.shape { color: var(--vp-c-text); font-size: 14px; }
.node-note { color: var(--infra-diagram-muted); font-size: 13px; }
.wire-label { fill: currentColor; font-size: 14px; }
figcaption { padding: .8rem 1rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: .85rem; line-height: 1.65; }
</style>
