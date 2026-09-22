<script setup lang="ts">
import DiTArchitecture from "./DiTArchitecture.vue"
import DiTPatchify from "./DiTPatchify.vue"
import DiffusionFigure, { type DiffusionDiagram } from "./DiffusionFigure.vue"
type View = "overview" | "patchify" | "block" | "cache"
withDefaults(defineProps<{ view?: View }>(), { view: "overview" })
const cacheDiagram: DiffusionDiagram = {
  "title": "Condition Cache：先分清跨步复用与逐步预计算",
  "height": 640,
  "caption": "条件路径依赖类别、每个时间步及固定权重。类别编码可以跨步复用；调制参数按 step、按层分别保存。下方图像状态依赖当前 latent，每次前向重新计算。",
  "boxes": [
    {
      "x": 25,
      "y": 20,
      "w": 310,
      "label": "类别编码 eᵧ",
      "detail": "固定 y → [B, D] · 跨步复用",
      "tone": "condition"
    },
    {
      "x": 465,
      "y": 20,
      "w": 310,
      "label": "时间步编码 eₜ",
      "detail": "时间表中的每个 t → [B, D] · 逐步准备",
      "tone": "condition"
    },
    {
      "x": 245,
      "y": 150,
      "w": 310,
      "label": "全局条件 c(t) = eₜ + eᵧ",
      "detail": "[B, D] · 随时间步变化",
      "tone": "condition"
    },
    {
      "x": 465,
      "y": 280,
      "w": 310,
      "label": "各层独立的条件映射",
      "detail": "Block：[B, 6D]；Final：[B, 2D]",
      "tone": "condition"
    },
    {
      "x": 465,
      "y": 410,
      "w": 310,
      "label": "调制参数缓存",
      "detail": "按当前 step、当前层读取对应系数",
      "tone": "condition"
    },
    {
      "x": 25,
      "y": 280,
      "w": 310,
      "label": "当前 latent zₜ",
      "detail": "[B, C, Hₗ, Wₗ] · 随采样更新",
      "tone": "image"
    },
    {
      "x": 25,
      "y": 410,
      "w": 310,
      "label": "当前层图像 hidden state",
      "detail": "[B, Nᵢ, D] · 随本次前向更新",
      "tone": "image"
    },
    {
      "x": 245,
      "y": 545,
      "w": 310,
      "label": "本层图像计算",
      "detail": "归一化、条件调制与子层计算",
      "tone": "image"
    }
  ],
  "wires": [
    {
      "path": "M180 92 V122 H340 V150",
      "tone": "condition"
    },
    {
      "path": "M620 92 V122 H460 V150",
      "tone": "condition"
    },
    {
      "path": "M400 222 V251 H620 V280",
      "tone": "condition"
    },
    {
      "path": "M620 352 V410",
      "tone": "condition"
    },
    {
      "path": "M620 482 V581 H555",
      "tone": "condition",
      "dashed": true
    },
    {
      "path": "M180 352 V410",
      "tone": "image"
    },
    {
      "path": "M180 482 V581 H245",
      "tone": "image"
    }
  ],
  "annotations": [
    {
      "x": 400,
      "y": 391,
      "label": "固定权重，已知时间表 → 可提前准备条件路径",
      "anchor": "middle"
    }
  ],
  "legend": [
    {
      "tone": "image",
      "label": "━ 动态图像状态"
    },
    {
      "tone": "condition",
      "label": "━ 条件计算　┄ 缓存读取"
    }
  ]
}

</script>

<template>
  <DiTArchitecture v-if="view === 'overview' || view === 'block'" :standalone-block="view === 'block'" />
  <DiTPatchify v-else-if="view === 'patchify'" />
  <DiffusionFigure v-else :diagram="cacheDiagram" class="dit-diagram" />
</template>
