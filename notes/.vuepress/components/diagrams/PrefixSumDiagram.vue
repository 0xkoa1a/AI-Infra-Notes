<script setup lang="ts">
import { useId } from "vue"

defineProps<{ view: "doubling" | "tree" | "blocks" }>()
const id = useId()
const arrow = `url(#${id}-arrow)`
const rounds = [
  { label: "输入", values: [1, 2, 3, 4, 5, 6, 7, 8] },
  { label: "距离 1", values: [1, 3, 5, 7, 9, 11, 13, 15] },
  { label: "距离 2", values: [1, 3, 6, 10, 14, 18, 22, 26] },
  { label: "距离 4", values: [1, 3, 6, 10, 15, 21, 28, 36] },
]
const blocks = [
  { input: [1, 2, 3, 4], local: [1, 3, 6, 10], sum: 10, prefix: 10, offset: 0, output: [1, 3, 6, 10] },
  { input: [5, 6, 7, 8], local: [5, 11, 18, 26], sum: 26, prefix: 36, offset: 10, output: [15, 21, 28, 36] },
  { input: [9, 10, 11, 12], local: [9, 19, 30, 42], sum: 42, prefix: 78, offset: 36, output: [45, 55, 66, 78] },
]
const descriptions = {
  doubling: "输入为1到8。距离1、2、4三轮后，最后一项依次为15、26、36。连线跟踪最后一项每轮读取的左侧部分和。",
  tree: "左树由输入1、2、3、4向上求出区间和3、7及总和10。右树从根前缀0向下分发，得到exclusive结果0、1、3、6，再加原输入得到inclusive结果1、3、6、10。",
  blocks: "12个输入分成3块，每块4项。块总和为10、26、42，扫描后为10、36、78，因此三个块分别加0、10、36，得到完整前缀和。",
}
</script>

<template>
  <figure class="prefix-figure">
    <div class="scroll" role="region" tabindex="0" aria-label="Prefix Sum 数值图，可横向滚动">
      <svg :viewBox="`0 0 900 ${view === 'doubling' ? 350 : view === 'tree' ? 415 : 535}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
        <title :id="`${id}-title`">{{ view === 'doubling' ? '每轮扩大前缀范围' : view === 'tree' ? '区间和与前缀的树形传递' : '局部前缀加上块偏移' }}</title>
        <desc :id="`${id}-desc`">{{ descriptions[view] }}</desc>
        <defs>
          <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker>
        </defs>

        <g v-if="view === 'doubling'">
          <g v-for="(round, row) in rounds" :key="round.label">
            <text x="90" :y="57 + row * 85" text-anchor="end" class="label">{{ round.label }}</text>
            <g v-for="(value, col) in round.values" :key="col">
              <rect :x="125 + col * 92" :y="30 + row * 85" width="64" height="42" rx="5" :class="col === 7 ? 'result' : row < 3 && col === 7 - 2 ** row ? 'source' : 'cell'" />
              <text :x="157 + col * 92" :y="57 + row * 85" class="number">{{ value }}</text>
            </g>
          </g>
          <g v-for="(offset, row) in [1, 2, 4]" :key="offset">
            <path :d="`M${157 + (7 - offset) * 92} ${76 + row * 85} L789 ${109 + row * 85}`" class="edge" :marker-end="arrow" />
            <path :d="`M801 ${76 + row * 85} V${109 + row * 85}`" class="edge" :marker-end="arrow" />
            <text x="823" :y="99 + row * 85" class="label accent">+</text>
          </g>
        </g>

        <g v-else-if="view === 'tree'">
          <text x="230" y="30" class="center label">向上：子树总和</text>
          <text x="670" y="30" class="center label">向下：子树之前的和</text>
          <path d="M265 82 H635" class="edge" :marker-end="arrow" />
          <text x="450" y="69" class="center label">根置 0</text>
          <path d="M95 258 L160 184 M185 258 L160 184 M275 258 L300 184 M365 258 L300 184 M160 146 L230 104 M300 146 L230 104" class="edge" :marker-end="arrow" />
          <path d="M670 104 L600 146 M670 104 L740 146 M600 184 L535 258 M600 184 L625 258 M740 184 L715 258 M740 184 L805 258" class="edge" :marker-end="arrow" />
          <text x="743" y="119" class="small accent">+ 左和 3</text>
          <text x="645" y="224" class="small accent">+1</text>
          <text x="814" y="224" class="small accent">+3</text>
          <g v-for="(value, index) in [1, 2, 3, 4]" :key="`input-${index}`">
            <rect :x="65 + index * 90" y="263" width="60" height="38" rx="5" class="source" />
            <text :x="95 + index * 90" y="288" class="number">{{ value }}</text>
          </g>
          <g v-for="(node, index) in [{ x: 160, value: 3 }, { x: 300, value: 7 }, { x: 230, value: 10 }]" :key="`sum-${index}`">
            <rect :x="node.x - 30" :y="index === 2 ? 63 : 146" width="60" height="38" rx="5" class="source" />
            <text :x="node.x" :y="index === 2 ? 88 : 171" class="number">{{ node.value }}</text>
          </g>
          <g v-for="(node, index) in [{ x: 600, value: 0 }, { x: 740, value: 3 }, { x: 670, value: 0 }]" :key="`prefix-${index}`">
            <rect :x="node.x - 30" :y="index === 2 ? 63 : 146" width="60" height="38" rx="5" class="result" />
            <text :x="node.x" :y="index === 2 ? 88 : 171" class="number">{{ node.value }}</text>
          </g>
          <g v-for="(value, index) in [0, 1, 3, 6]" :key="`exclusive-${index}`">
            <rect :x="505 + index * 90" y="263" width="60" height="38" rx="5" class="result" />
            <text :x="535 + index * 90" y="288" class="number">{{ value }}</text>
            <path :d="`M${535 + index * 90} 305 V355`" class="edge" :marker-end="arrow" />
            <text :x="550 + index * 90" y="337" class="small accent">+{{ index + 1 }}</text>
            <rect :x="505 + index * 90" y="361" width="60" height="38" rx="5" class="result" />
            <text :x="535 + index * 90" y="386" class="number">{{ [1, 3, 6, 10][index] }}</text>
          </g>
          <text x="488" y="288" text-anchor="end" class="small">exclusive</text>
          <text x="488" y="386" text-anchor="end" class="small">inclusive</text>
        </g>

        <g v-else>
          <text x="105" y="69" text-anchor="end" class="label">输入</text>
          <text x="105" y="159" text-anchor="end" class="label">块内前缀</text>
          <text x="105" y="241" text-anchor="end" class="label">块总和</text>
          <text x="105" y="317" text-anchor="end" class="label">累计总和</text>
          <text x="105" y="409" text-anchor="end" class="label">块偏移</text>
          <text x="105" y="499" text-anchor="end" class="label">全局前缀</text>
          <g v-for="(block, col) in blocks" :key="col" :transform="`translate(${135 + col * 255},0)`">
            <text x="110" y="25" class="center label">block {{ col }}</text>
            <g v-for="(values, row) in [block.input, block.local, block.output]" :key="row">
              <g v-for="(value, index) in values" :key="index">
                <rect :x="index * 55" :y="row === 2 ? 472 : 42 + row * 90" width="55" height="42" :class="row === 2 ? 'result' : 'cell'" />
                <text :x="27.5 + index * 55" :y="row === 2 ? 499 : 69 + row * 90" class="number">{{ value }}</text>
              </g>
            </g>
            <path d="M110 88 V126" class="edge" :marker-end="arrow" />
            <path d="M192.5 178 V197 H110 V211" class="edge" :marker-end="arrow" />
            <rect x="80" y="217" width="60" height="38" rx="5" class="source" />
            <text x="110" y="242" class="number">{{ block.sum }}</text>
            <path d="M110 259 V286" class="edge" :marker-end="arrow" />
            <rect x="80" y="292" width="60" height="38" rx="5" class="result" />
            <text x="110" y="317" class="number">{{ block.prefix }}</text>
            <rect x="70" y="382" width="80" height="42" rx="5" class="result" />
            <text x="110" y="409" class="number">+{{ block.offset }}</text>
            <path d="M15 180 V450 H110" class="carry" />
            <path d="M110 428 V466" class="edge" :marker-end="arrow" />
            <g v-if="col < 2">
              <path d="M145 311 H330" class="edge" :marker-end="arrow" />
              <path d="M110 334 V356 H365 V376" class="edge" :marker-end="arrow" />
            </g>
          </g>
        </g>
      </svg>
    </div>
    <figcaption v-if="view === 'doubling'">箭头跟踪最后一项；橙色标出每轮读取的左侧部分和。</figcaption>
    <figcaption v-else-if="view === 'tree'">橙色为子树总和，蓝色为子树之前的和；叶子再加原输入，得到 inclusive 结果。</figcaption>
    <figcaption v-else>示例每块 4 项；将累计总和右移一块，作为下一块的偏移，首块偏移为 0。</figcaption>
  </figure>
</template>

<style scoped>
.prefix-figure { margin: 1.5rem 0; }
.scroll { overflow-x: auto; border: 1px solid var(--infra-diagram-line); border-radius: 8px; background: var(--vp-c-bg); }
svg { display: block; width: 100%; min-width: 720px; color: var(--vp-c-text); font-family: inherit; }
.cell { fill: var(--vp-c-bg-soft); stroke: var(--infra-diagram-line); }
.source { fill: color-mix(in srgb, var(--infra-weight-flow) 10%, var(--vp-c-bg)); stroke: var(--infra-weight-flow); }
.result { fill: color-mix(in srgb, var(--infra-token-flow) 10%, var(--vp-c-bg)); stroke: var(--infra-token-flow); }
.number { text-anchor: middle; fill: currentColor; font-size: 19px; font-variant-numeric: tabular-nums; }
.label { fill: currentColor; font-size: 17px; }
.small { fill: currentColor; font-size: 15px; }
.center { text-anchor: middle; }
.accent { fill: var(--infra-token-flow); }
.edge { fill: none; stroke: var(--infra-token-flow); stroke-width: 1.6; }
.carry { fill: none; stroke: var(--infra-diagram-muted); stroke-width: 1.4; stroke-dasharray: 4 4; }
.arrowhead { fill: var(--infra-token-flow); }
figcaption { margin-top: .5rem; color: var(--infra-diagram-muted); font-size: .9rem; }
</style>
