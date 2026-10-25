<script setup lang="ts">
import { computed, ref, useId } from "vue"

const props = defineProps<{ view: string }>()
const id = useId()
const step = ref(0)
const task = ref(0)
const layer = ref(0)
const stages = ["报到 1/4", "报到 4/4", "加载完成", "UMMA 完成"]
const arrow = `url(#${id}-arrow)`
const signalArrow = `url(#${id}-signal)`
const height = computed(() => props.view === "input-stage" ? 500 : props.view === "tmem-layout" ? 455 : 430)
const title = computed(() => props.view === "input-stage" ? "同一输入 stage：四方报到与传输完成" : props.view === "tmem-layout" ? "单 CTA 的 TMEM：两个累加槽与 scale 区" : `${layer.value === 0 ? 'L1' : 'L2'} 后处理：读出与写出使用不同缓冲`)
const description = computed(() => {
  if (props.view === "input-stage") return `两个CTA各自的warp4加载激活与scale，warp5加载权重与scale。当前${stages[step.value]}。leader的full需要四方报到且TMA传输完成才允许UMMA读取；UMMA完成后，两个CTA的empty允许覆盖本输入stage。`
  if (props.view === "tmem-layout") return `每个CTA的TMEM有128条lane，对应输出特征；column对应token位置。容量64行时，累加槽0占column0到63，槽1占64到127，后面两类scale各占4列。当前派发任务${task.value}，使用槽${task.value % 2}。任务2复用槽0之前，任务0的最后一次有效读取已经完成。`
  return `${layer.value === 0 ? 'L1从TMEM读入FP32寄存器，将gate/up转为BF16后执行SwiGLU和路由乘权，再量化为FP8，交替使用两份输出SMEM，由TMA写入本卡L2输入缓冲。' : 'L2将FP32结果转换为BF16，在单份输出SMEM重排，再读入向量寄存器，用global store写入源卡回传槽。'}两条路径都在最后一次有效TMEM读取完成后允许复用累加槽。`
})
</script>

<template>
  <div v-if="view === 'input-stage'" class="controls" role="group" aria-label="输入 stage 的完成条件">
    <button v-for="(label, index) in stages" :key="label" type="button" :aria-pressed="step === index" @click="step = index">{{ label }}</button>
  </div>
  <div v-else-if="view === 'tmem-layout'" class="controls" role="group" aria-label="使用 TMEM 的任务">
    <button v-for="index in [0, 1, 2]" :key="index" type="button" :aria-pressed="task === index" @click="task = index">任务 {{ index }}{{ index === 2 ? '：复用槽 0' : '' }}</button>
  </div>
  <div v-else class="controls" role="group" aria-label="后处理阶段">
    <button v-for="(name, index) in ['L1', 'L2']" :key="name" type="button" :aria-pressed="layer === index" @click="layer = index">{{ name }}</button>
  </div>
  <div class="scroll" role="region" tabindex="0" :aria-label="`${title}，可横向滚动`">
    <svg :viewBox="`0 0 900 ${height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
      <title :id="`${id}-title`">{{ title }}</title>
      <desc :id="`${id}-desc`">{{ description }}</desc>
      <defs>
        <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker>
        <marker :id="`${id}-signal`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="signal-head" /></marker>
      </defs>

      <g v-if="view === 'input-stage'">
        <text x="450" y="25" class="center strong">同一输入 stage · 两 CTA 协作</text>
        <g v-for="cta in [0, 1]" :key="cta">
          <rect :x="30 + cta * 450" y="45" width="390" height="130" rx="8" class="panel" />
          <text :x="225 + cta * 450" y="72" class="center">CTA {{ cta }} · 本地 SMEM</text>
          <g v-for="role in [0, 1]" :key="role">
            <rect :x="45 + cta * 450 + role * 190" y="92" width="170" height="58" rx="5" :class="step > 0 || cta === 0 && role === 0 ? (role === 0 ? 'tint acts' : 'tint weight') : 'pending'" />
            <text :x="130 + cta * 450 + role * 190" y="115" class="center small">warp {{ role + 4 }}</text>
            <text :x="130 + cta * 450 + role * 190" y="139" class="center">{{ role === 0 ? '激活 + scale' : '权重 + scale' }}</text>
            <path :d="`M${130 + cta * 450 + role * 190} 179 C${130 + cta * 450 + role * 190} 200,${320 + cta * 160 + role * 50} 190,${320 + cta * 160 + role * 50} 216`" class="signal" :class="{ dim: step === 0 && (cta !== 0 || role !== 0) }" :marker-end="signalArrow" />
          </g>
        </g>
        <rect x="280" y="220" width="340" height="70" rx="7" :class="step >= 2 ? 'tint ready' : 'pending'" />
        <text x="450" y="247" class="center">leader full · 报到 {{ step === 0 ? 1 : 4 }}/4</text>
        <text x="450" y="274" class="center small">传输字节：{{ step >= 2 ? '全部完成 ✓' : '尚未完成' }}</text>
        <path d="M450 294 V326" class="signal" :class="{ dim: step < 2 }" :marker-end="signalArrow" />
        <rect x="330" y="331" width="240" height="48" rx="7" :class="step >= 2 ? 'tint result' : 'pending'" />
        <text x="450" y="362" class="center">UMMA · {{ step < 2 ? '等待' : step === 2 ? '读取输入并计算' : '完成 ✓' }}</text>
        <path d="M450 383 V406 H225 V426 M450 406 H675 V426" class="signal" :class="{ dim: step < 3 }" :marker-end="signalArrow" />
        <g v-for="cta in [0, 1]" :key="cta">
          <rect :x="105 + cta * 450" y="431" width="240" height="44" rx="6" :class="step === 3 ? 'tint ready' : 'pending'" />
          <text :x="225 + cta * 450" y="459" class="center">CTA {{ cta }} empty {{ step === 3 ? '✓' : '○' }}</text>
        </g>
      </g>

      <g v-else-if="view === 'tmem-layout'">
        <text x="450" y="25" class="center strong">每个 CTA · 128 条 lane</text>
        <text x="110" y="76" class="center small">0</text>
        <text x="398" y="76" class="center small">64</text>
        <text x="686" y="76" class="center small">128</text>
        <text x="57" y="103" class="center small">lane 0</text>
        <text x="57" y="202" class="center small">输出特征</text>
        <text x="57" y="322" class="center small">lane 127</text>
        <g v-for="slot in [0, 1]" :key="slot">
          <rect :x="110 + slot * 288" y="85" width="288" height="240" :class="slot === task % 2 ? 'tint result' : 'panel'" />
          <path v-for="line in [1, 2, 3, 4, 5, 6, 7]" :key="line" :d="`M${110 + slot * 288} ${85 + line * 30} H${398 + slot * 288} M${110 + slot * 288 + line * 36} 85 V325`" class="edge" opacity="0.25" />
          <rect :x="148 + slot * 288" y="156" width="212" height="100" rx="6" class="box" />
          <text :x="254 + slot * 288" y="184" class="center strong">累加槽 {{ slot }}</text>
          <text :x="254 + slot * 288" y="212" class="center">{{ slot === 0 ? (task === 2 ? '任务 2' : '任务 0') : task === 0 ? '空闲' : '任务 1' }}</text>
          <text :x="254 + slot * 288" y="241" class="center small">128 × 64 FP32 · 32 KiB</text>
        </g>
        <rect x="686" y="85" width="18" height="240" class="tint acts" />
        <rect x="704" y="85" width="18" height="240" class="tint weight" />
        <path d="M695 329 V351 H735 M713 329 V388 H735" class="edge" />
        <text x="742" y="356" class="small">激活 scale · 4 列</text>
        <text x="742" y="393" class="small">权重 scale · 4 列</text>
        <text x="380" y="356" class="center small muted">column → token 位置</text>
        <text x="380" y="414" class="center">{{ task === 2 ? '任务 0 的末次有效读取完成 → 槽 0 可覆盖' : `当前任务 ${task} 使用槽 ${task}` }}</text>
      </g>

      <g v-else>
        <text x="450" y="25" class="center strong">{{ layer === 0 ? 'L1 · 中间激活写出' : 'L2 · 专家结果回传' }}</text>
        <g v-for="(name, index) in ['TMEM', '原始结果寄存器', layer === 0 ? 'SwiGLU × α' : '转为 BF16']" :key="index">
          <rect :x="25 + index * 215" y="70" width="175" height="83" rx="7" :class="index === 0 ? 'tint result' : 'box'" />
          <text :x="112 + index * 215" y="102" class="center">{{ name }}</text>
          <text :x="112 + index * 215" y="133" class="center small muted">{{ index < 2 ? 'FP32' : layer === 0 ? '量化为 FP8 + scale' : '等待输出 SMEM 可写' }}</text>
          <path :d="`M${204 + index * 215} 112 H${234 + index * 215}`" class="edge" :marker-end="arrow" />
        </g>
        <rect x="670" y="57" width="205" height="150" rx="7" class="panel" />
        <text x="772" y="83" class="center">输出 SMEM</text>
        <g v-for="slot in layer === 0 ? [0, 1] : [0]" :key="slot">
          <rect :x="690 + (layer === 0 ? slot * 86 : 22)" y="102" :width="layer === 0 ? 78 : 121" height="49" rx="5" class="tint acts" />
          <text :x="layer === 0 ? 729 + slot * 86 : 772" y="133" class="center small">{{ layer === 0 ? `槽 ${slot}` : '重排后的 BF16' }}</text>
        </g>
        <text x="772" y="181" class="center small muted">{{ layer === 0 ? '交替填充 / TMA 写出' : '一份暂存区' }}</text>
        <path d="M327 157 V268" class="signal" :marker-end="signalArrow" />
        <text x="315" y="213" text-anchor="end" class="small muted">末次有效读取完成</text>
        <rect x="217" y="273" width="220" height="52" rx="7" class="tint ready" />
        <text x="327" y="305" class="center">TMEM 允许覆盖</text>
        <g v-if="layer === 0">
          <path d="M772 211 V295" class="edge" :marker-end="arrow" />
          <text x="772" y="257" class="center small muted">TMA Store</text>
          <rect x="650" y="300" width="235" height="65" rx="7" class="tint acts" />
          <text x="767" y="326" class="center">本卡 L2 输入缓冲</text>
          <text x="767" y="351" class="center small">FP8 激活</text>
        </g>
        <g v-else>
          <path d="M772 211 V241" class="edge" :marker-end="arrow" />
          <rect x="670" y="246" width="205" height="50" rx="7" class="box" />
          <text x="772" y="277" class="center">向量寄存器</text>
          <path d="M772 300 V344 H650" class="edge" :marker-end="arrow" />
          <text x="767" y="328" class="center small muted">global store</text>
          <rect x="460" y="320" width="185" height="63" rx="7" class="tint result" />
          <text x="552" y="346" class="center">源卡回传槽</text>
          <text x="552" y="370" class="center small">按 GPU / token / s 定位</text>
        </g>
      </g>
    </svg>
  </div>
  <figcaption v-if="view === 'input-stage'">虚线传递完成通知；输入数据保存在两个 CTA 各自的 SMEM 中。</figcaption>
  <figcaption v-else-if="view === 'tmem-layout'">行块容量 64 时已使用的 TMEM 区域；两个累加槽独立交接。</figcaption>
  <figcaption v-else>{{ layer === 0 ? 'scale 由线程另行写入本卡缓冲。' : '' }}最后一次有效读取完成后，旧结果继续在寄存器与输出 SMEM 中处理。</figcaption>
</template>
