<script setup lang="ts">
import { computed, ref, useId } from "vue"

const props = defineProps<{ view: string }>()
const id = useId()
const layer = ref(0)
const completed = ref(2)
const phase = ref(0)
const heights: Record<string, number> = { compute: 630, partition: 475, readiness: 430, schedule: 420, ring: 600 }
const height = computed(() => heights[props.view] || 500)
const title = computed(() => {
  const titles: Record<string, string> = { compute: `GPU 0 / E0 · ${layer.value === 0 ? 'L1：输入到中间激活' : 'L2：中间激活到回传片段'}`, partition: "同一输入块，两个 CTA 分担输出列", readiness: "同一批 token：L1 列块就绪后，L2 按特征块推进", schedule: "块 q：任务已派发与数据已写完", ring: "固定物理槽，观察旧块释放与新块写入" }
  return titles[props.view] || "MegaMoE 计算"
})
const arrow = `url(#${id}-arrow)`
const signalArrow = `url(#${id}-signal)`
const phaseLabels = ["旧块消费", "复用 L1 槽", "复用 L2 槽", "旧块回传"]
const description = computed(() => {
  if (props.view === "compute") return `GPU0专家E0的有效行是a0、c1，块容量64，H为512，F为256。当前为${layer.value === 0 ? 'L1' : 'L2'}，激活和权重通过TMA进入SMEM，协作UMMA将结果累加到两份TMEM。${layer.value === 0 ? '后处理生成加权的FP8中间激活与scale，写入L2输入缓冲。' : '后处理分别写回GPU0的a0槽0和GPU2的c1槽1。'}`
  if (props.view === "partition") return "同卡两个CTA处理a0、c1两行。图中选取本次128个输入特征，CTA0加载输出列0到127的权重，CTA1加载128到255的权重。CTA0发射协作UMMA，共同使用输入，各CTA的TMEM分别累加两行的不同128列输出。形状按有效行表示。"
  if (props.view === "readiness") return `同一输入行块的四个L1原始列块各有64对gate/up，SwiGLU后各生成64个中间特征。当前前${completed.value}块已写完。L2的K0等待n0、n1，K1等待n2、n3；完整输出需要K0、K1共同累加。`
  if (props.view === "schedule") return "任务足够多时先预热若干L1，再交错派发L2与L1。下方跟踪块q：生成列0到255和256到511的两项L1任务均已派发后，允许派发L2(q)；n0和n1写完并设置就绪位后，L2(q)加载K0。K1按自己的就绪位等待。"
  return `当前为${phaseLabels[phase.value]}。两个输入环独立复用物理槽，旧块q与后续块q加R使用同一槽。来源记录按逻辑块保留。下方跟踪q的一项L2任务：TMEM末次读取完成后允许覆盖，寄存器中的片段继续按旧来源记录写回。`
})
</script>

<template>
  <div v-if="view === 'compute'" class="controls" role="group" aria-label="投影阶段">
    <button v-for="(name, index) in ['L1', 'L2']" :key="name" type="button" :aria-pressed="layer === index" @click="layer = index">{{ name }}</button>
  </div>
  <div v-else-if="view === 'readiness'" class="controls" role="group" aria-label="已写完的 L1 列块数">
    <span class="control-label">L1 已写完</span>
    <button v-for="count in [0, 1, 2, 3, 4]" :key="count" type="button" :aria-pressed="completed === count" @click="completed = count">{{ count }} / 4 块</button>
  </div>
  <div v-else-if="view === 'ring'" class="controls" role="group" aria-label="缓冲复用阶段">
    <button v-for="(name, index) in phaseLabels" :key="name" type="button" :aria-pressed="phase === index" @click="phase = index">{{ name }}</button>
  </div>
  <div class="scroll" role="region" tabindex="0" :aria-label="`${title}，可横向滚动`">
    <svg :viewBox="`0 0 900 ${height}`" role="img" :aria-labelledby="`${id}-title ${id}-desc`">
      <title :id="`${id}-title`">{{ title }}</title>
      <desc :id="`${id}-desc`">{{ description }}</desc>
      <defs>
        <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker>
        <marker :id="`${id}-signal`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="signal-head" /></marker>
      </defs>

      <g v-if="view === 'compute'">
        <text x="450" y="25" class="center strong">{{ title }}</text>
        <rect x="20" y="58" width="205" height="312" rx="8" class="panel" />
        <text x="122" y="83" class="center">本卡显存（HBM）</text>
        <text x="122" y="110" class="center acts ink">{{ layer === 0 ? 'L1 输入 X' : 'L2 输入 A' }}</text>
        <g v-for="row in 4" :key="row">
          <rect v-for="col in 4" :key="col" :x="78 + (col - 1) * 24" :y="122 + (row - 1) * 15" width="24" height="15" :class="row <= 2 ? 'tint acts' : 'pending'" />
          <text x="68" :y="134 + (row - 1) * 15" text-anchor="end" class="small muted">{{ row === 1 ? 'a0' : row === 2 ? 'c1' : row === 3 ? '…' : '预留' }}</text>
        </g>
        <text x="122" y="204" class="center small muted">[64, {{ layer === 0 ? 512 : 256 }}] + scale</text>
        <rect x="48" y="239" width="150" height="84" rx="5" class="tint weight" />
        <text x="123" y="265" class="center">{{ layer === 0 ? 'W₁ 权重片段' : 'W₂ 权重片段' }}</text>
        <text x="123" y="289" class="center small">[256, {{ layer === 0 ? 512 : 256 }}]</text>
        <text x="123" y="310" class="center small muted">输出特征 [0,256)</text>
        <text v-if="layer === 0" x="122" y="351" class="center small">路由权重 α：a0 / c1 各一份</text>

        <rect x="295" y="84" width="185" height="270" rx="8" class="panel" />
        <text x="387" y="109" class="center">配对 CTA 的 SMEM</text>
        <rect x="310" y="125" width="155" height="62" rx="5" class="tint acts" />
        <text x="387" y="151" class="center">激活 + scale</text>
        <text x="387" y="174" class="center small">每 CTA 容量 [32,128]</text>
        <rect x="310" y="245" width="155" height="62" rx="5" class="tint weight" />
        <text x="387" y="271" class="center">权重 + scale</text>
        <text x="387" y="294" class="center small">每 CTA [128,128]</text>
        <path d="M228 156 H290 M228 276 H290" class="edge" :marker-end="arrow" />
        <text x="260" y="143" class="center small muted">TMA</text>
        <text x="260" y="263" class="center small muted">TMA</text>

        <path d="M483 156 H510 V198 H529 M483 276 H510 V228 H529" class="edge" :marker-end="arrow" />
        <circle cx="600" cy="213" r="66" class="box" />
        <text x="600" y="200" class="center">Tensor Core</text>
        <text x="600" y="228" class="center strong">UMMA</text>
        <text x="600" y="307" class="center small muted">{{ layer === 0 ? '4 个输入特征块累加' : '2 个输入特征块累加' }}</text>
        <path d="M669 213 H730" class="edge" :marker-end="arrow" />
        <rect x="735" y="172" width="150" height="83" rx="7" class="tint result" />
        <text x="810" y="198" class="center">TMEM × 2</text>
        <text x="810" y="221" class="center small">FP32 累加</text>
        <text x="810" y="243" class="center small">每 CTA [64,128]</text>

        <path d="M810 259 V360" class="edge" :marker-end="arrow" />
        <text x="714" y="350" class="center small muted">后处理线程读取累加值</text>
        <rect x="550" y="365" width="335" height="70" rx="7" class="box" />
        <text x="717" y="393" class="center">{{ layer === 0 ? 'SwiGLU × α' : 'FP32 → BF16' }}</text>
        <text x="717" y="418" class="center small">{{ layer === 0 ? '量化为 FP8 + scale' : '按来源记录定位回传槽' }}</text>
        <path v-if="layer === 0" d="M228 345 H495 V400 H545" class="edge" :marker-end="arrow" />
        <g v-if="layer === 0">
          <path d="M717 439 V474 H480 V490" class="edge" :marker-end="arrow" />
          <rect x="280" y="495" width="400" height="87" rx="7" class="tint acts" />
          <text x="480" y="524" class="center">L2 输入缓冲 · A 的特征 [0,128)</text>
          <text x="480" y="550" class="center">[64,128] + scale</text>
          <text x="480" y="572" class="center small muted">有效行：a0、c1</text>
        </g>
        <g v-else>
          <path d="M717 439 V475 H375 V505 M717 475 H730 V505" class="edge" :marker-end="arrow" />
          <g v-for="(name, index) in ['GPU 0 · a0 / s0', 'GPU 2 · c1 / s1']" :key="name">
            <rect :x="265 + index * 355" y="510" width="220" height="74" rx="7" class="tint result" />
            <text :x="375 + index * 355" y="539" class="center">{{ name }}</text>
            <text :x="375 + index * 355" y="566" class="center small">回写输出列 [0,256)</text>
          </g>
        </g>
      </g>

      <g v-else-if="view === 'partition'">
        <text x="450" y="25" class="center strong">GPU 0 / E0 · 一项 L1 任务</text>
        <text x="450" y="56" class="center small muted">本次的 128 个输入特征</text>
        <g v-for="(name, row) in ['a0', 'c1']" :key="name">
          <text x="325" :y="87 + row * 30" text-anchor="end" class="small">{{ name }}</text>
          <rect v-for="col in [0, 1, 2, 3]" :key="col" :x="340 + col * 55" :y="66 + row * 30" width="55" height="30" class="tint acts" />
        </g>
        <path d="M450 130 V202" class="edge" :marker-end="arrow" />
        <g v-for="cta in [0, 1]" :key="cta">
          <rect :x="45 + cta * 625" y="157" width="185" height="112" rx="7" class="tint weight" />
          <text :x="137 + cta * 625" y="184" class="center">CTA {{ cta }} · 权重</text>
          <text :x="137 + cta * 625" y="215" class="center small">输出列 [{{ cta * 128 }},{{ (cta + 1) * 128 }})</text>
          <text :x="137 + cta * 625" y="246" class="center small">本次读取 [128,128]</text>
          <path :d="cta === 0 ? 'M234 233 H300' : 'M666 233 H600'" class="edge" :marker-end="arrow" />
          <rect :x="70 + cta * 490" y="345" width="270" height="94" rx="7" class="tint result" />
          <text :x="205 + cta * 490" y="370" class="center">CTA {{ cta }} · TMEM</text>
          <text :x="205 + cta * 490" y="395" class="center small">a0、c1 的输出列 [{{ cta * 128 }},{{ (cta + 1) * 128 }})</text>
          <text :x="205 + cta * 490" y="422" class="center small">有效结果 [2,128]</text>
        </g>
        <rect x="305" y="207" width="290" height="52" rx="7" class="box" />
        <text x="450" y="239" class="center">CTA 0 发射协作 UMMA</text>
        <path d="M385 263 V304 H205 V339 M515 263 V304 H695 V339" class="edge" :marker-end="arrow" />
      </g>

      <g v-else-if="view === 'readiness'">
        <text x="130" y="29" class="center">L1 原始列块</text>
        <text x="390" y="29" class="center">SwiGLU 后的特征</text>
        <text x="723" y="29" class="center">L2 输入特征块</text>
        <g v-for="block in [0, 1, 2, 3]" :key="block" :transform="`translate(0,${65 + block * 76})`">
          <rect x="25" width="210" height="52" rx="5" :class="block < completed ? 'tint ready' : 'pending'" />
          <text x="42" y="31" class="small">n{{ block }} {{ block < completed ? '✓' : '○' }}</text>
          <rect x="92" y="5" width="131" height="19" rx="2" class="tint acts" />
          <rect x="92" y="28" width="131" height="19" rx="2" class="tint acts" />
          <text x="157" y="19" class="center small">64 gate</text>
          <text x="157" y="42" class="center small">64 up</text>
          <path d="M239 26 H329" class="edge" :marker-end="arrow" />
          <rect x="335" width="110" height="52" rx="5" :class="block < completed ? 'tint ready' : 'pending'" />
          <text x="390" y="24" class="center small">{{ block * 64 }}–{{ (block + 1) * 64 - 1 }}</text>
          <text x="390" y="43" class="center small muted">64 特征</text>
        </g>
        <g v-for="pair in [0, 1]" :key="pair">
          <path :d="`M449 ${91 + pair * 152} H510 V${129 + pair * 152} H594 M449 ${167 + pair * 152} H510 V${129 + pair * 152}`" class="edge" :marker-end="arrow" :class="{ dim: completed < (pair + 1) * 2 }" />
          <rect x="600" :y="87 + pair * 152" width="245" height="84" rx="7" :class="completed >= (pair + 1) * 2 ? 'tint ready' : 'pending'" />
          <text x="722" :y="115 + pair * 152" class="center">K{{ pair }} · 特征 [{{ pair * 128 }},{{ (pair + 1) * 128 }})</text>
          <text x="722" :y="146 + pair * 152" class="center small">{{ completed >= (pair + 1) * 2 ? '可加载并累加 ✓' : `等待 n${pair * 2}、n${pair * 2 + 1}` }}</text>
        </g>
        <text x="285" y="403" class="center small muted">SwiGLU × α → FP8 + scale</text>
        <text x="722" y="403" class="center strong">完整输出需累加 K0 + K1</text>
      </g>

      <g v-else-if="view === 'schedule'">
        <text x="450" y="26" class="center">先预热 L1 → 再交错派发 L2 / L1</text>
        <g v-for="(stage, index) in ['L1', 'L1', '…', 'L2', 'L1', 'L2', 'L1']" :key="index">
          <rect :x="35 + index * 120" y="47" width="96" height="42" rx="5" :class="stage === 'L2' ? 'tint result' : stage === 'L1' ? 'tint acts' : 'box'" />
          <text :x="83 + index * 120" y="74" class="center">{{ stage }}</text>
          <path v-if="index < 6" :d="`M${135 + index * 120} 68 H${149 + index * 120}`" class="edge" :marker-end="arrow" />
        </g>
        <text x="450" y="127" class="center strong">输入块 q · 两层推进条件</text>
        <g v-for="task in [0, 1]" :key="task">
          <rect x="50" :y="153 + task * 60" width="325" height="47" rx="6" class="tint acts" />
          <text x="212" :y="182 + task * 60" class="center small">L1(q) · 列 [{{ task * 256 }},{{ (task + 1) * 256 }}) · 已派发</text>
        </g>
        <path d="M380 176 H420 V206 H550 M380 236 H420 V206" class="signal" :marker-end="signalArrow" />
        <rect x="555" y="181" width="285" height="50" rx="6" class="box" />
        <text x="697" y="212" class="center">允许派发 L2(q)</text>
        <path d="M697 235 V302" class="edge" :marker-end="arrow" />
        <rect x="50" y="307" width="325" height="58" rx="6" class="tint ready" />
        <text x="212" y="332" class="center">A(q) · K0 已写完</text>
        <text x="212" y="353" class="center small muted">n0、n1 的就绪位已设置</text>
        <path d="M380 336 H550" class="signal" :marker-end="signalArrow" />
        <rect x="555" y="307" width="285" height="58" rx="6" class="box" />
        <text x="697" y="332" class="center">L2(q) 加载并累加 K0</text>
        <text x="697" y="353" class="center small muted">K1 按自己的就绪位等待</text>
      </g>

      <g v-else>
        <text x="450" y="27" class="center strong">同一物理槽 · q 与 q + R</text>
        <g v-for="slot in [0, 1]" :key="slot" :transform="`translate(${80 + slot * 440},65)`">
          <rect width="300" height="166" rx="8" class="panel" />
          <text x="150" y="29" class="center strong">{{ slot === 0 ? 'L1 输入环' : 'L2 输入环' }}</text>
          <rect x="32" y="48" width="236" height="63" rx="6" :class="slot === 0 ? 'tint acts' : 'tint result'" />
          <text x="150" y="76" class="center">{{ slot === 0 ? (phase === 0 ? 'X(q)' : 'X(q + R)') : (phase < 2 ? 'A(q)' : 'A(q + R)') }}</text>
          <text x="150" y="99" class="center small">{{ slot === 0 ? '[64,H]' : '[64,F]' }}</text>
          <text x="150" y="140" class="center small">{{ slot === 0 ? (phase === 0 ? 'L1 消费旧输入' : 'L1 槽可覆盖 · 新输入搬入') : phase < 2 ? 'L2 仍消费旧中间值' : 'L2 槽可覆盖 · 新中间值写入' }}</text>
        </g>
        <text x="450" y="264" class="center small muted">下方跟踪 q 的一项 L2 任务及其回传片段</text>
        <rect x="65" y="290" width="250" height="70" rx="7" :class="phase === 3 ? 'pending' : 'tint result'" />
        <text x="190" y="318" class="center">TMEM · {{ phase < 2 ? '该任务正在累加' : phase === 2 ? '该任务结果就绪' : '允许覆盖' }}</text>
        <text x="190" y="344" class="center small muted">{{ phase === 3 ? '可供后续同槽任务覆盖' : '计算结果单独保存' }}</text>
        <path d="M320 325 H395" class="edge" :marker-end="arrow" />
        <rect x="400" y="290" width="200" height="70" rx="7" :class="phase === 3 ? 'tint result' : 'pending'" />
        <text x="500" y="318" class="center">寄存器 · 旧结果</text>
        <text x="500" y="343" class="center small muted">末次 TMEM 读取完成</text>
        <path d="M605 325 H680" class="edge" :marker-end="arrow" />
        <rect x="685" y="290" width="150" height="70" rx="7" class="box" />
        <text x="760" y="318" class="center">转换 / 写回</text>
        <text x="760" y="344" class="center small muted">写入源卡回传槽</text>
        <rect x="65" y="431" width="300" height="112" rx="7" class="box" />
        <text x="215" y="458" class="center">来源记录 q</text>
        <text x="215" y="490" class="center small">第 0 行 → GPU 0 / a0 / s0</text>
        <text x="215" y="520" class="center small">第 1 行 → GPU 2 / c1 / s1</text>
        <path d="M215 427 V392 H655 V345 H680" class="edge" :marker-end="arrow" />
        <g v-if="phase === 3">
          <path d="M760 365 V412 H545 V440 M760 412 V440" class="edge" :marker-end="arrow" />
          <g v-for="(name, index) in ['GPU 0 · a0 / s0', 'GPU 2 · c1 / s1']" :key="name">
            <rect :x="445 + index * 215" y="445" width="200" height="65" rx="6" class="tint result" />
            <text :x="545 + index * 215" y="484" class="center">{{ name }}</text>
          </g>
          <text x="650" y="568" class="center small muted">TMEM 已释放，寄存器中的片段继续回传</text>
        </g>
      </g>
    </svg>
  </div>
  <figcaption v-if="view === 'compute'">E0 有效行：a0、c1；块容量 64，H=512、F=256。蓝：激活；橙：权重；紫：累加或输出。</figcaption>
  <figcaption v-else-if="view === 'partition'">按有效行展示计算分工：蓝为输入，橙为权重，紫为不同输出列的累加结果。</figcaption>
  <figcaption v-else-if="view === 'readiness'">同一输入行块的特征依赖；gate/up 按配对关系分开示意。绿色为已写完，灰框为等待。</figcaption>
  <figcaption v-else-if="view === 'schedule'">任务足够多时的派发示意，预热数量由任务规模确定；虚线表示推进条件。</figcaption>
  <figcaption v-else>两环分别等待自己的消费者；来源记录按逻辑块保留。</figcaption>
</template>
