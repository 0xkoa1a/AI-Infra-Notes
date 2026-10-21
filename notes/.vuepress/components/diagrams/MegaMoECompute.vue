<script setup lang="ts">
import { computed, ref, useId } from "vue"

const props = defineProps<{ view: string }>()
const id = useId()
const layer = ref(0)
const completed = ref(2)
const phase = ref(0)
const heights: Record<string, number> = { compute: 630, partition: 520, readiness: 430, schedule: 365, ring: 600 }
const height = computed(() => heights[props.view] || 500)
const title = computed(() => {
  const titles: Record<string, string> = { compute: `GPU 0 / E0 · ${layer.value === 0 ? 'L1：输入到中间激活' : 'L2：中间激活到回传片段'}`, partition: "激活按行分放，两个 CTA 协作产生不同输出列", readiness: "L1 列块就绪后，L2 按输入特征块推进", schedule: "先派发生产任务，再等待具体数据", ring: "固定物理槽，观察旧块释放与新块写入" }
  return titles[props.view] || "MegaMoE 计算"
})
const arrow = `url(#${id}-arrow)`
const signalArrow = `url(#${id}-signal)`
const phaseLabels = ["旧块消费", "复用 L1 槽", "复用 L2 槽", "旧块回传"]
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
      <desc :id="`${id}-desc`">{{ view === 'compute' ? `跟踪 GPU0 专家 E0 的输入块，有效行 a0、c1，块容量64，H为512，F为256。当前为${layer === 0 ? 'L1' : 'L2'}，激活和权重通过 TMA 进入两 CTA 的 SMEM，leader 发射 UMMA，结果累加在两份 TMEM 中。${layer === 0 ? '后处理生成含路由权重的 FP8 中间激活与 scale，写入 L2 输入环。' : '后处理生成 BF16 结果片段，分别写回 GPU0 的 a0 槽0和 GPU2 的 c1 槽1。'}` : view === 'partition' ? '满64行输入块按行分成两半，CTA0的SMEM保存前32行，CTA1保存后32行，各自保存自己负责的128列对应权重。leader 发射2CTA UMMA，共同使用两边的激活，各CTA的本地TMEM存放不同的128列结果。' : view === 'readiness' ? `四个L1原始列块各有64对gate/up，SwiGLU后各产生64个中间特征。当前前${completed}块已写完；L2第一个输入特征块等待前两块，第二个等待后两块。完整输出需要两个输入特征块都参与累加。` : view === 'schedule' ? '顶部按派发先后排列L1预热与后续L2/L1交错。调度器确认生产所需输入的L1任务已派发，才继续派发L2；L2加载线程另等当前所需特征块写完。两种等待对应任务推进与数据可读。' : `当前为${phaseLabels[phase]}阶段。L1输入环和L2输入环各自固定一个物理槽，映射旧逻辑块q与新块q加R。L1槽先释放并装入新块，L2槽等自己的旧消费者结束后再复用。旧块的来源记录单独保留，用于把结果写回GPU0的a0槽0和GPU2的c1槽1。TMEM末次读取完成后可释放，寄存器中的旧结果继续转换与回传。` }}</desc>
      <defs>
        <marker :id="`${id}-arrow`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="arrowhead" /></marker>
        <marker :id="`${id}-signal`" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" class="signal-head" /></marker>
      </defs>

      <g v-if="view === 'compute'">
        <text x="450" y="25" class="center strong">{{ title }}</text>
        <rect x="20" y="58" width="205" height="312" rx="8" class="panel" />
        <text x="122" y="83" class="center">本卡 HBM</text>
        <text x="122" y="110" class="center acts ink">{{ layer === 0 ? 'L1 输入 X' : 'L2 输入 A' }}</text>
        <g v-for="row in 4" :key="row">
          <rect v-for="col in 4" :key="col" :x="78 + (col - 1) * 24" :y="122 + (row - 1) * 15" width="24" height="15" :class="row <= 2 ? 'tint acts' : 'pending'" />
          <text x="68" :y="134 + (row - 1) * 15" text-anchor="end" class="small muted">{{ row === 1 ? 'a0' : row === 2 ? 'c1' : row === 3 ? '…' : '补齐' }}</text>
        </g>
        <text x="122" y="204" class="center small muted">[64, {{ layer === 0 ? 512 : 256 }}] + scale</text>
        <rect x="48" y="239" width="150" height="84" rx="5" class="tint weight" />
        <text x="123" y="265" class="center">{{ layer === 0 ? 'W₁ 的输出列组' : 'W₂ 的输出列组' }}</text>
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
          <text x="480" y="524" class="center">L2 输入环 · A 的特征 [0,128)</text>
          <text x="480" y="550" class="center">[64,128] + scale</text>
          <text x="480" y="572" class="center small muted">行：a0、c1 与补齐位置</text>
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
        <g v-for="half in [0, 1]" :key="half">
          <rect x="325" :y="33 + half * 36" width="250" height="36" class="tint acts" />
          <text x="450" :y="57 + half * 36" class="center small">{{ half === 0 ? '前 32 行 · CTA 0 加载' : '后 32 行 · CTA 1 加载' }}</text>
          <path :d="half === 0 ? 'M321 51 C240 51,170 105,170 198' : 'M579 87 C660 87,600 125,600 198'" class="edge" :marker-end="arrow" />
        </g>
        <text x="450" y="126" class="center small muted">满块示意 · 64 行 × 128 输入特征</text>
        <g v-for="cta in [0, 1]" :key="cta" :transform="`translate(${80 + cta * 430},150)`">
          <rect width="310" height="154" rx="8" class="panel" />
          <text x="155" y="28" class="center strong">CTA {{ cta }} · SMEM</text>
          <rect x="25" y="53" width="130" height="62" class="tint acts" />
          <text x="90" y="89" class="center small">{{ cta === 0 ? '前 32 行' : '后 32 行' }}</text>
          <text x="90" y="139" class="center small muted">输入 [32,128]</text>
          <rect x="180" y="53" width="110" height="62" class="tint weight" />
          <text x="235" y="78" class="center small">本 CTA 权重</text>
          <text x="235" y="101" class="center small">[128,128]</text>
          <text x="235" y="139" class="center small muted">输出列 {{ cta * 128 }}–{{ (cta + 1) * 128 - 1 }}</text>
        </g>
        <path d="M235 308 V325 H370 V337 M665 308 V325 H530 V337" class="edge" :marker-end="arrow" />
        <rect x="300" y="342" width="300" height="43" rx="7" class="box" />
        <text x="450" y="369" class="center">2CTA UMMA · 共同使用两半输入</text>
        <path d="M370 389 V411 H235 V423 M530 389 V411 H665 V423" class="edge" :marker-end="arrow" />
        <g v-for="cta in [0, 1]" :key="cta">
          <rect :x="100 + cta * 430" y="428" width="270" height="70" rx="7" class="tint result" />
          <text :x="235 + cta * 430" y="455" class="center">CTA {{ cta }} · 本地 TMEM</text>
          <text :x="235 + cta * 430" y="481" class="center small">[64,128] · 不同输出列</text>
        </g>
      </g>

      <g v-else-if="view === 'readiness'">
        <text x="130" y="29" class="center">L1 原始列块</text>
        <text x="390" y="29" class="center">SwiGLU 后的特征</text>
        <text x="723" y="29" class="center">L2 输入特征块</text>
        <g v-for="block in [0, 1, 2, 3]" :key="block" :transform="`translate(0,${65 + block * 76})`">
          <rect x="25" width="210" height="52" rx="5" :class="block < completed ? 'tint ready' : 'pending'" />
          <text x="45" y="21" class="small">n{{ block }} {{ block < completed ? '✓' : '○' }}</text>
          <text x="135" y="37" class="center">64 对 gate / up</text>
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
        <text x="35" y="27" class="small muted">一段派发序列 →</text>
        <g v-for="(stage, index) in ['L1', 'L1', 'L1', 'L2', 'L1', 'L2', 'L1']" :key="index">
          <rect :x="35 + index * 120" y="49" width="96" height="48" rx="5" :class="stage === 'L1' ? 'tint acts' : 'tint result'" />
          <text :x="83 + index * 120" y="79" class="center strong">{{ stage }}</text>
          <path v-if="index < 6" :d="`M${135 + index * 120} 73 H${149 + index * 120}`" class="edge" :marker-end="arrow" />
        </g>
        <path d="M35 112 V123 H371 V112" class="edge" />
        <text x="203" y="148" class="center small muted">预热：先把生产任务派发出去</text>
        <g v-for="(name, index) in ['所需 L1 任务已派发', '当前 K 块已写完']" :key="name">
          <rect x="90" :y="185 + index * 95" width="290" height="48" rx="6" :class="index === 0 ? 'tint acts' : 'tint ready'" />
          <text x="235" :y="215 + index * 95" class="center">{{ name }}</text>
          <path :d="`M384 ${209 + index * 95} H555`" class="signal" :marker-end="signalArrow" />
          <rect x="560" :y="185 + index * 95" width="250" height="48" rx="6" class="box" />
          <text x="685" :y="215 + index * 95" class="center">{{ index === 0 ? '调度器派发 L2' : 'L2 加载线程读取' }}</text>
        </g>
        <path d="M685 237 V274" class="edge" :marker-end="arrow" />
      </g>

      <g v-else>
        <text x="450" y="27" class="center strong">同一物理槽 · q 与 q + R</text>
        <g v-for="slot in [0, 1]" :key="slot" :transform="`translate(${80 + slot * 440},65)`">
          <rect width="300" height="166" rx="8" class="panel" />
          <text x="150" y="29" class="center strong">{{ slot === 0 ? 'L1 输入环' : 'L2 输入环' }}</text>
          <rect x="32" y="48" width="236" height="63" rx="6" :class="slot === 0 ? 'tint acts' : 'tint result'" />
          <text x="150" y="76" class="center">{{ slot === 0 ? (phase === 0 ? 'X(q)' : 'X(q + R)') : (phase < 2 ? 'A(q)' : 'A(q + R)') }}</text>
          <text x="150" y="99" class="center small">{{ slot === 0 ? '[64,H]' : '[64,F]' }}</text>
          <text x="150" y="140" class="center small">{{ slot === 0 ? (phase === 0 ? 'L1 消费旧输入' : '旧 L1 已归还 · 新输入搬入') : phase < 2 ? 'L2 仍消费旧中间值' : '旧 L2 已归还 · 新中间值写入' }}</text>
        </g>
        <text x="450" y="264" class="center small muted">旧块的计算结果与来源记录继续保留</text>
        <rect x="65" y="290" width="250" height="70" rx="7" :class="phase === 3 ? 'pending' : 'tint result'" />
        <text x="190" y="318" class="center">TMEM · {{ phase < 2 ? 'q 的累加结果' : phase === 2 ? 'q 的 L2 结果' : '已释放' }}</text>
        <text x="190" y="344" class="center small muted">{{ phase === 3 ? '可供后续同槽任务覆盖' : '计算结果单独保存' }}</text>
        <path d="M320 325 H395" class="edge" :marker-end="arrow" />
        <rect x="400" y="290" width="200" height="70" rx="7" :class="phase === 3 ? 'tint result' : 'pending'" />
        <text x="500" y="318" class="center">寄存器 · 旧结果</text>
        <text x="500" y="343" class="center small muted">末次 TMEM 读取完成</text>
        <path d="M605 325 H680" class="edge" :marker-end="arrow" />
        <rect x="685" y="290" width="150" height="70" rx="7" class="box" />
        <text x="760" y="318" class="center">转换 / 写回</text>
        <text x="760" y="344" class="center small muted">peer store</text>
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
          <text x="650" y="568" class="center small muted">TMEM 已归还，旧结果继续回传</text>
        </g>
      </g>
    </svg>
  </div>
  <figcaption v-if="view === 'compute'">E0 有效行：a0、c1；块容量 64，H=512、F=256。蓝：激活；橙：权重；紫：累加或输出。</figcaption>
  <figcaption v-else-if="view === 'partition'">同一 GPU 内的一对 CTA；蓝：输入行；橙：权重；紫：输出列。</figcaption>
  <figcaption v-else-if="view === 'readiness'">绿色：已写完；灰框：等待。每个原始列块的 64 对 gate/up 生成 64 个特征。</figcaption>
  <figcaption v-else-if="view === 'schedule'">方块表示派发顺序；虚线表示允许下一步推进的条件。</figcaption>
  <figcaption v-else>两环分别等待自己的消费者；来源记录按逻辑块保留。</figcaption>
</template>
