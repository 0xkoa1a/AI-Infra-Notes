<script setup lang="ts">
withDefaults(
  defineProps<{
    view?: "ownership" | "storage" | "pipeline"
  }>(),
  { view: "ownership" },
)

const querySlices = [
  { rows: "0–127", cta: 0 },
  { rows: "128–255", cta: 1 },
  { rows: "256–383", cta: 0 },
  { rows: "384–511", cta: 1 },
]

const kvSlots = ["Kⱼ", "Vⱼ", "Kⱼ₊₁", "Vⱼ₊₁", "Kⱼ₊₂", "Vⱼ₊₂"]

const mmaSteps = [
  { operation: "QK₀,₀", block: "stage 0 · K₀" },
  { operation: "QK₁,₀", block: "stage 1 · K₀" },
  { operation: "PV₀,₀", block: "stage 0 · V₀" },
  { operation: "QK₀,₁", block: "stage 0 · K₁" },
  { operation: "PV₁,₀", block: "stage 1 · V₀" },
  { operation: "QK₁,₁", block: "stage 1 · K₁" },
  { operation: "PV₀,₁", block: "stage 0 · V₁" },
]
</script>

<template>
  <figure v-if="view === 'ownership'" class="fa4" aria-label="两个 CTA 与两个 Query stage 的行归属">
    <div class="fa4__intro">
      <strong>一个 work tile：512 条 Query</strong>
      <span>2 个 CTA × 2 个 stage × 每份 128 行</span>
    </div>

    <div class="fa4__scroll" tabindex="0" role="region" aria-label="连续 Query 行在 stage 和 CTA 之间的空间归属，可横向滚动">
      <div class="fa4__ownership">
        <div class="fa4__stage-span"><strong>stage 0</strong><span>一次联合 QK · 256 行</span></div>
        <div class="fa4__stage-span"><strong>stage 1</strong><span>一次联合 QK · 256 行</span></div>
        <div class="fa4__query-direction">Query 行号连续增加 →</div>
        <div v-for="slice in querySlices" :key="slice.rows" class="fa4__query-slice" :class="`fa4__query-slice--cta${slice.cta}`">
          <span>Query {{ slice.rows }}</span>
          <strong>128 行</strong>
        </div>
        <div v-for="slice in querySlices" :key="`cta-${slice.rows}`" class="fa4__slice-owner" :class="`fa4__slice-owner--cta${slice.cta}`">
          <strong>CTA{{ slice.cta }} · SM{{ slice.cta }}</strong>
          <span>本地 S / P / O</span>
        </div>
      </div>
    </div>

    <figcaption class="fa4__caption">
      行号从 0 开始，展示首个 work tile。两个 stage 属于不同 Query 区域；同一 CTA 的两份行区间中间隔着另一 CTA 的行。
      联合 UMMA 由 leader CTA 发射，结果仍按行留在两个 CTA 各自的 TMEM 中。
    </figcaption>
  </figure>

  <figure v-else-if="view === 'storage'" class="fa4" aria-label="每个 CTA 的 SMEM 和 TMEM 缓冲区及生命周期">
    <div class="fa4__intro">
      <strong>每个 CTA 的本地存储</strong>
      <span>CTA0 与 CTA1 结构相同，各保存自己的数据分片</span>
    </div>

    <div class="fa4__storage">
      <div class="fa4__memory-row">
        <div class="fa4__memory-label"><strong>SMEM · Q</strong><span>输入驻留</span></div>
        <div class="fa4__memory-content">
          <div class="fa4__two-slots">
            <div class="fa4__slot fa4__slot--input"><strong>sQ₀</strong><span>stage 0 · 128 行</span></div>
            <div class="fa4__slot fa4__slot--input"><strong>sQ₁</strong><span>stage 1 · 128 行</span></div>
          </div>
          <p>两份 Q 在当前 work tile 的整个 K/V 循环中保留。</p>
        </div>
      </div>

      <div class="fa4__memory-row">
        <div class="fa4__memory-label"><strong>SMEM · K/V</strong><span>共有 ring</span></div>
        <div class="fa4__memory-content">
          <div class="fa4__ring">
            <div v-for="(slot, index) in kvSlots" :key="index" class="fa4__slot fa4__slot--input">
              <small>槽 {{ index }}</small><strong>{{ slot }}</strong><span>16 KiB</span>
            </div>
          </div>
          <p><strong>6 个槽交替存 K 或 V。</strong>标签是一次占用示例；MMA 消费完成并归还槽后，Load 才能写入新事件。</p>
        </div>
      </div>

      <div class="fa4__memory-row">
        <div class="fa4__memory-label"><strong>TMEM</strong><span>计算状态</span></div>
        <div class="fa4__memory-content">
          <div class="fa4__two-slots">
            <div class="fa4__slot fa4__slot--state"><strong>S₀ → P₀</strong><span>同一工作区内复用</span></div>
            <div class="fa4__slot fa4__slot--state"><strong>S₁ → P₁</strong><span>同一工作区内复用</span></div>
          </div>
          <p>Softmax 读取 S 后写入低精度 P；PV 消费 P 完成后，下一次 QK 才能覆盖该工作区。</p>
          <div class="fa4__two-slots">
            <div class="fa4__slot fa4__slot--output"><strong>O₀ · FP32</strong><span>跨 K/V block 累加</span></div>
            <div class="fa4__slot fa4__slot--output"><strong>O₁ · FP32</strong><span>跨 K/V block 累加</span></div>
          </div>
          <p>每个 stage 的 O 保留独立空间，Correction 按需缩放已有贡献。</p>
        </div>
      </div>

      <div class="fa4__memory-row">
        <div class="fa4__memory-label"><strong>SMEM · O</strong><span>输出交接</span></div>
        <div class="fa4__memory-content">
          <div class="fa4__two-slots">
            <div class="fa4__slot fa4__slot--output"><strong>sO₀</strong><span>归一化、转换后的输出</span></div>
            <div class="fa4__slot fa4__slot--output"><strong>sO₁</strong><span>归一化、转换后的输出</span></div>
          </div>
          <p>Correction 写入 → Epilogue 发出 TMA store → store 完成对源 SMEM 的读取 → 归还该槽。</p>
        </div>
      </div>
    </div>

    <figcaption class="fa4__caption">
      参考配置为 BF16、D = Dv = 128、2CTA、q_stage = 2。6 个 K/V 槽合计 96 KiB，表示 6 个事件槽；一个 CTA pair 包含两套这样的本地存储。
      图示按用途分组，位置与面积不表示物理地址或容量比例。
    </figcaption>
  </figure>

  <figure v-else class="fa4" aria-label="CTA pair 从 prologue 开始的联合 UMMA、本地计算与异步搬运窗口">
    <div class="fa4__intro">
      <strong>从 prologue 开始的一段推进窗口</strong>
      <span>下标依次表示 stage、K/V 访问次序；可横向滚动</span>
    </div>

    <div class="fa4__scroll" tabindex="0" role="region" aria-label="流水线依赖示意，可横向滚动">
      <div class="fa4__pipeline">
        <div class="fa4__axis-label">推进方向 →</div>
        <div v-for="(_, index) in mmaSteps" :key="`step-${index}`" class="fa4__step">{{ index + 1 }}</div>

        <div class="fa4__resource"><strong>Tensor Core</strong><span>两个 SM 联合执行<br>leader CTA 发射</span></div>
        <div v-for="step in mmaSteps" :key="`${step.operation}-${step.block}`" class="fa4__operation fa4__operation--mma">
          <strong>{{ step.operation }}</strong><span>{{ step.block }}</span>
        </div>

        <div class="fa4__resource"><strong>CUDA Core / MUFU</strong><span>两 CTA 各算本地行<br>同一 SM 内共享资源</span></div>
        <div class="fa4__cuda-band">
          <div class="fa4__cta-note"><span>CTA0 / SM0：本地 128 行</span><span>CTA1 / SM1：本地 128 行</span></div>
          <div class="fa4__local-work fa4__local-work--s0">
            <strong>Softmax₀,₀</strong><span>准备 P₀,₀ 前段</span><small>首轮无需修正旧 O</small>
          </div>
          <div class="fa4__local-work fa4__local-work--tail0"><strong>P₀,₀ 尾段</strong><span>可与 PV₀,₀ 前段交错</span></div>
          <div class="fa4__local-work fa4__local-work--s1">
            <strong>Softmax₁,₀</strong><span>准备 P₁,₀ 前段</span><small>首轮无需修正旧 O</small>
          </div>
          <div class="fa4__local-work fa4__local-work--tail1"><strong>P₁,₀ 尾段</strong><span>可与 PV₁,₀ 前段交错</span></div>
          <div class="fa4__local-work fa4__local-work--next0"><strong>Softmax₀,₁</strong><span>row scale → Correction 0</span><small>P₀,₁ 与旧 O₀ 就绪后，PV₀,₁ 才能推进</small></div>
          <div class="fa4__local-work fa4__local-work--next1"><strong>Softmax₁,₁</strong><span>row scale → Correction 1</span><small>准备窗口外的 PV₁,₁</small></div>
          <p class="fa4__shared-note">Correction 按需缩放上一轮 PV 留下的 O。两组 Softmax 与 Correction 可以交错或部分重叠；同一 SM 内共享 CUDA Core / MUFU 吞吐。</p>
        </div>

        <div class="fa4__resource"><strong>TMA Load</strong><span>两 CTA 装入各自分片</span></div>
        <div class="fa4__transfer fa4__transfer--load">
          <strong>K₀ / V₀ → K₁ / V₁ → …</strong>
          <span>各自的 K/V ring 有空槽就能预取；MMA 消费完成后归还槽。</span>
        </div>

        <div class="fa4__resource"><strong>TMA Store</strong><span>两 CTA 写出本地结果</span></div>
        <div class="fa4__transfer fa4__transfer--store">
          <strong>上一 work tile 的 O → GMEM</strong>
          <span>异步写出可跨入当前 K/V 循环；源 sO 槽在 store 读完后归还。</span>
        </div>
      </div>
    </div>

    <figcaption class="fa4__caption">
      图示从当前 work tile 的两次 QK prologue 开始，按访问次序标记 block 0、1，展示一种保守的交错顺序。
      步骤宽度、条长均不表示实测耗时；两个 stage 没有图中所示的强制 Softmax 先后关系。
      每次联合 QK 产生两个 CTA 各 128 行的 S；各 CTA 随后对本地行执行对称的 Softmax 与 Correction。
      资源行概括 CTA pair 对应的执行引擎，省略 SM 内的发射细节；首轮没有旧 O 需要修正，后续 Correction 等对应的新 row scale 后按需执行。
    </figcaption>
  </figure>
</template>

<style scoped>
.fa4 {
  margin: 1.6rem 0 2rem;
  color: var(--vp-c-text-1);
  font-size: 0.84rem;
  line-height: 1.55;
}

.fa4 *,
.fa4 *::before,
.fa4 *::after {
  box-sizing: border-box;
}

.fa4__intro {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem 1rem;
  align-items: baseline;
  margin-bottom: 0.85rem;
}

.fa4__intro > span,
.fa4__caption,
.fa4__memory-label > span,
.fa4__resource > span,
.fa4__step,
.fa4__axis-label {
  color: var(--infra-diagram-muted);
}

.fa4__scroll {
  overflow-x: auto;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 0.7rem;
  background: var(--vp-c-bg);
}

.fa4__scroll:focus-visible {
  outline: 2px solid var(--infra-token-flow);
  outline-offset: 3px;
}

.fa4__ownership {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.5rem;
  min-width: 36rem;
  padding: 1rem;
}

.fa4__stage-span {
  display: flex;
  grid-column: span 2;
  flex-direction: column;
  padding: 0.45rem 0.5rem 0.65rem;
  border-right: 1px solid var(--infra-diagram-line);
  border-bottom: 2px solid var(--infra-diagram-line);
  border-left: 1px solid var(--infra-diagram-line);
  text-align: center;
}

.fa4__stage-span > span {
  color: var(--infra-diagram-muted);
  font-size: 0.75rem;
}

.fa4__query-direction {
  grid-column: 1 / 5;
  margin-top: 0.35rem;
  color: var(--infra-diagram-muted);
  font-size: 0.73rem;
}

.fa4__query-slice {
  display: flex;
  min-height: 5.2rem;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.35rem;
  padding: 0.6rem;
  border: 1.5px solid currentcolor;
  border-radius: 0.4rem;
}

.fa4__query-slice > span {
  color: var(--vp-c-text-1);
  font-size: 0.79rem;
}

.fa4__query-slice--cta0,
.fa4__slice-owner--cta0 {
  color: var(--infra-planning);
}

.fa4__query-slice--cta1,
.fa4__slice-owner--cta1 {
  color: var(--infra-execution);
}

.fa4__query-slice--cta0 {
  background: color-mix(in srgb, var(--infra-planning) 8%, transparent);
}

.fa4__query-slice--cta1 {
  background: color-mix(in srgb, var(--infra-execution) 8%, transparent);
}

.fa4__slice-owner {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 0.2rem;
}

.fa4__slice-owner > span {
  margin-top: 0.2rem;
  color: var(--infra-diagram-muted);
  font-size: 0.74rem;
}

.fa4__caption {
  margin-top: 0.8rem;
  font-size: 0.77rem;
  line-height: 1.7;
}

.fa4__storage {
  overflow: hidden;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 0.7rem;
  background: var(--vp-c-bg);
}

.fa4__memory-row {
  display: grid;
  grid-template-columns: 7.5rem minmax(0, 1fr);
  gap: 1rem;
  padding: 1rem;
}

.fa4__memory-row + .fa4__memory-row {
  border-top: 1px solid var(--infra-diagram-line);
}

.fa4__memory-label > strong,
.fa4__memory-label > span,
.fa4__resource > strong,
.fa4__resource > span {
  display: block;
}

.fa4__memory-label > span,
.fa4__resource > span {
  margin-top: 0.2rem;
  font-size: 0.73rem;
}

.fa4__two-slots {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.5rem;
}

.fa4__ring {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.35rem;
}

.fa4__slot {
  display: flex;
  flex-direction: column;
  gap: 0.12rem;
  justify-content: center;
  padding: 0.6rem 0.5rem;
  border: 1px solid currentcolor;
  border-radius: 0.4rem;
  text-align: center;
}

.fa4__slot > span,
.fa4__slot > small {
  color: var(--vp-c-text-1);
  font-size: 0.73rem;
}

.fa4__slot--input {
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 7%, transparent);
}

.fa4__slot--state {
  color: var(--infra-planning);
  background: color-mix(in srgb, var(--infra-planning) 7%, transparent);
}

.fa4__slot--output {
  color: var(--infra-execution);
  background: color-mix(in srgb, var(--infra-execution) 7%, transparent);
}

.fa4__memory-content > p {
  margin: 0.5rem 0 0;
  font-size: 0.77rem;
}

.fa4__memory-content > p + .fa4__two-slots {
  margin-top: 0.9rem;
}

.fa4__pipeline {
  display: grid;
  grid-template-columns: 8rem repeat(7, minmax(7rem, 1fr));
  gap: 0.65rem 0.4rem;
  min-width: 63rem;
  padding: 1rem;
}

.fa4__axis-label,
.fa4__step {
  font-size: 0.73rem;
}

.fa4__step {
  border-bottom: 1px solid var(--infra-diagram-line);
  text-align: center;
}

.fa4__resource {
  padding-top: 0.5rem;
}

.fa4__operation {
  display: flex;
  min-height: 4.5rem;
  flex-direction: column;
  justify-content: center;
  padding: 0.5rem;
  border: 1px solid var(--infra-execution);
  border-radius: 0.4rem;
  background: color-mix(in srgb, var(--infra-execution) 8%, transparent);
  text-align: center;
  font-size: 0.76rem;
}

.fa4__operation > strong {
  color: var(--infra-execution);
}

.fa4__cuda-band {
  display: grid;
  grid-column: 2 / 9;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 0.4rem;
  padding: 0.5rem 0;
  border-top: 1px solid var(--infra-diagram-line);
  border-bottom: 1px solid var(--infra-diagram-line);
  background: color-mix(in srgb, var(--infra-planning) 4%, transparent);
}

.fa4__cta-note {
  display: flex;
  grid-column: 1 / 8;
  justify-content: center;
  gap: 2rem;
  padding-bottom: 0.2rem;
  color: var(--infra-diagram-muted);
  font-size: 0.72rem;
}

.fa4__local-work {
  display: flex;
  grid-row: 2;
  flex-direction: column;
  gap: 0.3rem;
  padding: 0.5rem;
  border: 1px solid var(--infra-planning);
  border-radius: 0.4rem;
  font-size: 0.7rem;
}

.fa4__local-work > strong {
  color: var(--infra-planning);
  font-size: 0.73rem;
}

.fa4__local-work > small {
  font-size: inherit;
}

.fa4__local-work--s0 { grid-column: 2; }
.fa4__local-work--tail0 { grid-column: 3; }
.fa4__local-work--s1 { grid-column: 4; }
.fa4__local-work--tail1 { grid-column: 5; }
.fa4__local-work--next0 { grid-column: 6; }
.fa4__local-work--next1 { grid-column: 7; }

.fa4__shared-note {
  grid-column: 1 / 8;
  margin: 0.3rem 0.5rem 0;
  color: var(--infra-diagram-muted);
  font-size: 0.73rem;
}

.fa4__transfer {
  display: flex;
  grid-column: 2 / 9;
  flex-direction: column;
  justify-content: center;
  gap: 0.25rem;
  padding: 0.6rem 0.8rem;
  border: 1px dashed currentcolor;
  border-radius: 0.4rem;
  font-size: 0.76rem;
}

.fa4__transfer > span {
  color: var(--vp-c-text-1);
}

.fa4__transfer--load {
  color: var(--infra-token-flow);
  background: color-mix(in srgb, var(--infra-token-flow) 6%, transparent);
}

.fa4__transfer--store {
  color: var(--infra-weight-flow);
  background: color-mix(in srgb, var(--infra-weight-flow) 6%, transparent);
}

@media (max-width: 640px) {
  .fa4__memory-row {
    grid-template-columns: minmax(0, 1fr);
    gap: 0.65rem;
    padding: 0.8rem;
  }

  .fa4__memory-label {
    display: flex;
    align-items: baseline;
    gap: 0.7rem;
  }

  .fa4__ring {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media print {
  .fa4 {
    break-inside: avoid;
  }
}
</style>
