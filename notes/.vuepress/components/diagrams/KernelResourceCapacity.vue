<script setup lang="ts">
const capacity = 8
const fixedUsage = 2
const configurations = [
  { buffers: 2, blocks: ["A", "B"] },
  { buffers: 3, blocks: ["A"] },
]
</script>

<template>
  <figure class="capacity-figure" aria-label="共享内存容量对照教学示意：每块增加缓冲后，可驻留的块数从两个减少到一个">
    <p class="capacity-label">相同的 SM 共享内存容量：8 等份</p>
    <div class="panels">
      <div v-for="configuration in configurations" :key="configuration.buffers" class="panel">
        <p class="panel-label">每块 {{ configuration.buffers }} 组缓冲</p>
        <div class="capacity">
          <div
            v-for="block in configuration.blocks"
            :key="block"
            class="block"
            :style="{ flex: `${fixedUsage + configuration.buffers} 0 0` }"
          >
            <div class="fixed" :style="{ flex: `${fixedUsage} 0 0` }">块 {{ block }} · 固定占用 2 份</div>
            <div class="buffers" :style="{ flex: `${configuration.buffers} 0 0` }">输入缓冲 {{ configuration.buffers }} 份</div>
          </div>
          <div
            v-if="configuration.blocks.length === 1"
            class="free"
            :style="{ flex: `${capacity - fixedUsage - configuration.buffers} 0 0` }"
          >
            <span>剩余 3 份</span>
            <span>下一块需要 5 份</span>
          </div>
        </div>
        <p class="result">驻留 {{ configuration.blocks.length }} 个块</p>
      </div>
    </div>
    <figcaption>教学假设：每块固定占用 2 份，每组缓冲占用 1 份；其他驻留限制不生效。长度按这些假设比例绘制。</figcaption>
  </figure>
</template>

<style scoped>
.capacity-figure {
  margin: 1.5rem 0;
  border: 1px solid var(--infra-diagram-line);
  border-radius: 12px;
  background: var(--vp-c-bg);
  overflow: hidden;
}
.capacity-label { margin: 0; padding: 1rem 1rem 0; text-align: center; font-weight: 600; }
.panels { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.5rem; padding: 0.5rem 1.5rem 1rem; }
.panel-label { margin: 0.5rem 0; text-align: center; }
.capacity { height: 16rem; display: flex; flex-direction: column; border: 2px solid var(--infra-diagram-line); border-radius: 8px; overflow: hidden; }
.block { min-height: 0; display: flex; flex-direction: column; }
.block + .block { border-top: 2px solid var(--infra-diagram-line); }
.fixed, .buffers, .free { display: flex; align-items: center; justify-content: center; text-align: center; padding-inline: 0.4rem; font-size: 0.9rem; }
.fixed { background: var(--vp-c-bg-alt); }
.buffers { background: color-mix(in srgb, var(--infra-token-flow) 14%, var(--vp-c-bg)); color: var(--infra-token-flow); }
.free { flex-direction: column; color: var(--infra-diagram-muted); border-top: 1px dashed var(--infra-diagram-line); }
.result { margin: 0.6rem 0 0; text-align: center; font-weight: 600; }
figcaption { padding: 0.75rem 1rem; border-top: 1px solid var(--infra-diagram-line); color: var(--infra-diagram-muted); font-size: 0.85rem; line-height: 1.6; }
@media (max-width: 480px) {
  .panels { gap: 0.75rem; padding-inline: 0.75rem; }
  .fixed, .buffers, .free { font-size: 0.8rem; }
}
@media print { .capacity-figure { break-inside: avoid; } }
</style>
