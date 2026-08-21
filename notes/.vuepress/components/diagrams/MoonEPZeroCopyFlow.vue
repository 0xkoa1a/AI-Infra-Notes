<template>
  <figure class="zero-copy-flow" aria-label="传统 EP 中 dispatch、permute、专家计算、unpermute 和用户 buffer 拷贝的数据流">
    <div class="zero-copy-flow__steps">
      <div class="zero-copy-flow__step zero-copy-flow__step--dispatch">
        <strong>Dispatch</strong>
        <span>按源 Rank 排列</span>
      </div>
      <div class="zero-copy-flow__arrow" aria-hidden="true"><span></span></div>
      <div class="zero-copy-flow__step zero-copy-flow__step--movement">
        <strong>Permute</strong>
        <span>按 Expert 分组</span>
      </div>
      <div class="zero-copy-flow__arrow" aria-hidden="true"><span></span></div>
      <div class="zero-copy-flow__step zero-copy-flow__step--compute">
        <strong>Expert Compute</strong>
        <span>Grouped GEMM</span>
      </div>
      <div class="zero-copy-flow__arrow" aria-hidden="true"><span></span></div>
      <div class="zero-copy-flow__step zero-copy-flow__step--movement">
        <strong>Unpermute</strong>
        <span>恢复 token 顺序</span>
      </div>
      <div class="zero-copy-flow__arrow" aria-hidden="true"><span></span></div>
      <div class="zero-copy-flow__step zero-copy-flow__step--copy">
        <strong>Copy</strong>
        <span>写入用户 buffer</span>
      </div>
    </div>

    <figcaption>通信与计算之间存在三次额外搬动：按专家重排、按 token 恢复，以及从通信 buffer 拷到用户 buffer。</figcaption>
  </figure>
</template>

<style scoped>
.zero-copy-flow {
  margin: 1.5rem 0 1.8rem;
  color: var(--vp-c-text-1);
}

.zero-copy-flow__steps {
  display: grid;
  grid-template-columns: minmax(5.5rem, 1fr) 1.1rem minmax(5.5rem, 1fr) 1.1rem minmax(5.5rem, 1fr) 1.1rem minmax(5.5rem, 1fr) 1.1rem minmax(5.5rem, 1fr);
  align-items: center;
  gap: 0.35rem;
}

.zero-copy-flow__step {
  display: grid;
  min-width: 0;
  min-height: 3.4rem;
  align-content: center;
  gap: 0.2rem;
  padding: 0.45rem 0.35rem;
  border-top: 2px solid currentcolor;
  border-bottom: 1px solid var(--infra-diagram-line);
  text-align: center;
}

.zero-copy-flow__step strong {
  font-family: var(--code-font-family);
  font-size: 0.72rem;
  line-height: 1.25;
}

.zero-copy-flow__step span {
  color: var(--infra-diagram-muted);
  font-size: 0.68rem;
  line-height: 1.25;
}

.zero-copy-flow__step--dispatch {
  color: var(--infra-token-flow);
}

.zero-copy-flow__step--movement {
  color: var(--infra-reroute);
}

.zero-copy-flow__step--compute {
  color: var(--infra-execution);
}

.zero-copy-flow__step--copy {
  color: var(--infra-weight-flow);
}

.zero-copy-flow__arrow {
  position: relative;
  height: 1px;
  background: var(--infra-diagram-muted);
}

.zero-copy-flow__arrow span {
  position: absolute;
  top: -0.22rem;
  right: -0.04rem;
  border-top: 0.22rem solid transparent;
  border-bottom: 0.22rem solid transparent;
  border-left: 0.32rem solid var(--infra-diagram-muted);
}

.zero-copy-flow figcaption {
  margin-top: 0.85rem;
  color: var(--infra-diagram-muted);
  font-size: 0.78rem;
  line-height: 1.5;
  text-align: center;
}

@media print {
  .zero-copy-flow {
    break-inside: avoid;
  }
}
</style>
