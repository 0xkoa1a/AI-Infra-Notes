<script setup lang="ts">
import type { Slot } from "@vuepress/helper/client"
import { Content } from "vuepress/client"

import VPPageMeta from "@vuepress/theme-default/components/VPPageMeta.vue"
import VPPageNav from "@vuepress/theme-default/components/VPPageNav.vue"

import ArticleToc from "./ArticleToc.vue"

defineSlots<{
  top?: Slot
  bottom?: Slot
  "content-top"?: Slot
  "content-bottom"?: Slot
}>()
</script>

<template>
  <main class="vp-page">
    <slot name="top" />
    <ArticleToc />
    <div vp-content>
      <slot name="content-top" />
      <Content id="content" />
      <slot name="content-bottom" />
    </div>
    <VPPageMeta />
    <VPPageNav />
    <slot name="bottom" />
  </main>
</template>

<style lang="scss">
@use "@vuepress/theme-default/styles/mixins";
@use "@vuepress/theme-default/styles/variables" as *;

.vp-page {
  display: block;
  padding-top: var(--navbar-height);
  padding-bottom: 2rem;
  padding-inline-start: var(--sidebar-width);

  @media (max-width: $MQNarrow) {
    padding-inline-start: var(--sidebar-width-mobile);
  }

  @media (max-width: $MQMobile) {
    padding-inline-start: 0;
  }

  [vp-content] {
    @include mixins.content-wrapper;

    & {
      padding-top: 0;
    }
  }

  @media (min-width: 1440px) {
    --article-content-offset: max(
      1rem,
      calc((100% - var(--content-width) - 5rem - var(--article-toc-rail, 16rem)) / 2)
    );

    > [vp-content],
    > .vp-page-meta,
    > .vp-page-nav {
      margin-inline-start: var(--article-content-offset);
      margin-inline-end: 0;
    }

    > .vp-page-nav {
      padding-inline: 2.5rem;
    }
  }
}
</style>
