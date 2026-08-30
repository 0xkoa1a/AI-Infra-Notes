<script setup lang="ts">
import type { Slot } from "@vuepress/helper/client"
import { Content } from "vuepress/client"

import VPPageMeta from "@vuepress/theme-default/components/VPPageMeta.vue"
import VPPageNav from "@vuepress/theme-default/components/VPPageNav.vue"

import ArticleToc from "./ArticleToc.vue"

const isPortableExport = import.meta.env.VITE_PORTABLE_EXPORT === "1"

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
    <div class="vp-article-layout">
      <div class="vp-article-main">
        <div vp-content>
          <slot name="content-top" />
          <Content id="content" />
          <slot name="content-bottom" />
        </div>
        <VPPageMeta v-if="!isPortableExport" />
        <VPPageNav v-if="!isPortableExport" />
      </div>
      <ArticleToc />
    </div>
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

  .vp-article-main {
    min-width: 0;
  }

  @media (min-width: 1440px) {
    --article-column-width: calc(var(--content-width) + 5rem);
    --article-layout-gutter: 1rem;
    --article-toc-gap: 1rem;
    --article-toc-width: 15rem;
    --article-layout-width: calc(
      var(--article-column-width) + var(--article-toc-gap) +
        var(--article-toc-width)
    );

    .vp-article-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) var(--article-toc-width);
      column-gap: var(--article-toc-gap);
      align-items: start;
      width: calc(100% - 2 * var(--article-layout-gutter));
      max-width: var(--article-layout-width);
      margin-inline: auto;
    }

    .vp-article-main > [vp-content],
    .vp-article-main > .vp-page-meta,
    .vp-article-main > .vp-page-nav {
      box-sizing: border-box;
      width: 100%;
      max-width: none;
      margin-inline: 0;
    }

    .vp-article-main > .vp-page-nav {
      padding-inline: 2.5rem;
    }
  }

  @media (min-width: 1920px) {
    --content-width: 827px;
    --article-toc-gap: 80px;
    --article-toc-width: 220px;
  }
}
</style>
