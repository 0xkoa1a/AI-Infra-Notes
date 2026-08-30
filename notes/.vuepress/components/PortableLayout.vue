<script setup lang="ts">
import { useData } from "@theme/useData"
import VPHome from "@theme/VPHome.vue"
import VPPage from "@theme/VPPage.vue"

const { frontmatter, page } = useData()
</script>

<template>
  <div
    class="vp-theme-container no-navbar no-sidebar external-link-icon"
    vp-container
  >
    <VPHome v-if="frontmatter.home" />
    <VPPage v-else :key="page.path" />
  </div>
</template>
