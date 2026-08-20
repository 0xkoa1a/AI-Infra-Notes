<script setup lang="ts">
import { useHeaders } from "@vuepress/helper/client"
import type { PageHeader } from "vuepress/client"
import { ClientOnly, RouteLink, useRoute } from "vuepress/client"
import { h, onMounted, ref, watch } from "vue"

const headers = useHeaders({
  selector: "#content > :where(h2, h3)",
  levels: [2, 3],
})
const route = useRoute()
const toc = ref<HTMLElement>()
const tocMarkerTop = ref("-2rem")

const isActive = (header: PageHeader): boolean => route.hash === `#${header.slug}`

const scrollTo = (top: number): void => {
  toc.value?.scrollTo({ top, behavior: "smooth" })
}

const updateTocMarker = (): void => {
  const activeItem = document.querySelector<HTMLElement>("#toc .vp-toc-item.active")

  if (!toc.value || !activeItem) {
    tocMarkerTop.value = "-2rem"
    return
  }

  tocMarkerTop.value = `${
    activeItem.getBoundingClientRect().top -
    toc.value.getBoundingClientRect().top +
    toc.value.scrollTop
  }px`
}

const keepActiveItemVisible = (hash: string): void => {
  if (!toc.value || !hash) return

  const activeLink = document.querySelector<HTMLElement>(`#toc a[href$="${hash}"]`)
  if (!activeLink) return

  const tocRect = toc.value.getBoundingClientRect()
  const linkRect = activeLink.getBoundingClientRect()

  if (linkRect.top < tocRect.top) {
    scrollTo(toc.value.scrollTop + linkRect.top - tocRect.top)
  } else if (linkRect.bottom > tocRect.bottom) {
    scrollTo(toc.value.scrollTop + linkRect.bottom - tocRect.bottom)
  }
}

onMounted(() => {
  watch(
    () => route.hash,
    (hash) => keepActiveItemVisible(hash),
    { immediate: true, flush: "post" },
  )
  watch(
    () => route.fullPath,
    () => requestAnimationFrame(updateTocMarker),
    { immediate: true, flush: "post" },
  )
})

const renderHeader = (header: PageHeader) =>
  h(
    RouteLink,
    {
      to: `#${header.slug}`,
      class: ["vp-toc-link", `level${header.level}`],
    },
    () => header.title,
  )

const renderHeaders = (items: PageHeader[]): ReturnType<typeof h> | null =>
  items.length
    ? h(
        "ul",
        { class: "vp-toc-list" },
        items.flatMap((header) => {
          const children = renderHeaders(header.children)
          return [
            h(
              "li",
              { class: ["vp-toc-item", { active: isActive(header) }] },
              renderHeader(header),
            ),
            children ? h("li", { class: "vp-toc-children" }, children) : null,
          ]
        }),
      )
    : null
</script>

<template>
  <ClientOnly>
    <div v-if="headers.length" class="vp-toc-placeholder">
      <aside id="toc" vp-toc aria-label="On this Page">
        <div class="vp-toc-header">On this Page</div>
        <div ref="toc" class="vp-toc-wrapper">
          <component :is="renderHeaders(headers)" />
          <div class="vp-toc-marker" :style="{ top: tocMarkerTop }" />
        </div>
      </aside>
    </div>
  </ClientOnly>
</template>
