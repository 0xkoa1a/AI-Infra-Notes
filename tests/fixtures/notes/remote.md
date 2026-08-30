---
title: "External Mode Fixture"
---

<script setup>
import { onMounted } from "vue"

const endpoint = ["https://example.invalid", "portable.json"].join("/")
onMounted(() => {
  void fetch(endpoint)
})
</script>

# External Mode Fixture

This fixture proves that variable network calls require explicit external mode.
