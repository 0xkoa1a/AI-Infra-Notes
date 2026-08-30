---
title: "Portable Export Fixture"
---

<script setup>
import { ref } from "vue"

const count = ref(0)
</script>

# Portable Export Fixture

This page exercises local assets, formulas, charts, Vue interaction, and Outline.

## Local asset

![Portable page fixture](./images/portable.svg)

## Formula

$$
E = mc^2
$$

## Mermaid

```mermaid
flowchart LR
  Markdown --> HTML
  HTML --> Portable
```

## ECharts

```echarts
{
  "xAxis": { "type": "category", "data": ["A", "B", "C"] },
  "yAxis": { "type": "value" },
  "series": [{ "type": "bar", "data": [2, 5, 3] }]
}
```

## Vue interaction

<button id="portable-counter" type="button" @click="count += 1">
  Count: {{ count }}
</button>

### Nested outline entry

The exported page keeps its nested outline.
