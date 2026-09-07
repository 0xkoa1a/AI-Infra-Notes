import { defineMermaidConfig } from "@vuepress/plugin-markdown-chart/client"
import { defineClientConfig } from "vuepress/client"

import "katex/dist/katex.min.css"

import EPLoadBalancingMap from "./components/diagrams/EPLoadBalancingMap.vue"
import FlashAttentionV4Diagram from "./components/diagrams/FlashAttentionV4Diagram.vue"
import CriticalPathTimeline from "./components/diagrams/CriticalPathTimeline.vue"
import DeepEPBufferTransform from "./components/diagrams/DeepEPBufferTransform.vue"
import DeepEPDedupExplorer from "./components/diagrams/DeepEPDedupExplorer.vue"
import DeepEPHybridPipeline from "./components/diagrams/DeepEPHybridPipeline.vue"
import DeepEPPathExplorer from "./components/diagrams/DeepEPPathExplorer.vue"
import MoonEPBalanceIterations from "./components/diagrams/MoonEPBalanceIterations.vue"
import MoonEPWeightLayout from "./components/diagrams/MoonEPWeightLayout.vue"
import MoonEPZeroCopyFlow from "./components/diagrams/MoonEPZeroCopyFlow.vue"
import RankLoadDistribution from "./components/diagrams/RankLoadDistribution.vue"

defineMermaidConfig({
  flowchart: {
    curve: "linear",
    htmlLabels: true,
    useMaxWidth: true,
  },
  themeVariables: (isDarkMode) =>
    isDarkMode
      ? {
          background: "#1b1b1f",
          fontFamily: "inherit",
          lineColor: "#a8a8ad",
          primaryBorderColor: "#68686d",
          primaryColor: "#252529",
          primaryTextColor: "#dfdfe4",
          secondaryBorderColor: "#68686d",
          secondaryColor: "#303035",
          secondaryTextColor: "#dfdfe4",
          tertiaryBorderColor: "#68686d",
          tertiaryColor: "#252529",
          tertiaryTextColor: "#dfdfe4",
        }
      : {
          background: "#ffffff",
          fontFamily: "inherit",
          lineColor: "#60646c",
          primaryBorderColor: "#c2c2c4",
          primaryColor: "#ffffff",
          primaryTextColor: "#2c2c30",
          secondaryBorderColor: "#c2c2c4",
          secondaryColor: "#f6f6f7",
          secondaryTextColor: "#2c2c30",
          tertiaryBorderColor: "#c2c2c4",
          tertiaryColor: "#ffffff",
          tertiaryTextColor: "#2c2c30",
        },
})

export default defineClientConfig({
  enhance({ app }) {
    app.component("CriticalPathTimeline", CriticalPathTimeline)
    app.component("DeepEPBufferTransform", DeepEPBufferTransform)
    app.component("DeepEPDedupExplorer", DeepEPDedupExplorer)
    app.component("DeepEPHybridPipeline", DeepEPHybridPipeline)
    app.component("DeepEPPathExplorer", DeepEPPathExplorer)
    app.component("EPLoadBalancingMap", EPLoadBalancingMap)
    app.component("FlashAttentionV4Diagram", FlashAttentionV4Diagram)
    app.component("MoonEPBalanceIterations", MoonEPBalanceIterations)
    app.component("MoonEPWeightLayout", MoonEPWeightLayout)
    app.component("MoonEPZeroCopyFlow", MoonEPZeroCopyFlow)
    app.component("RankLoadDistribution", RankLoadDistribution)
  },
})
