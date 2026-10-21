import { defineMermaidConfig } from "@vuepress/plugin-markdown-chart/client"
import { defineClientConfig } from "vuepress/client"

import "katex/dist/katex.min.css"

import EPLoadBalancingMap from "./components/diagrams/EPLoadBalancingMap.vue"
import FlashAttentionV4Diagram from "./components/diagrams/FlashAttentionV4Diagram.vue"
import CriticalPathTimeline from "./components/diagrams/CriticalPathTimeline.vue"
import ComFuseDiagram from "./components/diagrams/ComFuseDiagram.vue"
import MPKDiagram from "./components/diagrams/MPKDiagram.vue"
import MPKTwoGpuMatMul from "./components/diagrams/MPKTwoGpuMatMul.vue"
import MPKTwoGpuFlow from "./components/diagrams/MPKTwoGpuFlow.vue"
import BlackwellGemmDiagram from "./components/diagrams/BlackwellGemmDiagram.vue"
import MegaMoEDiagram from "./components/diagrams/MegaMoEDiagram.vue"
import DeepEPBufferTransform from "./components/diagrams/DeepEPBufferTransform.vue"
import DeepEPDedupExplorer from "./components/diagrams/DeepEPDedupExplorer.vue"
import DeepEPHybridPipeline from "./components/diagrams/DeepEPHybridPipeline.vue"
import DeepEPPathExplorer from "./components/diagrams/DeepEPPathExplorer.vue"
import DiTDiagram from "./components/diagrams/DiTDiagram.vue"
import MMDiTDiagram from "./components/diagrams/MMDiTDiagram.vue"
import MoonEPBalanceIterations from "./components/diagrams/MoonEPBalanceIterations.vue"
import MoonEPWeightLayout from "./components/diagrams/MoonEPWeightLayout.vue"
import MoonEPZeroCopyFlow from "./components/diagrams/MoonEPZeroCopyFlow.vue"
import RankLoadDistribution from "./components/diagrams/RankLoadDistribution.vue"
import KernelResourceCapacity from "./components/diagrams/KernelResourceCapacity.vue"
import TensorParallelMatMul from "./components/diagrams/TensorParallelMatMul.vue"
import SequenceParallelLayout from "./components/diagrams/SequenceParallelLayout.vue"
import UlyssesLayout from "./components/diagrams/UlyssesLayout.vue"
import TensorParallelFlow from "./components/diagrams/TensorParallelFlow.vue"
import ContextParallelFlow from "./components/diagrams/ContextParallelFlow.vue"
import ContextParallelBalance from "./components/diagrams/ContextParallelBalance.vue"

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
    app.component("ComFuseDiagram", ComFuseDiagram)
    app.component("MPKDiagram", MPKDiagram)
    app.component("MPKTwoGpuMatMul", MPKTwoGpuMatMul)
    app.component("MPKTwoGpuFlow", MPKTwoGpuFlow)
    app.component("BlackwellGemmDiagram", BlackwellGemmDiagram)
    app.component("MegaMoEDiagram", MegaMoEDiagram)
    app.component("DeepEPBufferTransform", DeepEPBufferTransform)
    app.component("DeepEPDedupExplorer", DeepEPDedupExplorer)
    app.component("DeepEPHybridPipeline", DeepEPHybridPipeline)
    app.component("DeepEPPathExplorer", DeepEPPathExplorer)
    app.component("EPLoadBalancingMap", EPLoadBalancingMap)
    app.component("FlashAttentionV4Diagram", FlashAttentionV4Diagram)
    app.component("DiTDiagram", DiTDiagram)
    app.component("MMDiTDiagram", MMDiTDiagram)
    app.component("MoonEPBalanceIterations", MoonEPBalanceIterations)
    app.component("MoonEPWeightLayout", MoonEPWeightLayout)
    app.component("MoonEPZeroCopyFlow", MoonEPZeroCopyFlow)
    app.component("RankLoadDistribution", RankLoadDistribution)
    app.component("KernelResourceCapacity", KernelResourceCapacity)
    app.component("TensorParallelMatMul", TensorParallelMatMul)
    app.component("SequenceParallelLayout", SequenceParallelLayout)
    app.component("UlyssesLayout", UlyssesLayout)
    app.component("TensorParallelFlow", TensorParallelFlow)
    app.component("ContextParallelFlow", ContextParallelFlow)
    app.component("ContextParallelBalance", ContextParallelBalance)
  },
})
