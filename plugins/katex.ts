import { katex } from "@mdit/plugin-katex-slim"
import type { Plugin } from "vuepress"

export function katexOnlyPlugin(): Plugin {
  return {
    name: "katex-only",
    extendsMarkdown: (markdown) => {
      // VuePress rc.31 still exposes the preceding @types/markdown-it shape,
      // while the current KaTeX plugin uses markdown-it's bundled v15 types.
      // Runtime compatibility is covered by the build and formula smoke test.
      katex(markdown as never, {
        transformer: (content: string) =>
          content.replaceAll(/^(?<tag><[a-z]+ )/gu, "$<tag>v-pre "),
      })
    },
  }
}
