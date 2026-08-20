# Repository instructions

## Preserve content semantics

- Treat notes as source-faithful technical documents. Preserve unrelated prose, formulas, images, examples, and conclusions when changing presentation.
- Inspect the current file and dirty worktree before editing. Do not reset or overwrite unrelated user changes.
- Keep the distinction between current-step signals and historical or predicted signals explicit.

## Choose the presentation layer

- Use Markdown for prose, definitions, lists, quotations, and precise comparison tables.
- Use Mermaid when nodes, arrows, sequence, state transitions, or simple topology are the information.
- Use Vue components when grouping, semantic color, multi-region layout, interaction, or chart-library integration is part of the explanation.
- Do not add article-local `<style>` blocks or complex Raw HTML for layout. Move presentation code into `notes/.vuepress/components/` and keep the Markdown invocation small.
- Keep `text` fences only when monospaced spatial layout is itself meaningful. Do not mechanically convert source code, pseudocode, or useful ASCII diagrams.
- Prefer the site-level Mermaid theme and shared `--infra-*` semantic tokens over per-article decorative styling.

## Diagram components

- Put article-specific knowledge diagrams under `notes/.vuepress/components/diagrams/`.
- Reuse shared tokens before creating generic component primitives. Add primitives only after a repeated interface is demonstrated by multiple diagrams.
- Diagram markup must be semantic and accessible. Do not create headings that accidentally enter the article TOC.
- Keep exact comparisons and citations in Markdown even when a Vue component provides the visual intuition.
- Use a timeline figure for CUDA critical-path analysis when execution overlap is part of the explanation. Divide lanes vertically by the actual hardware resource or execution engine (for example, CUDA Core, Tensor Core, TMA, or communication), and place operations horizontally by time. Lane names, count, ordering, and stage labels must come from the analyzed implementation; do not promote one article's resource layout into a repository-wide fixed template. When operations can overlap but contend for a shared resource, show the requested conservative ordering and annotate the possible overlap plus the contention. `<CriticalPathTimeline />` is the current reusable EP implementation of this pattern; its labels and captions are configurable, while future resource layouts may require a dedicated variant.

## Validation

- Run `make check`, `pnpm run docs:build`, and `git diff --check` for site changes.
- Inspect the rendered target page in the browser, including light and dark modes when semantic colors change.
- After validation, run `make clean` so generated site output and VuePress caches are not left behind.
