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
- Keep all Ultra-Scale Playbook Chinese chapter translations in `notes/ultrascale-playbook/`, with local figures in its `images/` directory and chapter-order frontmatter. Preserve sentence-by-sentence correspondence with the source; do not add explanations or translator notes. Keep the English source archive in `references/ultrascale-playbook/`.

## Diagram components

- Put article-specific knowledge diagrams under `notes/.vuepress/components/diagrams/`.
- Reuse shared tokens before creating generic component primitives. Add primitives only after a repeated interface is demonstrated by multiple diagrams.
- Diagram markup must be semantic and accessible. Do not create headings that accidentally enter the article TOC.
- Keep exact comparisons and citations in Markdown even when a Vue component provides the visual intuition.
- Use a timeline figure for CUDA critical-path analysis when execution overlap is part of the explanation. Divide lanes vertically by the actual hardware resource or execution engine (for example, CUDA Core, Tensor Core, TMA, or communication), and place operations horizontally by time. Lane names, count, ordering, and stage labels must come from the analyzed implementation; do not promote one article's resource layout into a repository-wide fixed template. When operations can overlap but contend for a shared resource, show the requested conservative ordering and annotate the possible overlap plus the contention. `<CriticalPathTimeline />` is the current reusable EP implementation of this pattern; its labels and captions are configurable, while future resource layouts may require a dedicated variant.

## Git workflow

- After completing user-requested changes, commit and push the task changes automatically. The user has authorized this workflow for this repository.
- Stage only the task changes and their required dependencies. Preserve unrelated uncommitted changes, including when a file contains changes from several tasks.
- Use a normal push; do not force-push or rewrite shared history.

## Validation and live preview

- The user runs `make preview` while reading and editing. Preserve that process and its generated output and caches.
- Do not run builds, tests, browser checks, `make check`, `git diff --check`, or cache cleanup unless the user explicitly requests them.
- For document changes, reread the changed prose and diagram source for logic, language, citations, and consistency. State that rendered-page validation was not performed.
