import path from "node:path"

import { katex } from "@mdit/plugin-katex-slim"
import MarkdownIt from "markdown-it"

import { listNoteFiles, readNoteFile } from "../lib/content.js"

type Heading = {
  level: number
  line: number
  text: string
}

const sourceDir = path.resolve(process.env.VUEPRESS_SOURCE_DIR ?? "notes")
const markdown = new MarkdownIt()
katex(markdown as never)
const errors: string[] = []
const files = listNoteFiles(sourceDir)

for (const relativePath of files) {
  let note
  try {
    note = readNoteFile(sourceDir, relativePath)
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error))
    continue
  }

  const displayPath = path.posix.join(
    path.relative(process.cwd(), sourceDir).split(path.sep).join("/"),
    relativePath,
  )
  if (!note.hasFrontmatter) {
    errors.push(`${displayPath}: missing frontmatter`)
    continue
  }
  if (!note.title) {
    errors.push(`${displayPath}: missing title frontmatter`)
    continue
  }

  const tokens = markdown.parse(note.body, {})
  const headings: Heading[] = []
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]
    if (token.type !== "heading_open") continue
    const inline = tokens[index + 1]
    headings.push({
      level: Number.parseInt(token.tag.slice(1), 10),
      line: note.bodyStartLine + (token.map?.[0] ?? 0),
      text: inline?.type === "inline" ? inline.content.trim() : "",
    })
  }

  const h1 = headings.filter((heading) => heading.level === 1)
  if (h1.length !== 1) {
    errors.push(`${displayPath}: expected one H1, found ${h1.length}`)
  }
  if (h1[0] && h1[0].text !== note.title) {
    errors.push(
      `${displayPath}:${h1[0].line}: H1 "${h1[0].text}" does not match title "${note.title}"`,
    )
  }
  if (headings[0]?.level !== 1) {
    errors.push(`${displayPath}: first heading must be H1`)
  }
  for (let index = 1; index < headings.length; index += 1) {
    const previous = headings[index - 1]
    const current = headings[index]
    if (current.level > previous.level + 1) {
      errors.push(
        `${displayPath}:${current.line}: heading jumps from H${previous.level} to H${current.level}`,
      )
    }
  }
}

if (errors.length) {
  console.error(errors.join("\n"))
  process.exit(1)
}
console.log(`heading check passed: ${files.length} Markdown files`)
