import fs from "node:fs"
import path from "node:path"

import { parseDocument } from "yaml"

export const notePagePatterns = [
  "**/*.md",
  "!**/[Rr][Ee][Aa][Dd][Mm][Ee].md",
  "!**/.*",
  "!**/.*/**",
] as const

export type NoteDocument = {
  absolutePath: string
  body: string
  bodyStartLine: number
  data: Record<string, unknown>
  hasFrontmatter: boolean
  relativePath: string
  source: string
}

export type NotePage = NoteDocument & {
  order: number
  title: string | null
}

const isIgnoredEntry = (name: string): boolean =>
  name.startsWith(".") || name.toLowerCase() === "readme.md"

const toPosix = (value: string): string => value.split(path.sep).join("/")

export function listNoteFiles(sourceDir: string): string[] {
  const visit = (directory: string): string[] =>
    fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
      if (isIgnoredEntry(entry.name)) return []

      const absolutePath = path.join(directory, entry.name)
      if (entry.isDirectory()) return visit(absolutePath)
      if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== ".md") {
        return []
      }
      return [toPosix(path.relative(sourceDir, absolutePath))]
    })

  return visit(sourceDir).sort((left, right) =>
    left.localeCompare(right, "zh-CN"),
  )
}

export function parseNoteDocument(
  source: string,
  absolutePath: string,
  relativePath = toPosix(path.basename(absolutePath)),
): NoteDocument {
  const match = source.match(
    /^---[^\S\r\n]*\r?\n([\s\S]*?)\r?\n---[^\S\r\n]*(?:\r?\n|$)/u,
  )
  if (!match) {
    return {
      absolutePath,
      body: source,
      bodyStartLine: 1,
      data: {},
      hasFrontmatter: false,
      relativePath,
      source,
    }
  }

  const document = parseDocument(match[1], { prettyErrors: true })
  if (document.errors.length) {
    throw new Error(
      `${relativePath}: invalid frontmatter: ${document.errors
        .map((error) => error.message)
        .join("; ")}`,
    )
  }
  const parsed = document.toJS()
  if (parsed !== null && (typeof parsed !== "object" || Array.isArray(parsed))) {
    throw new Error(`${relativePath}: frontmatter must be a mapping`)
  }

  return {
    absolutePath,
    body: source.slice(match[0].length),
    bodyStartLine: (match[0].match(/\n/gu)?.length ?? 0) + 1,
    data: (parsed ?? {}) as Record<string, unknown>,
    hasFrontmatter: true,
    relativePath,
    source,
  }
}

export function readNoteFile(
  sourceDir: string,
  relativePath: string,
): NotePage {
  const absolutePath = path.join(sourceDir, relativePath)
  const document = parseNoteDocument(
    fs.readFileSync(absolutePath, "utf8"),
    absolutePath,
    relativePath,
  )
  const rawTitle = document.data.title
  const rawOrder = document.data.order
  const title =
    typeof rawTitle === "string" && rawTitle.trim() ? rawTitle.trim() : null
  const order =
    typeof rawOrder === "number" && Number.isFinite(rawOrder)
      ? rawOrder
      : Number.MAX_SAFE_INTEGER

  return { ...document, order, title }
}
