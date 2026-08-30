import assert from "node:assert/strict"
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import test from "node:test"

import {
  listNoteFiles,
  parseNoteDocument,
  readNoteFile,
} from "../../lib/content.js"

test("note discovery matches VuePress page exclusions", async (context) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "notes-content-"))
  context.after(() => rm(root, { force: true, recursive: true }))
  await mkdir(path.join(root, "topic", "nested"), { recursive: true })
  await mkdir(path.join(root, ".drafts"), { recursive: true })
  await writeFile(path.join(root, "index.md"), "---\ntitle: Home\n---\n# Home\n")
  await writeFile(path.join(root, "README.md"), "not a page")
  await writeFile(path.join(root, ".hidden.md"), "not a page")
  await writeFile(
    path.join(root, "topic", "README.md"),
    "also not a page",
  )
  await writeFile(
    path.join(root, ".drafts", "secret.md"),
    "---\ntitle: Secret\n---\n# Secret\n",
  )
  await writeFile(
    path.join(root, "topic", "nested", "page.md"),
    "---\ntitle: Page\n---\n# Page\n",
  )

  assert.deepEqual(listNoteFiles(root), ["index.md", "topic/nested/page.md"])
})

test("frontmatter parser handles YAML and preserves body line numbers", () => {
  const source =
    "---\ntitle: \"A: precise title\"\norder: 3\ntags:\n  - test\n---\n# A: precise title\n"
  const note = parseNoteDocument(source, "/tmp/page.md", "page.md")
  assert.equal(note.hasFrontmatter, true)
  assert.equal(note.data.title, "A: precise title")
  assert.equal(note.bodyStartLine, 7)
  assert.match(note.body, /^# A: precise title/mu)
})

test("page metadata uses typed order and title", async (context) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "notes-metadata-"))
  context.after(() => rm(root, { force: true, recursive: true }))
  await writeFile(
    path.join(root, "page.md"),
    "---\ntitle: Test page\norder: 2\n---\n# Test page\n",
  )
  const page = readNoteFile(root, "page.md")
  assert.equal(page.title, "Test page")
  assert.equal(page.order, 2)
})
