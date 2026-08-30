import assert from "node:assert/strict"
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import test from "node:test"

import { parseArguments } from "../../scripts/portable/cli.mjs"
import {
  inspectBundledJavaScript,
  inspectPortableSources,
} from "../../scripts/portable/javascript.mjs"
import {
  collectPortabilityIssues,
  injectStrictRuntimePolicy,
} from "../../scripts/portable/policy.mjs"

test("CLI parses strict and explicit external modes", () => {
  assert.deepEqual(parseArguments(["--page", "topic/page.md"]), {
    allowExternal: false,
    output: null,
    page: "topic/page.md",
    sourceDir: "notes",
  })
  assert.equal(
    parseArguments(["page.md", "--allow-external"]).allowExternal,
    true,
  )
})

test("author source scan catches network calls with variable URLs", async (context) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "portable-source-"))
  context.after(() => rm(root, { force: true, recursive: true }))
  const configDirectory = path.join(root, "notes", ".vuepress")
  const componentDirectory = path.join(configDirectory, "components")
  await mkdir(componentDirectory, { recursive: true })
  const sourcePath = path.join(root, "notes", "page.md")
  await writeFile(sourcePath, "---\ntitle: Page\n---\n# Page\n")
  await writeFile(
    path.join(componentDirectory, "Remote.vue"),
    '<script setup lang="ts">\nconst endpoint = "https://example.invalid/data"; fetch(endpoint)\n</script>\n',
  )

  const issues = await inspectPortableSources({
    configDirectory,
    rootDirectory: root,
    sourcePath,
  })
  assert.equal(issues.length, 1)
  assert.match(issues[0], /fetch\(动态参数\)/u)
})

test("bundle audit distinguishes provable issues from guarded dynamic calls", () => {
  const inspected = inspectBundledJavaScript(
    'fetch(target); fetch("https://example.invalid/data");',
  )
  assert.deepEqual(inspected.issues, [
    "运行时资源 fetch(https://example.invalid/data)",
  ])
  assert.deepEqual(inspected.warnings, [
    "产物包含 fetch(动态参数)，由运行期断网守卫兜底",
  ])
})

test("strict HTML policy reports external assets and injects CSP guard", () => {
  const audit = collectPortabilityIssues(
    '<html><head></head><body><img src="https://example.invalid/a.png"></body></html>',
    new Set(),
  )
  assert.deepEqual(audit.issues, [
    "外部资源 https://example.invalid/a.png",
  ])
  const guarded = injectStrictRuntimePolicy(
    "<html><head></head><body></body></html>",
  )
  assert.match(guarded, /Content-Security-Policy/u)
  assert.match(guarded, /__PORTABLE_NETWORK_BLOCKS__/u)
  assert.match(guarded, /connect-src 'none'/u)
})

test("metadata scripts and canonical links are not treated as loaded resources", () => {
  const audit = collectPortabilityIssues(
    '<html><head><link rel="canonical" href="https://example.invalid/page"><script type="application/ld+json">{"url":"https://example.invalid/page"}</script></head></html>',
    new Set(),
  )
  assert.deepEqual(audit.issues, [])
  assert.deepEqual(audit.warnings, [])
})
