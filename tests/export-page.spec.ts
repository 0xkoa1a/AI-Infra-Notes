import { execFile } from "node:child_process"
import { readFile, rm, stat } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"
import { promisify } from "node:util"

import { expect, test } from "@playwright/test"

const execute = promisify(execFile)
const repositoryRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
)
const smokeDirectory = path.join(repositoryRoot, "_exports", ".smoke")
const richOutput = path.join(smokeDirectory, "portable.html")
const plainOutput = path.join(smokeDirectory, "plain.html")
const componentOutput = path.join(smokeDirectory, "component.html")

const exportPage = async (arguments_: string[]): Promise<void> => {
  await execute(
    process.execPath,
    [path.join(repositoryRoot, "scripts/export-page.mjs"), ...arguments_],
    {
      cwd: repositoryRoot,
      env: process.env,
      maxBuffer: 10 * 1024 * 1024,
    },
  )
}

test.beforeAll(async () => {
  await exportPage([
    "--source-dir",
    "tests/fixtures/notes",
    "--page",
    "portable.md",
    "--output",
    "_exports/.smoke/portable.html",
  ])
  await exportPage([
    "--page",
    "technical-info.md",
    "--output",
    "_exports/.smoke/plain.html",
  ])
  await exportPage([
    "--page",
    "parallel/DeepEP.md",
    "--output",
    "_exports/.smoke/component.html",
  ])
})

test.afterAll(async () => {
  await rm(smokeDirectory, { force: true, recursive: true })
})

test("strict export hydrates from file URL without network", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 })
  const externalRequests: string[] = []
  const consoleErrors: string[] = []
  page.on("request", (request) => {
    if (/^https?:/iu.test(request.url())) externalRequests.push(request.url())
  })
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text())
  })

  await page.goto(pathToFileURL(richOutput).href, { waitUntil: "load" })
  await expect(page).toHaveTitle(/Portable Export Fixture/u)
  await expect(page.locator("h1")).toHaveText("Portable Export Fixture")
  await expect(page.locator(".katex")).toBeVisible()
  await expect(page.locator(".mermaid-content svg")).toBeVisible({
    timeout: 30_000,
  })
  await expect(page.locator(".echarts-container canvas")).toBeVisible({
    timeout: 30_000,
  })

  const imageLoaded = await page
    .getByRole("img", { name: "Portable page fixture" })
    .evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth > 0)
  expect(imageLoaded).toBe(true)

  const counter = page.locator("#portable-counter")
  await expect(counter).toHaveText(/Count: 0/u)
  await counter.click()
  await expect(counter).toHaveText(/Count: 1/u)

  await expect(page.locator("#toc")).toBeVisible()
  const formulaLink = page
    .getByLabel("本页目录")
    .getByRole("link", { name: "Formula", exact: true })
  await formulaLink.click()
  await expect(formulaLink.locator("..")).toHaveClass(/active/u)
  await expect(page.locator("#formula")).toBeInViewport()

  await expect(page.locator(".vp-navbar")).toHaveCount(0)
  await expect(page.locator(".vp-sidebar")).toHaveCount(0)
  await expect(page.locator(".vp-page-nav")).toHaveCount(0)
  await expect(page.locator(".vp-page-meta")).toHaveCount(0)
  expect(externalRequests).toEqual([])
  expect(consoleErrors).toEqual([])
  expect(
    await page.evaluate(
      () =>
        (window as Window & { __PORTABLE_NETWORK_BLOCKS__?: string[] })
          .__PORTABLE_NETWORK_BLOCKS__ ?? [],
    ),
  ).toEqual([])
})

test("portable layout remains centered and overflow-free on mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(pathToFileURL(richOutput).href, { waitUntil: "load" })
  await expect(page.locator("h1")).toBeVisible()
  const geometry = await page.evaluate(() => ({
    bodyWidth: document.body.scrollWidth,
    viewportWidth: window.innerWidth,
  }))
  expect(geometry.bodyWidth).toBeLessThanOrEqual(geometry.viewportWidth)
})

test("unused chart runtimes stay out of plain exports", async () => {
  const richBytes = (await stat(richOutput)).size
  const plainBytes = (await stat(plainOutput)).size
  expect(plainBytes).toBeLessThan(richBytes)
})

test("registered knowledge diagrams hydrate in a real exported note", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 1000 })
  const externalRequests: string[] = []
  page.on("request", (request) => {
    if (/^https?:/iu.test(request.url())) externalRequests.push(request.url())
  })
  await page.goto(pathToFileURL(componentOutput).href, { waitUntil: "load" })
  const diagram = page.locator(".deep-path")
  await expect(diagram).toBeVisible()
  const lowLatency = diagram.getByRole("button", { name: "Low-Latency" })
  await lowLatency.click()
  await expect(lowLatency).toHaveAttribute("aria-pressed", "true")
  await diagram.getByRole("button", { name: "2 接收空间" }).click()
  await expect(
    diagram.getByText("Expert × Source Rank × T_max", { exact: true }),
  ).toBeVisible()
  await expect(page.locator("#toc")).toBeVisible()
  await expect(page.locator(".vp-navbar, .vp-sidebar, .vp-page-nav, .vp-page-meta")).toHaveCount(0)
  expect(externalRequests).toEqual([])
})

test("variable network calls require explicit external mode", async () => {
  const arguments_ = [
    "--source-dir",
    "tests/fixtures/notes",
    "--page",
    "remote.md",
    "--output",
    "_exports/.smoke/remote.html",
  ]
  await expect(exportPage(arguments_)).rejects.toThrow(
    /作者源码包含运行时网络依赖/u,
  )
  await exportPage([...arguments_, "--allow-external"])
  const html = await readFile(
    path.join(smokeDirectory, "remote.html"),
    "utf8",
  )
  expect(html).not.toContain("Content-Security-Policy")
  expect(html).toContain('name="portable-source"')
})
