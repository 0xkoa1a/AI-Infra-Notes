import path from "node:path"

import { defineConfig, devices } from "@playwright/test"

export default defineConfig({
  forbidOnly: Boolean(process.env.CI),
  outputDir: path.join(
    process.env.TMPDIR ?? "/tmp",
    "ai-infra-notes-playwright",
  ),
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"], channel: "chromium" },
    },
  ],
  reporter: "line",
  retries: process.env.CI ? 1 : 0,
  testDir: "./tests",
  timeout: 120_000,
  use: {
    headless: true,
    trace: "retain-on-failure",
  },
})
