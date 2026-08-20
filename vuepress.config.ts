import path from "node:path";
import { fileURLToPath } from "node:url";

import { viteBundler } from "@vuepress/bundler-vite";
import { activeHeaderLinksPlugin } from "@vuepress/plugin-active-header-links";
import { markdownChartPlugin } from "@vuepress/plugin-markdown-chart";
import { markdownExtPlugin } from "@vuepress/plugin-markdown-ext";
import { markdownMathPlugin } from "@vuepress/plugin-markdown-math";
import { slimsearchPlugin } from "@vuepress/plugin-slimsearch";
import { defaultTheme } from "@vuepress/theme-default";
import { defineUserConfig } from "vuepress";

import { createSidebar } from "./plugins/sidebar.js";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const sourceDir = path.join(rootDir, "notes");
const base = process.env.GITHUB_ACTIONS === "true" ? "/AI-Infra-Notes/" : "/";

// SlimSearch rc.131 currently traverses text below <pre> even though code is
// documented as excluded. Give its synchronous index pass a sanitized copy,
// then restore the rendered page before VuePress writes the HTML output.
const filterSearchPage = (page: { contentRendered: string }): boolean => {
  const rendered = page.contentRendered;
  page.contentRendered = rendered.replace(/<pre\b[^>]*>[\s\S]*?<\/pre>/gi, "");
  queueMicrotask(() => {
    page.contentRendered = rendered;
  });
  return true;
};

export default defineUserConfig({
  base,
  lang: "zh-CN",
  title: "AI Infra 知识库",
  description: "AI 推理与训练基础设施学习笔记",
  dest: path.join(rootDir, "_site"),
  pagePatterns: ["**/*.md", "!**/README.md"],
  alias: {
    "@theme/VPPage.vue": path.join(rootDir, "notes/.vuepress/components/VPPage.vue"),
  },
  bundler: viteBundler(),
  theme: defaultTheme({
    navbar: [{ text: "技术信息", link: "/technical-info.html" }],
    sidebar: createSidebar(sourceDir),
    sidebarDepth: 0,
    contributors: false,
    lastUpdatedText: "最近更新",
    editLink: false,
    themePlugins: {
      activeHeaderLinks: false,
      backToTop: true,
      mediumZoom: true,
    },
  }),
  plugins: [
    activeHeaderLinksPlugin({
      headerLinkSelector: "a.vp-sidebar-item, a.vp-toc-link",
    }),
    markdownMathPlugin({ type: "katex" }),
    markdownExtPlugin({ tasklist: true }),
    markdownChartPlugin({
      echarts: true,
      DANGEROUS_ALLOW_SCRIPT_EXECUTION: true,
      DANGEROUS_SCRIPT_EXECUTION_ALLOWLIST: ["model/SwiGLU.md"],
    }),
    slimsearchPlugin({
      indexContent: true,
      filter: filterSearchPage,
      locales: {
        "/": { placeholder: "搜索文档" },
      },
    }),
  ],
});
