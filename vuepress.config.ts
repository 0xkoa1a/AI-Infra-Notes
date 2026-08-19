import path from "node:path";
import { fileURLToPath } from "node:url";

import { containerPlugin } from "@vuepress/plugin-container";
import { searchPlugin } from "@vuepress/plugin-search";
import { copyCodePlugin } from "vuepress-plugin-copy-code2";
import { mdEnhancePlugin } from "vuepress-plugin-md-enhance";
import { defaultTheme, type UserConfig, viteBundler } from "vuepress";

import { createSidebar } from "./plugins/sidebar.js";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const sourceDir = path.join(rootDir, "notes");
const base = process.env.GITHUB_ACTIONS === "true" ? "/AI-Infra-Notes/" : "/";

const config: UserConfig = {
  base,
  lang: "zh-CN",
  title: "AI Infra 知识库",
  description: "AI 推理与训练基础设施学习笔记",
  dest: path.join(rootDir, "_site"),
  pagePatterns: ["**/*.md", "!**/README.md"],
  bundler: viteBundler(),
  theme: defaultTheme({
    navbar: [{ text: "技术信息", link: "/technical-info.html" }],
    sidebar: createSidebar(sourceDir),
    sidebarDepth: 0,
    contributors: false,
    lastUpdatedText: "最近更新",
    editLink: false,
    themePlugins: {
      backToTop: true,
      mediumZoom: true,
    },
  }),
  plugins: [
    containerPlugin({ type: "tip", locales: { "/": { defaultInfo: "提示" } } }),
    containerPlugin({ type: "warning", locales: { "/": { defaultInfo: "注意" } } }),
    containerPlugin({ type: "danger", locales: { "/": { defaultInfo: "危险" } } }),
    mdEnhancePlugin({ katex: true, tasklist: true, flowchart: true, echarts: true }),
    searchPlugin({
      locales: { "/": { placeholder: "搜索文档" } },
      maxSuggestions: 10,
    }),
    copyCodePlugin(),
  ],
};

export default config;
