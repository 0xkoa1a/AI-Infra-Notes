import fs from "node:fs";
import path from "node:path";

import type { SidebarConfig, SidebarGroup } from "vuepress";

type PageMeta = {
  title: string;
  order: number;
  link: string;
};

type Section = {
  directory: string;
  text: string;
};

const SECTIONS: Section[] = [
  { directory: "EP-load-balancing", text: "EP Load Balancing" },
  { directory: "misc", text: "Misc" },
  { directory: "cuda", text: "CUDA" },
  { directory: "model", text: "Model" },
  { directory: "parallel", text: "Parallel" },
  { directory: "flash-attn", text: "FlashAttention" },
];

function unquote(value: string): string {
  return value.replace(/^(["'])(.*)\1$/, "$2");
}

function readPage(sourceDir: string, relativePath: string): PageMeta {
  const content = fs.readFileSync(path.join(sourceDir, relativePath), "utf8");
  const frontmatter = content.match(/^---\s*\n([\s\S]*?)\n---/u)?.[1] ?? "";
  const title = frontmatter.match(/^title:\s*(.+)$/mu)?.[1]?.trim();
  const order = frontmatter.match(/^order:\s*(\d+)$/mu)?.[1];

  return {
    title: title ? unquote(title) : path.basename(relativePath, ".md"),
    order: order ? Number.parseInt(order, 10) : Number.MAX_SAFE_INTEGER,
    link: `/${relativePath.replace(/\.md$/u, ".html")}`,
  };
}

function listSection(sourceDir: string, section: Section): SidebarGroup {
  const pages = fs
    .readdirSync(path.join(sourceDir, section.directory), { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
    .map((entry) => readPage(sourceDir, `${section.directory}/${entry.name}`))
    .sort((left, right) => left.order - right.order || left.title.localeCompare(right.title, "zh-CN"));

  return {
    text: section.text,
    collapsible: true,
    children: pages.map((page) => ({ text: page.title, link: page.link })),
  };
}

export function createSidebar(sourceDir: string): SidebarConfig {
  const roots = ["XStage.md", "XStage-modeling.md"].map((file) => readPage(sourceDir, file));

  return [
    { text: "技术信息", link: "/technical-info.html" },
    {
      text: "建模基础",
      collapsible: true,
      children: roots.map((page) => ({ text: page.title, link: page.link })),
    },
    ...SECTIONS.map((section) => listSection(sourceDir, section)),
  ];
}
