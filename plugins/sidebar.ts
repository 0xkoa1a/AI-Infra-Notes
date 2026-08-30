import path from "node:path";

import type {
  SidebarGroupOptions,
  SidebarOptions,
} from "@vuepress/theme-default";

import { listNoteFiles, readNoteFile } from "../lib/content.js";

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
  { directory: "X-Stage", text: "X-Stage" },
  { directory: "EP-load-balancing", text: "EP Load Balancing" },
  { directory: "misc", text: "Misc" },
  { directory: "cuda", text: "CUDA" },
  { directory: "model", text: "Model" },
  { directory: "parallel", text: "Parallel" },
  { directory: "flash-attn", text: "FlashAttention" },
  { directory: "MoE-Overlap", text: "MoE Overlap" },
];

function readPage(sourceDir: string, relativePath: string): PageMeta {
  const note = readNoteFile(sourceDir, relativePath);

  return {
    title: note.title ?? path.basename(relativePath, ".md"),
    order: note.order,
    link: `/${relativePath.replace(/\.md$/u, ".html")}`,
  };
}

function listSection(sourceDir: string, section: Section): SidebarGroupOptions {
  const sectionPages = listNoteFiles(sourceDir)
    .filter((relativePath) => path.posix.dirname(relativePath) === section.directory)
    .map((relativePath) => readPage(sourceDir, relativePath))
    .sort((left, right) => left.order - right.order || left.title.localeCompare(right.title, "zh-CN"));

  return {
    text: section.text,
    collapsible: true,
    children: sectionPages.map((page) => ({ text: page.title, link: page.link })),
  };
}

export function createSidebar(sourceDir: string): SidebarOptions {
  return SECTIONS.map((section) => listSection(sourceDir, section));
}
