# AI 推理 Infra 学习笔记

本仓库存放 AI 推理与训练基础设施学习笔记·，并使用 VuePress 2、Vue 3、Vite 与 default theme 生成本地 HTML 知识库。技术栈与页面骨架参考 [MyCppTutorial](https://github.com/guyutongxue/MyCppTutorial)。

## 环境与使用

项目锁定 Node.js `22.18.0` 和 pnpm `11.19.0`。进入仓库后安装依赖并启动本地站点：

```bash
fnm use
corepack enable
make install
make preview
```

常用命令：

```bash
make check    # 检查 Node、pnpm 与 VuePress
make install  # 按 pnpm-lock.yaml 安装依赖
make render   # 完整生成 _site/index.html
make export PAGE=parallel/DeepEP.md  # 导出自包含单页 HTML
make test     # 运行类型、单元、构建与单页导出回归
make preview  # 启动带热更新的增量预览
make clean    # 清理 _site/ 与 VuePress 缓存
```

## 新增笔记

发布页面统一放在 `notes/` 下并使用 `.md` 扩展名。每篇笔记提供标题和排序元数据，并在正文开头保留同名一级标题：

```markdown
---
title: "笔记标题"
order: 1
---

# 笔记标题

正文从这里开始。
```

专题目录中的 Markdown 会按 `order` 自动加入左侧导航；新增一级专题目录时，在 `plugins/sidebar.ts` 的 `SECTIONS` 中登记一次。图片放在所属专题旁的 `images/` 目录，并使用相对路径引用。

根目录的 `README.md`、`TODO.md`、源码仓、PDF/PPTX 和附件不属于站点输入。生成的 `_site/`、VuePress 缓存和 `node_modules/` 不进入 Git。

## 导出单篇笔记

使用 `make export PAGE=<相对 notes/ 的 Markdown 路径>` 生成 `_exports/<同路径>.html`。导出结果是可直接通过 `file://` 打开的自包含文件：保留正文、图片、公式、Vue 组件、页面样式和本页目录，移除左侧导航、搜索、上一篇/下一篇和更新时间。

严格模式会内联本地资源，并拒绝可识别的运行时网络依赖；确实需要保留外部请求时，可显式执行：

```bash
make export PAGE=path/to/note.md ALLOW_EXTERNAL=1
```

此时导出文件不再承诺完全离线。严格模式还会加入 CSP 与运行时断网守卫，但它是保守的工程审计，不等同于对任意 JavaScript 的数学证明。导出的 HTML 带有来源路径、提交和生成时间元数据，默认输出目录不进入 Git。

## 写作与呈现

正文知识优先使用 Markdown；明确的数据流、时序和简单拓扑使用 Mermaid；只有当颜色、分组、复杂布局或交互本身参与解释时才使用 Vue 组件。文章内不维护用于排版的大段 `<style>` 或 Raw HTML。

完整的内容保护、组件组织与验证规则见 [AGENTS.md](./AGENTS.md)。
