---
title: "技术信息"
---

# 技术信息

本知识库使用 VuePress 2 生成静态网站，并由 Vue 3 与 Vite 提供开发和构建能力。页面外观与交互直接使用 VuePress default theme。

## 组件

- VuePress 2 与 default theme：站点结构、导航、侧栏、暗色模式、页内锚点与前后页导航。
- VuePress Local Search：在浏览器中提供本地全文搜索。
- `vuepress-plugin-md-enhance`：KaTeX 数学公式、任务列表和流程图。
- `vuepress-plugin-copy-code2`：代码块复制按钮。
- default theme 内置插件：活动标题链接、外链图标、Git 信息、图片缩放、页面加载进度条和返回顶部。

## 本地使用

```bash
make install
make preview
make render
```

`make preview` 提供增量热更新；`make render` 将完整静态站点输出到根目录的 `_site/`。
