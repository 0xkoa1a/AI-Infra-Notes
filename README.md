# AI 推理 Infra 学习笔记

本仓库存放我在 AI 推理 Infra 领域实习期间的学习笔记，并使用 Quarto 生成本地 HTML 知识库。

## 使用

前置依赖：[Quarto](https://quarto.org/docs/download/)。站点不需要 Python、Jupyter、Node 或数据分析依赖。

```bash
make check    # 检查 Quarto 环境
make render   # 生成 _site/index.html
make preview  # 启动本地实时预览
make clean    # 清理站点输出与 Quarto 缓存
```

## 新增笔记

发布到网站的笔记统一放在 `notes/` 下并使用 `.qmd` 扩展名。目录与页面会被 Quarto 自动加入左侧导航，不需要修改其他配置。

每篇笔记至少提供标题元数据：

```markdown
---
title: "笔记标题"
---

正文从这里开始。
```

图片放在所属专题旁的 `images/` 目录中，并从 QMD 使用相对路径引用，例如：

```markdown
![图片说明](images/example.png)
```

根目录的 `README.md`、`TODO.md`、源码仓和附件不属于站点输入。生成的 `_site/` 仅供本地浏览，不进入 Git。
