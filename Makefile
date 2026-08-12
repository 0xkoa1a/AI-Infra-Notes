.PHONY: help check render preview clean

QUARTO      := quarto
SITE_OUTPUT := $(CURDIR)/_site
SITE_SOURCE := $(CURDIR)/notes

help:
	@echo ""
	@echo "  AI 推理 Infra 学习笔记 —— 可用命令"
	@echo ""
	@echo "  make check    检查 Quarto 环境"
	@echo "  make render   渲染完整站点到 _site/"
	@echo "  make preview  启动本地实时预览"
	@echo "  make clean    清理站点输出与 Quarto 缓存"
	@echo ""

check:
	@command -v $(QUARTO) >/dev/null 2>&1 || { echo "quarto 未安装 -> https://quarto.org/docs/download/"; exit 1; }
	@$(QUARTO) --version | sed 's/^/quarto /'

render: check
	@test "$(SITE_OUTPUT)" = "$(CURDIR)/_site"
	rm -rf "$(SITE_OUTPUT)"
	cd "$(SITE_SOURCE)" && $(QUARTO) render
	@test -f "$(SITE_OUTPUT)/index.html"
	@echo ""
	@echo "已生成 _site/index.html"
	@echo "打开: open _site/index.html"

preview: check
	cd "$(SITE_SOURCE)" && $(QUARTO) preview

clean:
	@test "$(SITE_OUTPUT)" = "$(CURDIR)/_site"
	rm -rf "$(SITE_OUTPUT)" "$(SITE_SOURCE)/.quarto"
	@echo "已清理 _site/ 与 notes/.quarto/"
