.PHONY: help check install render preview clean

NODE           := node
PNPM           := pnpm
SITE_OUTPUT    := $(CURDIR)/_site
VUEPRESS_TMP   := $(CURDIR)/notes/.vuepress/.temp
VUEPRESS_CACHE := $(CURDIR)/notes/.vuepress/.cache

help:
	@echo ""
	@echo "  AI 推理 Infra 学习笔记 —— 可用命令"
	@echo ""
	@echo "  make check    检查 Node、pnpm 与 VuePress 环境"
	@echo "  make install  安装锁定版本的前端依赖"
	@echo "  make render   完整构建静态站点到 _site/"
	@echo "  make preview  启动本地增量预览"
	@echo "  make clean    清理站点输出与 VuePress 缓存"
	@echo ""

check:
	@command -v $(NODE) >/dev/null 2>&1 || { echo "Node.js 未安装"; exit 1; }
	@command -v $(PNPM) >/dev/null 2>&1 || { echo "pnpm 未安装"; exit 1; }
	@$(NODE) -e 'const major = Number(process.versions.node.split(".")[0]); if (major !== 22) { console.error(`需要 Node 22，当前为 $${process.versions.node}`); process.exit(1); }'
	@$(NODE) --version | sed 's/^/node /'
	@$(PNPM) --version | sed 's/^/pnpm /'
	@test -x "$(CURDIR)/node_modules/.bin/vuepress" || { echo "依赖未安装，请先运行 make install"; exit 1; }

install:
	$(PNPM) install --frozen-lockfile

render: check
	@test "$(SITE_OUTPUT)" = "$(CURDIR)/_site"
	rm -rf "$(SITE_OUTPUT)"
	$(PNPM) run docs:build
	@test -f "$(SITE_OUTPUT)/index.html"
	@echo ""
	@echo "已生成 _site/index.html"

preview: check
	$(PNPM) run docs:dev

clean:
	@test "$(SITE_OUTPUT)" = "$(CURDIR)/_site"
	@test "$(VUEPRESS_TMP)" = "$(CURDIR)/notes/.vuepress/.temp"
	@test "$(VUEPRESS_CACHE)" = "$(CURDIR)/notes/.vuepress/.cache"
	rm -rf "$(SITE_OUTPUT)" "$(VUEPRESS_TMP)" "$(VUEPRESS_CACHE)"
	@echo "已清理 _site/ 与 VuePress 缓存"
