export function fail(message) {
  throw new Error(message)
}

export function parseArguments(argv) {
  const result = {
    allowExternal: false,
    output: null,
    page: null,
    sourceDir: "notes",
  }

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === "--") {
      continue
    } else if (argument === "--allow-external") {
      result.allowExternal = true
    } else if (argument === "--page") {
      result.page = argv[++index] ?? fail("--page 需要 Markdown 路径")
    } else if (argument === "--output") {
      result.output = argv[++index] ?? fail("--output 需要 HTML 路径")
    } else if (argument === "--source-dir") {
      result.sourceDir = argv[++index] ?? fail("--source-dir 需要目录路径")
    } else if (!argument.startsWith("-") && !result.page) {
      result.page = argument
    } else {
      fail("未知参数：" + argument)
    }
  }

  if (!result.page) {
    fail("缺少页面路径，例如：pnpm export:page -- parallel/DeepEP.md")
  }
  return result
}
