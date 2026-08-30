import { readFile, readdir } from "node:fs/promises"
import path from "node:path"

import { parse } from "acorn"
import { simple } from "acorn-walk"
import ts from "typescript"

const networkFunctions = new Set(["fetch"])
const networkConstructors = new Set([
  "EventSource",
  "SharedWorker",
  "WebSocket",
  "Worker",
])

const isInlineRuntimeUrl = (value) =>
  /^(?:data:|blob:|about:blank|#)/iu.test(value.trim())

function typescriptCallName(expression) {
  if (ts.isIdentifier(expression)) return expression.text
  if (
    ts.isPropertyAccessExpression(expression) &&
    ts.isIdentifier(expression.name)
  ) {
    return expression.name.text
  }
  return null
}

function typescriptLiteral(node) {
  return node && (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node))
    ? node.text
    : null
}

function inspectTypescript(source, fileName) {
  const issues = []
  const sourceFile = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
    fileName.endsWith(".js") ? ts.ScriptKind.JS : ts.ScriptKind.TS,
  )

  const report = (node, apiName, argument) => {
    const literal = typescriptLiteral(argument)
    if (literal !== null && isInlineRuntimeUrl(literal)) return
    const position = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile))
    const target = literal === null ? "动态参数" : literal
    issues.push(
      fileName +
        ":" +
        (position.line + 1) +
        ": 运行时网络 API " +
        apiName +
        "(" +
        target +
        ")",
    )
  }

  const visit = (node) => {
    if (ts.isCallExpression(node)) {
      const name = typescriptCallName(node.expression)
      if (name && networkFunctions.has(name)) {
        report(node, name, node.arguments[0])
      } else if (name === "sendBeacon") {
        report(node, "navigator.sendBeacon", node.arguments[0])
      } else if (node.expression.kind === ts.SyntaxKind.ImportKeyword) {
        report(node, "import", node.arguments[0])
      }
    } else if (ts.isNewExpression(node)) {
      const name = typescriptCallName(node.expression)
      if (name && networkConstructors.has(name)) {
        report(node, name, node.arguments?.[0])
      }
    }
    ts.forEachChild(node, visit)
  }
  visit(sourceFile)
  return issues
}

function extractEmbeddedScripts(source) {
  return [...source.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/giu)].map(
    (match) => match[1],
  )
}

async function sourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const result = []
  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue
    const absolute = path.join(directory, entry.name)
    if (entry.isDirectory()) result.push(...(await sourceFiles(absolute)))
    else if (
      entry.isFile() &&
      [".js", ".ts", ".vue"].includes(path.extname(entry.name).toLowerCase())
    ) {
      result.push(absolute)
    }
  }
  return result
}

export async function inspectPortableSources({
  configDirectory,
  rootDirectory,
  sourcePath,
}) {
  const files = [sourcePath, ...(await sourceFiles(configDirectory))]
  const issues = []
  for (const file of [...new Set(files)]) {
    const source = await readFile(file, "utf8")
    const relative = path.relative(rootDirectory, file).split(path.sep).join("/")
    if (file.endsWith(".vue") || file.endsWith(".md")) {
      const scripts = extractEmbeddedScripts(source)
      for (let index = 0; index < scripts.length; index += 1) {
        issues.push(
          ...inspectTypescript(
            scripts[index],
            relative + "#script-" + (index + 1),
          ),
        )
      }
    } else {
      issues.push(...inspectTypescript(source, relative))
    }
  }
  return [...new Set(issues)].sort()
}

function acornCallName(expression) {
  if (expression?.type === "Identifier") return expression.name
  if (
    (expression?.type === "MemberExpression" ||
      expression?.type === "OptionalMemberExpression") &&
    !expression.computed &&
    expression.property?.type === "Identifier"
  ) {
    return expression.property.name
  }
  return null
}

function acornLiteral(node) {
  if (node?.type === "Literal" && typeof node.value === "string") {
    return node.value
  }
  if (node?.type === "TemplateLiteral" && node.expressions.length === 0) {
    return node.quasis[0]?.value.cooked ?? ""
  }
  return null
}

export function inspectBundledJavaScript(source) {
  const issues = new Set()
  const warnings = new Set()
  const ast = parse(source, {
    allowHashBang: true,
    ecmaVersion: "latest",
    sourceType: "module",
  })

  const inspectNetworkCall = (node, name, argument) => {
    const literal = acornLiteral(argument)
    if (literal === null) {
      warnings.add("产物包含 " + name + "(动态参数)，由运行期断网守卫兜底")
    } else if (!isInlineRuntimeUrl(literal)) {
      issues.add("运行时资源 " + name + "(" + literal + ")")
    }
  }

  simple(ast, {
    CallExpression(node) {
      const name = acornCallName(node.callee)
      if (name && networkFunctions.has(name)) {
        inspectNetworkCall(node, name, node.arguments[0])
      } else if (name === "sendBeacon") {
        inspectNetworkCall(node, "navigator.sendBeacon", node.arguments[0])
      }
    },
    ImportExpression(node) {
      const literal = acornLiteral(node.source)
      issues.add(
        literal === null
          ? "仍存在动态 import(动态参数)"
          : "仍存在动态 import(" + literal + ")",
      )
    },
    NewExpression(node) {
      const name = acornCallName(node.callee)
      if (name && networkConstructors.has(name)) {
        inspectNetworkCall(node, name, node.arguments[0])
      }
    },
  })

  return {
    issues: [...issues].sort(),
    warnings: [...warnings].sort(),
  }
}
