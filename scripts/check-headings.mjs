import fs from "node:fs";
import path from "node:path";

const root = path.resolve("notes");

function filesIn(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === ".vuepress" ? [] : filesIn(file);
    return entry.isFile() && entry.name.endsWith(".md") ? [file] : [];
  });
}

function parseTitle(frontmatter, file) {
  const match = frontmatter.match(/^title:\s*(?:"([^"]*)"|'([^']*)'|(.*))\s*$/mu);
  const title = match?.[1] ?? match?.[2] ?? match?.[3]?.trim();
  if (!title) throw new Error(`${file}: missing title frontmatter`);
  return title;
}

const errors = [];
for (const file of filesIn(root).sort()) {
  const relative = path.relative(process.cwd(), file);
  const source = fs.readFileSync(file, "utf8");
  const frontmatterMatch = source.match(/^---\s*\n([\s\S]*?)\n---\s*\n?/u);
  if (!frontmatterMatch) {
    errors.push(`${relative}: missing frontmatter`);
    continue;
  }

  const title = parseTitle(frontmatterMatch[1], relative);
  const body = source.slice(frontmatterMatch[0].length).split("\n");
  const headings = [];
  let fenced = false;
  for (let index = 0; index < body.length; index += 1) {
    const line = body[index];
    if (/^\s*(```|~~~)/u.test(line)) {
      fenced = !fenced;
      continue;
    }
    if (!fenced) {
      const match = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/u);
      if (match) headings.push({ level: match[1].length, text: match[2], line: index + 1 });
    }
  }

  const h1 = headings.filter((heading) => heading.level === 1);
  if (h1.length !== 1) errors.push(`${relative}: expected one H1, found ${h1.length}`);
  if (h1[0] && h1[0].text !== title) errors.push(`${relative}: H1 "${h1[0].text}" does not match title "${title}"`);
  if (headings[0]?.level !== 1) errors.push(`${relative}: first heading must be H1`);
  for (let index = 1; index < headings.length; index += 1) {
    if (headings[index].level > headings[index - 1].level + 1) {
      errors.push(`${relative}:${headings[index].line}: heading jumps from H${headings[index - 1].level} to H${headings[index].level}`);
    }
  }
}

if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log(`heading check passed: ${filesIn(root).length} Markdown files`);
