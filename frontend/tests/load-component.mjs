// Render actual TSX panels in Node using the project's own React/TypeScript.
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";
import ts from "typescript";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nodeRequire = createRequire(import.meta.url);
const cache = new Map();

export function loadComponent(name) {
  const file = join(root, name);
  if (cache.has(file)) return cache.get(file);
  const { outputText } = ts.transpileModule(readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
    fileName: file,
  });
  const module = { exports: {} };
  // "@/x" is the project root; "./x" (lib modules import each other so) is
  // relative to the importing file.
  const local = (rel) => loadComponent(rel + (existsSync(join(root, rel + ".tsx")) ? ".tsx" : ".ts"));
  const require = (id) => id.startsWith("@/")
    ? local(id.slice(2))
    : id.startsWith("./") || id.startsWith("../")
      ? local(relative(root, join(dirname(file), id)))
      : nodeRequire(id);
  new Script(`(function(require,module,exports){${outputText}\n})`, { filename: file })
    .runInThisContext()(require, module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
