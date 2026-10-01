// Render actual TSX panels in Node using the project's own React/TypeScript.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
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
  const require = (id) => id.startsWith("@/")
    ? loadComponent(id.slice(2) + (id.startsWith("@/components/") ? ".tsx" : ".ts"))
    : nodeRequire(id);
  new Script(`(function(require,module,exports){${outputText}\n})`, { filename: file })
    .runInThisContext()(require, module, module.exports);
  cache.set(file, module.exports);
  return module.exports;
}
