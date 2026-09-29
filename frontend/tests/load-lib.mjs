// Load frontend/lib/*.ts in plain Node for the unit tests (node --test).
//
// The lib modules are framework-free TypeScript with relative imports, so the
// project's own TypeScript compiler transpiles them to CommonJS in a temporary
// directory, from where they are required. No test framework or extra
// dependency is needed.

import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const LIB = join(dirname(fileURLToPath(import.meta.url)), "..", "lib");

let outDir = null;

function build() {
  outDir = mkdtempSync(join(tmpdir(), "frontend-lib-"));
  process.on("exit", () => rmSync(outDir, { recursive: true, force: true }));
  for (const name of readdirSync(LIB)) {
    if (!name.endsWith(".ts")) continue;
    const { outputText } = ts.transpileModule(readFileSync(join(LIB, name), "utf8"), {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2020,
        esModuleInterop: true,
      },
      fileName: name,
    });
    writeFileSync(join(outDir, name.replace(/\.ts$/, ".js")), outputText);
  }
}

// The compiled module lib/<name>.ts.
export function loadLib(name) {
  if (outDir === null) build();
  return createRequire(join(outDir, "index.js"))(`./${name}.js`);
}
