// The comps card's "Suggest peers" notes (components/CompsPanel.tsx) describe
// the peer box as the suggestion filled it, so they go once a peer set is
// applied (Update peers) or the applied peers change some other way. A small
// hooks runtime renders the card in Node, with the API mocked; child
// components are not rendered, and the test calls the handlers the card gives
// them, as a click would. Run with: npm test

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Script } from "node:vm";
import ts from "typescript";
import { loadComponent } from "./load-component.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nodeRequire = createRequire(import.meta.url);
const file = join(root, "components", "CompsPanel.tsx");
const compsModule = new Script(`(function(require,module,exports){${ts.transpileModule(
  readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
    fileName: file,
  }).outputText}\n})`, { filename: file }).runInThisContext();

globalThis.document = { activeElement: null };

// useState, useRef and useEffect, with a re-render batched into the next
// macrotask after a state change and effects run after each render.
function mount(initialProps, api) {
  const slots = [];
  let at = 0;
  let effects = [];
  let scheduled = false;
  let props = initialProps;
  const changed = (a, b) => !a || !b || a.length !== b.length || a.some((x, i) => !Object.is(x, b[i]));
  const hooks = {
    useState(init) {
      const slot = slots[at++] ??= { value: typeof init === "function" ? init() : init };
      slot.set ??= (v) => {
        const next = typeof v === "function" ? v(slot.value) : v;
        if (Object.is(next, slot.value)) return;
        slot.value = next;
        schedule();
      };
      return [slot.value, slot.set];
    },
    useRef(init) {
      return (slots[at++] ??= { current: init });
    },
    useEffect(fn, deps) {
      const k = at++;
      if (!slots[k] || changed(slots[k].deps, deps)) effects.push({ k, fn, deps });
    },
  };
  const module = { exports: {} };
  compsModule((id) => {
    if (id === "react") return { __esModule: true, default: hooks, ...hooks };
    if (id === "@/lib/api") return api;
    if (id === "@/components/ui") return loadComponent("components/ui.tsx");
    if (id.startsWith("@/lib/")) return loadComponent(`lib/${id.slice("@/lib/".length)}.ts`);
    return nodeRequire(id);
  }, module, module.exports);
  const CompsPanel = module.exports.default;

  const card = { tree: null };
  function render() {
    scheduled = false;
    at = 0;
    effects = [];
    card.tree = CompsPanel(props);
    for (const { k, fn, deps } of effects) {
      slots[k]?.cleanup?.();
      const cleanup = fn();
      slots[k] = { deps, cleanup: typeof cleanup === "function" ? cleanup : undefined };
    }
  }
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    setImmediate(render);
  }
  render();

  function* elements(node) {
    if (Array.isArray(node)) {
      for (const n of node) yield* elements(n);
    } else if (node && typeof node === "object" && node.props) {
      yield node;
      yield* elements(node.props.children);
    }
  }
  // Every string in a rendered tree, joined.
  const text = (node) => {
    if (typeof node === "string") return node;
    if (Array.isArray(node)) return node.map(text).join(" ");
    return node && typeof node === "object" && node.props ? text(node.props.children) : "";
  };
  Object.assign(card, {
    text: () => text(card.tree),
    button: (label) => [...elements(card.tree)].find((e) => e.type?.name === "Button"
      && (e.props.children === label))?.props,
    async idle() {
      for (let quiet = 0, n = 0; quiet < 3 && n < 200; n++) {
        await new Promise((r) => setImmediate(r));
        quiet = scheduled ? 0 : quiet + 1;
      }
    },
    // The page re-renders the card with new props (a recompute landed).
    async rerender(next) {
      props = { ...props, ...next };
      render();
      await card.idle();
    },
  });
  return card;
}

const NOTE = "No peers discovered: Yahoo's Consumer Electronics list has no name within "
  + "0.02-50x the target's size.";
const report = (peers) => ({
  summary: { currency: "USD", ticker: "AAPL" }, current_price: 200,
  company: { ticker: "AAPL" }, warnings: [], comps: null,
  assumptions_used: { peers, comps_basis: "fy" },
});

function setup() {
  const calls = [];
  const api = { __esModule: true,
    fetchPeerSuggestions: async () => ({ ticker: "AAPL", peers: [], notes: [NOTE] }) };
  const card = mount({ report: report(null), assumptions: {}, recomputing: false,
    setAssumptions: (a) => calls.push(["set", a]), onRecompute: (a) => calls.push(["recompute", a]) },
  api);
  return { card, calls };
}

test("Update peers clears the suggestion's notes", async () => {
  const { card, calls } = setup();
  card.button("Suggest peers").onClick();
  await card.idle();
  assert.match(card.text(), /No peers suggested\./);
  assert.ok(card.text().includes(NOTE));
  card.button("Update peers").onClick();
  await card.idle();
  assert.doesNotMatch(card.text(), /No peers suggested|No peers discovered/);
  assert.deepEqual(calls.map(([k]) => k), ["set", "recompute"]);
});

test("the notes go when the applied peers change some other way", async () => {
  const { card } = setup();
  card.button("Suggest peers").onClick();
  await card.idle();
  assert.match(card.text(), /No peers suggested\./);
  // A report for the same peer set keeps them.
  await card.rerender({ report: report(null) });
  assert.match(card.text(), /No peers suggested\./);
  // FMP's peers filled in on load (or a reset, or restored research).
  await card.rerender({ report: report("MSFT,GOOGL") });
  assert.doesNotMatch(card.text(), /No peers suggested|No peers discovered/);
});
