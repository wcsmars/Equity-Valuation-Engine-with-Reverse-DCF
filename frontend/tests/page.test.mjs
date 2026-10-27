// The dashboard page itself (app/page.tsx), driven against a mocked API: a
// ticker load waits for earlier research saves before it switches the view,
// and a recompute, research note or AI digest that finishes meanwhile lands
// on (and is saved to) its own ticker only. A small hooks runtime renders the
// page in Node, with no browser or extra dependency. Child panels are not
// rendered; the test calls the callbacks the page hands them, as a click
// would. Run with: npm test

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
const pageFile = join(root, "app", "page.tsx");
const pageModule = new Script(`(function(require,module,exports){${ts.transpileModule(
  readFileSync(pageFile, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
    fileName: pageFile,
  }).outputText}\n})`, { filename: pageFile }).runInThisContext();

// The page's browser surroundings: the API health probe and page listeners.
globalThis.fetch = async () => ({ ok: true,
  json: async () => ({ fmp_enabled: false, anthropic_enabled: true }) });
globalThis.window = { addEventListener() {}, removeEventListener() {} };
globalThis.document = { addEventListener() {}, removeEventListener() {}, visibilityState: "visible" };

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const NAMES = { AAPL: "Apple Inc.", NOK: "Nokia Oyj", GM: "General Motors" };
const valuation = (t, assumptions = {}) => ({
  report_id: `${t}-${Math.random()}`,
  summary: { ticker: t, name: NAMES[t] ?? t, currency: "USD", current_price: 10,
    blended_target: 12, blended_upside: 0.2, recommendation: "BUY" },
  company: { market: {} },
  comps: { peers: [] },
  assumptions_used: { rf: 0.04, erp: 0.05, forecast_years: 5, terminal_growth: 0.025,
    terminal_method: "gordon", ...assumptions },
});

// The backend as lib/api.ts reaches it. Each call is logged with a deferred
// result; while `auto[kind]` is true it is answered at once, otherwise the
// test settles it. Research saves merge into `store` once they succeed, as
// the backend's store does.
function mockApi({ watchlist = [], store = {} } = {}) {
  const api = {
    auto: { research: true, valuation: true, save: true },
    store,
    calls: { research: [], valuation: [], save: [], note: [], watchlist: [] },
  };
  const call = (kind, args, answer) => {
    const d = deferred();
    api.calls[kind].push({ args, ...d });
    if (api.auto[kind]) d.resolve(answer());
    return d.promise;
  };
  let list = watchlist;
  api.module = {
    __esModule: true,
    apiUrl: (path) => path,
    errorDetail: async () => "error",
    downloadExport: async () => {},
    fetchEnrichment: async () => ({ enabled: false }),
    fetchWatchlist: async () => list,
    updateWatchlist: async (action, ticker, snapshot) => {
      api.calls.watchlist.push({ action, ticker, name: snapshot?.name });
      list = action === "remove" ? list.filter((w) => w.ticker !== ticker)
        : [...list.filter((w) => w.ticker !== ticker), { ticker, ...snapshot }];
      return list;
    },
    fetchResearchState: (t) => call("research", [t], () => structuredClone(api.store[t] ?? {})),
    fetchValuation: (t, a) => call("valuation", [t, a], () => valuation(t)),
    postResearchNote: (body) => call("note", [body], () => ({})),
    saveResearchState: (t, state) => {
      const d = deferred();
      api.calls.save.push({ ticker: t, state: structuredClone(state), ...d });
      if (api.auto.save) d.resolve();
      return d.promise.then(() => {
        api.store[t] = { ...api.store[t], ...structuredClone(state) };
      });
    },
  };
  return api;
}

// Child components are stand-ins named after the real ones (never rendered);
// components/ui stays real for its class-name helper.
const stand = (name) => Object.defineProperty(() => null, "name", { value: name });
function components(file) {
  if (file === "ui") return loadComponent("components/ui.tsx");
  return new Proxy({ __esModule: true }, {
    get: (o, k) => (k in o ? o[k] : (o[k] = stand(k === "default" ? file : String(k)))),
  });
}

// A minimal hooks runtime: useState, useRef, useCallback and useEffect, a
// re-render batched into the next macrotask after state changes (as React
// schedules one), and effects run after each render.
function mount(api) {
  const slots = [];
  let at = 0;
  let effects = [];
  let scheduled = false;
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
    useCallback(fn, deps) {
      const k = at++;
      if (!slots[k] || changed(slots[k].deps, deps)) slots[k] = { fn, deps };
      return slots[k].fn;
    },
    useEffect(fn, deps) {
      const k = at++;
      if (!slots[k] || changed(slots[k].deps, deps)) effects.push({ k, fn, deps });
    },
  };
  const module = { exports: {} };
  pageModule((id) => {
    if (id === "react") return { __esModule: true, default: hooks, ...hooks };
    if (id === "@/lib/api") return api.module;
    if (id.startsWith("@/components/")) return components(id.slice("@/components/".length));
    if (id.startsWith("@/lib/")) return loadComponent(`lib/${id.slice("@/lib/".length)}.ts`);
    return nodeRequire(id);
  }, module, module.exports);
  const Home = module.exports.default;

  const page = { tree: null };
  function render() {
    scheduled = false;
    at = 0;
    effects = [];
    page.tree = Home();
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

  // Every element of the last render, depth first.
  function* elements(node) {
    if (Array.isArray(node)) {
      for (const n of node) yield* elements(n);
    } else if (node && typeof node === "object" && node.props) {
      yield node;
      yield* elements(node.props.children);
    }
  }
  const find = (match) => [...elements(page.tree)].find(match);
  Object.assign(page, {
    find,
    // The props of the button (a plain one or ui's Button) reading `text`.
    button: (text) => find((e) => (e.type === "button" || e.type?.name === "Button")
      && e.props.children === text)?.props,
    // Lets every pending callback and render run.
    async idle() {
      for (let quiet = 0, n = 0; quiet < 3 && n < 500; n++) {
        await new Promise((r) => setImmediate(r));
        quiet = scheduled ? 0 : quiet + 1;
      }
    },
    // The props the page gave a child component, if it is rendered.
    panel: (name) => find((e) => e.type?.name === name)?.props,
    // Types a ticker and presses Analyze.
    async analyze(t) {
      find((e) => e.type === "input").props.onChange({ target: { value: t } });
      await page.idle();
      find((e) => e.type === "form").props.onSubmit({ preventDefault() {} });
    },
    async openTab(tab) {
      page.panel("TabBar").onChange(tab);
      await page.idle();
    },
    // The company named in the header, and the error banner's text.
    shownName: () => page.panel("AIResearchPanel")?.report?.summary?.name,
    error: () => {
      const alert = page.panel("Alert");
      return alert ? [alert.children].flat().filter((c) => typeof c === "string").join("") : "";
    },
  });
  return page;
}

const digest = {
  summary: "Margins expanded on mix.",
  sentiment: "bullish",
  key_facts: [{ fact: "Q3 gross margin 46.2%", source: "Q3 FY2026 earnings call" }],
  risks: ["China demand"],
  catalysts: [],
  suggested_assumptions: [],
};
const savesFor = (api, t) => api.calls.save.filter((s) => s.ticker === t);

// Load AAPL and type into its research log, so a save is waiting (debounced).
async function aaplWithEdit(t, api) {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const page = mount(api);
  await page.analyze("AAPL");
  await page.idle();
  assert.equal(page.shownName(), "Apple Inc.");
  page.panel("AIResearchPanel").setResearchNotes("My view.");
  await page.idle();
  return page;
}

test("a recompute clicked while a switch waits for saves never lands on the next ticker", async (t) => {
  const api = mockApi({ watchlist: [{ ticker: "NOK", name: "Nokia Oyj" }] });
  const page = await aaplWithEdit(t, api);
  api.auto.save = false;
  api.auto.valuation = false;
  await page.analyze("NOK"); // waits for AAPL's save
  await page.idle();
  assert.equal(savesFor(api, "AAPL").length, 1);
  assert.equal(page.shownName(), "Apple Inc."); // still on screen and live
  await page.openTab("Valuation");
  page.panel("DCFPanel").onRecompute(); // on AAPL's view
  await page.idle();
  savesFor(api, "AAPL")[0].resolve(); // the switch to NOK goes ahead
  await page.idle();
  const [, recompute, nok] = api.calls.valuation;
  assert.deepEqual([recompute.args[0], nok.args[0]], ["AAPL", "NOK"]);
  nok.resolve(valuation("NOK"));
  await page.idle();
  recompute.resolve(valuation("AAPL")); // AAPL's recompute finishes last
  await page.idle();
  assert.equal(page.shownName(), "Nokia Oyj");
  assert.ok(api.calls.watchlist.every((w) => w.name !== "Apple Inc."), JSON.stringify(api.calls.watchlist));
  assert.ok(api.calls.watchlist.some((w) => w.ticker === "NOK" && w.name === "Nokia Oyj"));
});

test("a refused load leaves the ticker on screen working: its note, digest and recompute land", async (t) => {
  const api = mockApi();
  const page = await aaplWithEdit(t, api);
  api.auto.save = false;
  page.panel("AIResearchPanel").onGenerateNote(); // takes a while
  await page.analyze("NOK");
  await page.idle();
  page.panel("FilingsPanel").onDigested("Q3 call", digest); // during the wait
  await page.idle();
  savesFor(api, "AAPL")[0].reject(new Error("503 store locked"));
  await page.idle();
  assert.match(page.error(), /has not finished saving/);
  assert.equal(page.shownName(), "Apple Inc.");
  api.calls.note[0].resolve({ title: "AAPL note" });
  await page.idle();
  const research = page.panel("AIResearchPanel");
  assert.deepEqual(research.note, { title: "AAPL note" });
  assert.equal(research.noteLoading, false);
  assert.equal(research.digests.length, 1);
  assert.match(research.researchNotes, /^My view\.\n\n\[Q3 call\] Margins expanded on mix\.\n- Q3 gross margin 46\.2% \(source: Q3 FY2026 earnings call\)\n- Risk: China demand$/);
  // A recompute still lands on AAPL.
  const before = page.panel("AIResearchPanel").report;
  api.auto.valuation = true;
  await page.openTab("Valuation");
  page.panel("DCFPanel").onRecompute();
  await page.idle();
  assert.notEqual(page.panel("AIResearchPanel").report, before);
  assert.equal(page.panel("DCFPanel").recomputing, false);
});

test("a digest that finishes while a switch waits is saved to its own ticker", async (t) => {
  const api = mockApi();
  const page = await aaplWithEdit(t, api);
  const onDigested = page.panel("FilingsPanel").onDigested;
  api.auto.save = false;
  await page.analyze("NOK");
  await page.idle();
  onDigested("Q3 call", digest);
  api.auto.save = true;
  savesFor(api, "AAPL")[0].resolve();
  await page.idle();
  assert.equal(page.shownName(), "Nokia Oyj");
  assert.deepEqual(page.panel("AIResearchPanel").digests, []); // not on NOK
  const saved = api.store.AAPL;
  assert.equal(saved.digests.length, 1);
  assert.equal(saved.digests[0].source, "Q3 call");
  assert.match(saved.notes, /^My view\.\n\n\[Q3 call\] Margins expanded on mix\.\n- Q3 gross margin/);
  assert.equal(api.store.NOK, undefined);
  // Back on AAPL, the saved research has it.
  await page.analyze("AAPL");
  await page.idle();
  assert.equal(page.panel("AIResearchPanel").digests.length, 1);
});

test("a note that finishes while a switch waits is saved to its own ticker", async (t) => {
  const api = mockApi();
  const page = await aaplWithEdit(t, api);
  page.panel("AIResearchPanel").onGenerateNote();
  api.auto.save = false;
  await page.analyze("NOK");
  await page.idle();
  // The note finishes, and the waited-on save lands before the page renders
  // again (so before the autosave could pick the note up).
  api.calls.note[0].resolve({ title: "AAPL note" });
  for (let i = 0; i < 10; i++) await Promise.resolve();
  api.auto.save = true;
  savesFor(api, "AAPL")[0].resolve();
  await page.idle();
  assert.equal(page.shownName(), "Nokia Oyj");
  assert.equal(page.panel("AIResearchPanel").note, null);
  assert.deepEqual(api.store.AAPL.note, { title: "AAPL note" });
});

test("re-analyzing a ticker keeps a digest that finishes while its research is re-read", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const api = mockApi();
  const page = mount(api);
  await page.analyze("AAPL");
  await page.idle();
  const onDigested = page.panel("FilingsPanel").onDigested;
  api.auto.research = false;
  await page.analyze("AAPL");
  await page.idle();
  onDigested("Q3 call", digest); // the saved research is being re-read
  await page.idle();
  api.calls.research.at(-1).resolve(structuredClone(api.store.AAPL ?? {}));
  await page.idle();
  assert.equal(page.panel("AIResearchPanel").digests.length, 1);
  t.mock.timers.tick(1500); // the autosave runs
  await page.idle();
  assert.equal(api.store.AAPL.digests.length, 1);
});

test("a research log saved with summary-only digest lines gets their facts on restore", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const api = mockApi({ store: { AAPL: {
    notes: "My view.\n\n[Q3 call] Margins expanded on mix.",
    digests: [{ source: "Q3 call", digest, at: "2026-10-01T00:00:00Z" }],
  } } });
  const page = mount(api);
  await page.analyze("AAPL");
  await page.idle();
  assert.equal(page.panel("AIResearchPanel").researchNotes,
    "My view.\n\n[Q3 call] Margins expanded on mix.\n"
    + "- Q3 gross margin 46.2% (source: Q3 FY2026 earnings call)\n- Risk: China demand");
});

// The engine's defaults echo (parse_assumptions({})[5]) for the mock's keys.
const DEFAULT_ECHO = { rf: 0.04, erp: 0.05, forecast_years: 5, terminal_growth: 0.025,
  terminal_method: "gordon", tax_rate: null, peers: null };

test("the header flags custom inputs, and Reset to defaults keeps only the peer set", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const api = mockApi({ store: { AAPL: { assumptions: { tax_rate: 0.3, peers: "MSFT" } } } });
  // The backend echoes what it used, with its defaults beside it.
  api.module.fetchValuation = (tk, a = {}) => {
    api.calls.valuation.push({ args: [tk, a] });
    return Promise.resolve({ ...valuation(tk, { tax_rate: a.tax_rate ?? null, peers: a.peers ?? null }),
      assumptions_default: DEFAULT_ECHO });
  };
  const page = mount(api);
  await page.analyze("AAPL");
  await page.idle();
  assert.deepEqual(api.calls.valuation[0].args, ["AAPL", { tax_rate: 0.3, peers: "MSFT" }]);
  const badge = page.button("Custom inputs (2)");
  assert.ok(badge, "no custom-inputs badge");
  assert.equal(badge["aria-expanded"], false);
  // aria-controls names the list only while it exists.
  assert.equal(badge["aria-controls"], undefined);
  badge.onClick();
  await page.idle();
  const list = page.find((e) => e.props.id === "custom-inputs");
  assert.ok(list, "the list did not open");
  // The list is placed against the header card, card-wide below lg: a
  // 320 px panel aligned to the button ran off the left edge wherever the
  // card wraps and the button sits mid-row (390 px: AAPL at -71 px).
  const cls = (e) => new Set(String(e?.props.className ?? "").split(/\s+/));
  const panel = cls(list);
  for (const c of ["absolute", "inset-x-3", "top-full", "z-20", "lg:left-auto", "lg:right-5",
    "lg:w-80"]) assert.ok(panel.has(c), `the list lacks ${c}`);
  for (const c of ["right-0", "w-80", "max-w-[90vw]"]) assert.ok(!panel.has(c), `the list has ${c}`);
  const wrapper = page.find((e) => e.type === "div" && [e.props.children].flat()
    .some((k) => k?.props?.children === "Custom inputs (2)"));
  assert.ok(wrapper, "no badge wrapper");
  assert.ok(![...cls(wrapper)].some((c) => /(^|:)(relative|absolute|fixed|sticky)$/.test(c)),
    "the badge's wrapper is positioned, so the list would hang from the button");
  const card = page.find((e) => e.type === "div" && cls(e).has("rounded-xl")
    && cls(e).has("mt-5"));
  assert.ok(cls(card).has("relative"), "the company header card is not positioned");
  assert.equal(page.button("Custom inputs (2)")["aria-controls"], "custom-inputs");
  assert.equal(page.button("Custom inputs (2)")["aria-expanded"], true);
  assert.match(JSON.stringify(list.props.children), /Tax rate.*30\.0%.*default .*derived.*Peers.*MSFT.*Reset keeps the peer set/s);
  page.button("Reset to defaults").onClick();
  await page.idle();
  // The existing "default assumptions" reload: saved drivers dropped, peers kept.
  assert.deepEqual(api.calls.valuation.at(-1).args, ["AAPL", { peers: "MSFT" }]);
  assert.ok(page.button("Custom inputs (1)"), "the peer set is still custom");
  assert.equal(page.find((e) => e.props.id === "custom-inputs"), undefined);
  // An ordinary default valuation shows no badge.
  api.store.NOK = {};
  await page.analyze("NOK");
  await page.idle();
  assert.equal(page.find((e) => typeof e.props.children === "string"
    && e.props.children.startsWith("Custom inputs")), undefined);
});

test("Add peers opens the Valuation tab and focuses the comps card's peer input once", async () => {
  const api = mockApi();
  const page = mount(api);
  await page.analyze("GM");
  await page.idle();
  await page.openTab("Valuation");
  const ref = page.panel("CompsPanel").inputRef;
  let focused = 0;
  ref.current = { focus: () => { focused += 1; }, scrollIntoView() {} };
  await page.openTab("Overview");
  page.panel("VerdictExplainer").onAddPeers();
  await page.idle();
  assert.equal(page.panel("TabBar").value, "Valuation");
  assert.equal(focused, 1);
  // Opening the tab again later does not steal the focus.
  await page.openTab("Overview");
  await page.openTab("Valuation");
  assert.equal(focused, 1);
  // The DDM card gets the shared draft and recompute.
  const ddm = page.panel("DDMFCFEPanel");
  assert.equal(typeof ddm.setAssumptions, "function");
  assert.equal(typeof ddm.onRecompute, "function");
});

test("a ticker submitted while another loads replaces it; the one loading is not asked twice", async () => {
  const api = mockApi();
  api.auto.valuation = false;
  const page = mount(api);
  await page.analyze("PEP");
  await page.idle();
  // Analyze stays enabled (the input does too), so Enter submits.
  const analyze = page.find((e) => e.type?.name === "Button" && e.props.type === "submit").props;
  assert.equal(analyze.disabled, undefined);
  await page.analyze("pep "); // the ticker already loading: nothing new
  await page.idle();
  assert.deepEqual(api.calls.valuation.map((c) => c.args[0]), ["PEP"]);
  await page.analyze("KO");
  await page.idle();
  assert.deepEqual(api.calls.valuation.map((c) => c.args[0]), ["PEP", "KO"]);
  const [pep, ko] = api.calls.valuation;
  ko.resolve(valuation("KO"));
  await page.idle();
  pep.resolve(valuation("PEP")); // the replaced load finishes last: dropped
  await page.idle();
  assert.equal(page.shownName(), "KO");
  assert.equal(page.find((e) => e.type === "input").props.value, "KO");
  // Once KO has loaded it can be analyzed again.
  api.auto.valuation = true;
  await page.analyze("KO");
  await page.idle();
  assert.equal(api.calls.valuation.length, 3);
});

test("a ticker opened from the watchlist shows in the ticker input", async () => {
  const api = mockApi({ watchlist: [{ ticker: "AAPL", name: "Apple Inc." }] });
  const page = mount(api);
  await page.idle();
  page.find((e) => e.type === "input").props.onChange({ target: { value: "ZZZZQX" } });
  await page.idle();
  page.panel("WatchlistGrid").onOpen("AAPL");
  await page.idle();
  assert.equal(page.shownName(), "Apple Inc.");
  assert.equal(page.find((e) => e.type === "input").props.value, "AAPL");
});

test("an implausible ticker is refused before any request and the ticker on screen stays", async () => {
  const api = mockApi();
  const page = mount(api);
  await page.analyze("AAPL");
  await page.idle();
  const before = [api.calls.research.length, api.calls.valuation.length];
  await page.analyze("AAPL$");
  await page.idle();
  assert.deepEqual([api.calls.research.length, api.calls.valuation.length], before);
  assert.match(page.error(), /^"AAPL\$" is not a ticker: use up to 12 letters, digits, dots or hyphens/);
  assert.equal(page.shownName(), "Apple Inc.");
  assert.equal(page.button("Retry"), undefined);
});

test("a ticker with no market data shows the backend's message and a plain Retry only", async () => {
  const api = mockApi();
  const NO_DATA = "No market data for ZZZZQX; check the symbol, or try again shortly if the "
    + "market-data service is unreachable.";
  const refuse = (status, message) => () => {
    api.calls.valuation.push({ args: [] });
    return Promise.reject(Object.assign(new Error(message), { status }));
  };
  api.module.fetchValuation = refuse(404, NO_DATA);
  const page = mount(api);
  await page.analyze("ZZZZQX");
  await page.idle();
  assert.equal(page.error(), NO_DATA);
  assert.ok(page.button("Retry"), "no Retry");
  assert.equal(page.button("Retry with default assumptions"), undefined);
  // A refused input (400) can be the saved assumptions: both retries.
  api.module.fetchValuation = refuse(400, "terminal_growth must be below the cost of capital.");
  page.button("Retry").onClick();
  await page.idle();
  assert.equal(page.error(), "terminal_growth must be below the cost of capital.");
  assert.ok(page.button("Retry with default assumptions"));
  // An error that is not the failed load offers neither.
  await page.analyze("BAD TICKER");
  await page.idle();
  assert.match(page.error(), /is not a ticker/);
  assert.equal(page.button("Retry"), undefined);
});

test("the app has an icon, so browsers do not ask for a missing /favicon.ico", () => {
  // Next serves app/icon.png and links it from every page (the desktop
  // app's icon at 64 px).
  const png = readFileSync(join(root, "app", "icon.png"));
  assert.deepEqual([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  assert.deepEqual([png.readUInt32BE(16), png.readUInt32BE(20)], [64, 64]);
  assert.ok(png.length < 16_000);
});
