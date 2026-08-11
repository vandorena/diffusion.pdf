/**
 * Run the exact JavaScript that ships inside the PDF, outside the PDF.
 *
 * The builder writes the fully-substituted payload to out/diffusion.js from the
 * same Python string it hands to create_script, so this file is byte-identical
 * to what lands in the document -- no PDF parser, and no chance of the tested
 * code drifting from the shipped code.
 *
 * The viewer's scripting API is stubbed just deeply enough to run: getField
 * returns recorders, app.setTimeOut queues expressions that are drained in
 * order, and app.alert is a hard failure because the payload's outer catch
 * calls it, which makes every in-document exception show up here as a
 * non-zero exit.
 *
 *   node tools/harness.mjs [--js out/diffusion.js] [--ref out/py_trace.json]
 */

import { readFileSync, writeFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import process from "node:process";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i === -1 ? fallback : process.argv[i + 1];
}

const JS_PATH = arg("js", "out/diffusion.js");
const REF_PATH = arg("ref", "out/py_trace.json");
const OUT_PATH = arg("out", "out/js_trace.json");
const VERBOSE = process.argv.includes("--verbose");

/** One form field. Records every write so the paint path can be asserted. */
class FieldStub {
  constructor(name, log) {
    this.name = name;
    this._log = log;
    this._value = "";
    this._fill = null;
    this.display = 0;
    this.hidden = false;
    this.readonly = false;
    this.textColor = null;
    this.borderStyle = null;
    this.rect = [0, 0, 0, 0];
  }
  get value() { return this._value; }
  set value(v) {
    this._value = String(v);
    this._log.push({ op: "value", name: this.name, v: this._value });
  }
  get fillColor() { return this._fill; }
  set fillColor(c) {
    // Catching a malformed colour array here matters: in a real viewer a bad
    // assignment is silently ignored and the pixel simply never changes.
    if (!Array.isArray(c) || c.length === 0) {
      throw new Error(`${this.name}: fillColor must be an array, got ${JSON.stringify(c)}`);
    }
    const space = c[0];
    const arity = { T: 0, G: 1, RGB: 3, CMYK: 4 }[space];
    if (arity === undefined) {
      throw new Error(`${this.name}: unknown colour space ${JSON.stringify(space)}`);
    }
    if (c.length !== arity + 1) {
      throw new Error(`${this.name}: ${space} needs ${arity} components, got ${c.length - 1}`);
    }
    for (let i = 1; i < c.length; i++) {
      if (typeof c[i] !== "number" || !isFinite(c[i]) || c[i] < 0 || c[i] > 1) {
        throw new Error(`${this.name}: component ${i} out of [0,1]: ${c[i]}`);
      }
    }
    this._fill = c.slice();
    this._log.push({ op: "fillColor", name: this.name, color: this._fill });
  }
  buttonSetCaption() {}
  setFocus() {}
}

export function makeSandbox({ lenient = false } = {}) {
  const writes = [];
  const fields = new Map();
  const alerts = [];
  const timers = [];

  function getField(name) {
    if (!fields.has(name)) {
      if (!lenient && !/^(px|row|console)_\d+$/.test(name) &&
          !["wordInput", "seedInput", "stepsInput", "guidanceInput",
            "status", "liveStatus"].includes(name)) {
        throw new Error(`getField("${name}"): no such field in the built PDF`);
      }
      fields.set(name, new FieldStub(name, writes));
    }
    return fields.get(name);
  }

  const sandbox = {
    console: {
      log: (...a) => VERBOSE && console.log("[pdf]", ...a),
      println: (...a) => VERBOSE && console.log("[pdf]", ...a),
    },
    app: {
      alert: (msg) => { alerts.push(String(msg)); },
      setTimeOut: (expr, _ms) => { timers.push(expr); return { __timer: timers.length }; },
      clearTimeOut: () => {},
      setInterval: () => ({}),
      clearInterval: () => {},
      viewerType: "Chrome",
      viewerVersion: 0,
    },
    color: {
      transparent: ["T"], black: ["G", 0], white: ["G", 1],
      red: ["RGB", 1, 0, 0], blue: ["RGB", 0, 0, 1],
    },
    display: { visible: 0, hidden: 1, noPrint: 2, noView: 3 },
    util: {
      printf: (fmt, ...a) => { let i = 0; return String(fmt).replace(/%[\d.]*[dfs]/g, () => String(a[i++])); },
      printd: () => "",
    },
    Math, JSON, Date, isFinite, isNaN, parseInt, parseFloat,
  };
  sandbox.globalThis = sandbox;
  sandbox.getField = getField;
  sandbox.this = sandbox;

  return { sandbox, writes, fields, alerts, timers };
}

export function runPayload(src, opts = {}) {
  const env = makeSandbox(opts);
  const ctx = createContext(env.sandbox);
  runInContext(src, ctx, { filename: "diffusion.js" });
  return { ctx, ...env };
}

/** Drain queued app.setTimeOut expressions in order, as a viewer would. */
export function drainTimers(ctx, timers, limit = 10000) {
  let n = 0;
  while (timers.length && n < limit) {
    const expr = timers.shift();
    runInContext(expr, ctx, { filename: "timer" });
    n++;
  }
  if (n >= limit) throw new Error("timer queue did not drain -- infinite loop?");
  return n;
}

function fail(msg) { console.error(`FAIL  ${msg}`); process.exitCode = 1; }
function pass(msg) { console.log(`pass  ${msg}`); }

function maxAbsDiff(a, b) {
  let m = 0, at = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) { m = d; at = i; }
  }
  return { max: m, at };
}

function main() {
  let src;
  try {
    src = readFileSync(JS_PATH, "utf8");
  } catch {
    console.error(`no ${JS_PATH}. Build it first:\n` +
      `  python3 scripts/generateDiffusionPDF.py --emit-js ${JS_PATH}`);
    process.exit(2);
  }

  const { ctx, writes, alerts, timers } = runPayload(src);
  if (alerts.length) {
    fail(`the payload threw on load: ${alerts[0].split("\n")[0]}`);
    process.exit(1);
  }
  pass(`payload loaded and ran (${src.length.toLocaleString()} chars)`);

  let ref = null;
  try {
    ref = JSON.parse(readFileSync(REF_PATH, "utf8"));
  } catch {
    console.error(`no ${REF_PATH}. Generate it first:\n` +
      `  python3 -m tools.reference --out ${REF_PATH}`);
    process.exit(2);
  }

  // G1 -- the PRNG stream. Shifts and xors only, so any difference is a
  // coercion bug and shows up here before anything else can mask it.
  if (ref.prng) {
    const js = ctx.dsPrngStream(ref.prng.seed, ref.prng.values.length);
    const same = js.every((v, i) => v === ref.prng.values[i]);
    same ? pass(`G1 PRNG stream matches exactly (${js.length} draws)`)
         : fail(`G1 PRNG diverges at draw ${js.findIndex((v, i) => v !== ref.prng.values[i])}`);
  }

  // G2 -- the initial noise. Table lookup plus one multiply-add, so this is
  // required to be exact, not merely close.
  if (ref.x_init) {
    const js = ctx.dsInitialNoise(ref.word, ref.seed);
    const d = maxAbsDiff(js, ref.x_init);
    d.max === 0 ? pass("G2 initial noise is bit-identical")
                : fail(`G2 initial noise differs by ${d.max.toExponential(3)} at pixel ${d.at}`);
  }

  // G3 -- the trajectory, per step. Comparing only the final image would say
  // "wrong" without saying where; this localises a divergence to one step.
  if (ref.traj) {
    const js = ctx.dsTrajectory(ref.word, ref.seed, ref.cls, ref.steps, ref.guidance);
    let worst = 0, worstStep = -1, worstPix = -1;
    for (let i = 0; i < ref.traj.length; i++) {
      const d = maxAbsDiff(js[i].x, ref.traj[i].x);
      if (d.max > worst) { worst = d.max; worstStep = i; worstPix = d.at; }
    }
    const tol = 1e-9;
    worst <= tol
      ? pass(`G3 trajectory matches over ${ref.traj.length} steps (max ${worst.toExponential(2)})`)
      : fail(`G3 diverges at step ${worstStep}, pixel ${worstPix}, by ${worst.toExponential(3)} (tol ${tol})`);
  }

  // G4 -- what the user actually sees. Integer levels, so exact or broken.
  // Painting goes through the same function the document uses, so a bug in the
  // quantisation or the inversion shows up here rather than only on screen.
  if (ref.levels) {
    ctx.dsPaintCurrent();
    const fill = new Map(), text = new Map();
    for (const w of writes) {
      if (w.op === "fillColor") fill.set(w.name, w.color);
      else if (w.op === "value") text.set(w.name, w.v);
    }
    const n = ref.levels.length;
    const grid = ref.grid;

    // The chars build paints `grid` text rows rather than `grid*grid` colour
    // widgets, so the same gate has to read whichever path the build uses.
    // Keyed on row_0 alone: the boot diagnostic probes px_0's fillColor in
    // both builds, so the absence of colour writes is not a reliable signal.
    const charsMode = text.has("row_0");

    if (charsMode) {
      const RAMP = " .:-=+*#%@";
      let wrong = 0, firstWrong = -1, missing = 0;
      for (let r = 0; r < grid; r++) {
        const got = text.get(`row_${r}`);
        if (got === undefined) { missing++; continue; }
        let want = "";
        for (let c = 0; c < grid; c++) {
          const u = (ref.image[r * grid + c] + 1) * 0.5;
          let q = Math.floor(u * (RAMP.length - 1) + 0.5);
          q = Math.max(0, Math.min(RAMP.length - 1, q));
          const ch = RAMP.charAt(RAMP.length - 1 - q);
          want += ch + ch;
        }
        if (got !== want) { wrong++; if (firstWrong < 0) firstWrong = r; }
      }
      if (missing) fail(`G4 ${missing} of ${grid} character rows were never written`);
      else if (wrong) fail(`G4 ${wrong} character rows differ, first at row ${firstWrong}`);
      else pass(`G4 all ${grid} character rows match the reference exactly`);
    } else {
      let missing = 0, wrong = 0, firstWrong = -1;
      for (let i = 0; i < n; i++) {
        const c = fill.get(`px_${i}`);
        if (!c) { missing++; continue; }
        const lvl = Math.round(c[1] * (ref.n_levels - 1));
        if (lvl !== ref.levels[i]) { wrong++; if (firstWrong < 0) firstWrong = i; }
      }
      if (missing) fail(`G4 ${missing} of ${n} pixels were never painted`);
      else if (wrong) fail(`G4 ${wrong} pixels differ, first at ${firstWrong}`);
      else pass(`G4 all ${n} painted levels match the reference exactly`);
    }
  }

  if (alerts.length) fail(`app.alert fired: ${alerts[0].split("\n")[0]}`);
  if (timers.length) console.log(`note  ${timers.length} timer expressions left queued`);

  writeFileSync(OUT_PATH, JSON.stringify({ writes: writes.length }, null, 1));
  console.log(process.exitCode ? "\nsome checks failed" : "\nall checks passed");
}

if (import.meta.url === `file://${process.argv[1]}`) main();
