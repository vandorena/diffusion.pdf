try {
    // diffusion.pdf -- a denoising diffusion model that runs in the viewer.
    //
    // Everything on the sampling path is built from +, -, *, / and sqrt, which
    // IEEE-754 requires to be correctly rounded and which therefore give the
    // same bits here as in numpy. No log, exp, cos or pow is called: the
    // ECMAScript spec leaves those implementation-approximated, and using one
    // would trade exact verification for nothing. That is why the activation is
    // ReLU, why the timestep conditioning is a lookup table rather than a
    // sinusoid, why the noise comes from an inverse-CDF table rather than
    // Box-Muller, and why the schedule arrives as decimal literals computed
    // offline.

    var GRID = __GRID__;
    var NPIX = GRID * GRID;
    var LEVELS = __LEVELS__;
    var CONSOLE_LINES = __CONSOLE_LINE_COUNT__;
    var PAINT_MODE = "__PAINT_MODE__";
    var NORM_BINS = __NORM_BINS__;

    var ABAR = __ABAR__;
    var NORM = __NORM__;
    var CLASSES = __CLASSES__;
    var SYNONYMS = __SYNONYMS__;
    var TENSORS = __TENSORS__;
    var SCALES = __SCALES__;
    var BIASES = __BIASES__;
    var LN = __LN__;
    var LN_EPS = __LN_EPS__;
    var BLOCKS = __BLOCKS__;
    var NULL_CLASS = __NULL_CLASS__;
    var DEF_STEPS = __STEPS_DEFAULT__;
    var DEF_GUIDANCE = __GUIDANCE_DEFAULT__;
    var DEF_SEED = __SEED_DEFAULT__;
    var DEF_WORD = "__WORD_DEFAULT__";
    var PAINT_EVERY = __PAINT_EVERY__;
    var NUDGE = "__NUDGE__";   // none | caption | display | value

    // ------------------------------------------------------------------
    // viewer shim -- the only place that knows which viewer this is
    // ------------------------------------------------------------------

    function field(name) {
        try {
            if (typeof globalThis !== "undefined" && globalThis.getField) {
                return globalThis.getField(name);
            }
        } catch (e) { /* fall through to the Acrobat form */ }
        try { return this.getField(name); } catch (e2) { return null; }
    }

    var lines = [];
    function say(msg) {
        lines.push(String(msg));
        while (lines.length > CONSOLE_LINES) lines.shift();
        for (var i = 0; i < lines.length; i++) {
            var f = field("console_" + i);
            if (f) f.value = lines[i];
        }
    }

    // ------------------------------------------------------------------
    // deterministic noise
    // ------------------------------------------------------------------

    // xorshift32. Shifts and xors only: JS coerces those through ToInt32 and
    // ToUint32, so the wrapping is exact and matches Python masked to 32 bits.
    // There is no multiply that could exceed 2^53 and diverge, which is why the
    // harness can assert integer equality on the raw stream.
    function Rng(seed) {
        this.s = seed >>> 0;
        if (this.s === 0) this.s = 0x9E3779B9;
    }
    Rng.prototype.next = function () {
        var x = this.s;
        x ^= (x << 13);
        x ^= (x >>> 17);
        x ^= (x << 5);
        this.s = x >>> 0;
        return this.s;
    };

    // h * 33 + c stays far below 2^53, so plain multiplication is exact here
    // and Math.imul is not needed.
    function djb2(text) {
        var h = 5381;
        for (var i = 0; i < text.length; i++) {
            h = (h * 33 + text.charCodeAt(i)) % 4294967296;
        }
        return h >>> 0;
    }

    function seedFor(word, seed) {
        var combined = (djb2(word) ^ (seed >>> 0)) >>> 0;
        return combined === 0 ? 0x9E3779B9 : combined;
    }

    var NORM_SHIFT = 32 - Math.round(Math.log(NORM_BINS) / Math.LN2);
    var NORM_MASK = (1 << NORM_SHIFT) - 1;
    var NORM_SCALE = 1.0 / (1 << NORM_SHIFT);

    // Inverse-CDF lookup with linear interpolation. One uint32 per value: the
    // top bits index the table, the rest interpolate. Pure index arithmetic and
    // a multiply-add, so it is exact; Box-Muller would need log and cos.
    function gaussian(count, seed) {
        var rng = new Rng(seed);
        var out = new Float64Array(count);
        for (var i = 0; i < count; i++) {
            var u = rng.next();
            var idx = u >>> NORM_SHIFT;
            var frac = (u & NORM_MASK) * NORM_SCALE;
            out[i] = NORM[idx] + frac * (NORM[idx + 1] - NORM[idx]);
        }
        return out;
    }

    // ------------------------------------------------------------------
    // weights
    // ------------------------------------------------------------------

    // Hand-rolled because atob is not guaranteed in a PDF sandbox. Adapted from
    // the decoder already proven in src/template.js.
    function b64ToInt8(str) {
        var lut = new Uint8Array(123);
        var alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (var i = 0; i < alpha.length; i++) lut[alpha.charCodeAt(i)] = i;

        var out = new Int8Array(Math.floor(str.length / 4) * 3);
        var o = 0;
        for (var p = 0; p < str.length; p += 4) {
            var v1 = lut[str.charCodeAt(p)], v2 = lut[str.charCodeAt(p + 1)];
            var v3 = lut[str.charCodeAt(p + 2)], v4 = lut[str.charCodeAt(p + 3)];
            out[o++] = ((v1 << 2) | (v2 >> 4)) << 24 >> 24;
            if (str.charAt(p + 2) !== "=") {
                out[o++] = (((v2 & 15) << 4) | (v3 >> 2)) << 24 >> 24;
                if (str.charAt(p + 3) !== "=") {
                    out[o++] = (((v3 & 3) << 6) | v4) << 24 >> 24;
                }
            }
        }
        return out.subarray(0, o);
    }

    var W = {};   // name -> Int8Array, kept int8
    var SHAPE = {};

    // Substituted last by the builder: it is by far the largest thing in the
    // file, and replacing it first would make every later substitution rescan
    // several megabytes. Base64's alphabet contains no backslash or parenthesis,
    // so it needs no escaping when pdfrw writes it as a PDF literal string.
    var WEIGHTS_B64 = "__WEIGHTS_B64__";

    function unpackWeights() {
        var flat = b64ToInt8(WEIGHTS_B64);
        var off = 0;
        for (var i = 0; i < TENSORS.length; i++) {
            var spec = TENSORS[i];
            var rows = spec[1], cols = spec[2];
            var n = rows * cols;
            W[spec[0]] = flat.subarray(off, off + n);
            SHAPE[spec[0]] = [rows, cols];
            off += n;
        }
        if (off !== flat.length) {
            throw new Error("weight blob is " + flat.length + " bytes, tensors want " + off);
        }
        return off;
    }

    // y = scale_row * (w_row . x) + bias_row.
    //
    // The weights stay int8 and the per-row scale folds into the accumulator
    // once, rather than materialising a dequantised float copy: 740k weights as
    // float64 would be 6 MB of viewer heap, and this repo already has a commit
    // about Chrome's PDF viewer capping memory.
    function matvec(name, x, out, relu) {
        var w = W[name], s = SCALES[name], b = BIASES[name];
        var rows = SHAPE[name][0], cols = SHAPE[name][1];
        for (var i = 0; i < rows; i++) {
            var acc = 0, o = i * cols;
            for (var j = 0; j < cols; j++) acc += w[o + j] * x[j];
            acc = acc * s[i] + (b ? b[i] : 0);
            out[i] = relu && acc < 0 ? 0 : acc;
            }
        return out;
    }

    // Accumulate into an existing vector (for the conditioning projections,
    // which are summed into another layer's output before the activation).
    function matvecAdd(name, x, out) {
        var w = W[name], s = SCALES[name];
        var rows = SHAPE[name][0], cols = SHAPE[name][1];
        for (var i = 0; i < rows; i++) {
            var acc = 0, o = i * cols;
            for (var j = 0; j < cols; j++) acc += w[o + j] * x[j];
            out[i] += acc * s[i];
        }
        return out;
    }

    function embed(name, index, out) {
        var w = W[name], s = SCALES[name], cols = SHAPE[name][1];
        var o = index * cols, scale = s[index];
        for (var j = 0; j < cols; j++) out[j] = w[o + j] * scale;
        return out;
    }

    function embedAdd(name, index, out) {
        var w = W[name], s = SCALES[name], cols = SHAPE[name][1];
        var o = index * cols, scale = s[index];
        for (var j = 0; j < cols; j++) out[j] += w[o + j] * scale;
        return out;
    }

    // ------------------------------------------------------------------
    // painting -- isolated so a viewer quirk is a one-function fix
    // ------------------------------------------------------------------

    var PX = null;          // cached field objects
    var LAST = null;        // last level written, for dirty-skipping
    var GRAYS = [];         // preallocated colour arrays; never allocate per write

    function initPaint() {
        PX = [];
        for (var i = 0; i < NPIX; i++) PX.push(field("px_" + i));
        LAST = new Int16Array(NPIX);
        for (var k = 0; k < NPIX; k++) LAST[k] = -1;
        GRAYS = [];
        for (var g = 0; g < LEVELS; g++) GRAYS.push(["G", g / (LEVELS - 1)]);
    }

    // The one place that touches a widget's appearance.
    //
    // Assigning fillColor sets /MK/BG, but a viewer does not necessarily
    // regenerate the appearance stream just because the property changed. NUDGE
    // picks what else to poke to force it. Keep this the only paint path: when
    // a viewer turns out to need something different, this is the single line
    // that changes.
    function paintPixel(i, level) {
        var f = PX[i];
        if (!f) return;
        f.fillColor = GRAYS[level];
        if (NUDGE === "caption") {
            // A pushbutton's appearance is built from /MK, so re-setting the
            // caption is the most direct way to ask for it to be rebuilt.
            try { f.buttonSetCaption(""); } catch (e) { NUDGE = "none"; }
        } else if (NUDGE === "display") {
            try { f.display = display.visible; } catch (e) { NUDGE = "none"; }
        } else if (NUDGE === "value") {
            try { f.value = f.value; } catch (e) { NUDGE = "none"; }
        }
    }

    var RAMP = " .:-=+*#%@";

    // force = true rewrites every pixel even if its level is unchanged. The
    // dirty-skip is a big saving during a run, but it also means a repaint
    // cannot be requested for its own sake, which is what Clear needs.
    function paintAll(x, force) {
        var painted = 0;
        if (PAINT_MODE === "chars") {
            // Two characters per pixel: Courier advances 0.6 em, so doubling
            // makes the drawing come out roughly square instead of squashed.
            for (var r = 0; r < GRID; r++) {
                var s = "";
                for (var c = 0; c < GRID; c++) {
                    var u = (x[r * GRID + c] + 1) * 0.5;
                    var q = Math.floor(u * (RAMP.length - 1) + 0.5);
                    if (q < 0) q = 0; else if (q > RAMP.length - 1) q = RAMP.length - 1;
                    var ch = RAMP.charAt(RAMP.length - 1 - q);
                    s += ch + ch;
                }
                var f = field("row_" + r);
                if (f) { f.value = s; painted++; }
            }
            return painted;
        }
        for (var i = 0; i < NPIX; i++) {
            var v = (x[i] + 1) * 0.5;
            var lvl = Math.floor(v * LEVELS);
            if (lvl < 0) lvl = 0; else if (lvl > LEVELS - 1) lvl = LEVELS - 1;
            // Invert so a high value reads as dark ink on a light page.
            lvl = LEVELS - 1 - lvl;
            if (force || lvl !== LAST[i]) {
                paintPixel(i, lvl);
                LAST[i] = lvl;
                painted++;
            }
        }
        return painted;
    }

    // Blank the grid. Also the honest way to tell whether painting works at
    // all: if Clear does not visibly wipe the picture, no fillColor write is
    // reaching the screen and the problem is the paint path, not the model.
    function dsClear() {
        if (!PX) initPaint();
        for (var i = 0; i < NPIX; i++) {
            paintPixel(i, LEVELS - 1);
            LAST[i] = LEVELS - 1;
        }
        if (PAINT_MODE === "chars") {
            for (var r = 0; r < GRID; r++) {
                var f = field("row_" + r);
                if (f) f.value = "";
            }
        }
        S = null;
        var st = field("status");
        if (st) st.value = "cleared";
        say("cleared. if the picture did not go blank, fillColor is not");
        say("repainting in this viewer -- try the Nudge button.");
    }

    // Cycle the repaint strategy at runtime. Whether a viewer needs a nudge,
    // and which one, is not something that can be settled without a viewer, so
    // this makes it a click rather than a rebuild.
    function dsNudge() {
        var order = ["none", "caption", "display", "value"];
        var at = 0;
        for (var i = 0; i < order.length; i++) if (order[i] === NUDGE) at = i;
        NUDGE = order[(at + 1) % order.length];
        say("repaint nudge: " + NUDGE + " -- redrawing");
        var st = field("status");
        if (st) st.value = "nudge = " + NUDGE;
        if (S && S.x) paintAll(S.x, true);
        else dsClear();
    }

    // ------------------------------------------------------------------
    // word -> class
    // ------------------------------------------------------------------

    function normaliseWord(w) {
        var s = String(w == null ? "" : w).toLowerCase();
        var out = "";
        for (var i = 0; i < s.length; i++) {
            var ch = s.charAt(i);
            if ((ch >= "a" && ch <= "z") || ch === " ") out += ch;
        }
        return out.replace(/\s+/g, " ").replace(/^ | $/g, "");
    }

    function indexOfClass(w) {
        for (var i = 0; i < CLASSES.length; i++) if (CLASSES[i] === w) return i;
        if (SYNONYMS[w] !== undefined) {
            for (var j = 0; j < CLASSES.length; j++) if (CLASSES[j] === SYNONYMS[w]) return j;
        }
        return -1;
    }

    function lookupClass(word) {
        var clean = normaliseWord(word);
        if (clean === "") return { cls: 0, how: "empty", word: CLASSES[0] };

        var tries = [clean];
        if (clean.length > 2 && clean.charAt(clean.length - 1) === "s" &&
            clean.charAt(clean.length - 2) !== "s") {
            tries.push(clean.substring(0, clean.length - 1));
        }
        if (clean.length > 3 && clean.substring(clean.length - 2) === "es") {
            tries.push(clean.substring(0, clean.length - 2));
        }
        for (var i = 0; i < tries.length; i++) {
            var hit = indexOfClass(tries[i]);
            if (hit >= 0) return { cls: hit, how: "exact", word: CLASSES[hit] };
        }

        var tokens = clean.split(" ");
        for (var t = 0; t < tokens.length; t++) {
            var tok = tokens[t];
            var h = indexOfClass(tok);
            if (h < 0 && tok.length > 2 && tok.charAt(tok.length - 1) === "s") {
                h = indexOfClass(tok.substring(0, tok.length - 1));
            }
            if (h >= 0) return { cls: h, how: "token", word: CLASSES[h] };
        }

        // Unknown. Pick deterministically and say so rather than quietly
        // drawing the wrong thing -- the honest ceiling here is "nearest of N
        // known objects", and the UI should not pretend otherwise.
        var pick = djb2(clean) % CLASSES.length;
        return { cls: pick, how: "hash", word: CLASSES[pick] };
    }

    // ------------------------------------------------------------------
    // the sampler, as a step machine rather than a loop
    // ------------------------------------------------------------------

    var S = null;
    var TIMER_MODE = "SYNC";
    var TIMER_HANDLE = null;

    function timesteps(steps) {
        var T = ABAR.length;
        var stride = Math.floor(T / steps);
        var ts = [];
        for (var i = 0; i < steps; i++) ts.push(T - 1 - i * stride);
        return ts;
    }

    // LayerNorm, matching torch: biased variance, eps inside the sqrt. Mean,
    // variance, sqrt and divide are all correctly rounded in IEEE-754, so this
    // stays bit-comparable with the numpy reference -- which is the whole
    // reason normalisation is allowed here at all.
    function layerNorm(x, name, out) {
        var g = LN[name].g, b = LN[name].b, n = x.length;
        var mu = 0;
        for (var i = 0; i < n; i++) mu += x[i];
        mu /= n;
        var varsum = 0;
        for (i = 0; i < n; i++) { var d = x[i] - mu; varsum += d * d; }
        var inv = 1.0 / Math.sqrt(varsum / n + LN_EPS);
        for (i = 0; i < n; i++) out[i] = (x[i] - mu) * inv * g[i] + b[i];
        return out;
    }

    function predictV(x, t, cls, out) {
        var e = SCRATCH.e, h = SCRATCH.h, hn = SCRATCH.hn, inner = SCRATCH.inner;
        embed("temb", t, e);
        embedAdd("cemb", cls, e);

        matvec("inp", x, h, false);
        matvecAdd("c1", e, h);
        var i;
        for (i = 0; i < h.length; i++) if (h[i] < 0) h[i] = 0;

        for (var blk = 0; blk < BLOCKS; blk++) {
            layerNorm(h, "n" + blk, hn);
            matvec("a" + blk, hn, inner, false);
            matvecAdd("c" + blk, e, inner);
            for (i = 0; i < inner.length; i++) if (inner[i] < 0) inner[i] = 0;
            matvec("b" + blk, inner, SCRATCH.t1, false);
            for (i = 0; i < h.length; i++) h[i] += SCRATCH.t1[i];
        }

        layerNorm(h, "nf", hn);
        return matvec("out", hn, out, false);
    }

    var SCRATCH = null;

    function allocScratch() {
        var hidden = SHAPE["inp"][0];
        var cond = SHAPE["temb"][1];
        SCRATCH = {
            e: new Float64Array(cond),
            h: new Float64Array(hidden),
            hn: new Float64Array(hidden),
            inner: new Float64Array(hidden),
            t1: new Float64Array(hidden),
            vc: new Float64Array(NPIX),
            vn: new Float64Array(NPIX),
            x0: new Float64Array(NPIX)
        };
    }

    function dsStep() {
        var i = S.i, t = S.ts[i];
        var abarT = ABAR[t];
        var abarPrev = (i + 1 < S.ts.length) ? ABAR[S.ts[i + 1]] : 1.0;

        var v;
        if (S.guidance === 1.0) {
            v = predictV(S.x, t, S.cls, SCRATCH.vc);
        } else {
            predictV(S.x, t, S.cls, SCRATCH.vc);
            predictV(S.x, t, NULL_CLASS, SCRATCH.vn);
            v = SCRATCH.vc;
            for (var k = 0; k < NPIX; k++) {
                v[k] = SCRATCH.vn[k] + S.guidance * (SCRATCH.vc[k] - SCRATCH.vn[k]);
            }
        }

        var a = Math.sqrt(abarT), bb = Math.sqrt(1.0 - abarT);
        var ap = Math.sqrt(abarPrev), bp = Math.sqrt(1.0 - abarPrev);
        var x0 = SCRATCH.x0;
        for (var j = 0; j < NPIX; j++) {
            var val = a * S.x[j] - bb * v[j];
            if (val < -1) val = -1; else if (val > 1) val = 1;
            x0[j] = val;
        }
        // eps is re-derived from the clipped x0, not taken from the model.
        // DDIM assumes x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps; clipping x0
        // without re-deriving eps breaks that identity and the error compounds
        // until the picture dissolves into noise with plausible statistics.
        for (j = 0; j < NPIX; j++) {
            var eps = (S.x[j] - a * x0[j]) / bb;
            S.x[j] = ap * x0[j] + bp * eps;
        }
        S.i++;
    }

    function statusLine() {
        var f = field("status");
        if (!f) return;
        f.value = "step " + S.i + "/" + S.ts.length +
            "  " + S.label +
            (S.msPerStep ? "  " + S.msPerStep.toFixed(0) + " ms/step" : "");
    }

    function dsBegin() {
        if (!SCRATCH) allocScratch();
        if (!PX) initPaint();

        var word = (field("wordInput") || { value: DEF_WORD }).value || DEF_WORD;
        var seed = parseInt((field("seedInput") || { value: DEF_SEED }).value, 10);
        if (isNaN(seed)) seed = DEF_SEED;
        var steps = parseInt((field("stepsInput") || { value: DEF_STEPS }).value, 10);
        if (isNaN(steps) || steps < 1) steps = DEF_STEPS;
        var guidance = parseFloat((field("guidanceInput") || { value: DEF_GUIDANCE }).value);
        if (isNaN(guidance)) guidance = DEF_GUIDANCE;

        var T = ABAR.length;
        if (T % steps !== 0) {
            steps = DEF_STEPS;
            say("steps must divide " + T + "; using " + steps);
        }

        var found = lookupClass(word);
        S = {
            x: gaussian(NPIX, seedFor(normaliseWord(word), seed)),
            i: 0,
            ts: timesteps(steps),
            cls: found.cls,
            guidance: guidance,
            label: '"' + found.word + '"',
            msPerStep: 0,
            t0: Date.now()
        };

        say("");
        if (found.how === "hash") {
            say('"' + normaliseWord(word) + '" is not in the vocabulary.');
            say('drawing "' + found.word + '" instead (chosen by hash).');
        } else {
            say('drawing "' + found.word + '"' +
                (found.how === "token" ? " (matched one word of it)" : ""));
        }
        say(steps + " steps, guidance " + guidance + ", seed " + seed);

        paintAll(S.x, true);
        statusLine();
        dsPump();
    }

    function dsPump() {
        if (!S || S.i >= S.ts.length) return;

        var chunkStart = Date.now();
        var budget = (TIMER_MODE === "SYNC") ? 1e9 : 50;
        var did = 0;
        while (S.i < S.ts.length && (Date.now() - chunkStart) < budget) {
            dsStep();
            did++;
            if (TIMER_MODE === "SYNC" && PAINT_EVERY > 0 && (S.i % PAINT_EVERY) === 0) {
                paintAll(S.x);
                statusLine();
            }
            if (did >= 1 && TIMER_MODE !== "SYNC") break;
        }

        var elapsed = Date.now() - chunkStart;
        S.msPerStep = did ? elapsed / did : S.msPerStep;
        paintAll(S.x);
        statusLine();

        if (S.i < S.ts.length) {
            if (TIMER_MODE === "TIMER_ACRO") {
                // A string expression, which both Acrobat and PDFium accept,
                // and the handle is kept alive so Acrobat cannot collect it.
                TIMER_HANDLE = app.setTimeOut("dsPump()", 1);
            } else if (TIMER_MODE === "TIMER_DOM") {
                TIMER_HANDLE = setTimeout(dsPump, 0);
            } else {
                dsPump();
            }
        } else {
            var total = (Date.now() - S.t0) / 1000;
            say("done in " + total.toFixed(1) + "s (" +
                S.msPerStep.toFixed(0) + " ms/step)");
        }
    }

    // One step per click. Useful on its own -- watching a person advance the
    // denoiser reads better than an animation -- and it is the debugging tool
    // for every viewer that turns out not to repaint mid-script.
    function dsStepOnce() {
        if (!S || S.i >= S.ts.length) { dsBegin1(); return; }
        dsStep();
        paintAll(S.x);
        statusLine();
        if (S.i >= S.ts.length) say("done.");
    }

    function dsBegin1() {
        var saved = TIMER_MODE;
        TIMER_MODE = "MANUAL";
        dsBeginNoPump();
        TIMER_MODE = saved;
    }

    function dsBeginNoPump() {
        if (!SCRATCH) allocScratch();
        if (!PX) initPaint();
        var word = (field("wordInput") || { value: DEF_WORD }).value || DEF_WORD;
        var seed = parseInt((field("seedInput") || { value: DEF_SEED }).value, 10);
        if (isNaN(seed)) seed = DEF_SEED;
        var steps = parseInt((field("stepsInput") || { value: DEF_STEPS }).value, 10);
        if (isNaN(steps) || steps < 1 || (ABAR.length % steps)) steps = DEF_STEPS;
        var guidance = parseFloat((field("guidanceInput") || { value: DEF_GUIDANCE }).value);
        if (isNaN(guidance)) guidance = DEF_GUIDANCE;
        var found = lookupClass(word);
        S = {
            x: gaussian(NPIX, seedFor(normaliseWord(word), seed)),
            i: 0, ts: timesteps(steps), cls: found.cls, guidance: guidance,
            label: '"' + found.word + '"', msPerStep: 0, t0: Date.now()
        };
        say('stepping "' + found.word + '" -- click Step to advance');
        paintAll(S.x, true);
        statusLine();
    }

    // ------------------------------------------------------------------
    // hooks the verification harness calls; unused inside the document
    // ------------------------------------------------------------------

    function dsPrngStream(seed, n) {
        var rng = new Rng(seed), out = [];
        for (var i = 0; i < n; i++) out.push(rng.next());
        return out;
    }

    function dsInitialNoise(word, seed) {
        var x = gaussian(NPIX, seedFor(normaliseWord(word), seed));
        return Array.prototype.slice.call(x);
    }

    // Paint whatever state the sampler is currently in. The harness calls this
    // after dsTrajectory so the gate on the painted levels exercises the real
    // paint path rather than a reimplementation of it.
    function dsPaintCurrent() {
        if (!PX) initPaint();
        return paintAll(S.x);
    }

    function dsTrajectory(word, seed, cls, steps, guidance) {
        if (!SCRATCH) allocScratch();
        S = {
            x: gaussian(NPIX, seedFor(normaliseWord(word), seed)),
            i: 0, ts: timesteps(steps), cls: cls, guidance: guidance,
            label: "", msPerStep: 0, t0: 0
        };
        var out = [];
        while (S.i < S.ts.length) {
            var t = S.ts[S.i];
            dsStep();
            out.push({ t: t, x: Array.prototype.slice.call(S.x) });
        }
        return out;
    }

    // ------------------------------------------------------------------
    // boot
    // ------------------------------------------------------------------

    var nbytes = unpackWeights();
    allocScratch();

    if (typeof app !== "undefined" && typeof app.setTimeOut === "function") {
        TIMER_MODE = "TIMER_ACRO";
    } else if (typeof setTimeout === "function") {
        TIMER_MODE = "TIMER_DOM";
    } else {
        TIMER_MODE = "SYNC";
    }

    // Boot diagnostic. "Nothing appears" has two very different causes -- the
    // widgets not resolving at all, versus resolving but not repainting -- and
    // guessing between them from the outside costs a rebuild each time.
    initPaint();
    var resolved = 0, canFill = false;
    for (var q = 0; q < NPIX; q++) if (PX[q]) resolved++;
    try {
        if (PX[0]) { PX[0].fillColor = ["G", 1]; canFill = true; }
    } catch (e) { canFill = false; }

    say("=== diffusion.pdf ===");
    say("a diffusion model, running in this document. no network.");
    say("");
    say(GRID + "x" + GRID + ", " + (nbytes / 1024).toFixed(0) + " kB of int8 weights, " +
        ABAR.length + " trained steps");
    say("scheduler: " + TIMER_MODE + ", paint: " + PAINT_MODE + ", nudge: " + NUDGE);
    say("widgets resolved: " + resolved + "/" + NPIX +
        ", fillColor assignable: " + canFill);
    if (resolved === 0) {
        say("!! getField cannot see the pixel widgets. Nothing can paint.");
    } else if (!canFill) {
        say("!! this viewer rejects fillColor. Use the chars build instead.");
    }
    say("");
    say("known words: " + CLASSES.join(", "));
    say("");
    say("Type one and press Generate. Step advances one denoising step.");
    say("Anything unknown maps to the nearest known object, and says so.");
} catch (e) { app.alert(e.stack || e); }
