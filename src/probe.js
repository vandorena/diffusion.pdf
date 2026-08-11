try {
    // probe.pdf -- measures the things the diffusion PDF is budgeted against.
    //
    // Nothing here is guessed. Every number this prints replaces an assumption
    // in docs/diffusion.md. Run it in Chrome first, then Acrobat if you care
    // about that path.
    //
    // Results go into form fields rather than app.alert so the whole report can
    // be screenshotted or copied out in one go.

    var GRID = __GRID__;
    var NPIX = GRID * GRID;
    var CONSOLE_LINES = __CONSOLE_LINE_COUNT__;

    var lines = [];

    // The one viewer-specific call in the file. PDFium exposes the document
    // scripting object on the global; Acrobat requires `this`. Everything else
    // goes through this shim so the Acrobat path stays a one-line change.
    function field(name) {
        try {
            if (typeof globalThis !== "undefined" && globalThis.getField) {
                return globalThis.getField(name);
            }
        } catch (e) { /* fall through */ }
        try { return this.getField(name); } catch (e2) { return null; }
    }

    function print_msg(msg) {
        lines.push(String(msg));
        while (lines.length > CONSOLE_LINES) lines.shift();
        for (var i = 0; i < lines.length; i++) {
            var f = field("console_" + i);
            if (f) f.value = lines[i];
        }
    }

    function pad(s, n) {
        s = String(s);
        while (s.length < n) s += " ";
        return s;
    }

    function fixed(x, n) {
        // toFixed exists in every ES3 engine, but be defensive about Infinity.
        if (!isFinite(x)) return String(x);
        return x.toFixed(n);
    }

    // ---------------------------------------------------------------- A
    // What does this engine actually have? llama.js proves typed arrays and
    // Math.imul exist in Chrome's viewer; this confirms it in situ and covers
    // the ones nothing in the repo exercises.

    function probeCapabilities() {
        print_msg("=== A. capabilities ===");
        var names = [
            "Float64Array", "Float32Array", "Int32Array", "Int8Array",
            "Uint8Array", "ArrayBuffer", "DataView", "JSON", "BigInt",
            "setTimeout", "setInterval"
        ];
        var row = "";
        for (var i = 0; i < names.length; i++) {
            var present;
            try { present = (eval("typeof " + names[i]) !== "undefined"); }
            catch (e) { present = false; }
            row += names[i] + "=" + (present ? "y" : "N") + " ";
            if (row.length > 60) { print_msg("  " + row); row = ""; }
        }
        if (row) print_msg("  " + row);

        var maths = ["imul", "fround", "clz32", "hypot", "log2"];
        row = "";
        for (var j = 0; j < maths.length; j++) {
            row += "Math." + maths[j] + "=" + (typeof Math[maths[j]] !== "undefined" ? "y" : "N") + " ";
        }
        print_msg("  " + row);

        var appBits = "";
        try { appBits += "app=" + (typeof app) + " "; } catch (e) { appBits += "app=err "; }
        try { appBits += "setTimeOut=" + (typeof app.setTimeOut) + " "; } catch (e) { appBits += "setTimeOut=err "; }
        try { appBits += "viewer=" + app.viewerType + " " + app.viewerVersion; } catch (e) { appBits += "viewer=?"; }
        print_msg("  " + appBits);

        var gf = (typeof globalThis !== "undefined" && globalThis.getField) ? "globalThis" : "this";
        print_msg("  getField resolves via: " + gf);
        print_msg("");
    }

    // ---------------------------------------------------------------- B
    // Compute. idea.md's 27M MAC/s came from llm.pdf's asm.js; the diffusion
    // model is plain JS over typed arrays, and the gap between those two is
    // the single number the whole parameter budget rests on.

    var ROWS = 256, COLS = 512;

    function benchMatmul(label, mkW, mkX, mkY, body) {
        var W = mkW(), x = mkX(), y = mkY();
        var iters = 0, t0 = Date.now(), elapsed = 0;
        // Always run for at least 300 ms: Date.now() is coarse, and in some
        // viewers deliberately coarsened, so a single pass would time as 0.
        while (elapsed < 300) {
            body(W, x, y);
            iters++;
            elapsed = Date.now() - t0;
        }
        var macs = ROWS * COLS * iters;
        var rate = macs / (elapsed / 1000);
        print_msg("  " + pad(label, 26) + pad(fixed(rate / 1e6, 2) + " MMAC/s", 16) +
                  iters + " passes / " + elapsed + " ms");
        return rate;
    }

    function plainW() {
        var W = [];
        for (var i = 0; i < ROWS; i++) {
            var r = [];
            for (var j = 0; j < COLS; j++) r.push((i * 7 + j * 13) % 255 / 255 - 0.5);
            W.push(r);
        }
        return W;
    }
    function typedW(Ctor) {
        var W = new Ctor(ROWS * COLS);
        for (var i = 0; i < ROWS * COLS; i++) W[i] = (i % 255) / 255 - 0.5;
        return W;
    }
    function int8W() {
        var W = new Int8Array(ROWS * COLS);
        for (var i = 0; i < ROWS * COLS; i++) W[i] = (i % 255) - 127;
        return W;
    }
    function plainX() {
        var x = [];
        for (var i = 0; i < COLS; i++) x.push((i % 100) / 100);
        return x;
    }
    function typedX() {
        var x = new Float64Array(COLS);
        for (var i = 0; i < COLS; i++) x[i] = (i % 100) / 100;
        return x;
    }

    function probeCompute() {
        print_msg("=== B. compute (" + ROWS + "x" + COLS + " matvec) ===");

        benchMatmul("B1 plain Array", plainW, plainX,
            function () { return new Array(ROWS); },
            function (W, x, y) {
                for (var i = 0; i < ROWS; i++) {
                    var s = 0, Wi = W[i];
                    for (var j = 0; j < COLS; j++) s += Wi[j] * x[j];
                    y[i] = s;
                }
            });

        benchMatmul("B2 Float64Array", function () { return typedW(Float64Array); }, typedX,
            function () { return new Float64Array(ROWS); },
            function (W, x, y) {
                for (var i = 0; i < ROWS; i++) {
                    var s = 0, o = i * COLS;
                    for (var j = 0; j < COLS; j++) s += W[o + j] * x[j];
                    y[i] = s;
                }
            });

        benchMatmul("B3 Float32Array w", function () { return typedW(Float32Array); }, typedX,
            function () { return new Float64Array(ROWS); },
            function (W, x, y) {
                for (var i = 0; i < ROWS; i++) {
                    var s = 0, o = i * COLS;
                    for (var j = 0; j < COLS; j++) s += W[o + j] * x[j];
                    y[i] = s;
                }
            });

        // B4 is the path the diffusion model actually takes: int8 weights kept
        // int8 in memory, one float scale per output row folded into the
        // accumulator once. Never materialise a float copy -- 740k weights as
        // float64 is 6 MB, and Chrome's viewer has already had a memory-cap
        // commit in this repo.
        var rate4 = benchMatmul("B4 Int8Array + scale", int8W, typedX,
            function () { return new Float64Array(ROWS); },
            function (W, x, y) {
                for (var i = 0; i < ROWS; i++) {
                    var s = 0, o = i * COLS;
                    for (var j = 0; j < COLS; j++) s += W[o + j] * x[j];
                    y[i] = s * 0.0078125;
                }
            });

        benchMatmul("B5 B4 + ReLU", int8W, typedX,
            function () { return new Float64Array(ROWS); },
            function (W, x, y) {
                for (var i = 0; i < ROWS; i++) {
                    var s = 0, o = i * COLS;
                    for (var j = 0; j < COLS; j++) s += W[o + j] * x[j];
                    s = s * 0.0078125;
                    y[i] = s > 0 ? s : 0;
                }
            });

        // Translate straight into the number that matters.
        var perEval = __MAC_PER_EVAL__;
        var evals = __EVALS_PER_IMAGE__;
        var total = perEval * evals;
        print_msg("");
        print_msg("  model is " + (perEval / 1e3).toFixed(0) + "k MAC/eval x " + evals +
                  " evals = " + (total / 1e6).toFixed(1) + "M MAC/image");
        print_msg("  => " + fixed(total / rate4, 2) + " s per image at the B4 rate");
        print_msg("");
    }

    // ---------------------------------------------------------------- C
    // Painting. Whether assigning fillColor repaints at all, and what it
    // costs, is idea.md's longest-standing open question.

    var CACHE = null;

    function cacheFields() {
        CACHE = [];
        for (var i = 0; i < NPIX; i++) CACHE.push(field("px_" + i));
        var missing = 0;
        for (var k = 0; k < CACHE.length; k++) if (!CACHE[k]) missing++;
        return missing;
    }

    function timeIt(label, n, fn) {
        var t0 = Date.now();
        fn();
        var dt = Date.now() - t0;
        print_msg("  " + pad(label, 30) + pad(dt + " ms", 10) +
                  fixed(dt * 1000 / n, 1) + " us/write");
        return dt;
    }

    function probePaint() {
        print_msg("=== C. paint (" + NPIX + " widgets) ===");
        var missing = cacheFields();
        if (missing) print_msg("  WARNING: " + missing + " of " + NPIX + " widgets not found");

        var grays = [];
        for (var g = 0; g < 16; g++) grays.push(["G", g / 15]);

        timeIt("P3 getField lookups only", NPIX, function () {
            var sink = 0;
            for (var i = 0; i < NPIX; i++) { if (field("px_" + i)) sink++; }
        });

        timeIt("P1 .value writes (cached)", NPIX, function () {
            for (var i = 0; i < NPIX; i++) if (CACHE[i]) CACHE[i].value = "#";
        });

        timeIt("P2 .fillColor (cached)", NPIX, function () {
            for (var i = 0; i < NPIX; i++) if (CACHE[i]) CACHE[i].fillColor = grays[i % 16];
        });

        timeIt("P4 .fillColor (uncached)", NPIX, function () {
            for (var i = 0; i < NPIX; i++) {
                var f = field("px_" + i);
                if (f) f.fillColor = grays[(i + 8) % 16];
            }
        });
        print_msg("");
    }

    // Does the assignment actually show? Four variants on four visible bands.
    // Look at the page after clicking: whichever bands changed colour is the
    // answer, and the timings say what the nudge costs.
    function probeNudge() {
        print_msg("=== C2. does fillColor repaint? look at the grid ===");
        if (!CACHE) cacheFields();
        var band = Math.floor(NPIX / 4);

        function paintBand(k, label, extra) {
            var t0 = Date.now();
            for (var i = k * band; i < (k + 1) * band && i < NPIX; i++) {
                var f = CACHE[i];
                if (!f) continue;
                f.fillColor = ["G", k / 4];
                if (extra) extra(f);
            }
            print_msg("  band " + k + " " + pad(label, 22) + (Date.now() - t0) + " ms");
        }

        paintBand(0, "(a) plain", null);
        paintBand(1, "(b) + value touch", function (f) { f.value = f.value; });
        paintBand(2, "(c) + display", function (f) {
            try { f.display = display.visible; } catch (e) { }
        });
        paintBand(3, "(d) + setCaption", function (f) {
            try { f.buttonSetCaption(""); } catch (e) { }
        });
        print_msg("  four bands should be four different greys, dark to light.");
        print_msg("");
    }

    // ---------------------------------------------------------------- D
    // Liveness. A 16-step sampler wants to show progress. Whether that is
    // possible at all decides whether the PDF animates or freezes and then
    // shows a picture.

    var liveState = null;

    function probeSyncLiveness() {
        print_msg("=== D1. synchronous repaint test ===");
        print_msg("  counting to 10 with ~200ms of work between each.");
        print_msg("  if the counter animates, PDFium flushes during a script.");
        var status = field("liveStatus");
        for (var k = 1; k <= 10; k++) {
            var t0 = Date.now();
            var sink = 0;
            while (Date.now() - t0 < 200) { sink += Math.sqrt(sink + 1); }
            if (status) status.value = "sync count " + k + "/10";
        }
        print_msg("  done. did it animate, or jump straight to 10/10?");
        print_msg("");
    }

    function probeTimerLiveness() {
        print_msg("=== D2. app.setTimeOut test ===");
        if (typeof app === "undefined" || typeof app.setTimeOut !== "function") {
            print_msg("  app.setTimeOut is NOT available -- sampler must run");
            print_msg("  synchronously, or step by step from a button.");
            print_msg("");
            return;
        }
        liveState = { k: 0, timer: null };
        print_msg("  scheduling 10 chunks via app.setTimeOut...");
        pumpTimer();
    }

    // Called by name from the timer, so it has to be global.
    function pumpTimer() {
        var status = field("liveStatus");
        liveState.k++;
        var t0 = Date.now(), sink = 0;
        while (Date.now() - t0 < 150) { sink += Math.sqrt(sink + 1); }
        if (status) status.value = "timer count " + liveState.k + "/10";
        if (liveState.k < 10) {
            // Acrobat wants a string expression, and will garbage-collect the
            // timer object if nothing holds a reference to it.
            liveState.timer = app.setTimeOut("pumpTimer()", 1);
        } else {
            print_msg("  timer chunks finished -- did the counter animate?");
            print_msg("");
        }
    }

    function probeAll() {
        lines = [];
        probeCapabilities();
        probeCompute();
        probePaint();
        print_msg("all measurements done. C2 and D are separate buttons.");
    }

    print_msg("=== probe.pdf ===");
    print_msg("grid " + GRID + "x" + GRID + " = " + NPIX + " widgets");
    print_msg("");
    print_msg("Run All      capabilities + compute + paint costs");
    print_msg("Repaint?     four grey bands; says if fillColor shows");
    print_msg("Sync Live    does a long script repaint as it runs");
    print_msg("Timer Live   does app.setTimeOut exist and animate");
    print_msg("");
    print_msg("ready.");
} catch (e) { app.alert(e.stack || e); }
