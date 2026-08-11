# Handoff

A denoising diffusion model that runs inside a PDF. Type a word, press
Generate, and the sampler executes in the PDF viewer's own JavaScript engine,
painting a 28x28 drawing onto form widgets. No network, no plugin.

This document is for whoever picks this up next. It covers what works, what is
verified, what is *not* verified, and the handful of things that will waste your
afternoon if nobody tells you.

---

## 1. Status in one screen

| | |
|---|---|
| Model | class-conditional pixel-space diffusion, v-prediction, DDIM |
| Size | **55.4M int8 weights**, 2048 wide, 6 blocks -> **75 MB PDF** |
| Grid | 28x28 = 784 pixels, Quick, Draw!'s native resolution |
| Vocabulary | **131 categories**, 209 words + 161 emoji |
| Sampling | 8 DDIM steps, guidance 2.5, 886M MAC per image |
| Quality | **0.874 +/- 0.020** class accuracy, against a classifier scoring **0.790 on real Quick, Draw! drawings** |
| Verified | fingerprint + PRNG exact, noise bit-identical, full trajectory to 3.5e-14, painted levels exact -- on **both** builds |
| **Works today** | `out/diffusion-chars.pdf` -- renders as a character grid |
| **Broken today** | the colour grid: `fillColor` writes do not repaint in Chrome (see S6) |
| **Never run** | `out/probe.pdf` -- needs a human with a viewer (see S7) |

Samples score *above* the classifier's accuracy on real human doodles, which is
the expected effect of classifier-free guidance: the drawings come out more
canonical than the real thing. Remaining errors are dominated by categories that
genuinely collide at this resolution -- wheel/pizza, motorbike/bicycle,
key/eyeglasses.

A smaller 6.4M / 32-category model reached 0.973 against a 0.956 baseline. The
two numbers are not comparable: 131 classes is a much harder problem than 32,
and the baseline moves with it.

---

## 2. Run it

```sh
python3 -m pip install -r requirements.txt -r requirements-train.txt

python3 -m train.train_diffusion                 # -> train/model.json
python3 -m train.eval_samples                    # quality verdict + drawings

# the build that works in Chrome today
python3 scripts/generateDiffusionPDF.py --paint-mode chars \
    --output out/diffusion-chars.pdf --emit-js out/diffusion-chars.js

# the colour build (see S6 before trusting it)
python3 scripts/generateDiffusionPDF.py --bake-initial --word umbrella

# prove the JS inside the PDF matches numpy
python3 -m tools.reference --word umbrella --seed 7
node tools/harness.mjs --js out/diffusion-chars.js
```

Vocabulary edits do **not** need a retrain -- words and emoji are metadata:

```sh
# edit SYNONYMS / EMOJI in train/train_diffusion.py, then
python3 -m tools.update_vocab
python3 -m tools.update_vocab --show
python3 -m train.data --list        # all 345 Quick, Draw! categories
```

---

## 3. Where things live

```
train/
  data.py             Quick, Draw! fetch via HTTP Range; CATEGORIES lives here
  diffusion.py        schedule, q_sample, DDIM loop. THE numpy reference
  train_diffusion.py  torch training; SYNONYMS and EMOJI tables live here
  export_weights.py   quantiser + wire format. Only file that knows the layout
  eval_samples.py     denoising-skill metric, classifier accuracy, verdict
scripts/
  pdf_helpers.py      PDF primitives shared with the original llm.pdf builder
  generateDiffusionPDF.py   the builder
  generateProbePDF.py + src/probe.js   the measurement PDF
src/
  diffusion.js        everything that runs inside the document
tools/
  harness.mjs         runs the shipped JS in Node against the numpy reference
  reference.py        dumps the reference trajectory
  update_vocab.py     rewrites vocabulary in model.json without retraining
  test_builder_identical.sh   proves the pdf_helpers refactor changed no bytes
docs/diffusion.md     the full write-up, with all measured numbers
```

`out/`, `train/data/` and `train/model.json` are gitignored (9-74 MB artifacts
and a few hundred MB of cached drawings).

---

## 4. Design decisions you should not casually undo

**Nothing on the sampling path may call `log`, `exp`, `cos` or `pow`.**
IEEE-754 requires `+ - * /` and `sqrt` to be correctly rounded, so they give
identical bits in numpy and in the viewer. The transcendentals are explicitly
implementation-approximated by the ECMAScript spec and are *not* bit-identical.
This single rule is why the harness can assert equality instead of defending a
tolerance, and it is why:

- the activation is **ReLU** (SiLU measured slightly better, but costs ~41k
  `Math.exp` per image)
- timestep conditioning is a **learned 64x128 table**, not a sinusoid
- the noise comes from an **inverse-CDF table**, not Box-Muller
- the PRNG is **xorshift32** -- shifts and xors only, so no multiply can exceed
  2^53 and diverge
- the schedule is computed with `cos` **once, offline** and shipped as decimals

LayerNorm *is* allowed: mean, variance, sqrt and divide are all exact.

**The weight blob must stay a multiple of 3 bytes.** Base64 of a length
divisible by three has zero padding. `cond_rows()` adds spare class-embedding
rows until this holds; do not "simplify" it away.

**Weights stay `Int8Array` in the viewer.** The per-row scale folds into the
accumulator once. Materialising a float64 copy of 55M weights would be 440 MB of
viewer heap, and this repo already carries a commit about Chrome's PDF viewer
capping memory.

**The payload must be 7-bit ASCII.** `render_template()` asserts it. If a single
non-ASCII character sneaks in, pdfrw silently re-encodes the *entire* payload as
UTF-16 hex and doubles the file. This has already fired once in anger: a literal
U+FE0F in a regex in `src/diffusion.js`. Emoji are safe only because
`json.dumps` escapes them to `\uXXXX`, which are ASCII source characters.

---

## 5. Bugs already found and fixed -- do not reintroduce

**DDIM consistency after clipping.** The sampler clamps predicted `x0` to
[-1,1], which is correct and bounds quantisation error. It must then **re-derive
`eps` from the clamped `x0`**, because DDIM assumes
`x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps`. Using the model's original `eps` after
clipping breaks that identity and the error compounds: the x0 prediction started
at correlation 1.000 with a correct drawing and decayed to 0.368, ending as
noise whose mean and variance still looked entirely plausible. This cost hours
because every training metric looked healthy. See `train/diffusion.py`.

**Widgets must be indirect objects.** Without `f.indirect = True` before
building `/AcroForm`, pdfrw inlines every field dict into *both* `page.Annots`
and `AcroForm.Fields` -- two copies of every widget, two fields with the same
name, and a 45% larger file.

**numpy 2.0.2 emits spurious FP flags.** `np.ones((256,784)) @ np.ones(784)`
reports a divide-by-zero on this platform while returning exactly 784.0. It is a
SIMD padding-lane artifact. `predict_v` suppresses it locally and `sample()`
checks the result is finite, so real divergence is still caught. Do not
`np.seterr(all="raise")` globally and conclude the model is broken.

**The character ramp is not inverted; the grey grid is.** They look like the
same operation and are not. Ink is a high value: on a white page the grey grid
must flip it so ink paints dark, but text is already dark on white, so ink takes
the *dense* end of the ramp and background stays blank. Inverting both filled
the page with '@' and carved the drawing out in whitespace -- a negative image.
The harness replicates this formula, so `src/diffusion.js` and `tools/harness.mjs`
must be changed together or the gate certifies the bug.

**Class index 0 is falsy.** `direct(token) or fallback` silently misroutes the
first category. Use `is not None`.

**A stale reference fails G3, and it looks exactly like a sampler bug.**
`out/py_trace.json` is tied to the weights it was computed from. Rebuild the PDF
after a retrain without regenerating it and G3 reports a numeric divergence at
some step, which sends you hunting a bug that is not there. This cost real time
once. Both the payload and the reference now carry a **model fingerprint** (a
SHA-256 prefix of the weight blob) and the harness checks it first, so the
failure is now a one-line message telling you to regenerate. Order is always:

```sh
train -> python3 -m tools.reference -> build the PDF -> node tools/harness.mjs
```

---

## 6. The open bug: the colour grid does not repaint

**Symptom.** In Chrome, `out/diffusion.pdf` shows the baked-in initial drawing
(that comes from `/MK/BG`, not from script), but nothing repaints afterwards.
Clear does not blank the grid, so *no* `fillColor` write is reaching the screen.

**Already tried:** removing the ReadOnly flag from the pixel widgets. They were
`/Ff 65537` (pushbutton | ReadOnly); a pushbutton is not a data field so
ReadOnly bought nothing, and PDFium may treat a ReadOnly widget as inert. They
are now `/Ff 65536`. This did not fix it on its own.

**Still to try, in order:**

1. Open `out/diffusion.pdf` and read the console line it prints on boot:
   `widgets resolved: N/784, fillColor assignable: true|false`. This separates
   the two remaining causes -- `getField` not seeing the widgets at all, versus
   the writes being accepted but not repainted.
2. The **Nudge** button cycles `none -> caption -> display -> value` and
   redraws. If one of those works, bake it in via `--nudge`.
3. `out/probe.pdf` -> "Repaint?" paints four grey bands with the four different
   nudges. One look identifies the working one definitively.

**The workaround that already works:** `--paint-mode chars` renders the drawing
as a 28-row character grid using `.value` writes on text fields -- the exact
mechanism llm.pdf's console depends on, so it is known-good in this viewer. It
is a complete, working artifact, not a consolation prize.

All painting goes through **one function** (`paintPixel` in `src/diffusion.js`)
precisely so this is a one-line fix once the answer is known.

---

## 7. Never measured -- needs a human with a viewer

`out/probe.pdf` (176 kB) exists to answer these and has never been opened:

1. **Will a viewer paint 784 widgets?** 256 is the largest previously
   demonstrated. If 784 chokes, `--grid 16` is a retrain, not a rewrite.
2. **What is the real MAC/s in PDFium?** Measured **729M MAC/s in Node** on the
   current 55M model (1508M on the 6.4M one -- larger weights are less cache
   friendly). Both are far above idea.md's asm.js-derived 27M estimate. PDFium's
   JIT policy is unknown, so the truth is somewhere below Node. At 729M an image
   takes 1.2 s; **if PDFium is 10x slower that is 12 s, at 30x it is 36 s** --
   still inside llm.pdf's ~50 s precedent, but this is the number the whole
   75 MB budget rests on and it is unverified.
3. **Does `fillColor` repaint, and with which nudge?** See S6.
4. **Does anything repaint during a long synchronous script, and does
   `app.setTimeOut` exist?** Decides whether denoising animates or just appears.

---

## 8. The current model, and a timing warning

```sh
python3 -m train.train_diffusion --hidden 2048 --blocks 6 \
    --per-class 2500 --epochs 45
```

55.2M parameters, 131 categories, 327,500 images, final loss 0.2476. This is
what `train/model.json` holds.

**It reported 45,034 s (12.5 h) of wall clock. Benchmarks say the configuration
is about 0.9 h of compute.** The 14x gap is not explained. Ruled out by
measurement: EMA costs 1.0x, and holding the full 1 GB dataset resident on the
GPU costs ~4%. The run spanned overnight, so the most likely explanation is that
elapsed time included system sleep or idle -- `time.time()` counts both -- or
contention from other work on the same GPU. **Budget a clean run at roughly an
hour, but do not be alarmed by a much larger reported figure**, and prefer
`time.monotonic()` plus a step-rate counter if you want a trustworthy estimate.

45 epochs turned out to be enough, contrary to my initial worry: this model
reached loss 0.2810 by epoch 15, which took the 9x smaller model ~55 epochs.
Larger models here are markedly more sample-efficient per epoch.

An older 32-category model may still be at `/tmp/model_baseline.json`
(will not survive a reboot).

---

## 9. What actually moves quality, measured

In descending order of effect. Full tables in `docs/diffusion.md`.

1. **Capacity.** The one large lever. 774k params gave samples that were noise
   (2.44x sharper than real line art); 6.4M gave real line art (1.07x). Nothing
   else came close.
2. **Guidance scale.** 0.891 -> 0.969 accuracy going from cfg 1.5 to 2.5. Free,
   no retrain. Past 2.5 the metric keeps rising while samples visibly collapse
   toward one prototype -- do not optimise this number alone.
3. **Category selection.** Most remaining errors are categories that genuinely
   collide at 28x28: mushroom/tree, banana/moon, bird/airplane. Choosing
   shape-distinct categories beats more training.
4. **EMA.** +1.2 points, free at inference. Worth having, not transformative.
5. **Sampling steps: no effect, or negative.** 8 steps beat 16 and 32 on both
   accuracy and sharpness, and halve the compute. Do not assume more steps help.
6. **Learning rate: no.** 5e-4 measured *worse* than 2e-3.

**Methodology warning.** The class-accuracy metric has ~3 points of noise at 96
samples. A first EMA comparison at that size said EMA was *worse*; at 256
samples it was clearly better. Do not act on differences under ~5 points without
a few hundred samples.

---

## 9b. A known defect in the wire format

`tensor_order()` emits **`c1` twice** -- once as the input conditioning
projection, once as block 1's. The model builds blocks with
`setattr(self, f"c{i}")`, so `self.c1` is overwritten and the input projection
silently shares one weight matrix with block 1.

It is consistent across torch, numpy and JavaScript, so training works and every
gate passes. But it is an unintended weight tie, and the blob carries the same
262,144 bytes twice (verified byte-identical at offsets 1,630,848 and
18,932,352).

Fix: rename the input projection (e.g. `cin`) in `Denoiser.__init__`, its
`forward`, `tensor_order()`, and `predictV` in `src/diffusion.js`. Requires a
retrain, so it has not been done.

---

## 10. Untried, roughly by promise

- **Convolutions.** Far more parameter-efficient for images and the obvious next
  architecture. A UNet comparison was started and killed once the MLP result
  settled things. Compute headroom is much larger than originally assumed
  (S7.2), so the original reason for ruling them out no longer holds. Needs conv
  loops in `src/diffusion.js` and new ops in the wire format.
- **A compressed embedding table** for open-vocabulary words -- idea.md's second
  tier. Today an unknown word hash-picks a category and says so.
- **Acrobat.** The `/AcroForm` and the `getField` shim keep the path reachable;
  nobody has opened it there.
- **Bigger grid.** 28x28 is Quick, Draw!'s native resolution, so going larger
  adds no information without a different dataset.

---

## 11. Two claims in idea.md that are wrong

Worth knowing, because they shaped the original design and both were disproved
in this repo.

1. **"The iterative sampler is the fatal part."** True at Stable Diffusion's
   scale, false here. A sampler multiplies compute but costs **zero extra file
   size** -- the same weights are reused every step. Given that idea.md
   establishes file size as the binding constraint, diffusion spends the
   abundant resource to buy the scarce one.
2. **"File size is ceiling #1", framed around 179 kB.** That framing comes from
   a much smaller sibling artifact. This repository ships PDFs of 18, 25 and
   51 MB, and llm.pdf computes for roughly 50 s per click. The real budget was
   always one to two orders of magnitude larger than assumed.

The substrate question idea.md leaves open is also already closed:
`llama/llama.js` asserts `Math.imul` and binds `Float64Array`/`Int8Array`, so
the Chromium target is a modern V8. Typed arrays are available; asm.js is
unnecessary.
