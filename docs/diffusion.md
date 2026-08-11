# A diffusion model inside a PDF

Type a word, press Generate, and a denoising diffusion model runs sixteen
sampling steps in the PDF viewer's own JavaScript engine and paints the result
onto 784 form widgets. No network, no plugin, no external file.

Every number below was measured on this machine, except where it says
otherwise. The things that still need a real viewer are collected at the end.

## What it is

| | |
|---|---|
| Model | 6,341,776 parameters, pixel-space, class-conditional |
| Weights in the document | **6,424,704 int8 bytes** → 8,566,272 base64 chars, **zero padding** |
| Grid | 28x28 = 784 widgets, Quick, Draw!'s native resolution |
| Objective | v-prediction, cosine schedule with the SNR shifted by 64/grid |
| Trained timesteps | 64 |
| Sampler | deterministic DDIM, eta = 0, 16 steps |
| Guidance | classifier-free, scale 1.5, label dropout 0.1 |
| Compute | 6.41M MAC per evaluation, **205M MAC per image** (16 steps x 2 for CFG) |
| PDF | **9.1 MB**, 803 widgets |
| Training | 418 s on an M-series GPU via MPS, 128,000 drawings, 80 epochs |

Categories: apple, book, car, cat, clock, cloud, door, envelope, eye, fish,
house, ladder, lightning, star, tree, umbrella.

## Why an iterative sampler is affordable

`idea.md` argues that "the iterative sampler is the fatal part." That is true at
Stable Diffusion's scale and false at this one, for a reason the document itself
supplies: it establishes file size as the binding constraint and compute as "a
distant third."

**An iterative sampler multiplies compute but costs zero extra file size.** The
same weights are reused at every step. Diffusion therefore spends the resource
in surplus to buy quality in the one that is scarce, which is the opposite of
the trade the document assumed it was making.

Two further corrections fell out of building it:

- **File size was not the constraint either.** `idea.md`'s 179 kB framing comes
  from a much smaller artifact. This repository ships PDFs of 18, 25 and 51 MB,
  and llm.pdf computes for roughly 50 seconds per click. Against that
  precedent, 9.1 MB and a few seconds is unremarkable.
- **The substrate fork is already closed.** `llama/llama.js` asserts
  `Math.imul` at startup and binds `Float64Array`, `Float32Array`, `Int32Array`
  and `Int8Array`. The Chromium target is a modern V8, so typed arrays are
  available and asm.js is unnecessary.

## Tuning: what is worth changing, in order

Measured with a reference classifier trained on the real drawings, scoring the
fraction of samples it assigns to the category that was asked for. That
classifier gets **0.921 on real Quick, Draw! data**, which is the number to
compare against -- not 1.0.

**Guidance is the dominant knob, and more sampling steps do not help.**

| | cfg 1.0 | cfg 1.5 | cfg 2.5 | cfg 4.0 |
|---|---|---|---|---|
| 8 steps | 0.703 | 0.891 | **0.969** | 0.969 |
| 16 steps | 0.641 | 0.891 | 0.938 | 0.969 |
| 32 steps | 0.625 | 0.859 | 0.938 | 0.969 |

Both defaults moved as a result: **8 steps, guidance 2.5**. That is better than
the old 16/1.5 on every axis at once -- more recognisable, edge statistics
closer to the real data, and half the compute in the viewer.

Two things worth understanding about that table:

- **Steps are not free quality.** Going 8 -> 32 makes samples slightly *worse*
  and slightly noisier. DDIM with a strided subset of 64 trained timesteps does
  not benefit from finer stepping here, and the extra passes only give
  quantisation error more opportunities to accumulate.
- **Past cfg 2.5 the metric lies.** Accuracy keeps climbing to 0.969 at cfg 4.0
  while the samples visibly collapse toward one prototype per class. Scoring
  only classifier accuracy would pick the worse model; sharpness and a look at
  the pictures are what stop that.

Above the real-data score the model is producing drawings that are *more*
canonical than real human doodles. At that point "accuracy" is no longer the
thing to optimise, and the remaining gains are in how the drawing looks, not
what it is.

### Training changes, measured

| change | class-accuracy | note |
|---|---|---|
| 80 epochs, 4k/class, no EMA | 0.961 +/- 0.012 | |
| **140 epochs, 6k/class, EMA 0.999** | **0.973 +/- 0.010** | current |

EMA is worth having and costs nothing at inference -- the averaged weights are
what gets quantised and shipped -- but the gain is about one point, not a
transformation. A first comparison at 96 samples per model showed EMA *losing*;
that was noise. Anything under roughly 5 points needs a few hundred samples
before it means anything.

### What the remaining errors actually are

At 256 samples per model, nearly all the misses are **categories that genuinely
collide at 28x28**, not model failures:

| category | mistaken for |
|---|---|
| mushroom | tree -- stalk under a canopy is the same silhouette |
| banana | moon -- a crescent is a crescent |
| bird | airplane, scissors -- a V-shape with two wings |
| lightning | sword, flower -- a zigzag stroke |

Two of those are pure category-selection mistakes on my part. Replacing one of
each colliding pair is a larger accuracy win than any amount of extra training,
and costs a retrain rather than a redesign.

## What actually decided sample quality

Capacity, by a wide margin. The same architecture, schedule and training recipe,
changing only width and depth:

| parameters | denoising skill at mid noise | sample sharpness vs. real data |
|---|---|---|
| 774k | 24.8% better than the class mean | **2.44x** — noise |
| 2.0M | 31.9% better | — |
| **6.4M** | **41.9% better** | **1.07x** — real line art |

At 774k parameters the samples were noise with plausible mean and variance. The
fix was not the learning rate (5e-4 measured *worse* than 2e-3) and only
partly LayerNorm (+2 points). It was size.

The metric in that table is deliberately not the training loss. A model can
predict a perfect class prototype from pure noise and still generate garbage,
because sampling feeds the model its own output — what matters is whether it can
sharpen a half-noisy image. `train/eval_samples.py` measures that directly and
prints a verdict.

### The bug that made this hard to see

The first samples were noise even though the model was fine. The x₀ prediction
at maximum noise was a clean, recognisable house; by the middle of the chain it
had dissolved.

The cause was in the sampler, not the network. DDIM's update assumes

    x_t = sqrt(abar) * x0 + sqrt(1 - abar) * eps

and the implementation clamped `x0` to [-1, 1] — which is right, and is what
bounds quantisation error — while still using the model's original `eps`. After
the clamp that pair no longer describes the state the sampler is in. The error
compounds a little at every step: the x₀ prediction started at correlation
1.000 with the correct drawing and decayed monotonically to 0.368, ending as
noise whose mean and variance still looked entirely reasonable.

`eps` is now re-derived from the clamped `x0`. See `train/diffusion.py`.

## Exactness

Every operation on the sampling path is `+`, `-`, `*`, `/` or `sqrt`. IEEE-754
requires those to be correctly rounded, so they produce identical bits in numpy
and in the viewer. `log`, `exp`, `cos` and `pow` are explicitly
implementation-approximated by the ECMAScript spec and appear nowhere on that
path. Concretely:

- **ReLU only.** `max(0, x)` is exact. SiLU measured slightly better but costs
  tens of thousands of `Math.exp` calls per image.
- **A learned timestep table**, not a sinusoidal embedding — no trigonometry,
  fewer parameters, and measured equivalent. Legal because DDIM samples a
  strided subset of the 64 trained indices.
- **xorshift32** for the noise: shifts and xors only, so there is no multiply
  that could exceed 2⁵³ and diverge between JS and Python.
- **An inverse-CDF table** for the Gaussian, not Box–Muller, which would need
  `log` and `cos`. 2049 entries from `torch.erfinv`.
- **LayerNorm is allowed**, because mean, variance, `sqrt` and divide are all
  correctly rounded.
- The schedule is computed with `cos` once, offline, and shipped as decimals.

This is why the harness can assert equality rather than defend a tolerance.

## Verification

`tools/harness.mjs` runs the JavaScript that ships inside the PDF, outside the
PDF. The builder writes the fully-substituted payload to `out/diffusion.js` from
the same string it hands to `create_script`, so the tested code is
byte-identical to the shipped code by construction — no PDF parser, no drift.

The viewer's API is stubbed just deeply enough to run: `getField` returns
recorders, `app.setTimeOut` queues expressions drained in order, and `app.alert`
is a hard failure, because the payload's outer catch calls it and that turns
every in-document exception into a non-zero exit.

Four gates, hardest first:

| gate | check | result |
|---|---|---|
| G1 | PRNG uint32 stream, 512 draws | **exact integer equality** |
| G2 | initial noise `x_T` | **bit-identical float64** |
| G3 | every intermediate `x_t`, all 16 steps | **max 1.72e-14** (tolerance 1e-9) |
| G4 | the 784 painted grey levels | **exact** |

G3 is per-step rather than end-to-end, so a divergence localises to one step and
one pixel instead of producing a single unhelpful diff.

    python3 -m train.train_diffusion          # -> train/model.json
    python3 -m tools.reference --word umbrella
    python3 scripts/generateDiffusionPDF.py --bake-initial --word umbrella
    node tools/harness.mjs

Two further checks run on their own:

- `train_diffusion.py` reloads what it just wrote and asserts the numpy forward
  pass reproduces the torch one (currently 3.3% of output range, which is the
  int8 quantisation error). A tensor-order bug fails the build.
- `tools/test_builder_identical.sh` proves the `pdf_helpers.py` refactor left
  `generatePDF.py` emitting the same bytes, using a dummy model file so it needs
  no GGUF download.

## Inside the document

The sampler is a step machine, not a loop: `dsBegin` / `dsStep` / `dsPump`,
where `dsStep` is pure and never touches the UI. The scheduler is chosen once at
page-open — `app.setTimeOut` if it exists, then `setTimeout`, then fully
synchronous — so a viewer without timers still works, just without animation.
A **Step** button advances one denoising step per click; it is a good demo in
its own right and the debugging tool for any viewer that turns out not to
repaint mid-script.

Painting is isolated in one function. The pixels are borderless read-only
pushbuttons: a pushbutton's entire appearance *is* `/MK/BG`, which is exactly
what `field.fillColor` writes, so it needs no font or text layout to regenerate.
`/H /N` stops the click-flash; `/F 4` lets it print. The build bakes a reference
render into `/MK/BG`, so the grid shows a real drawing the moment the file opens
and the first Generate click is itself an end-to-end check.

Unlike llm.pdf, the document carries an `/AcroForm`. Without one, widgets exist
only in `page.Annots`, which PDFium resolves and Acrobat generally does not.

### The honest ceiling

A typed word is matched against the category list, then a small hand-written
synonym table, then a crude singularisation, then each token in turn. Anything
unmatched picks a category by hash **and says so in the status line**. This maps
a word to the nearest of sixteen known objects; it is not open-vocabulary
generation, and the UI should not imply otherwise.

## Still unmeasured — needs a real viewer

`out/probe.pdf` exists to answer these and has to be opened by a person.

1. **Will a viewer paint 784 widgets?** 256 is the largest count previously
   demonstrated. If 784 chokes, `--grid 16` is a retrain, not a rewrite.
2. **What is the real MAC/s?** The 205M MAC per image figure is meaningful only
   against a measured rate. `idea.md`'s ~27M MAC/s came from asm.js; this is
   plain JS over typed arrays, and the gap is unknown. At 27M MAC/s an image
   takes 7.6 s; ten times slower would be 76 s.
3. **Does `field.fillColor` repaint, and does it need a nudge?** The probe tests
   four variants on four visible bands.
4. **Does anything repaint during a long synchronous script**, and does
   `app.setTimeOut` exist? This decides whether the denoising animates.

If the answers are bad, the fallback ladder is cheap and each rung is a rebuild
rather than a rewrite: fewer steps (16 → 8 halves the compute), drop CFG
(another half), narrower model, `--paint-mode chars`, `--grid 16`.

## Not done

- **Convolutions.** Far more parameter-efficient for images, and the compute
  headroom is larger than the plan assumed. A UNet comparison was started and
  stopped once the MLP result settled the architecture; a conv model might reach
  the same quality in a few hundred kB instead of 9 MB.
- **A compressed embedding table** for open-vocabulary words, `idea.md`'s
  second tier.
- **Acrobat.** The `/AcroForm` and the `getField` shim keep the path reachable,
  but it has not been tried.
