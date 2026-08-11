# How diffusion.pdf works

You open a PDF. You type `octopus`. You press a button drawn on the page. A few
seconds later an octopus appears, drawn one pixel at a time on the page itself.

Nothing was downloaded. There is no server, no plugin, no embedded image of an
octopus. The file contains 55 million numbers and a program, and the program
runs inside your PDF viewer.

This document explains how, from the bottom up.

---

## 1. A PDF can run code

This is the part most people do not know, and everything else rests on it.

PDF is not only a page-description format. Since Acrobat 3 it has carried a
scripting layer, and viewers implement enough of it to be interesting: a
document can declare JavaScript that runs when a page opens or when a widget is
clicked. Chrome's built-in viewer (PDFium) implements this, which is why this
project targets it.

Two mechanisms matter here.

**The page-open action.** A page object can carry an `/AA` ("additional
actions") dictionary, and its `/O` entry names an action to run when the page is
displayed. Ours is a `/JavaScript` action whose `/JS` value is the entire
program:

```
2 0 obj
<< /Type /Page
   /AA << /O << /S /JavaScript /JS ( ...75 million characters... ) >> >>
   /Annots [ ... 805 widgets ... ] >>
```

**Widget actions.** Form fields are annotations. Each has its own `/AA`, whose
`/U` entry fires on mouse-up. Ours contain one expression each, e.g.
`dsBegin()`. Those functions were defined by the page-open script, and they are
still in scope, because a document shares one JavaScript context for its whole
session. The page-open action defines the program; the buttons call into it.

The weights ride along inside that same `/JS` string as base64. There is no
separate resource, no attachment, no stream — one enormous PDF literal string.
`builds/pythia-llm.pdf` in this repository is 51 MB of exactly this, so the
approach is known to scale.

This trick is inherited from [llm.pdf](https://github.com/EvanZhouDev/llm.pdf),
which this repository is a fork of, and before that from ading2210's DoomPDF and
LinuxPDF. What is new here is what the program does.

---

## 2. What the program is

A **denoising diffusion model**: the same family as Stable Diffusion, five
orders of magnitude smaller.

The idea is simpler than its reputation. Take a drawing, add a little Gaussian
noise, then more, then more, until after enough steps nothing is left but noise.
That destruction process is fixed and known. Now train a network to undo one
step of it: given a noisy image and a number saying *how* noisy, predict what
was added.

Once you have that, you can generate. Start from pure noise — which costs
nothing, it is just random numbers — and repeatedly ask the network to take one
step back toward a clean image. After enough steps you have a drawing that never
existed.

Ours is **class-conditional**: the network also receives *which of 131 objects*
it should be denoising toward. That is what makes typing a word do anything.

### The schedule

The noise level at step `t` is described by `abar[t]`, the fraction of the
original signal still present:

```
x_t = sqrt(abar[t]) * x_0  +  sqrt(1 - abar[t]) * noise
```

`abar[0] = 0.9938` (nearly clean) down to `abar[63] = 1.9e-9` (pure noise). The
curve between them is the *cosine schedule*, with one adjustment: the standard
version is calibrated for 32x32 images and up. At a coarser grid each pixel
carries more of the picture, so a given noise level destroys less, and much of
the schedule gets spent somewhere trivially easy. We divide the signal-to-noise
ratio by `(64/grid)^2`, which shifts the whole schedule toward noisier. On
16x16 that single line moved class accuracy from 0.978 to 0.995.

There are **64 trained noise levels**, not the usual 1000. Fewer levels means
the timestep conditioning can be a lookup table (§4), and the shipped schedule
is 64 numbers instead of a thousand.

### v-prediction

The network could predict the noise that was added (`eps`), the clean image
(`x0`), or a particular blend of the two called `v`:

```
v = sqrt(abar) * eps  -  sqrt(1 - abar) * x0
```

We predict `v`, and it is not a close call — measured on an identical network
and seed, class accuracy was **0.978 for v against 0.716 for eps**.

The reason is worth understanding. At high noise `x_t` is nearly pure noise, so
`eps` is nearly `x_t`: predicting it is almost the identity function, and the
model burns capacity learning to copy its input. Then recovering the image
requires `x0 = (x_t - sqrt(1-abar)*eps) / sqrt(abar)`, dividing by a number
approaching zero and amplifying whatever error remains. `v` has bounded variance
across the whole schedule, and recovering `x0` from it is a blend with no
division at all.

### Sampling: DDIM

Generation runs **8 steps of DDIM with eta = 0** — fully deterministic. Each
step:

1. ask the network for `v` at the current noise level
2. derive the implied clean image `x0`, and **clamp it to [-1, 1]**
3. re-derive the noise `eps` **from the clamped `x0`**
4. re-noise to the next, lower level

Step 3 is not optional, and getting it wrong is the single worst bug this
project had. DDIM's update assumes `x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps`.
Clamping `x0` while keeping the model's original `eps` breaks that identity —
the pair no longer describes the state the sampler is actually in — and the
error compounds. The x0 prediction began at correlation 1.000 with a correct
drawing and decayed to 0.368, producing noise whose mean and variance still
looked entirely reasonable. Every training metric was healthy throughout.

Determinism matters beyond reproducibility: it is what lets the whole trajectory
be checked against a reference implementation (§8).

### Classifier-free guidance

Each step actually runs the network **twice**: once told the class, once told
nothing (a dedicated null class the model was trained on 10% of the time). The
two are then extrapolated apart:

```
v = v_null + w * (v_class - v_null)
```

`w = 2.5`. This doubles the compute and is worth it: it moves class accuracy
from 0.703 to 0.969. Past about 2.5 the metric keeps climbing while the drawings
visibly collapse toward a single prototype per class — the accuracy number alone
would pick a worse model.

---

## 3. The constraint that shapes everything

The program has to produce **the same numbers in the PDF viewer as in numpy**,
or there is no way to know it is correct.

IEEE-754 requires `+`, `-`, `*`, `/` and `sqrt` to be *correctly rounded*: given
the same inputs, every conforming implementation returns bit-identical results.
It requires nothing of the sort for `log`, `exp`, `cos`, `pow` or `tanh` — the
ECMAScript spec explicitly permits implementation-defined approximations, and
V8's differ from numpy's in the last bits.

So: **nothing on the sampling path may call a transcendental function.** That
one rule explains most of the model's more unusual choices.

| choice | why |
|---|---|
| **ReLU** everywhere | `max(0,x)` is exact. SiLU measured slightly better but needs ~41,000 `Math.exp` calls per image |
| **Learned timestep table**, not sinusoidal embeddings | no `sin`/`cos`. Cheaper too, and legal because DDIM only ever uses the 64 trained indices |
| **Inverse-CDF table** for Gaussian noise | Box–Muller needs `log` and `cos`. A 2049-entry table plus linear interpolation is index arithmetic and one multiply-add |
| **xorshift32** for randomness | shifts and xors only. No multiply that could exceed 2^53 and diverge between JS numbers and Python integers |
| **Schedule shipped as decimals** | `cos` is called once, offline; the 64 resulting values are emitted as decimal literals that both languages parse identically |

**LayerNorm is allowed**, which surprises people: mean, variance, `sqrt` and
divide are all correctly rounded, so it is exact.

The payoff is that verification asserts *equality*, not a tolerance. The one
place a tolerance survives is the trajectory comparison, and only because numpy
uses BLAS whose summation order differs from a naive JS loop. Observed:
**3.5e-14** over a full 8-step trajectory.

---

## 4. The network

A residual MLP, 2048 wide, six blocks. No convolutions (see §10).

```
t (0..63)  ──▶ temb[t] ─┐
                        ├──▶ e (128)        two table lookups, zero multiplies
class      ──▶ cemb[y] ─┘

x_t (784) ──▶ inp ──┐
                    ⊕ ──▶ ReLU ──▶ h (2048)
           e ──▶ c1 ┘

    ┌── six times ────────────────────────────────┐
    │  hn = LayerNorm(h)                          │
    │  u  = ReLU( a_i(hn) + c_i(e) )              │
    │  h  = h + b_i(u)                            │
    └─────────────────────────────────────────────┘

h ──▶ LayerNorm ──▶ out ──▶ v̂ (784)
```

The class and timestep are summed into a single 128-dimensional conditioning
vector, which is projected into the trunk at the input **and again inside every
block**. Injecting it once is not enough: with only one entry point the class
signal has to survive six layers, and high guidance scales produce artifacts.

Both embeddings are lookups, not matrix multiplies, so they cost bytes but no
arithmetic.

| tensor | shape | bytes |
|---|---|---|
| `temb` | 64 x 128 | 8,192 |
| `cemb` | 133 x 128 | 17,024 |
| `inp` | 2048 x 784 | 1,605,632 |
| `a_i`, `b_i` (x6 each) | 2048 x 2048 | 50,331,648 |
| `c_i` (x6) + input `c1` | 2048 x 128 | 1,835,008 |
| `out` | 784 x 2048 | 1,605,632 |
| **total** | | **55,403,136** |

The grid is **28x28**, which is Quick, Draw!'s native resolution. That is
deliberate: the dataset ships 784-byte bitmaps, so at this size there is no
resampling at all. Earlier attempts at 16x16 had to downsample, and a 1-pixel
doodle stroke either averages into grey mush or barely survives — most of the
"the model can't draw" problem turned out to be a resampling artifact.

---

## 5. Getting 55 million weights into a text file

Weights are quantised to **int8, symmetric, with one scale per output row**:

```
scale_j = max|W[j,:]| / 127
W_int8[j,k] = round(W[j,k] / scale_j)
```

A row is one output neuron's weight vector, which is exactly the unit the
forward pass consumes — `y_j = scale_j * dot(W[j], x) + b_j` — so the scale
folds into the accumulator once instead of touching every weight. An embedding
table is a Linear with a one-hot input, so a row is one embedding vector and the
identical code applies with no special case.

Quantisation error was the risk I expected to bite hardest, because a diffusion
sampler feeds its own output back in eight times, so errors have the opportunity
to compound in a way they cannot in a single forward pass. Measured on a smaller
model, it does not: FD 5.87 float32 against 5.89 int8, class accuracy unchanged.
Three reasons — DDIM is contractive toward the data manifold, the x0 clamp is a
hard bound on per-step drift, and v-prediction has no `1/sqrt(abar)` division to
amplify anything.

Every tensor goes into **one concatenated byte buffer, base64'd exactly once**.
This matters: base64 pads to a multiple of three, so encoding tensors separately
would sprinkle padding through the file. One buffer whose length is divisible by
three encodes with **zero padding**. The class embedding is sized to make that
true — `cond_rows()` adds spare rows until `total % 3 == 0`, which is why
`cemb` has 133 rows for 131 categories plus one null class.

Scales, biases and LayerNorm parameters stay as plain JSON numbers. They are a
few percent of the file, and quantising them would buy a second dequantisation
path in the JavaScript for nothing.

### The ASCII rule

The payload must be **7-bit ASCII**, and the builder asserts it.

pdfrw writes a Python string as a PDF literal `(...)` when it can. If any
character is not encodable in PDFDocEncoding it silently falls back to
**UTF-16 hex**, which doubles the file. A single smart quote in a comment would
turn a 75 MB PDF into 150 MB with no error message.

This fired once for real: a literal U+FE0F variation selector inside a regex.
Emoji in the vocabulary are safe only because `json.dumps` escapes them to
`\uXXXX` — ASCII source characters that JavaScript reads back as the original
codepoint.

---

## 6. What happens when you click Generate

**At page open**, once:

1. the base64 blob is decoded to an `Int8Array` by a hand-rolled decoder — `atob`
   is not guaranteed in a PDF sandbox
2. the tensor list is walked and each name bound to a **subarray view** of that
   one buffer, so nothing is copied
3. scratch buffers are allocated once and reused for every step of every run
4. the scheduler is chosen by feature detection (below)

Weights stay `Int8Array` for the life of the document. Materialising a float64
copy of 55M weights would be 440 MB of viewer heap, and this repository already
carries a commit titled *"Reduce memory use slightly to prevent Chrome memory
capping"*.

**On click:**

1. the typed word is resolved to a class (§7)
2. the word and the seed field are hashed into a 32-bit PRNG state, and 784
   Gaussian samples are drawn — this is `x_T`, the starting noise
3. eight DDIM steps run, each evaluating the network twice for guidance
4. the result is painted

**The seed field ships holding `auto`**, which draws a fresh seed from the
clock — mixed through djb2, because xorshift32 correlates in its first few
outputs for nearby seeds and `x_T` only draws 784 of them. One word producing
one drawing forever reads as a lookup table rather than a model, which is why
this is the default rather than an option.

That clock read is the *only* thing in the document that varies between runs,
and it is deliberately confined to the one place it can be. The sampler stays
bit-deterministic. `auto` leaves the word in the field rather than writing the
number back — otherwise the next click would reuse it and the variation would
stop after one press — and the number it picked goes to the console, so a
drawing you liked is still reproducible: type the number in. Any integer in the
box restores the old behaviour exactly.

The harness calls `dsTrajectory`, `dsInitialNoise` and `dsPrngStream` with an
explicit seed and never reads a field, so all five gates in §9 are untouched.
The one thing this does cost is the free end-to-end check in §7, which now
needs the baked seed typed in first.

Measured, because the §2 note on guidance predicts otherwise: at guidance 2.5
three seeds of `cat` give three visibly different animals, mean absolute
difference 0.21 over [-1, 1]. Diversity survives at the shipped scale, so the
seed alone is enough and the sampler does not need `eta > 0`.

### The sampler is a state machine, not a loop

`dsStep()` performs exactly one denoising step and never touches the UI.
`dsPump()` decides how to keep going. That separation exists because a PDF
viewer's threading model is not something you get to assume:

| mode | condition | behaviour |
|---|---|---|
| `TIMER_ACRO` | `app.setTimeOut` exists | run a ~50 ms chunk, paint, reschedule |
| `TIMER_DOM` | `setTimeout` exists | same, via the DOM timer |
| `SYNC` | neither | run straight through, painting every Nth step |

Acrobat's `setTimeOut` takes a *string expression*, not a function, and will
garbage-collect a timer object nobody holds a reference to — so the handle is
kept in a global and the callback is passed as `"dsPump()"`.

There is also a **Step** button that advances exactly one denoising step per
click. It is the fallback if no scheduler yields to the compositor, but it is
also a better demo than an animation: you watch noise resolve into a drawing at
your own pace, and it is the debugging tool you want anyway.

---

## 7. Painting: the picture is 784 buttons

The drawing surface is not an image. PDF has no way for a script to draw
arbitrary pixels. What it has is **form widgets**, and a widget's background
colour is scriptable.

So the page carries **784 borderless pushbuttons** in a 28x28 grid. A
pushbutton's entire appearance is its `/MK/BG` entry — no font, no text layout,
no value string — which makes it the cheapest widget to repaint. `/H /N`
suppresses the click-flash so the picture does not invert under the cursor;
`/F 4` lets it print.

The build **bakes a reference render into `/MK/BG`**, so the grid shows a real
drawing the instant the file opens, before any script runs. That gives a free
end-to-end check: type `7` into the seed box — the seed the appearance was
baked from — click Generate, and if the picture does not visibly change, the
JavaScript agrees with numpy. It used to need no typing at all, because the
seed box shipped holding that number; the box now defaults to `auto` (§6), so
the first click is *supposed* to draw something else and the check costs one
extra keystroke.

Painting is isolated in exactly one function, because whether assigning
`fillColor` actually *repaints* is viewer-dependent and, in Chrome, currently
does not work (§9). Everything else — dirty-pixel skipping, painting every Nth
step, the runtime nudge cycler — exists to keep that one function cheap and
patchable.

### The character fallback

`--paint-mode chars` renders the drawing as 28 rows of text instead, two
characters per pixel through the ramp `" .:-=+*#%@"`. Courier advances 0.6 em,
so doubling each character makes the result roughly square.

This is not a consolation prize. Writing `.value` on a text field is the update
path llm.pdf's entire console depends on, so it is *known* to work in this
viewer — which is why it is the build that currently works end to end.

One trap here, which bit: the grey grid **inverts** and the character ramp must
**not**. Ink is a high value; on a white page the grey grid has to flip it so
ink paints dark. But text is already dark on white, so ink takes the *dense* end
of the ramp and background stays blank. Inverting both produced a negative
image — a page full of `@` with the drawing carved out in whitespace.

---

## 8. From a word to a category

The model knows 131 objects. A typed word has to become one of them.

1. strip the U+FE0F variation selector, then check the raw input against the
   category list and a hand-written table of **209 words and 161 emoji**
2. if that fails, walk the string by codepoint looking for a known emoji —
   surrogate pairs are two UTF-16 units, so this steps by codepoint, not index
3. normalise to letters and spaces, retry, then try crude singularisation
   (`cats` → `cat`, `boxes` → `box`)
4. try each token in turn, so `a big red car` finds `car`
5. otherwise pick a category by hashing the word — **and say so in the status
   line**

That last point is a deliberate honesty constraint. The real ceiling here is
*"map the word to the nearest of 131 known objects and draw it"* — this is not
open-vocabulary generation, and the interface should not imply otherwise. Typing
`dinosaur` gets you a bus, and the document tells you it guessed.

The vocabulary is metadata, not weights, so `tools/update_vocab.py` rewrites it
in an existing model file without retraining.

**All 131 categories are printed on page 2 of the document.** They used to be a
single `known words:` line in the console, which at 131 categories is about
1,100 characters in a field 684pt wide and 11pt tall: silently truncated to the
first handful, and eating a twelfth of a twelve-row scrollback to do it. Page 2
is static text, so it costs no widgets, survives printing, and cannot scroll
away.

---

## 9. How we know it is right

The builder writes the fully-substituted JavaScript to `out/diffusion.js` **from
the same Python string it hands to the PDF**. The tested code is therefore
byte-identical to the shipped code by construction — no PDF parser, no
possibility of drift.

`tools/harness.mjs` runs that file in Node under `vm.createContext`, with the
viewer's API stubbed just deeply enough: `getField` returns recorders that log
every write, `app.setTimeOut` queues expressions drained in order, and
`app.alert` is a **hard failure** — the payload's outer catch calls it, so every
in-document exception surfaces as a non-zero exit.

Five gates, hardest first:

| gate | check | result |
|---|---|---|
| fingerprint | payload and reference are the same model | SHA-256 prefix |
| G1 | PRNG stream, 512 draws | **exact integer equality** |
| G2 | initial noise `x_T` | **bit-identical float64** |
| G3 | every intermediate `x_t`, all 8 steps | **3.5e-14** (tolerance 1e-9) |
| G4 | the painted greys / character rows | **exact** |

G3 is per-step rather than end-to-end so a divergence localises to one step and
one pixel instead of producing a single unhelpful diff.

The fingerprint gate exists because of a real failure: rebuilding the PDF after
a retrain without regenerating the reference makes G3 report a numeric
divergence, which reads exactly like a sampler bug and sends you hunting one.

**What this does not prove.** The harness reimplements the character-ramp
formula in order to check it, so for a while both copies were inverted together
and G4 passed on a wrong rendering. A gate that reimplements what it checks only
proves the two copies agree. That is now cross-checked against the independent
Python renderer.

Two further checks run outside the harness: training reloads what it just wrote
and asserts the numpy forward pass reproduces the torch one (a tensor-order bug
fails the build), and `tools/test_builder_identical.sh` proves the shared-helper
refactor left the original llm.pdf builder emitting identical bytes.

---

## 10. Things that are wrong or unfinished

**A tensor name collision.** `tensor_order()` emits `c1` twice — once as the
input conditioning projection, once as block 1's. Because the model builds
blocks with `setattr(self, f"c{i}")`, `self.c1` is overwritten, so the input
projection and block 1 **silently share one weight matrix**. It is consistent
across torch, numpy and JavaScript, so everything trains and every gate passes,
but it is an unintended weight tie and it writes 262,144 bytes into the blob
twice. Fixing it means renaming the input projection and retraining.

**The colour grid does not repaint in Chrome.** `fillColor` writes are accepted
and never reach the screen. The ReadOnly flag has been ruled out. The character
build works and is what ships.

**The compute budget is unverified.** Everything is sized against 729M MAC/s
measured in Node, where the JIT is fully on. PDFium's policy is unknown. At that
rate an image takes 1.2 s; ten times slower is 12 s, thirty times is 36 s — all
survivable against llm.pdf's ~50 s precedent, but nobody has measured it.
`out/probe.pdf` exists to answer this and has never been run.

**Convolutions were never tried properly.** They are far more parameter-efficient
for images and are the obvious next architecture. They were ruled out early on a
compute argument that later measurement invalidated — the real budget turned out
to be one to two orders of magnitude larger than assumed. A comparison was
started and killed once the MLP result settled the immediate question.

---

## 11. Two things the original plan got wrong

This project began from `idea.md`, which argued that a diffusion model in a PDF
was the wrong shape. Two of its load-bearing claims turned out to be false, and
the corrections are what made this work.

**"The iterative sampler is the fatal part."** True at Stable Diffusion's scale.
False here, for a reason the document itself supplies: it establishes file size
as the binding constraint. An iterative sampler multiplies *compute* but costs
**zero extra file size** — the same weights are reused at every step. Diffusion
spends the abundant resource to buy the scarce one, which is the opposite of the
trade it was assumed to be making.

**"File size is ceiling #1," framed around 179 kB.** That framing came from a
much smaller sibling artifact. This repository ships PDFs of 18, 25 and 51 MB,
and llm.pdf computes for roughly 50 seconds per click. The budget was always one
to two orders of magnitude larger than assumed — which is how a 75 MB model
became reasonable.

Its open question about the execution substrate was also already answered, in
the repository: `llama/llama.js` asserts `Math.imul` at startup and binds
`Float64Array`, `Float32Array`, `Int32Array` and `Int8Array`. The Chromium
target is a modern V8. Typed arrays are available and asm.js is unnecessary.

---

## 12. The numbers

| | |
|---|---|
| Parameters | 55,403,136 int8 |
| Base64 | 73,870,848 characters, zero padding |
| PDF | 75 MB |
| Vocabulary | 131 categories, 209 words, 161 emoji |
| Grid | 28x28 = 784 widgets |
| Sampling | 8 DDIM steps, guidance 2.5 |
| Compute | 886M MAC per image |
| Speed | 1.2 s per image in Node (729M MAC/s) |
| Quality | 0.874 class accuracy, against 0.790 for a classifier on real drawings |
| Training | 45 epochs, 327,500 drawings, final loss 0.2476 |
| Agreement with numpy | 3.5e-14 over the full trajectory |

The quality figure deserves a note: samples score *higher* than the classifier
manages on real human doodles. That is not the model beating reality, it is what
classifier-free guidance does — the drawings come out more canonical than the
real thing, which is the same knob that destroys diversity if you push it
further.
