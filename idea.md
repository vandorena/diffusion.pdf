# Idea: a generative image model that runs inside a PDF

Inspired by [LinuxPDF](https://github.com/ading2210/linuxpdf), which runs a full Linux
emulator inside a PDF, and by [llm.pdf](https://github.com/EvanZhouDev/llm.pdf), which
runs a real language model inside one. This is the image version: type a word, and a
decoder embedded in the document turns it into a picture, in the viewer's own
JavaScript engine, with no network of any kind.

The counterpart to [slop.pdf](docs/http-demo.md), which cannot produce a single
character without a relay holding an API key.

## What the first draft of this file got wrong

The original argument was: an LLM in a PDF is impractical, because total cost is
`(cost per forward pass) x (number of tokens)` and PDF JS engines run with the JIT
off, so an image decoder's single forward pass is the only tractable option.

**llm.pdf disproves the "impractical" half.** It compiles llama.cpp to asm.js with
Emscripten, base64s a Q8 GGUF model into the document, and runs it — around
**135M parameters at roughly 5 seconds per token**.

The *reasoning* survives: 5s/token means a sentence takes minutes, and the token loop
is still a multiplier we do not have to pay. One pass per image remains the right
shape for this. But "impossible" was wrong, and the correction matters, because
llm.pdf hands us the number this project was missing.

## The number that changes everything

A dense transformer costs about 2 FLOP per parameter per token. 135M parameters at
5 s/token is therefore roughly:

```
2 x 135e6 FLOP / 5 s  ~=  54 MFLOP/s  ~=  27M multiply-accumulates per second
```

Order of magnitude, via asm.js over typed arrays. Against that budget:

| | MACs per image | at ~27M MAC/s |
|---|---|---|
| our decoder today | 73,728 | ~3 ms |
| a 10x bigger decoder | ~700k | ~30 ms |
| a 20M parameter 1-step model | ~20M | ~1 s |
| SD-Turbo, 1 step, 860M params | ~340 G | ~3.5 hours |
| Stable Diffusion, 20 steps | ~14 T | ~70 hours |

**Compute was never our constraint.** We are using something like 0.3% of the
available budget. The real ceilings are, in order:

1. **File size.** Weights go into the document as base64: about **1.4 bytes of PDF per
   int8 parameter**. A 10 MB PDF buys ~7M parameters; 50 MB buys ~35M. llm.pdf's 135M
   Q8 model implies a ~180 MB file, which is the honest upper bound of what anyone has
   shown a viewer will swallow.
2. **Field count.** The display is form widgets, and how many a viewer will paint is
   still unmeasured. 16x16 is 256 and works; 28x28 is 784 and is untested.
3. **Compute**, a distant third.

## Status

Working and verified end to end:

- `train/train_pca.py` (numpy, closed form) then `train/train_vae.py` (torch, 14
  seconds) → `train/model.json`, a 32 → 256 → 256 decoder, int8 with per-output-row
  scales, 73,728 weights, 98,304 characters of base64.
- `imagemodel.js` decodes that in the viewer, folds the scales in once at open, and
  paints 256 widgets through `field.fillColor`. `--mode chars` is the fallback.
- `out/imagepdf.pdf`: **179 kB, 259 fields, no network at all.**
- `tools/bench_forward.mjs` runs the JavaScript **extracted back out of the built PDF**
  and checks it against a numpy reference: forward pass agrees to **3.3e-16**, the
  text→latent hash to **5.6e-17**, and the painted widgets reproduce
  `out/preview.png` exactly.

Full detail in [docs/image-model.md](docs/image-model.md).

**The known weakness: the text is decorative.** `hash_seed("brown")` throws away every
bit of meaning and returns a stable-but-arbitrary latent. `brown` renders a `0` by
accident. The model has never seen a word, and MNIST contains nothing but digits, so
there is no object for a word to resemble even in principle.

## Where this goes: Stable Diffusion's interface, not its architecture

The goal is arbitrary text in, a picture of that thing out. SD's actual architecture
is out of reach — the table above puts a real one at ~70 hours per image — but its
three parts fail very differently:

- **The iterative sampler is the fatal part.** Denoising over ~20 steps is the same
  multiplier that makes text painful, applied to a far larger network.
- **The text encoder runs once**, not per step. Right shape, wrong size.
- **The VAE decoder is a single forward pass** — architecturally the thing we already
  have running.

Distillation collapses the fatal part: consistency models and adversarial distillation
reach 4, 2, and 1 step. **A 1-step diffusion model is not iterative — it is a
conditional decoder**, which is exactly what fits here. So: keep the interface, drop
the sampler, and shrink the network by five orders of magnitude.

### Content: Quick, Draw!

MNIST cannot participate in this. [Quick, Draw!](https://quickdraw.withgoogle.com/data)
is 345 categories of 28x28 doodles of actual objects — cat, car, tree, house, apple,
guitar — and doodle line-art survives a coarse grid far better than photographs would.

Confirmed reachable at
`https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<category>.npy`.
The `.npy` bitmap format is a header followed by raw 784-byte rows, so an HTTP **Range
request pulls only the first N drawings** — ~1.6 MB per class instead of ~100 MB.

### Meaning: where word understanding comes from

SD's power is CLIP, trained on 400M image-text pairs. We cannot train that, and we
cannot ship a transformer to run it. We can **distill it offline**: precompute
embeddings for a vocabulary with a real model, compress them to int8, and ship a
lookup table. That is how "tiger" lands near "cat" without any text encoder in the
document.

Two honest tiers:

- **Names and synonyms** — a hand-written table, ~10 kB, unknown words fall back to
  today's hash. Demonstrable immediately, no generalisation.
- **A compressed embedding table** — ~6k words at int8, ~250 kB, nearest category by
  cosine. Any common noun lands somewhere sensible.

The ceiling to be honest about either way: this is *"map the word to the nearest of N
known objects and draw it"*, not open-vocabulary generation.

## The execution substrate fork

This is the decision the numbers above force, and it is not yet made:

| | headroom | cost |
|---|---|---|
| plain JS, plain arrays (today) | unmeasured, certainly the slowest | none — works, 179 kB, debuggable |
| plain JS, typed arrays | likely a large multiple, for a small change | needs `Float64Array` to exist |
| Emscripten → asm.js, the llm.pdf route | the full ~27M MAC/s, **and convolutions become viable** | a C toolchain and much larger files |

Convolutions were ruled out in the first draft as too slow without a JIT. llm.pdf
shows the asm.js substrate carries them, and they are dramatically more
parameter-efficient for images — which matters enormously when **file size is the
ceiling**.

## Open questions

- **What does `out/probe.pdf` say?** Still unanswered, and now the most valuable
  measurement in the project: it times a 73,728-MAC pass in the real engine. Dividing
  llm.pdf's ~27M MAC/s by that number gives the exact payoff for moving to typed
  arrays or asm.js — the difference between budgeting 70k parameters and 7M.
- Does `field.fillColor` assign *and repaint*, in which viewers, and with what nudge?
- How many widgets will a viewer paint before it chokes? 256 works. Is 784?
- How large a PDF is acceptable? This sets the parameter count directly.
- Is a 100 kB+ string literal safe in every viewer? The probe tests 120,000 characters.

## Plan

1. Run the probe in Acrobat, Chrome and Firefox. Record the timings. **Everything
   below is budgeted off that number.**
2. Pick the substrate: typed arrays if they are fast enough, asm.js only if they are
   not. Keep the hand-written path if it fits — the current one is 179 kB and fully
   verified.
3. Swap MNIST for Quick, Draw! via Range requests, at whatever grid the field-count
   answer allows.
4. Train a **conditional** decoder: class in, image out. This alone is the difference
   between a word seeding noise and a word choosing an object.
5. Ship names-and-synonyms lookup, then the compressed embedding table.
6. Grow the decoder toward the file-size ceiling — and revisit convolutions if the
   substrate is asm.js.

Steps 3 through 6 each replace `train/model.json` and nothing else. The format, the
quantiser, the paint path and the verification harness were built to survive exactly
this.
