"""Dump the reference trajectory the JS harness must reproduce.

Everything here comes from train/diffusion.py and train/export_weights.py --
the same code that trains and exports -- so the reference cannot drift away
from what the model actually is. What it emits is deliberately more than the
final image:

  prng    the raw uint32 stream, so a coercion bug is caught before it can
          hide inside a plausible-looking picture
  x_init  the starting noise on its own, so an RNG fault and a sampler fault
          do not look identical
  traj    every intermediate x_t, so a divergence localises to one step
  levels  the quantised greys actually painted, which is what a user sees

    python3 -m tools.reference --word house --seed 7
"""

import argparse
import json
import os

import numpy as np

from train import diffusion as D
from train import export_weights as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(model_path, word, seed, steps=None, guidance=None, levels=16, n_prng=512):
    model = E.load(model_path)
    grid = model["grid"]
    npix = grid * grid
    steps = steps or model["sampler"]["steps"]
    guidance = model["guidance"]["scale"] if guidance is None else guidance
    cls, how = E.lookup_class(model, word)

    state = D.seed_for(word, seed)
    gen = D.xorshift32(state)
    prng_values = [next(gen) for _ in range(n_prng)]

    table = np.asarray(model["norm"], dtype=np.float64)
    x_init = D.gaussian(npix, state, table)

    trace = []
    image = E.sample(model, word, seed, cls=cls, steps=steps,
                     guidance=guidance, trace=trace)

    # The paint step, matching the JS exactly, inversion included.
    #
    # Quick, Draw! bitmaps are ink-on-black, so after scaling to [-1,1] ink is
    # +1 and background is -1. A PDF page is white, so the level is flipped:
    # ink becomes grey 0 (black) and background becomes grey levels-1 (white).
    # The reference has to record the flipped value, because what G4 asserts is
    # the colour actually written to the widget.
    unit = np.clip((image + 1.0) * 0.5, 0.0, 1.0)
    lvl = np.clip(np.floor(unit * levels), 0, levels - 1).astype(int)
    painted = (levels - 1) - lvl

    # Fingerprint the weights the reference was computed from. Rebuilding the
    # PDF without regenerating the reference otherwise fails G3 with a numeric
    # divergence, which reads like a sampler bug rather than what it is.
    import hashlib

    with open(model_path) as f:
        blob = json.load(f)["w"]
    fingerprint = hashlib.sha256(blob.encode("ascii")).hexdigest()[:16]

    return {
        "model_id": fingerprint,
        "word": word,
        "seed": int(seed),
        "cls": int(cls),
        "lookup": how,
        "steps": int(steps),
        "guidance": float(guidance),
        "grid": grid,
        "n_levels": levels,
        "prng": {"seed": int(state), "values": prng_values},
        "x_init": [float(v) for v in x_init],
        "traj": [{"t": s["t"], "x": [float(v) for v in s["x"]]} for s in trace],
        "image": [float(v) for v in image],
        "levels": [int(v) for v in painted],
    }


def main():
    p = argparse.ArgumentParser(description="Dump the reference DDIM trajectory")
    p.add_argument("--model", default=os.path.join(ROOT, "train", "model.json"))
    p.add_argument("--word", default="house")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--levels", type=int, default=16)
    p.add_argument("--out", default=os.path.join(ROOT, "out", "py_trace.json"))
    p.add_argument("--pgm", default=None, help="also write a preview PGM")
    a = p.parse_args()

    ref = build(a.model, a.word, a.seed, a.steps, a.guidance, a.levels)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(ref, f)

    if a.pgm:
        from train import data as Data
        Data.write_pgm(a.pgm, np.asarray(ref["image"]), ref["grid"])
        print(f"wrote {a.pgm}")

    print(f"wrote {a.out}")
    print(f"  {a.word!r} -> class {ref['cls']} ({ref['lookup']}), "
          f"{ref['steps']} steps, guidance {ref['guidance']}")
    print(f"  {len(ref['traj'])} trajectory snapshots, {len(ref['levels'])} pixels")


if __name__ == "__main__":
    main()
