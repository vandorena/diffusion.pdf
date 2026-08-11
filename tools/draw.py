"""Draw a word with the trained model, without a PDF in the way.

The fastest loop for judging a model: no build, no viewer, just the numpy
reference sampler printing the picture. What it renders is what the document
would render -- it runs on the dequantised int8 weights, so it shows what the
PDF can actually produce rather than what training produced.

    python3 -m tools.draw cat
    python3 -m tools.draw octopus lighthouse guitar
    python3 -m tools.draw cat --seeds 4          # variety for one word
    python3 -m tools.draw cat --cfg 1.0 2.5 5.0  # what guidance does
    python3 -m tools.draw --words                # what it knows
    python3 -m tools.draw cat --pgm out/cat.pgm
"""

import argparse
import os

from train import data as Data
from train import export_weights as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="Draw a word with the trained model")
    p.add_argument("words", nargs="*", help="words or emoji to draw")
    p.add_argument("--model", default=os.path.join(ROOT, "train", "model.json"))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seeds", type=int, default=1,
                   help="draw this many different seeds per word")
    p.add_argument("--cfg", type=float, nargs="*", default=None,
                   help="guidance scale(s); repeat to compare side by side")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--pgm", default=None, help="also write a PGM of the first drawing")
    p.add_argument("--words", dest="list_words", action="store_true",
                   help="list every category and every word/emoji that maps to one")
    a = p.parse_args()

    model = E.load(a.model)
    grid = model["grid"]

    if a.list_words:
        classes = model["classes"]
        syn = model.get("synonyms", {})
        print(f"{len(classes)} categories:\n")
        for i in range(0, len(classes), 4):
            print("  " + "".join(f"{c:<20}" for c in classes[i:i + 4]).rstrip())
        by_class = {}
        for k, v in syn.items():
            by_class.setdefault(v, []).append(k)
        print(f"\n{len(syn)} other words and emoji map onto those, e.g.")
        for cls in sorted(by_class)[:12]:
            print(f"  {cls:<14} {' '.join(sorted(by_class[cls])[:6])}")
        print("\nAnything else picks a category by hash, and says so.")
        return

    if not a.words:
        p.error("give at least one word, or --words to see the vocabulary")

    scales = a.cfg or [model["guidance"]["scale"]]
    first = None

    for word in a.words:
        cls, how = E.lookup_class(model, word)
        name = model["classes"][cls]
        note = {"exact": "", "token": "  (matched one word of it)",
                "emoji": "  (emoji)", "hash": "  <-- NOT in the vocabulary, "
                "picked by hash"}.get(how, "")
        print(f"\n=== {word!r} -> {name}{note} ===")

        for cfg in scales:
            for k in range(a.seeds):
                seed = a.seed + k * 101
                img = E.sample(model, word, seed, cls=cls,
                               steps=a.steps, guidance=cfg)
                if first is None:
                    first = img
                label = []
                if len(scales) > 1:
                    label.append(f"guidance {cfg}")
                if a.seeds > 1:
                    label.append(f"seed {seed}")
                if label:
                    print(f"-- {', '.join(label)} --")
                print(Data.to_ascii(img, grid))

    if a.pgm and first is not None:
        Data.write_pgm(a.pgm, first, grid)
        print(f"\nwrote {a.pgm}  (open with Preview, or `qlmanage -p {a.pgm}`)")


if __name__ == "__main__":
    main()
