"""Refresh the word and emoji table in an existing model.json.

The vocabulary is metadata: it decides which class a typed word selects, and
nothing about it touches the weights. Editing it should not cost a retrain, so
this rewrites just that field in place.

    python3 -m tools.update_vocab                 # apply train_diffusion's tables
    python3 -m tools.update_vocab --show          # print the current vocabulary
"""

import argparse
import json
import os

from train.train_diffusion import EMOJI, SYNONYMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="Update the vocabulary in model.json")
    p.add_argument("--model", default=os.path.join(ROOT, "train", "model.json"))
    p.add_argument("--show", action="store_true")
    a = p.parse_args()

    with open(a.model) as f:
        model = json.load(f)
    classes = model["classes"]

    if a.show:
        print(f"{len(classes)} categories: {', '.join(classes)}\n")
        current = model.get("synonyms", {})
        words = {k: v for k, v in current.items() if k.isascii()}
        glyphs = {k: v for k, v in current.items() if not k.isascii()}
        print(f"{len(words)} words:")
        for k, v in sorted(words.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"  {k:<14} -> {v}")
        print(f"\n{len(glyphs)} emoji:")
        by_class = {}
        for k, v in glyphs.items():
            by_class.setdefault(v, []).append(k)
        for cls in sorted(by_class):
            print(f"  {cls:<12} {' '.join(by_class[cls])}")
        return

    merged = {k: v for k, v in {**SYNONYMS, **EMOJI}.items() if v in classes}
    dropped = sorted({v for v in {**SYNONYMS, **EMOJI}.values()} - set(classes))

    before = len(model.get("synonyms", {}))
    model["synonyms"] = merged
    with open(a.model, "w") as f:
        json.dump(model, f, separators=(",", ":"))

    words = sum(1 for k in merged if k.isascii())
    print(f"{a.model}: {before} -> {len(merged)} entries "
          f"({words} words, {len(merged) - words} emoji)")
    if dropped:
        print(f"  ignored (no such category in this model): {', '.join(dropped)}")
    print("  rebuild the PDF to pick this up:")
    print("    python3 scripts/generateDiffusionPDF.py --paint-mode chars "
          "--output out/diffusion-chars.pdf --emit-js out/diffusion-chars.js")


if __name__ == "__main__":
    main()
