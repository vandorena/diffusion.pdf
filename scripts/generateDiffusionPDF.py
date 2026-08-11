"""Build diffusion.pdf -- a denoising diffusion model inside a PDF.

Separate from generatePDF.py on purpose. The two share the PDF primitives in
pdf_helpers.py and nothing else: different template, placeholders, layout,
field topology and CLI. Merging them would put the working, video-documented
llm.pdf builder at risk every time this layout changes.

The fully-substituted JavaScript is written to --emit-js from the same string
handed to create_script, so the verification harness runs code that is
byte-identical to what ships, with no PDF parser in the loop.

    python3 scripts/generateDiffusionPDF.py
    python3 scripts/generateDiffusionPDF.py --no-pdf   # just the JS, for the harness
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_helpers import (  # noqa: E402
    PdfArray,
    PdfDict,
    PdfWriter,
    attach_acroform,
    create_action_buttons,
    create_button,
    create_field,
    create_page,
    create_script,
    create_text,
    render_template,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WIDTH = HEIGHT = 700
CONSOLE_ROWS = 12
CONSOLE_H = 11


def js_literal(value):
    """A JS literal, not a JSON string to be parsed at runtime.

    JSON almost certainly exists in this engine, but nothing in the repo
    exercises it, and emitting a literal costs nothing and needs no evidence.
    """
    return json.dumps(value, separators=(",", ":"))


def build_pixel_grid(grid, cell, x0, top, initial=None):
    """The pixel widgets, px_0..px_N, row 0 at the top.

    Borderless read-only pushbuttons. For a pushbutton the entire appearance is
    /MK/BG -- exactly what field.fillColor writes -- so repainting one is the
    cheapest thing a widget can do, and it needs no font or text layout. /H /N
    stops the click-flash across the picture; /F 4 lets it print.

    `initial` bakes a reference render into the file, so the grid shows a real
    drawing the instant it opens and the first Generate click is itself an
    end-to-end check: if the picture does not visibly change, the JS agrees
    with numpy.
    """
    fields = []
    for r in range(grid):
        for c in range(grid):
            i = r * grid + c
            shade = 1.0 if initial is None else initial[i]
            fields.append(create_button(
                f"px_{i}",
                x0 + c * cell, top - (r + 1) * cell, cell, cell,
                None,
                background=[round(shade, 4)],
                highlight="N",
                flags=4,
                read_only=True,
            ))
    return fields


def main():
    p = argparse.ArgumentParser(description="Build the diffusion PDF")
    p.add_argument("--model", default=os.path.join(ROOT, "train", "model.json"))
    p.add_argument("--template", default=os.path.join(ROOT, "src", "diffusion.js"))
    p.add_argument("--output", default=os.path.join(ROOT, "out", "diffusion.pdf"))
    p.add_argument("--emit-js", default=os.path.join(ROOT, "out", "diffusion.js"))
    p.add_argument("--no-pdf", action="store_true",
                   help="write only the JS, for fast harness iteration")
    p.add_argument("--paint-mode", choices=["gray", "chars"], default="gray")
    p.add_argument("--levels", type=int, default=16)
    p.add_argument("--paint-every", type=int, default=2)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--word", default="house")
    p.add_argument("--cell", type=int, default=None)
    p.add_argument("--bake-initial", action="store_true",
                   help="render the default word into /MK/BG at build time")
    a = p.parse_args()

    with open(a.model) as f:
        model = json.load(f)

    grid = model["grid"]
    npix = grid * grid
    steps = a.steps or model["sampler"]["steps"]
    guidance = model["guidance"]["scale"] if a.guidance is None else a.guidance

    with open(a.template) as f:
        template = f.read()

    tensors = [[n, r, c] for n, r, c in model["tensors"]]

    replacements = {
        "__GRID__": grid,
        "__LEVELS__": a.levels,
        "__CONSOLE_LINE_COUNT__": CONSOLE_ROWS,
        "__PAINT_MODE__": a.paint_mode,
        "__PAINT_EVERY__": a.paint_every,
        "__NORM_BINS__": model["norm_bins"],
        "__ABAR__": js_literal(model["abar"]),
        "__NORM__": js_literal(model["norm"]),
        "__CLASSES__": js_literal(model["classes"]),
        "__SYNONYMS__": js_literal(model.get("synonyms", {})),
        "__TENSORS__": js_literal(tensors),
        "__SCALES__": js_literal(model["s"]),
        "__BIASES__": js_literal(model["b"]),
        "__LN__": js_literal(model["ln"]),
        "__LN_EPS__": repr(model["ln_eps"]),
        "__BLOCKS__": model["blocks"],
        "__NULL_CLASS__": model["guidance"]["null_class"],
        "__STEPS_DEFAULT__": steps,
        "__GUIDANCE_DEFAULT__": guidance,
        "__SEED_DEFAULT__": a.seed,
        "__WORD_DEFAULT__": a.word,
        # last, and largest
        "__WEIGHTS_B64__": model["w"],
    }

    js = render_template(template, replacements)

    if a.emit_js:
        os.makedirs(os.path.dirname(a.emit_js), exist_ok=True)
        with open(a.emit_js, "w") as f:
            f.write(js)
        print(f"wrote {a.emit_js} ({len(js):,} chars)")

    if a.no_pdf:
        return

    initial = None
    if a.bake_initial:
        sys.path.insert(0, ROOT)
        from train import export_weights as E
        print(f"rendering {a.word!r} for the initial appearance...")
        loaded = E.load(a.model)
        img = E.sample(loaded, a.word, a.seed, steps=steps, guidance=guidance)
        # same quantisation the JS paints with, and the same inversion
        initial = []
        for v in img:
            unit = min(max((v + 1.0) * 0.5, 0.0), 1.0)
            lvl = min(int(unit * a.levels), a.levels - 1)
            initial.append((a.levels - 1 - lvl) / (a.levels - 1))

    cell = a.cell or max(4, 448 // grid)
    span = grid * cell
    x0 = (WIDTH - span) // 2
    top = HEIGHT - 34

    writer = PdfWriter()
    page = create_page(WIDTH, HEIGHT)
    page.Contents = PdfDict()
    page.Contents.stream = create_text(8, HEIGHT - 24, 15, "diffusion.pdf")
    page.Contents.stream += create_text(
        150, HEIGHT - 22, 8,
        f"{grid}x{grid} denoising diffusion, {steps} steps, no network.")

    fields = build_pixel_grid(grid, cell, x0, top, initial)

    # Character-ramp rows, hidden by default. Value writes are the update path
    # llm.pdf already depends on, so if fillColor turns out not to repaint,
    # this is the fallback that still animates.
    if a.paint_mode == "chars":
        for r in range(grid):
            fields.append(create_field(
                f"row_{r}", x0, top - (r + 1) * cell, span, cell, "",
                read_only=True))

    controls_y = top - span - 34
    labels_y = controls_y + 27

    page.Contents.stream += create_text(x0, labels_y, 8, "Word:")
    fields.append(create_field("wordInput", x0, controls_y, 150, 22, a.word))

    page.Contents.stream += create_text(x0 + 156, labels_y, 8, "Seed:")
    fields.append(create_field("seedInput", x0 + 156, controls_y, 52, 22, str(a.seed)))

    page.Contents.stream += create_text(x0 + 213, labels_y, 8, "Steps:")
    fields.append(create_field("stepsInput", x0 + 213, controls_y, 44, 22, str(steps)))

    page.Contents.stream += create_text(x0 + 262, labels_y, 8, "CFG:")
    fields.append(create_field("guidanceInput", x0 + 262, controls_y, 44, 22, str(guidance)))

    fields.extend(create_action_buttons([
        {"name": "generateButton", "x": x0 + 313, "y": controls_y,
         "width": 76, "height": 22, "label": "Generate", "js_function": "dsBegin()"},
        {"name": "stepButton", "x": x0 + 394, "y": controls_y,
         "width": 54, "height": 22, "label": "Step", "js_function": "dsStepOnce()"},
    ]))

    status_y = controls_y - 26
    fields.append(create_field("status", x0, status_y, span, 20, "ready",
                               read_only=True))

    console_top = status_y - 6
    for i in range(CONSOLE_ROWS):
        fields.append(create_field(
            f"console_{i}", 8, console_top - (i + 1) * CONSOLE_H,
            WIDTH - 16, CONSOLE_H, "", read_only=True))

    page.Contents.stream += create_text(
        8, 12, 7,
        "Runs entirely in the PDF viewer. Best in a Chromium-based browser.")

    page.AA = PdfDict()
    page.AA.O = create_script(js)
    page.Annots = PdfArray(fields)

    writer.addpage(page)
    attach_acroform(writer, fields)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    writer.write(a.output)

    size = os.path.getsize(a.output)
    macs = sum(r * c for n, r, c in tensors if n not in ("temb", "cemb"))
    print(f"wrote {a.output}")
    print(f"  {len(fields)} widgets ({npix} pixels + {len(fields) - npix} ui)")
    print(f"  {size:,} bytes ({size / 1e6:.1f} MB)")
    print(f"  {macs * steps * 2 / 1e6:.0f}M MAC per image")


if __name__ == "__main__":
    main()
