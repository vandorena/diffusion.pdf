"""Build probe.pdf -- the measurement PDF.

It carries no model. Its whole job is to replace the assumptions the diffusion
build is budgeted against with numbers from a real viewer:

  * are typed arrays / Math.imul / app.setTimeOut actually there
  * what is the real MAC/s, versus the 27M/s that idea.md inferred from asm.js
  * does assigning field.fillColor repaint, and does it need a nudge
  * what does painting a grid of widgets cost
  * will the viewer paint 784 widgets at all

    python3 scripts/generateProbePDF.py --grid 28
"""

import argparse
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
CONSOLE_ROWS = 22
CONSOLE_H = 12


def build_grid(grid, cell, x0, top):
    """The pixel widgets, flat-named px_0..px_N, row 0 at the top.

    Borderless read-only pushbuttons: for a pushbutton the whole appearance is
    /MK/BG, which is exactly what fillColor writes, and it needs no font or
    text layout to regenerate. /H /N stops the click-flash, /F 4 lets it print.
    """
    fields = []
    for r in range(grid):
        for c in range(grid):
            i = r * grid + c
            shade = 0.25 + 0.5 * ((r + c) % 2)  # checkerboard, so "did it repaint" is obvious
            fields.append(create_button(
                f"px_{i}",
                x0 + c * cell,
                top - (r + 1) * cell,
                cell, cell,
                None,
                background=[shade],
                highlight="N",
                flags=4,
                read_only=True,
            ))
    return fields


def main():
    p = argparse.ArgumentParser(description="Build the measurement PDF")
    p.add_argument("--grid", type=int, default=28,
                   help="widgets per side. 28 is what the model needs; try 32 to test the ceiling.")
    p.add_argument("--cell", type=int, default=None)
    p.add_argument("--mac-per-eval", type=int, default=729088,
                   help="MACs in one denoiser evaluation, for the budget line")
    p.add_argument("--evals", type=int, default=32,
                   help="denoiser evaluations per image (steps x 2 for CFG)")
    p.add_argument("--template", default=os.path.join(ROOT, "src", "probe.js"))
    p.add_argument("--output", default=os.path.join(ROOT, "out", "probe.pdf"))
    a = p.parse_args()

    grid = a.grid
    cell = a.cell or max(4, 420 // grid)
    span = grid * cell
    x0 = (WIDTH - span) // 2
    top = HEIGHT - 30

    with open(a.template) as f:
        template = f.read()

    js = render_template(template, {
        "__GRID__": grid,
        "__CONSOLE_LINE_COUNT__": CONSOLE_ROWS,
        "__MAC_PER_EVAL__": a.mac_per_eval,
        "__EVALS_PER_IMAGE__": a.evals,
    })

    writer = PdfWriter()
    page = create_page(WIDTH, HEIGHT)
    page.Contents = PdfDict()
    page.Contents.stream = create_text(8, HEIGHT - 20, 13, "probe.pdf")
    page.Contents.stream += create_text(
        110, HEIGHT - 20, 8,
        f"{grid}x{grid} = {grid * grid} widgets. Numbers print below.")

    fields = build_grid(grid, cell, x0, top)

    console_top = top - span - 12
    for i in range(CONSOLE_ROWS):
        fields.append(create_field(
            f"console_{i}", 8, console_top - (i + 1) * CONSOLE_H,
            WIDTH - 16, CONSOLE_H, "", read_only=True))

    status_y = console_top - CONSOLE_ROWS * CONSOLE_H - 30
    fields.append(create_field("liveStatus", 8, status_y, 200, 22, "idle",
                               read_only=True))

    buttons = [
        ("probeButton", "Run All", "probeAll()", 214),
        ("nudgeButton", "Repaint?", "probeNudge()", 310),
        ("syncButton", "Sync Live", "probeSyncLiveness()", 406),
        ("timerButton", "Timer Live", "probeTimerLiveness()", 502),
    ]
    fields.extend(create_action_buttons([
        {"name": n, "x": x, "y": status_y, "width": 92, "height": 22,
         "label": label, "js_function": fn}
        for n, label, fn, x in buttons
    ]))

    page.AA = PdfDict()
    page.AA.O = create_script(js)
    page.Annots = PdfArray(fields)

    writer.addpage(page)
    attach_acroform(writer, fields)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    writer.write(a.output)

    size = os.path.getsize(a.output)
    print(f"wrote {a.output}")
    print(f"  {len(fields)} widgets ({grid * grid} pixels + {CONSOLE_ROWS} console + 5 ui)")
    print(f"  {size:,} bytes, {len(js):,} chars of JS")


if __name__ == "__main__":
    main()
