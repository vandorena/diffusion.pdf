"""Train the denoiser. Writes train/model.json and nothing else.

Everything downstream -- the numpy reference, the JS in the document, the
verification harness -- reads only that file, so a retrain can change the
architecture without touching another line of code.

    python3 -m train.train_diffusion --grid 28 --epochs 60
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import data as Data
from . import diffusion as D
from . import export_weights as E

HERE = os.path.dirname(os.path.abspath(__file__))

# Words that should land on a category without being that category's name.
# Deliberately small: this is the honest, hand-written tier. The compressed
# embedding table that would generalise is a later step.
SYNONYMS = {
    "kitty": "cat", "kitten": "cat", "feline": "cat", "tabby": "cat",
    "auto": "car", "automobile": "car", "vehicle": "car", "truck": "car",
    "home": "house", "cottage": "house", "building": "house",
    "oak": "tree", "pine": "tree", "forest": "tree",
    "parasol": "umbrella", "brolly": "umbrella",
    "mail": "envelope", "letter": "envelope",
    "watch": "clock", "time": "clock",
    "bolt": "lightning", "thunder": "lightning", "storm": "lightning",
    "fruit": "apple", "novel": "book", "reading": "book",
    "sea": "fish", "ocean": "fish", "salmon": "fish",
    "sky": "cloud", "rain": "cloud",
    "vision": "eye", "seeing": "eye",
    "steps": "ladder", "climb": "ladder",
    "entrance": "door", "gate": "door",
}


def cond_rows(n_classes, npix, hidden, cond, blocks, T=D.T_TRAIN):
    """Rows for the class embedding: the categories, a null one for CFG, then
    however many spares make the whole weight buffer a multiple of three.

    Base64 of a length divisible by 3 carries no padding, which is the
    difference between the blob encoding exactly or picking up stray '=' bytes.
    A spare row costs `cond` bytes and buys that alignment.
    """
    fixed = T * cond + npix * hidden + hidden * cond \
        + blocks * (hidden * hidden * 2 + hidden * cond) + npix * hidden
    rows = n_classes + 1
    while (fixed + rows * cond) % 3:
        rows += 1
    return rows


class Denoiser(nn.Module):
    """Residual MLP with pre-norm blocks. ReLU everywhere, deliberately.

    ReLU is max(0, x), exact in IEEE-754 and therefore identical in numpy and
    in the viewer's JS. SiLU measured slightly better but costs tens of
    thousands of Math.exp calls per image, and exp is explicitly
    implementation-approximated -- it would trade bit-exact verification for a
    fraction of a point of sample quality. LayerNorm is fine on the same test:
    mean, variance, sqrt and divide are all correctly rounded.

    Width and depth matter more than anything else here. At 774k parameters the
    samples were noise; the same architecture at 6.4M produces line art with
    the same edge statistics as the training data.
    """

    def __init__(self, npix, hidden=768, cond=128, T=D.T_TRAIN, n_cond_rows=17,
                 blocks=4):
        super().__init__()
        self.blocks = blocks
        self.temb = nn.Embedding(T, cond)
        self.cemb = nn.Embedding(n_cond_rows, cond)
        self.inp = nn.Linear(npix, hidden)
        self.c1 = nn.Linear(cond, hidden, bias=False)
        for i in range(blocks):
            setattr(self, f"n{i}", nn.LayerNorm(hidden))
            setattr(self, f"a{i}", nn.Linear(hidden, hidden))
            setattr(self, f"b{i}", nn.Linear(hidden, hidden))
            setattr(self, f"c{i}", nn.Linear(cond, hidden, bias=False))
        self.nf = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, npix)

        nn.init.normal_(self.temb.weight, std=0.02)
        nn.init.normal_(self.cemb.weight, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, y):
        e = self.temb(t) + self.cemb(y)
        h = torch.relu(self.inp(x) + self.c1(e))
        for i in range(self.blocks):
            hn = getattr(self, f"n{i}")(h)
            inner = torch.relu(getattr(self, f"a{i}")(hn) + getattr(self, f"c{i}")(e))
            h = h + getattr(self, f"b{i}")(inner)
        return self.out(self.nf(h))


def main():
    p = argparse.ArgumentParser(description="Train the in-PDF diffusion model")
    p.add_argument("--grid", type=int, default=28)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--cond", type=int, default=128)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--per-class", type=int, default=8000)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--drop", type=float, default=0.1, help="CFG label dropout")
    p.add_argument("--steps", type=int, default=16, help="default DDIM steps")
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--shift", type=float, default=None, help="schedule SNR shift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(HERE, "model.json"))
    a = p.parse_args()

    torch.manual_seed(a.seed)
    npix = a.grid * a.grid
    # 64/grid reproduces the shift that measured best at 16x16, and scales it
    # correctly as the grid grows: a finer grid needs less correction.
    shift = a.shift if a.shift is not None else 64.0 / a.grid

    print(f"loading Quick, Draw! at {a.grid}x{a.grid}")
    images, labels, categories = Data.load(a.categories, a.per_class, a.grid, a.seed)
    n_rows = cond_rows(len(categories), npix, a.hidden, a.cond, a.blocks)
    null_class = len(categories)
    print(f"{len(images)} images, {len(categories)} categories, cemb has {n_rows} rows")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    abar = D.cosine_abar(D.T_TRAIN, shift=shift)
    abar_t = torch.tensor(abar, dtype=torch.float32, device=dev)
    X = torch.tensor(images, dtype=torch.float32, device=dev)
    Y = torch.tensor(labels, dtype=torch.long, device=dev)

    model = Denoiser(npix, a.hidden, a.cond, D.T_TRAIN, n_rows, a.blocks).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"{n_param:,} parameters, training on {dev}, snr shift {shift:.3f}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    per_epoch = max(1, len(X) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * per_epoch, pct_start=0.25
    )

    t0 = time.time()
    for epoch in range(a.epochs):
        order = torch.randperm(len(X), device=dev)
        total = 0.0
        for i in range(per_epoch):
            idx = order[i * a.batch : (i + 1) * a.batch]
            x0, y = X[idx], Y[idx].clone()

            # classifier-free guidance: some fraction of the batch is trained
            # unconditionally, so sampling can extrapolate away from it.
            y[torch.rand(len(y), device=dev) < a.drop] = null_class

            t = torch.randint(0, D.T_TRAIN, (len(x0),), device=dev)
            eps = torch.randn_like(x0)
            at = abar_t[t].unsqueeze(1)
            x_t = at.sqrt() * x0 + (1.0 - at).sqrt() * eps
            target = at.sqrt() * eps - (1.0 - at).sqrt() * x0

            loss = ((model(x_t, t, y) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()

        if epoch % 5 == 0 or epoch == a.epochs - 1:
            print(f"  epoch {epoch:>3}  loss {total / per_epoch:.4f}")

    print(f"trained in {time.time() - t0:.1f}s")

    w = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    tensors = {name: w[f"{name}.weight"] for name in E.tensor_order(a.blocks)}
    biases = {name: w[f"{name}.bias"] for name in E.biased(a.blocks)}
    norms = {name: (w[f"{name}.weight"], w[f"{name}.bias"])
             for name in E.norm_names(a.blocks)}

    saved = E.save(a.out, tensors, biases, norms, {
        "grid": a.grid, "hidden": a.hidden, "cond": a.cond, "blocks": a.blocks,
        "T": D.T_TRAIN, "abar": abar, "norm": D.norm_table(),
        "steps": a.steps, "guidance": a.guidance, "null_class": null_class,
        "classes": categories,
        "synonyms": {k: v for k, v in SYNONYMS.items() if v in categories},
    })

    # Does the exported file reproduce the network that was just trained?
    # Quantisation makes this approximate, not exact, so the check is against
    # the spread of the output rather than a fixed epsilon. Catching a layout
    # bug here is the difference between a wrong picture and a confusing one.
    reloaded = E.load(a.out)
    model.eval()
    with torch.no_grad():
        probe_x = torch.tensor(images[:1], dtype=torch.float32, device=dev)
        probe_t = torch.tensor([D.T_TRAIN // 2], device=dev)
        probe_y = torch.tensor([int(labels[0])], device=dev)
        want = model(probe_x, probe_t, probe_y)[0].cpu().numpy().astype(np.float64)
    got = E.predict_v(reloaded, images[0].astype(np.float64),
                      D.T_TRAIN // 2, int(labels[0]))
    err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-9)
    if err > 0.05:
        raise RuntimeError(
            f"exported model does not match the trained one (relative error "
            f"{err:.3f}). The tensor order or the forward pass disagree."
        )
    print(f"  export check: numpy matches torch to {err * 100:.2f}% of output range")

    n_bytes = sum(r * c for _, r, c in saved["tensors"])
    macs = n_bytes - sum(r * c for n, r, c in saved["tensors"] if n in E.EMBEDDINGS)
    size = os.path.getsize(a.out)
    print(f"wrote {a.out}")
    print(f"  {n_bytes:,} int8 weights, {len(saved['w']):,} base64 chars "
          f"({saved['w'].count('=')} padding)")
    print(f"  {macs / 1e6:.2f}M MAC per evaluation, "
          f"{macs * a.steps * 2 / 1e6:.0f}M MAC per image at {a.steps} steps with CFG")
    print(f"  {size:,} bytes on disk")


if __name__ == "__main__":
    main()
