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
# Emoji are just synonyms with unusual spelling. They survive the build's
# 7-bit-ASCII assertion because json.dumps escapes them to \uXXXX surrogate
# pairs, which are ASCII source characters that JS reads back as the original
# codepoint -- so no UTF-16 fallback, and no doubled file.
#
# Keys are stored without the U+FE0F variation selector; the lookup strips it,
# so both the bare and the "emoji-presentation" form of a glyph match.
EMOJI = {
    "\U0001F34E": "apple", "\U0001F34F": "apple",
    "\U0001F355": "pizza",
    "\U0001F369": "donut",
    "\U0001F366": "ice cream", "\U0001F368": "ice cream",
    "\U0001F354": "hamburger",
    "\U0001F955": "carrot",
    "\U0001F353": "strawberry",
    "\U0001F349": "watermelon",
    "\U0001F35E": "bread",
    "\U0001F36A": "cookie",
    "\U0001F36D": "lollipop",
    "\U0001F361": "popsicle",
    "\U0001F377": "wine glass",
    "\U00002615": "coffee cup",
    "\U0001F382": "birthday cake",
    "\U0001F37A": "mug",
    "\U0001FAD6": "teapot",
    "\U0001F374": "fork",
    "\U0001F944": "spoon",
    "\U0001F52A": "knife",
    "\U0001F333": "tree", "\U0001F332": "tree",
    "\U0001F338": "flower", "\U0001F337": "flower", "\U0001F339": "flower",
    "\U0001F335": "cactus",
    "\U0001F343": "leaf", "\U0001F342": "leaf",
    "\U0001F334": "palm tree",
    "\U0001F34D": "pineapple",
    "\U00002601": "cloud", "\U0001F325": "cloud",
    "\U000026A1": "lightning", "\U0001F329": "lightning",
    "\U0001F319": "moon", "\U0001F31B": "moon",
    "\U00002B50": "star", "\U0001F31F": "star",
    "\U00002600": "sun", "\U0001F31E": "sun",
    "\U0001F308": "rainbow",
    "\U0001F32A": "tornado",
    "\U00002744": "snowflake",
    "\U0001F431": "cat", "\U0001F408": "cat", "\U0001F63A": "cat",
    "\U0001F436": "dog", "\U0001F415": "dog",
    "\U0001F426": "bird", "\U0001F424": "bird",
    "\U0001F41F": "fish", "\U0001F420": "fish",
    "\U0001F98B": "butterfly",
    "\U0001F41D": "bee",
    "\U0001F40C": "snail",
    "\U0001F40D": "snake",
    "\U0001F577": "spider",
    "\U0001F419": "octopus",
    "\U0001F980": "crab",
    "\U0001F433": "whale", "\U0001F40B": "whale",
    "\U0001F988": "shark",
    "\U0001F418": "elephant",
    "\U0001F992": "giraffe",
    "\U0001F427": "penguin",
    "\U0001F989": "owl",
    "\U0001F986": "duck",
    "\U0001F438": "frog",
    "\U0001F422": "sea turtle",
    "\U0001F6AA": "door",
    "\U0001F3E0": "house", "\U0001F3E1": "house",
    "\U0001FA91": "chair",
    "\U0001F6CF": "bed",
    "\U0001F550": "clock", "\U000023F0": "clock",
    "\U0001F4FA": "television",
    "\U0001F4DE": "telephone", "\U0000260E": "telephone",
    "\U0001F511": "key",
    "\U0001F56F": "candle",
    "\U0000231B": "hourglass", "\U000023F3": "hourglass",
    "\U00002702": "scissors",
    "\U0001F528": "hammer",
    "\U0001FA9B": "screwdriver",
    "\U0001F58C": "paintbrush",
    "\U0000270F": "pencil",
    "\U0001F9F9": "broom",
    "\U0001FAA3": "bucket",
    "\U0001FA9C": "ladder",
    "\U00002602": "umbrella", "\U0001F302": "umbrella",
    "\U0001F4D5": "book", "\U0001F4D6": "book",
    "\U00002709": "envelope", "\U0001F4E7": "envelope",
    "\U0001F4F7": "camera",
    "\U0001F3A7": "headphones",
    "\U0001F453": "eyeglasses",
    "\U0001F4A1": "light bulb",
    "\U0001F697": "car", "\U0001F699": "car",
    "\U0001F6B2": "bicycle",
    "\U00002708": "airplane",
    "\U000026F5": "sailboat",
    "\U0001F682": "train", "\U0001F686": "train",
    "\U0001F68C": "bus",
    "\U0001F681": "helicopter",
    "\U0001F6E5": "submarine",
    "\U0001F6F6": "canoe",
    "\U0001F3CD": "motorbike",
    "\U0001F3A1": "wheel",
    "\U0001F455": "t-shirt",
    "\U0001F456": "pants",
    "\U0001F45F": "shoe", "\U0001F45E": "shoe",
    "\U0001F9E6": "sock",
    "\U0001F3A9": "hat", "\U0001F452": "hat",
    "\U0001F451": "crown",
    "\U0001F5E1": "sword", "\U00002694": "sword",
    "\U0001F4FF": "necklace",
    "\U0000231A": "wristwatch",
    "\U0001F3F0": "castle",
    "\U000026EA": "church",
    "\U0001F5FC": "lighthouse",
    "\U0001F309": "bridge",
    "\U000026FA": "tent",
    "\U0001F3E2": "skyscraper",
    "\U000026F0": "mountain", "\U0001F3D4": "mountain",
    "\U0001F441": "eye", "\U0001F440": "eye",
    "\U0000270B": "hand", "\U0001F91A": "hand",
    "\U0001F9B6": "foot",
    "\U0001F442": "ear",
    "\U0001F443": "nose",
    "\U0001F9B7": "tooth",
    "\U0001F480": "skull",
    "\U0001F468": "moustache",
    "\U0001F48E": "diamond",
    "\U0001F53A": "triangle",
    "\U00002B21": "hexagon",
    "\U0001F642": "smiley face", "\U0001F600": "smiley face",
    "\U000026C4": "snowman", "\U00002603": "snowman",
    "\U0001F6A6": "traffic light",
    "\U0001F6D1": "stop sign",
    "\U0001F3B8": "guitar",
    "\U0001F3BB": "violin",
    "\U0001F3B9": "piano",
    "\U0001F941": "drums",
    "\U0001F3BA": "trumpet",
    "\U0001F3B7": "saxophone",
}

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
    "plane": "airplane", "jet": "airplane", "flying": "airplane",
    "bike": "bicycle", "cycle": "bicycle",
    "sparrow": "bird", "eagle": "bird", "robin": "bird",
    "wax": "candle", "flame": "candle",
    "king": "crown", "queen": "crown", "royal": "crown",
    "doughnut": "donut",
    "rose": "flower", "tulip": "flower", "daisy": "flower", "bloom": "flower",
    "lock": "key", "unlock": "key",
    "lunar": "moon", "crescent": "moon", "night": "moon",
    "hill": "mountain", "peak": "mountain", "everest": "mountain",
    "slice": "pizza", "pepperoni": "pizza",
    "shears": "scissors", "cut": "scissors",
    "snow": "snowman", "winter": "snowman",
    "blade": "sword", "sabre": "sword", "katana": "sword",
    "puppy": "dog", "hound": "dog", "canine": "dog",
    "kettle": "teapot", "brew": "teapot",
    "spectacles": "eyeglasses", "glasses": "eyeglasses",
    "idea": "light bulb", "lamp": "light bulb", "bulb": "light bulb",
    "boat": "sailboat", "yacht": "sailboat", "ship": "sailboat",
    "kayak": "canoe", "paddle": "canoe",
    "chopper": "helicopter",
    "motorcycle": "motorbike", "moped": "motorbike",
    "sneaker": "shoe", "boot": "shoe", "trainer": "shoe",
    "cap": "hat", "beanie": "hat",
    "palace": "castle", "fortress": "castle",
    "cathedral": "church", "chapel": "church", "temple": "church",
    "beacon": "lighthouse",
    "camping": "tent",
    "tower": "skyscraper", "office": "skyscraper",
    "smile": "smiley face", "happy": "smiley face", "face": "smiley face",
    "death": "skull", "bone": "skull", "skeleton": "skull",
    "gem": "diamond", "jewel": "diamond",
    "music": "guitar", "rock": "guitar",
    "fiddle": "violin",
    "keyboard": "piano", "keys": "piano",
    "horn": "trumpet", "brass": "trumpet",
    "sax": "saxophone", "jazz": "saxophone",
    "beach": "palm tree", "tropical": "palm tree",
    "desert": "cactus", "succulent": "cactus",
    "sunshine": "sun", "solar": "sun", "day": "sun",
    "twister": "tornado", "cyclone": "tornado",
    "ice": "snowflake", "frost": "snowflake",
    "insect": "bee", "wasp": "bee", "honey": "bee",
    "bug": "spider", "web": "spider", "arachnid": "spider",
    "squid": "octopus", "tentacles": "octopus",
    "shell": "sea turtle", "tortoise": "sea turtle",
    "burger": "hamburger", "cheeseburger": "hamburger",
    "sundae": "ice cream", "gelato": "ice cream",
    "coffee": "coffee cup", "tea": "coffee cup", "espresso": "coffee cup",
    "beer": "mug", "pint": "mug",
    "wine": "wine glass", "glass": "wine glass",
    "cake": "birthday cake", "birthday": "birthday cake",
    "biscuit": "cookie",
    "toast": "bread", "loaf": "bread",
    "nail": "hammer", "tool": "hammer",
    "sweep": "broom", "witch": "broom",
    "pail": "bucket",
    "write": "pencil", "draw": "pencil", "pen": "pencil",
    "paint": "paintbrush", "art": "paintbrush",
    "photo": "camera", "picture": "camera",
    "audio": "headphones", "listen": "headphones",
    "sand": "hourglass", "timer": "hourglass",
    "seat": "chair", "stool": "chair",
    "sleep": "bed", "mattress": "bed",
    "desk": "table",
    "tv": "television", "screen": "television",
    "phone": "telephone", "call": "telephone",
    "signal": "traffic light",
    "halt": "stop sign", "stop": "stop sign",
    "necktie": "necklace", "jewellery": "necklace",
    "shirt": "t-shirt", "tee": "t-shirt",
    "trousers": "pants", "jeans": "pants",
    "toes": "foot", "feet": "foot",
    "hearing": "ear",
    "smell": "nose",
    "dentist": "tooth", "teeth": "tooth",
    "beard": "moustache",
    "span": "bridge", "crossing": "bridge",
    "mill": "windmill",
    "ferris": "wheel", "tyre": "wheel", "tire": "wheel",
    "underwater": "submarine",
    "locomotive": "train", "railway": "train",
    "coach": "bus",
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
    p.add_argument("--steps", type=int, default=8, help="default DDIM steps")
    p.add_argument("--guidance", type=float, default=2.5)
    p.add_argument("--shift", type=float, default=None, help="schedule SNR shift")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema", type=float, default=0.999,
                   help="EMA decay for the exported weights; 0 disables")
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

    # Exponential moving average of the weights. Standard practice in
    # diffusion and close to free: SGD leaves the parameters rattling around
    # the minimum, and averaging over the tail of training lands somewhere
    # better than any single step visited. It costs nothing at inference --
    # the averaged weights are what gets exported and quantised.
    ema = ({k: v.detach().clone().float() for k, v in model.state_dict().items()}
           if a.ema > 0 else None)

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

            if ema is not None:
                # Warm up the decay so the average is not anchored to the
                # random initialisation for the first few thousand steps.
                step = epoch * per_epoch + i + 1
                d = min(a.ema, (1.0 + step) / (10.0 + step))
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

        if epoch % 5 == 0 or epoch == a.epochs - 1:
            print(f"  epoch {epoch:>3}  loss {total / per_epoch:.4f}")

    print(f"trained in {time.time() - t0:.1f}s")

    if ema is not None:
        model.load_state_dict({k: v.to(dev) for k, v in ema.items()})
        print(f"exporting EMA weights (decay {a.ema})")

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
        "synonyms": {k: v for k, v in {**SYNONYMS, **EMOJI}.items()
                     if v in categories},
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
