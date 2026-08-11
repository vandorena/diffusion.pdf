"""The wire format, the quantiser, and a numpy forward pass that matches it.

This is the only file that knows how weights are laid out in the document. If
something downstream breaks after a retrain, the bug is here.

All the int8 weights go into one concatenated buffer, base64'd exactly once.
Encoding the tensors separately would pad each one to a multiple of three and
inflate the result; one buffer whose length is divisible by three encodes with
no padding at all -- which is why the class embedding is sized to make the
total land on a multiple of 3.

Scales, biases and LayerNorm parameters stay as plain JSON numbers. They are a
few percent of the file, and quantising them would cost a second dequantisation
path in the JS for nothing.
"""

import base64
import json

import numpy as np

from . import diffusion as D

LN_EPS = 1e-5  # torch.nn.LayerNorm default; the JS must use the same constant

# Tensors that are indexed rather than multiplied: bytes but no MACs.
EMBEDDINGS = ("temb", "cemb")


def tensor_order(blocks):
    """Blob order. This IS the format -- the JS slices the buffer in sequence."""
    names = ["temb", "cemb", "inp", "c1"]
    for i in range(blocks):
        names += [f"a{i}", f"b{i}", f"c{i}"]
    names.append("out")
    return names


def biased(blocks):
    """Layers carrying a bias. c1/ci are summed into another layer that has one."""
    return ["inp"] + [f"{k}{i}" for i in range(blocks) for k in ("a", "b")] + ["out"]


def norm_names(blocks):
    return [f"n{i}" for i in range(blocks)] + ["nf"]


def quantise(w):
    """Symmetric int8 with one scale per output row.

    A row is one output neuron's weight vector, which is the unit the forward
    pass consumes: y_j = scale_j * dot(w_j, x) + b_j, so the scale folds into
    the accumulator once instead of touching every weight. An embedding table
    is a Linear with a one-hot input, so a row is one embedding vector and the
    same code applies with no special case.
    """
    w = np.asarray(w, dtype=np.float64)
    peak = np.abs(w).max(axis=1)
    peak[peak == 0.0] = 1.0  # a dead row would otherwise divide by zero
    scale = peak / 127.0
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32).astype(np.float64)


def _floats(a):
    """float32 precision is plenty and keeps the JSON to a third of the size."""
    return [float(f"{float(v):.9g}") for v in np.asarray(a).ravel()]


def save(path, tensors, biases, norms, meta):
    blob, shapes, scales = [], [], {}
    for name in tensor_order(meta["blocks"]):
        q, scale = quantise(tensors[name])
        blob.append(q.reshape(-1))
        shapes.append([name, int(q.shape[0]), int(q.shape[1])])
        scales[name] = _floats(scale)

    flat = np.concatenate(blob)
    if len(flat) % 3:
        raise RuntimeError(
            f"weight buffer is {len(flat)} bytes, not a multiple of 3, so base64 "
            "would pad. Adjust the spare rows in cemb to realign."
        )

    model = {
        "kind": "diffusion-v2",
        "grid": meta["grid"],
        "hidden": meta["hidden"],
        "cond": meta["cond"],
        "blocks": meta["blocks"],
        "range": [-1, 1],
        "tensors": shapes,
        "w": base64.b64encode(flat.tobytes()).decode("ascii"),
        "s": scales,
        "b": {k: _floats(v) for k, v in biases.items()},
        "ln": {k: {"g": _floats(v[0]), "b": _floats(v[1])} for k, v in norms.items()},
        "ln_eps": LN_EPS,
        "T": meta["T"],
        "abar": _floats(meta["abar"]),
        "norm": _floats(meta["norm"]),
        "norm_bins": D.NORM_BINS,
        "pred": "v",
        "sampler": {"type": "ddim", "steps": meta["steps"], "eta": 0.0},
        "guidance": {"scale": meta["guidance"], "null_class": meta["null_class"]},
        "classes": meta["classes"],
        "synonyms": meta.get("synonyms", {}),
    }

    with open(path, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    return model


def load(path):
    """Read model.json back as dequantised float64 matrices.

    Dequantised deliberately: the preview and the reference trajectory must
    show what the document can actually produce, not what training produced.
    """
    with open(path) as f:
        model = json.load(f)

    flat = np.frombuffer(base64.b64decode(model["w"]), dtype=np.int8)
    tensors, offset = {}, 0
    for name, rows, cols in model["tensors"]:
        n = rows * cols
        q = flat[offset:offset + n].reshape(rows, cols).astype(np.float64)
        offset += n
        scale = np.asarray(model["s"][name], dtype=np.float64)
        tensors[name] = q * scale[:, None]

    if offset != len(flat):
        raise RuntimeError(f"blob has {len(flat)} bytes, tensors consume {offset}")

    model["W"] = tensors
    model["B"] = {k: np.asarray(v, dtype=np.float64) for k, v in model["b"].items()}
    model["LN"] = {
        k: (np.asarray(v["g"], dtype=np.float64), np.asarray(v["b"], dtype=np.float64))
        for k, v in model["ln"].items()
    }
    return model


def layer_norm(x, gain, bias, eps=LN_EPS):
    """Matches torch.nn.LayerNorm: biased variance, eps inside the sqrt.

    Every operation here is +, -, *, / or sqrt, all of which IEEE-754 requires
    to be correctly rounded, so this stays bit-comparable with the JS.
    """
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    return (x - mu) / np.sqrt(var + eps) * gain + bias


def predict_v(model, x, t, y):
    """One denoiser evaluation, in the same order the JS does it.

    The errstate is not hiding a numerical problem: numpy 2.0.2's vectorised
    matmul sets FP exception flags from its SIMD padding lanes on this platform
    -- `np.ones((256,784)) @ np.ones(784)` reports a divide by zero while
    returning exactly 784.0. The arithmetic agrees with a manual dot product to
    4e-16, and sample() checks the result is finite.
    """
    W, B, LN = model["W"], model["B"], model["LN"]
    blocks = model["blocks"]

    with np.errstate(all="ignore"):
        e = W["temb"][t] + W["cemb"][y]

        h = W["inp"] @ x + B["inp"] + W["c1"] @ e
        h = np.maximum(h, 0.0)

        for i in range(blocks):
            hn = layer_norm(h, *LN[f"n{i}"], model["ln_eps"])
            inner = W[f"a{i}"] @ hn + B[f"a{i}"] + W[f"c{i}"] @ e
            inner = np.maximum(inner, 0.0)
            h = h + (W[f"b{i}"] @ inner + B[f"b{i}"])

        h = layer_norm(h, *LN["nf"], model["ln_eps"])
        return W["out"] @ h + B["out"]


def sample(model, word, seed, cls=None, steps=None, guidance=None, trace=None):
    """Full guided DDIM sample. Returns the image in [-1, 1]."""
    abar = np.asarray(model["abar"], dtype=np.float64)
    table = np.asarray(model["norm"], dtype=np.float64)
    steps = steps or model["sampler"]["steps"]
    guidance = model["guidance"]["scale"] if guidance is None else guidance
    null_class = model["guidance"]["null_class"]

    if cls is None:
        cls = lookup_class(model, word)[0]

    npix = model["grid"] ** 2
    state = D.seed_for(word, seed)
    x_T = D.gaussian(npix, state, table)

    def predict(x, t):
        v_cond = predict_v(model, x, t, cls)
        if guidance == 1.0:
            return v_cond
        v_null = predict_v(model, x, t, null_class)
        return v_null + guidance * (v_cond - v_null)

    out = D.ddim_sample(predict, x_T, abar, steps, trace=trace)
    if not np.isfinite(out).all():
        raise RuntimeError(
            f"sampler produced non-finite values for {word!r} (class {cls}). "
            "The int8 weights or the schedule are wrong."
        )
    return out


def lookup_class(model, word):
    """Map a typed word to a category index.

    Returns (index, how) so the caller can be honest in the UI about which rule
    fired -- an unknown word silently drawing the wrong object is worse than
    one that says it guessed.
    """
    classes, synonyms = model["classes"], model.get("synonyms", {})

    def direct(w):
        if not w:
            return None
        if w in classes:
            return classes.index(w)
        if w in synonyms and synonyms[w] in classes:
            return classes.index(synonyms[w])
        return None

    # Emoji first, on the raw input: the normalisation below keeps only letters
    # and would delete them entirely. U+FE0F is stripped so the bare glyph and
    # its emoji-presentation form both match.
    raw = word.strip().replace("️", "")
    hit = direct(raw)
    if hit is not None:
        return hit, "emoji" if not raw.isascii() else "exact"
    for ch in raw:
        hit = direct(ch)
        if hit is not None:
            return hit, "emoji"

    clean = "".join(c for c in raw.lower() if c.isalpha() or c == " ")
    clean = " ".join(clean.split())

    candidates = [clean]
    if clean.endswith("s") and not clean.endswith("ss"):
        candidates.append(clean[:-1])
    if clean.endswith("es"):
        candidates.append(clean[:-2])
    for candidate in candidates:
        hit = direct(candidate)
        if hit is not None:
            return hit, "exact"

    for token in clean.split():
        # `or` would be wrong here: category 0 is a legitimate hit and falsy.
        hit = direct(token)
        if hit is None and token.endswith("s"):
            hit = direct(token[:-1])
        if hit is not None:
            return hit, "token"

    return D.djb2(clean) % len(classes), "hash"
