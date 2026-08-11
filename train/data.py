"""Quick, Draw! bitmaps, fetched a few megabytes at a time.

The full numpy_bitmap files are ~100 MB per category, but they are raw uint8
rows behind a short header, so an HTTP Range request pulls only the first N
drawings. Sixteen categories at 8000 drawings each is ~100 MB of source data
reduced to ~100 MB / 12.

Nothing here ships. The PDF carries the exported weights and nothing else.
"""

import gzip
import os
import struct
import urllib.error
import urllib.request

import numpy as np

URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

NATIVE = 28  # quickdraw bitmaps are 28x28. at --grid 28 there is no resampling.
ROW_BYTES = NATIVE * NATIVE

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Chosen for silhouettes that survive a coarse grid and read differently from
# each other. Curated in eval_samples.py against what actually renders.
CATEGORIES = [
    "apple", "book", "car", "cat",
    "clock", "cloud", "door", "envelope",
    "eye", "fish", "house", "ladder",
    "lightning", "star", "tree", "umbrella",
]


def _read_npy_header(url):
    """Return (data_offset, n_rows) by pulling only the first 128 bytes.

    .npy is a 6-byte magic, 2 version bytes, a little-endian uint16 header
    length, then that many bytes of a Python dict literal. The dict carries
    the row count, which we need to know how many drawings actually exist.
    """
    head = _range_get(url, 0, 127)
    if head[:6] != b"\x93NUMPY":
        raise RuntimeError(f"not a .npy file: {url}")
    hlen = struct.unpack("<H", head[8:10])[0]
    header = head[10 : 10 + hlen].decode("latin-1")

    if "'descr': '|u1'" not in header:
        raise RuntimeError(f"expected uint8 bitmaps, got header: {header}")
    if "'fortran_order': False" not in header:
        raise RuntimeError(f"expected C order, got header: {header}")

    shape = header.split("'shape': (")[1].split(")")[0]
    n_rows, row_len = (int(v) for v in shape.split(",")[:2])
    if row_len != ROW_BYTES:
        raise RuntimeError(f"expected {ROW_BYTES}-byte rows, got {row_len}")

    return 10 + hlen, n_rows


def _range_get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status not in (200, 206):
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        return resp.read()


def fetch_category(name, count, cache_dir=CACHE_DIR):
    """First `count` drawings of one category as uint8 (count, 784).

    Cached on disk, so a re-run costs nothing. Asking for more than the cache
    holds refetches; asking for fewer slices what is already there.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{name}.npy.gz")

    if os.path.exists(cache):
        with gzip.open(cache, "rb") as f:
            cached = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, ROW_BYTES)
        if len(cached) >= count:
            return cached[:count].copy()

    url = URL.format(name.replace(" ", "%20"))
    try:
        offset, available = _read_npy_header(url)
        n = min(count, available)
        raw = _range_get(url, offset, offset + n * ROW_BYTES - 1)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"no such Quick, Draw! category: {name!r}. "
                "Names must match exactly, e.g. 'ice cream', 't-shirt'. "
                "Full list: https://storage.googleapis.com/quickdraw_dataset/full/categories.txt"
            ) from e
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(
            f"could not reach the Quick, Draw! dataset ({e}).\n"
            f"Download {url} manually, then place it at {cache} "
            "(gzipped raw rows, no .npy header)."
        ) from e

    if len(raw) < n * ROW_BYTES:
        raise RuntimeError(f"short read for {name}: {len(raw)} bytes, wanted {n * ROW_BYTES}")

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, ROW_BYTES)
    with gzip.open(cache, "wb", compresslevel=6) as f:
        f.write(arr.tobytes())
    return arr.copy()


def _dilate(imgs):
    """Thicken strokes by one pixel, 3x3 max. Only used when downsampling.

    A 1px doodle stroke does not survive a 2x2 pool; thickening first is what
    keeps the line art legible at 16x16.
    """
    n = len(imgs)
    sq = imgs.reshape(n, NATIVE, NATIVE)
    padded = np.pad(sq, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(sq)
    for dy in range(3):
        for dx in range(3):
            np.maximum(out, padded[:, dy : dy + NATIVE, dx : dx + NATIVE], out=out)
    return out.reshape(n, ROW_BYTES)


def _resample(imgs, grid):
    """28x28 -> grid x grid. Max-pool, not mean: mean fades thin strokes out.

    Pads 28 up to a multiple of `grid` first so the pool factor is an integer.
    """
    if grid == NATIVE:
        return imgs

    n = len(imgs)
    factor = int(np.ceil(NATIVE / grid))
    padded_size = grid * factor
    pad = padded_size - NATIVE
    lo, hi = pad // 2, pad - pad // 2

    sq = _dilate(imgs).reshape(n, NATIVE, NATIVE)
    sq = np.pad(sq, ((0, 0), (lo, hi), (lo, hi)))
    pooled = sq.reshape(n, grid, factor, grid, factor).max(axis=(2, 4))
    return pooled.reshape(n, grid * grid)


def load(categories=None, per_class=8000, grid=NATIVE, seed=0, cache_dir=CACHE_DIR):
    """Return (images, labels, categories).

    images is float32 in [-1, 1], shape (N, grid*grid). Diffusion wants a
    roughly zero-mean signal; the raw [0, 1] bitmaps are ~95% background and
    the schedule maths degrades badly on them.
    """
    categories = list(categories or CATEGORIES)

    chunks, labels = [], []
    for label, name in enumerate(categories):
        arr = fetch_category(name, per_class, cache_dir)
        chunks.append(_resample(arr, grid))
        labels.append(np.full(len(arr), label, dtype=np.int64))
        print(f"  {name:<12} {len(arr):>6} drawings")

    images = np.concatenate(chunks).astype(np.float32) / 255.0
    images = images * 2.0 - 1.0
    labels = np.concatenate(labels)

    order = np.random.default_rng(seed).permutation(len(images))
    return images[order], labels[order], categories


def to_ascii(image, grid, ramp=" .:-=+*#%@"):
    """One image (in [-1, 1]) as text, for looking at things without Pillow."""
    px = (image.reshape(grid, grid) + 1.0) * 0.5
    idx = np.clip((px * (len(ramp) - 1) + 0.5).astype(int), 0, len(ramp) - 1)
    return "\n".join("".join(ramp[i] * 2 for i in row) for row in idx)


def write_pgm(path, image, grid):
    """Binary PGM. Pillow is not installed and this is ten lines."""
    px = np.clip((image.reshape(grid, grid) + 1.0) * 0.5, 0.0, 1.0)
    body = (px * 255.0 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (grid, grid))
        f.write(body.tobytes())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch and preview Quick, Draw! data")
    p.add_argument("--per-class", type=int, default=64)
    p.add_argument("--grid", type=int, default=NATIVE)
    p.add_argument("--categories", nargs="*", default=None)
    a = p.parse_args()

    imgs, labs, cats = load(a.categories, a.per_class, a.grid)
    print(f"\n{len(imgs)} images, {a.grid}x{a.grid}, range [{imgs.min():.1f}, {imgs.max():.1f}]\n")
    for label, name in enumerate(cats):
        first = np.argmax(labs == label)
        print(f"--- {name} ---")
        print(to_ascii(imgs[first], a.grid))
