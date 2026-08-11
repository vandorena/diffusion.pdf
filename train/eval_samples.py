"""Does the model actually work? Metrics that fail loudly, not eyeballing.

The metric that matters is denoising skill at *intermediate* noise. A model can
predict a perfect class prototype from pure noise and still generate garbage,
because sampling feeds the model its own output: if it cannot sharpen a
half-noisy image, the chain degrades instead of converging. Reconstruction
error at abar ~ 0.1-0.5 predicts sample quality; loss does not.

    python3 -m train.eval_samples --model train/model.json
"""

import argparse
import os

import numpy as np

from . import data as Data
from . import diffusion as D
from . import export_weights as E

HERE = os.path.dirname(os.path.abspath(__file__))


def recon_skill(model, images, labels, seed=999):
    """Mean absolute x0 error at each noise level, against two baselines.

    `mean` is the error from predicting the dataset average image -- a model
    that cannot beat it has learned nothing at that noise level.
    """
    abar = np.asarray(model["abar"], dtype=np.float64)
    table = np.asarray(model["norm"], dtype=np.float64)
    npix = model["grid"] ** 2
    mean_image = images.mean(axis=0)

    rows = []
    for t in D.ddim_timesteps(8, len(abar)):
        errs = []
        for k in range(len(images)):
            x0 = images[k].astype(np.float64)
            eps = D.gaussian(npix, seed + k, table)
            x_t = D.q_sample(x0, eps, abar[t])
            v = E.predict_v(model, x_t, t, int(labels[k]))
            x0_hat = np.clip(D.x0_from_v(x_t, v, abar[t]), -1.0, 1.0)
            errs.append(np.abs(x0_hat - x0).mean())
        base = np.abs(mean_image - images).mean()
        rows.append((t, abar[t], float(np.mean(errs)), float(base)))
    return rows


def sample_stats(model, n=8, steps=None, guidance=None, seed=1000):
    """Sharpness and class separation of actual samples.

    `sharpness` is the mean absolute neighbour difference: line art has high
    values, an over-smoothed conditional mean has low ones. Comparing it to
    the real data's value catches the classic diffusion failure of producing
    a plausible blur.
    """
    classes = model["classes"]
    imgs = {}
    for name in classes[:n]:
        imgs[name] = E.sample(model, name, seed, steps=steps, guidance=guidance)

    def sharp(a):
        g = a.reshape(model["grid"], model["grid"])
        return float((np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean()) / 2)

    names = list(imgs)
    between = [
        float(np.abs(imgs[a] - imgs[b]).mean())
        for i, a in enumerate(names) for b in names[i + 1:]
    ]
    return {
        "sharpness": float(np.mean([sharp(v) for v in imgs.values()])),
        "class_separation": float(np.mean(between)) if between else 0.0,
        "images": imgs,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate the trained denoiser")
    p.add_argument("--model", default=os.path.join(HERE, "model.json"))
    p.add_argument("--n", type=int, default=24, help="images for the recon test")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--show", nargs="*", default=None, help="categories to print")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    model = E.load(a.model)
    grid = model["grid"]
    images, labels, _ = Data.load(model["classes"], 32, grid, seed=7)
    images, labels = images[: a.n], labels[: a.n]

    real_sharp = float(np.mean([
        (np.abs(np.diff(im.reshape(grid, grid), axis=0)).mean()
         + np.abs(np.diff(im.reshape(grid, grid), axis=1)).mean()) / 2
        for im in images
    ]))

    print("denoising skill (lower is better; must beat the mean-image baseline)")
    worst = 0.0
    for t, ab, err, base in recon_skill(model, images, labels):
        margin = (base - err) / base
        worst = max(worst, 1.0 - margin) if ab < 0.9 else worst
        flag = "  <-- no better than the mean" if err > base * 0.95 else ""
        print(f"  t={t:>2} abar={ab:.3f}  MAE={err:.4f}  baseline={base:.4f}  "
              f"{margin * 100:>5.1f}% better{flag}")

    stats = sample_stats(model, steps=a.steps, guidance=a.guidance)
    print(f"\nsamples: sharpness={stats['sharpness']:.4f} (real data {real_sharp:.4f})")
    print(f"         class separation={stats['class_separation']:.4f}")

    ratio = stats["sharpness"] / real_sharp
    if ratio > 1.6:
        print("  VERDICT: samples are noise -- sharper than real line art")
    elif ratio < 0.35:
        print("  VERDICT: samples are over-smoothed blurs")
    elif stats["class_separation"] < 0.05:
        print("  VERDICT: conditioning is not working -- classes look alike")
    else:
        print("  VERDICT: plausible")

    if not a.quiet:
        for name in (a.show or model["classes"][:4]):
            print(f"\n--- {name} ---")
            print(Data.to_ascii(E.sample(model, name, 1000, steps=a.steps,
                                         guidance=a.guidance), grid))


if __name__ == "__main__":
    main()
