"""The diffusion process, in numpy. This is the reference.

train_diffusion.py trains against it, export_weights.py samples through it to
produce the preview, and tools/reference.py dumps its trajectory for the JS
harness to match. One implementation, so they cannot drift apart.

Everything on the sampling path is built out of +, -, *, / and sqrt, which
IEEE-754 requires to be correctly rounded and which therefore agree bit for
bit between numpy and a JavaScript engine. Nothing here calls log, exp, cos
or pow at sample time -- those are explicitly implementation-approximated by
the ECMAScript spec and would cost us exactness for no benefit. The schedule
is built with cos() once, offline, and shipped as decimal literals.
"""

import numpy as np

T_TRAIN = 64  # trained timesteps. DDIM samples a strided subset of these.
NORM_BINS = 2048  # inverse-CDF table resolution; index is the top 11 bits


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------

def cosine_abar(T=T_TRAIN, s=0.008, shift=1.0):
    """Cumulative alpha for the cosine schedule, with an SNR shift.

    The plain cosine schedule is calibrated for 32x32 and up. At a coarser
    grid each pixel carries more of the image, so a given noise level destroys
    less information and much of the schedule is spent somewhere trivial.
    Dividing the SNR by shift**2 moves the whole schedule toward noisier, which
    is where the model should be spending its capacity.

    shift = 64 / grid reproduces the value that measured best at 16x16.
    """
    t = np.arange(T + 1, dtype=np.float64) / T
    f = np.cos((t + s) / (1.0 + s) * (np.pi / 2.0)) ** 2
    abar = f[1:] / f[0]
    abar = np.clip(abar, 1e-8, 1.0 - 1e-8)

    if shift != 1.0:
        snr = abar / (1.0 - abar) / (shift * shift)
        abar = snr / (1.0 + snr)

    # float32 round-trip: the JS side parses the same decimals we emit, and we
    # want the training-time schedule to be the one that actually ships.
    return abar.astype(np.float32).astype(np.float64)


def ddim_timesteps(steps, T=T_TRAIN):
    """Descending trained indices, evenly strided.

    Integer stride rather than linspace+round so the JS and numpy sides pick
    identical timesteps with no float comparison anywhere.
    """
    if T % steps != 0:
        raise ValueError(f"steps={steps} must divide T={T}")
    stride = T // steps
    return [T - 1 - i * stride for i in range(steps)]


# --------------------------------------------------------------------------
# v-parameterisation
#
#   v  = sqrt(abar)*eps - sqrt(1-abar)*x0
#   x0 = sqrt(abar)*x_t - sqrt(1-abar)*v
#   eps= sqrt(1-abar)*x_t + sqrt(abar)*v
#
# v has bounded target variance across the whole schedule, and recovering x0
# from it is a blend rather than a division by sqrt(abar). Predicting eps
# instead divides by a vanishing number at high noise and measured far worse
# (class accuracy 0.716 vs 0.978 on the same net and seed).
# --------------------------------------------------------------------------

def q_sample(x0, eps, abar_t):
    a = np.sqrt(abar_t)
    b = np.sqrt(1.0 - abar_t)
    return a * x0 + b * eps


def v_target(x0, eps, abar_t):
    a = np.sqrt(abar_t)
    b = np.sqrt(1.0 - abar_t)
    return a * eps - b * x0


def x0_from_v(x_t, v, abar_t):
    a = np.sqrt(abar_t)
    b = np.sqrt(1.0 - abar_t)
    return a * x_t - b * v


def eps_from_v(x_t, v, abar_t):
    a = np.sqrt(abar_t)
    b = np.sqrt(1.0 - abar_t)
    return b * x_t + a * v


# --------------------------------------------------------------------------
# deterministic sampler
# --------------------------------------------------------------------------

def ddim_sample(predict_v, x_T, abar, steps, trace=None):
    """DDIM with eta = 0.

    predict_v(x_t, t) -> v, already guidance-combined by the caller.
    With eta = 0 the trajectory is a pure function of x_T, so the harness can
    assert per-step agreement rather than only comparing final images.

    Clamping the predicted x0 to [-1, 1] every step is what stops int8
    quantisation error from accumulating: it is a hard bound on how far any
    single step can push the state off the data manifold.

    The clamp forces eps to be recomputed rather than taken from the model
    output. DDIM's update assumes x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps, and
    clipping x0 without re-deriving eps breaks that identity: the pair no
    longer describes the state the sampler is actually in, and the error
    compounds. Left inconsistent, the x0 prediction starts as a recognisable
    drawing and decorrelates from it a little more at every step, ending in
    noise that still has plausible-looking mean and variance.
    """
    x = np.array(x_T, dtype=np.float64, copy=True)
    ts = ddim_timesteps(steps, len(abar))

    for i, t in enumerate(ts):
        abar_t = abar[t]
        abar_prev = abar[ts[i + 1]] if i + 1 < len(ts) else 1.0

        v = predict_v(x, t)
        x0 = np.clip(x0_from_v(x, v, abar_t), -1.0, 1.0)
        eps = (x - np.sqrt(abar_t) * x0) / np.sqrt(1.0 - abar_t)
        x = np.sqrt(abar_prev) * x0 + np.sqrt(1.0 - abar_prev) * eps

        if trace is not None:
            trace.append({"t": int(t), "v": v.copy(), "x0": x0.copy(), "x": x.copy()})

    return x


# --------------------------------------------------------------------------
# deterministic noise
#
# xorshift32 uses only shifts and xors. JS coerces those through ToInt32 /
# ToUint32, so the wrapping is exact and matches Python masked to 32 bits --
# there is no multiply anywhere that could exceed 2**53 and diverge.
# --------------------------------------------------------------------------

def xorshift32(state):
    """Generator of uint32s. state must be a non-zero uint32."""
    x = int(state) & 0xFFFFFFFF
    if x == 0:
        x = 0x9E3779B9
    while True:
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        x &= 0xFFFFFFFF
        yield x


def djb2(text):
    """Stable string -> uint32. h*33 stays well under 2**53, so plain
    multiplication is exact in JS and needs no Math.imul."""
    h = 5381
    for ch in text:
        h = (h * 33 + ord(ch)) & 0xFFFFFFFF
    return h


def norm_table(bins=NORM_BINS):
    """Inverse normal CDF at bin midpoints, bins+1 entries.

    Box-Muller would need log and cos, neither of which is bit-identical
    across engines. A table plus linear interpolation is index arithmetic and
    one multiply-add: exact, and accurate to far below an int8 model's noise
    floor. torch.erfinv supplies the values (scipy is not installed).
    """
    import torch

    p = (np.arange(bins + 1, dtype=np.float64) + 0.5) / (bins + 1)
    inv = torch.erfinv(torch.from_numpy(2.0 * p - 1.0)).numpy() * np.sqrt(2.0)
    return inv.astype(np.float32).astype(np.float64)


def gaussian(count, seed, table, bins=NORM_BINS):
    """`count` standard normals from a uint32 seed, via the table.

    Consumes one uint32 per value: the top bits index the table, the rest
    interpolate. Mirrored exactly in the JS.
    """
    shift = 32 - int(np.log2(bins))  # 2048 bins -> index is the top 11 bits
    frac_mask = (1 << shift) - 1
    frac_scale = 1.0 / (1 << shift)

    rng = xorshift32(seed)
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        u = next(rng)
        idx = u >> shift
        frac = (u & frac_mask) * frac_scale
        out[i] = table[idx] + frac * (table[idx + 1] - table[idx])
    return out


def seed_for(word, seed):
    """Combine the typed word and the seed field into the PRNG state."""
    combined = (djb2(word) ^ (int(seed) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return combined or 0x9E3779B9
