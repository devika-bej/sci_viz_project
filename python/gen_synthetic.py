"""Generate synthetic scalar fields as raw files (x fastest, little-endian).

Examples:
  python python/gen_synthetic.py perlin --n 128 --res 8 --octaves 4 --seed 0
  python python/gen_synthetic.py gauss  --n 64  --k 10 --sigma 4 --seed 0
"""
import argparse
from pathlib import Path
import numpy as np


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin3d(n, res, rng):
    """One octave of 3D gradient noise on an n^3 grid with res cells per axis."""
    g = rng.normal(size=(res + 1, res + 1, res + 1, 3))      # gradients at lattice [z,y,x]
    g /= np.linalg.norm(g, axis=-1, keepdims=True)

    p = np.arange(n) * res / n                               # same coords on every axis
    i = np.floor(p).astype(np.int64)
    f = p - i
    u = fade(f)

    out = np.empty((n, n, n), dtype=np.float32)              # [z, y, x]
    yi, xi = i[:, None], i[None, :]
    yf, xf = f[:, None], f[None, :]
    v, w = u[:, None], u[None, :]                            # fades for y, x

    for z in range(n):                                       # slab-by-slab to keep memory low
        zi, zf, s = i[z], f[z], u[z]
        corner = {}
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    gr = g[zi + dz, yi + dy, xi + dx]        # (n, n, 3)
                    corner[dz, dy, dx] = (gr[..., 0] * (xf - dx)
                                          + gr[..., 1] * (yf - dy)
                                          + gr[..., 2] * (zf - dz))
        lerp = lambda a, b, t: a + t * (b - a)
        x00 = lerp(corner[0, 0, 0], corner[0, 0, 1], w)
        x01 = lerp(corner[0, 1, 0], corner[0, 1, 1], w)
        x10 = lerp(corner[1, 0, 0], corner[1, 0, 1], w)
        x11 = lerp(corner[1, 1, 0], corner[1, 1, 1], w)
        out[z] = lerp(lerp(x00, x01, v), lerp(x10, x11, v), s)
    return out


def fbm(n, res, octaves, rng):
    """Sum octaves: frequency doubles, amplitude halves -> many critical points."""
    total = np.zeros((n, n, n), dtype=np.float32)
    for o in range(octaves):
        total += (0.5 ** o) * perlin3d(n, res * 2 ** o, rng)
    return total


def gaussians(n, k, sigma, rng):
    """Sum of k Gaussian bumps: about k maxima if they are well separated."""
    c = rng.uniform(0.15 * n, 0.85 * n, size=(k, 3))
    ax = np.arange(n, dtype=np.float32)
    out = np.zeros((n, n, n), dtype=np.float32)
    for cz, cy, cx in c:
        out += (np.exp(-((ax - cz) ** 2) / (2 * sigma**2))[:, None, None]
                * np.exp(-((ax - cy) ** 2) / (2 * sigma**2))[None, :, None]
                * np.exp(-((ax - cx) ** 2) / (2 * sigma**2))[None, None, :])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["perlin", "gauss"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--res", type=int, default=8)
    ap.add_argument("--octaves", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.kind == "perlin":
        vol = fbm(a.n, a.res, a.octaves, rng)
        tag = f"perlin_r{a.res}_o{a.octaves}_s{a.seed}"
    else:
        vol = gaussians(a.n, a.k, a.sigma, rng)
        tag = f"gauss_k{a.k}_s{a.seed}"

    name = f"{tag}_{a.n}x{a.n}x{a.n}_float32.raw"
    Path(a.out).mkdir(exist_ok=True)
    vol.astype("<f4").tofile(Path(a.out) / name)           # C-order [z,y,x] => x fastest
    print(f"wrote {name}  min={vol.min():.4f} max={vol.max():.4f}")


if __name__ == "__main__":
    main()