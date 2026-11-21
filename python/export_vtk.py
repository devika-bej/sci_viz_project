"""Export ExTreeM outputs to legacy VTK files for ParaView.

usage:  python python/export_vtk.py data/<stem>.raw [--res vis/<stem>] [--eps 0.05]
needs:  ./build/vis/extreem data/<stem>.raw --algo vis      (run first)
writes into the vis folder:
  field.vtk               volume: value, region (descending manifolds), simplified (persistence >= eps*range)
  critical_points.vtk     points: type (0 max, 1 merge saddle, 2 loop saddle), value, persistence
  extremum_graph.vtk      lines saddle -> maximum, with value and is_max per vertex
  persistence_diagram.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DT = {"uint8": "<u1", "uint16": "<u2", "float32": "<f4", "float64": "<f8"}
SURFACE, INK2, REF, SERIES1 = "#fcfcfb", "#52514e", "#9a9893", "#2a78d6"


def parse_stem(stem):
    _, dims, dtype = stem.rsplit("_", 2)
    nx, ny, nz = map(int, dims.split("x"))
    return nx, ny, nz, dtype


def header(fh, title, lines):
    fh.write(b"# vtk DataFile Version 3.0\n" + title.encode() + b"\nBINARY\n")
    fh.write(("\n".join(lines) + "\n").encode())


def scalars(fh, name, arr, vtk_type):
    fh.write(f"SCALARS {name} {vtk_type} 1\nLOOKUP_TABLE default\n".encode())
    fh.write(np.ascontiguousarray(arr).astype(">f4" if vtk_type == "float" else ">i4").tobytes())
    fh.write(b"\n")


def coords(ids, nx, ny):
    ids = np.asarray(ids, dtype=np.int64)
    return np.stack([ids % nx, (ids // nx) % ny, ids // (nx * ny)], axis=1).astype(">f4")


def shuffled(labels, rng):
    """Relabel to 0..k-1 in random order so neighbouring regions get very different colours."""
    u, inv = np.unique(labels, return_inverse=True)
    return rng.permutation(len(u))[inv].astype(np.int32), len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--res")
    ap.add_argument("--eps", type=float, default=0.05, help="persistence threshold as fraction of value range")
    a = ap.parse_args()

    stem = Path(a.raw).stem
    nx, ny, nz, dtype = parse_stem(stem)
    n = nx * ny * nz
    res = Path(a.res or f"vis/{stem}")

    f = np.fromfile(a.raw, dtype=DT[dtype]).astype(np.float32)
    dm = np.fromfile(res / "dm.bin", dtype="<i4")
    assert f.size == n and dm.size == n, "size mismatch"
    pairs = pd.read_csv(res / "pairs_egp.csv")
    mx = pd.read_csv(res / "maxima.csv")
    sad = pd.read_csv(res / "saddles.csv")
    edges = pd.read_csv(res / "graph_edges.csv")
    rng = np.random.default_rng(0)

    # ---- persistence simplification: climb parent branches while persistence < threshold
    thr = a.eps * float(f.max() - f.min())
    parent = dict(zip(pairs.max_id.tolist(), pairs.parent_max_id.tolist()))
    pers = dict(zip(pairs.max_id.tolist(), pairs.persistence.tolist()))
    rep = {}
    for m in pairs.max_id.tolist():
        path, x = [], m
        while x not in rep and parent[x] != -1 and pers[x] < thr:
            path.append(x)
            x = parent[x]
        r = rep.get(x, x)
        rep[x] = r
        for p in path:
            rep[p] = r
    mids = np.array(sorted(rep), dtype=np.int64)
    mrep = np.array([rep[m] for m in mids], dtype=np.int64)
    simp = mrep[np.searchsorted(mids, dm)]

    region, k_full = shuffled(dm, rng)
    simplified, k_simp = shuffled(simp, rng)

    with open(res / "field.vtk", "wb") as fh:
        header(fh, f"{stem} ExTreeM",
               ["DATASET STRUCTURED_POINTS", f"DIMENSIONS {nx} {ny} {nz}",
                "ORIGIN 0 0 0", "SPACING 1 1 1", f"POINT_DATA {n}"])
        scalars(fh, "value", f, "float")
        scalars(fh, "region", region, "int")
        scalars(fh, "simplified", simplified, "int")

    # ---- critical points: 0 = maximum, 1 = merge saddle, 2 = loop saddle (removed by EGP)
    real = pairs[pairs.parent_max_id != -1]
    merge_pers = real.groupby("saddle_id").persistence.max()
    ids = np.concatenate([mx.max_id.to_numpy(), sad.saddle_id.to_numpy()])
    typ = np.concatenate([np.zeros(len(mx), np.int32),
                          np.where(sad.saddle_id.isin(merge_pers.index), 1, 2).astype(np.int32)])
    per = np.concatenate([mx.max_id.map(pers).to_numpy(np.float32),
                          sad.saddle_id.map(merge_pers).fillna(0).to_numpy(np.float32)])
    N = len(ids)
    with open(res / "critical_points.vtk", "wb") as fh:
        header(fh, f"{stem} critical points", ["DATASET POLYDATA", f"POINTS {N} float"])
        fh.write(coords(ids, nx, ny).tobytes() + b"\n")
        fh.write(f"VERTICES {N} {2 * N}\n".encode())
        fh.write(np.stack([np.ones(N, np.int32), np.arange(N, dtype=np.int32)], 1).astype(">i4").tobytes() + b"\n")
        fh.write(f"POINT_DATA {N}\n".encode())
        scalars(fh, "type", typ, "int")
        scalars(fh, "value", f[ids], "float")
        scalars(fh, "persistence", per, "float")

    # ---- extremum graph
    gids = np.unique(np.concatenate([edges.saddle_id.to_numpy(), edges.max_id.to_numpy()]))
    ia = np.searchsorted(gids, edges.saddle_id.to_numpy())
    ib = np.searchsorted(gids, edges.max_id.to_numpy())
    E = len(edges)
    with open(res / "extremum_graph.vtk", "wb") as fh:
        header(fh, f"{stem} extremum graph", ["DATASET POLYDATA", f"POINTS {len(gids)} float"])
        fh.write(coords(gids, nx, ny).tobytes() + b"\n")
        fh.write(f"LINES {E} {3 * E}\n".encode())
        fh.write(np.stack([np.full(E, 2, np.int32), ia, ib], 1).astype(">i4").tobytes() + b"\n")
        fh.write(f"POINT_DATA {len(gids)}\n".encode())
        scalars(fh, "value", f[gids], "float")
        scalars(fh, "is_max", np.isin(gids, mx.max_id.to_numpy()).astype(np.int32), "int")

    # ---- persistence diagram (single series; diagonal is a neutral reference)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    lo, hi = float(f.min()), float(f.max())
    ax.plot([lo, hi], [lo, hi], color=REF, lw=1)
    ax.scatter(pairs.saddle_value, pairs.max_value, s=8, alpha=0.6, color=SERIES1, linewidths=0)
    ax.set_xlabel("death (saddle value)")
    ax.set_ylabel("birth (maximum value)")
    ax.set_title(f"{stem}\n{len(pairs)} pairs", fontsize=9)
    ax.grid(color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(res / "persistence_diagram.png", dpi=200)
    plt.close(fig)

    print(f"{stem}: {k_full} regions -> {k_simp} after simplification (eps={a.eps}, thr={thr:.4g})")
    print(f"  critical points: {len(mx)} maxima, {int((typ == 1).sum())} merge saddles, "
          f"{int((typ == 2).sum())} loop saddles")
    print(f"  graph: {len(gids)} vertices, {E} edges")
    print(f"  wrote {res}/field.vtk, critical_points.vtk, extremum_graph.vtk, persistence_diagram.png")


if __name__ == "__main__":
    main()