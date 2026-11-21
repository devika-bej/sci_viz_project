"""Benchmark figures + summary table for the report.

usage:  python python/plots.py [--res results] [--ttk ttk_results] [--out report/figures]

reads   results/timings.csv, graph_stats.csv, memory.csv   (+ ttk_results/ttk_timings.csv if present)
writes  report/figures/*.png  and  report/figures/summary.{csv,md}

Timing conventions (medians over repeats):
  ExTreeM time  = pc + cp_lut + graph + egp_setup + egp_par
  baseline time = baseline   (serial union-find, 1 thread)
  load and order are shared preprocessing and reported separately
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# ---- palette (validated: lightness band, chroma, CVD and normal-vision separation all pass;
#      contrast vs surface is below 3:1 for slots 3-5, so every chart has a legend/labels and
#      the numbers are also written to summary.md)
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9a9893"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

STEPS = ["pc", "cp_lut", "graph", "egp_setup", "egp_par"]
STEP_LABEL = {"pc": "path compression", "cp_lut": "critical points", "graph": "extremum graph",
              "egp_setup": "EGP setup", "egp_par": "EGP"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6,
    })


def label(stem):
    name, dims, _ = stem.rsplit("_", 2)
    return f"{name} {dims}"


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print(f"  wrote {out / name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--ttk", default="ttk_results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    res, out = Path(a.res), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()

    # ---------------- load + aggregate ----------------
    t = pd.read_csv(res / "timings.csv")
    med = t.groupby(["dataset", "algo", "threads", "step"], as_index=False).seconds.median()
    ex = med[(med.algo == "egp") & med.step.isin(STEPS)]
    ex_total = ex.groupby(["dataset", "threads"]).seconds.sum()                    # (dataset, threads)
    ex_step = ex.set_index(["dataset", "threads", "step"]).seconds
    order = med[(med.algo == "egp") & (med.step == "order")].set_index(["dataset", "threads"]).seconds
    base = med[(med.algo == "baseline") & (med.step == "baseline")].set_index("dataset").seconds

    gs = pd.read_csv(res / "graph_stats.csv").drop_duplicates("dataset", keep="last").set_index("dataset")
    mem = pd.read_csv(res / "memory.csv")
    mem_mb = mem.assign(mb=mem.max_rss_kb / 1024).set_index(["dataset", "algo"]).mb

    datasets = [d for d in gs.sort_values("VK").index if d in ex_total.index.get_level_values(0)]
    threads = sorted(ex_total.index.get_level_values(1).unique())
    T1, TM = threads[0], threads[-1]

    # ---------------- 1. strong scaling, small multiples ----------------
    n = len(datasets)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, d in zip(axes.flat, datasets):
        s = ex_total.loc[d]
        sp = s.loc[T1] / s
        ax.plot(threads, [x / T1 for x in threads], ls="--", lw=1, color=REF)
        ax.plot(sp.index, sp.values, marker="o", color=SERIES[0])
        ax.annotate(f"{sp.iloc[-1]:.1f}×", (sp.index[-1], sp.iloc[-1]), textcoords="offset points",
                    xytext=(-4, 6), ha="right", color=INK2)
        ax.set_title(label(d), fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(threads)
        ax.set_xticklabels([str(x) for x in threads])
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("threads")
    for ax in axes[:, 0]:
        ax.set_ylabel("speed-up vs 1 thread")
    fig.suptitle("ExTreeM strong scaling (dashed: ideal)", fontsize=10)
    save(fig, out, "scaling.png")

    # ---------------- 2. where the time goes (share of ExTreeM time, max threads) ----------------
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * n + 1.2))
    left = np.zeros(n)
    tot = np.array([ex_total.loc[(d, TM)] for d in datasets])
    for i, st in enumerate(STEPS):
        vals = np.array([ex_step.get((d, TM, st), 0.0) for d in datasets]) / tot * 100
        ax.barh(range(n), vals, left=left, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5,
                label=STEP_LABEL[st], height=0.6)
        left += vals
    for y, s in enumerate(tot):
        ax.text(101, y, f"{s:.3f} s", va="center", color=INK2, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label(d) for d in datasets])
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(f"share of ExTreeM time at {TM} threads (%)   ·   right: total time")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, handlelength=1.2)
    save(fig, out, "breakdown.png")

    # ---------------- 3. speed-up over the serial baseline ----------------
    sp_base = pd.Series({d: base[d] / ex_total.loc[(d, TM)] for d in datasets if d in base.index})
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(sp_base) + 1.0))
    ax.barh(range(len(sp_base)), sp_base.values, color=SERIES[0], height=0.6)
    for y, v in enumerate(sp_base.values):
        ax.text(v, y, f"  {v:.1f}×", va="center", color=INK2, fontsize=8)
    ax.set_yticks(range(len(sp_base)))
    ax.set_yticklabels([label(d) for d in sp_base.index])
    ax.set_xlabel(f"speed-up of ExTreeM ({TM} threads) over serial union-find baseline")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, sp_base.max() * 1.18)
    save(fig, out, "speedup_vs_baseline.png")

    # ---------------- 4. speed-up vs extremum-graph size (paper-style) ----------------
    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = gs.loc[sp_base.index, "ratio_percent"]
    ax.scatter(x, sp_base.values, s=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for d in sp_base.index:
        ax.annotate(d.rsplit("_", 2)[0], (x[d], sp_base[d]), textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("extremum graph size  (|V(G)|+|E(G)|) / (|V(K)|+|E(K)|)  in %")
    ax.set_ylabel("speed-up over serial baseline")
    save(fig, out, "speedup_vs_graph_ratio.png")

    # ---------------- 5. per-step scaling on the largest dataset ----------------
    big = datasets[-1]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(threads, [x / T1 for x in threads], ls="--", lw=1, color=REF, label="ideal")
    for i, st in enumerate(STEPS):
        s = pd.Series({th: ex_step.get((big, th, st), np.nan) for th in threads})
        sp = s.loc[T1] / s
        ax.plot(threads, sp.values, marker="o", color=SERIES[i], label=STEP_LABEL[st])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(x) for x in threads])
    ax.set_xlabel("threads")
    ax.set_ylabel("speed-up vs 1 thread")
    ax.set_title(f"per-step scaling: {label(big)}")
    ax.legend(fontsize=8)
    save(fig, out, "step_scaling.png")

    # ---------------- 6. TTK comparison (optional) ----------------
    ttk_file = Path(a.ttk) / "ttk_timings.csv"
    ttk_tot = None
    if ttk_file.exists():
        tk = pd.read_csv(ttk_file)
        ttk_tot = tk[tk.step == "total"].groupby(["dataset", "backend", "threads"]).seconds.median()
        tds = [d for d in datasets if d in ttk_tot.index.get_level_values(0)]
        tthreads = sorted(set(ttk_tot.index.get_level_values(2)) & set(threads))
        if tds and tthreads:
            panels = [tthreads[0], tthreads[-1]] if len(tthreads) > 1 else tthreads
            fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 0.5 * len(tds) + 1.4),
                                     sharey=True, squeeze=False)
            names = ["ours (order + ExTreeM)", "TTK FTM", "TTK ExTreeM"]
            for ax, th in zip(axes[0], panels):
                for i, nm in enumerate(names):
                    xs, ys = [], []
                    for y, d in enumerate(tds):
                        if i == 0:
                            v = ex_total.get((d, th), np.nan) + order.get((d, th), np.nan)
                        else:
                            v = ttk_tot.get((d, "FTM" if i == 1 else "ExTreeM", th), np.nan)
                        xs.append(v)
                        ys.append(y)
                    ax.scatter(xs, ys, s=60, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5,
                               label=nm, zorder=3)
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.margins(x=0.15, y=0.35)
                ax.set_title(f"{th} thread{'s' if th > 1 else ''}")
                ax.set_xlabel("time (s, log scale)")
                ax.grid(axis="y", visible=False)
            axes[0, 0].set_yticks(range(len(tds)))
            axes[0, 0].set_yticklabels([label(d) for d in tds])
            handles, labels_ = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels_, fontsize=8, loc="upper center", ncol=3)
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            fig.savefig(out / "ttk_comparison.png", dpi=200)
            plt.close(fig)
            print(f"  wrote {out / 'ttk_comparison.png'}")

    # ---------------- summary table ----------------
    rows = []
    for d in datasets:
        r = {
            "dataset": d,
            "vertices": int(gs.loc[d, "VK"]),
            "maxima": int(gs.loc[d, "maxima"]),
            "saddles": int(gs.loc[d, "saddles"]),
            "graph_ratio_%": round(float(gs.loc[d, "ratio_percent"]), 4),
            f"order_{TM}T_s": round(float(order.get((d, TM), np.nan)), 4),
            "baseline_1T_s": round(float(base.get(d, np.nan)), 4),
            f"extreem_{T1}T_s": round(float(ex_total.loc[(d, T1)]), 4),
            f"extreem_{TM}T_s": round(float(ex_total.loc[(d, TM)]), 4),
            "self_speedup": round(float(ex_total.loc[(d, T1)] / ex_total.loc[(d, TM)]), 2),
            "speedup_vs_baseline": round(float(sp_base.get(d, np.nan)), 2),
            "mem_baseline_MB": round(float(mem_mb.get((d, "baseline"), np.nan)), 0),
            "mem_extreem_MB": round(float(mem_mb.get((d, "egp"), np.nan)), 0),
        }
        if ttk_tot is not None:
            r[f"ttk_ftm_{TM}T_s"] = round(float(ttk_tot.get((d, "FTM", TM), np.nan)), 4)
            r[f"ttk_extreem_{TM}T_s"] = round(float(ttk_tot.get((d, "ExTreeM", TM), np.nan)), 4)
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)
    cols = list(summ.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(v) for v in row) + " |" for row in summ.itertuples(index=False)]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print(f"  wrote {out / 'summary.csv'} and summary.md")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()