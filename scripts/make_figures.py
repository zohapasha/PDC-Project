"""Regenerates every figure (PDF) and table (LaTeX + Markdown) from results/*.csv only.

Usage: python scripts/make_figures.py
Colours follow the entity (a method keeps its colour in every figure); markers and line
styles give a second, colour-independent encoding for greyscale print.
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = "results"
FIG = os.path.join(R, "figures")
TAB = os.path.join(R, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# validated categorical palette, fixed slot order, assigned per method (never cycled)
STYLE = {
    "ours":              ("#2a78d6", "o", "-",  "Ours"),
    "ours_two_level":    ("#2a78d6", "s", "--", "Ours (two-level, option b)"),
    "ours_two_level_b":  ("#2a78d6", "s", "--", "Ours (two-level, option b)"),
    "ours_two_level_a":  ("#6250d6", "<", ":",  "Ours (two-level, option a)"),
    "B1_offline":        ("#eb6834", "^", "-",  "B1 offline / full recompute"),
    "B1_full_recompute": ("#eb6834", "^", "-",  "B1 full recompute"),
    "B3_no_correction":  ("#1baf7a", "D", "-.", "B3 no correction"),
    "B4_no_constraints": ("#eda100", "v", ":",  "B4 no constraints"),
    "B2_component":      ("#e87ba4", "P", "-",  "B2 component recompute"),
    "B5_centralized":    ("#008300", "X", "-",  "B5 centralized"),
    "centralized":       ("#008300", "X", "-",  "Centralized (single instance)"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.markersize": 4.5, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
W1, W2 = 3.5, 7.16          # IEEE single / double column width (inches)


def load(name):
    p = os.path.join(R, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def line(ax, df, x, y, method, label=None):
    c, m, ls, lab = STYLE[method]
    g = df[df.method == method].groupby(x)[y].agg(["mean", "std"]).reset_index()
    if g.empty:
        return
    ax.errorbar(g[x], g["mean"], yerr=g["std"].fillna(0), color=c, marker=m, ls=ls,
                capsize=2, elinewidth=0.8, label=label or lab)


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=200)
    plt.close(fig)
    print("  figure", name)


def ms(series):
    return f"{series.mean():.3f} ± {series.std():.3f}"


# ------------------------------------------------------------------------- E1 table
def e1():
    for name in ("e1_accuracy", "e1_accuracy_hard"):
        d = load(name)
        if d is None:
            continue
        order = ["ours", "ours_two_level", "ours_two_level_a", "B1_offline", "B3_no_correction",
                 "B4_no_constraints"]
        cols = [("pw_f1", "Pairwise F1"), ("b3_f1", "B³ F1"), ("idf1", "IDF1"),
                ("violations", "Violations"), ("agree_offline", "Agreement w/ B1")]
        rows = []
        for m in order:
            s = d[d.method == m]
            if s.empty:
                continue
            r = [STYLE[m][3]]
            for c, _ in cols:
                r.append(f"{s[c].mean():.1f} ± {s[c].std():.1f}" if c == "violations" else ms(s[c]))
            rows.append(r)
        hdr = ["Method"] + [c[1] for c in cols]
        n = d.n_tracklets.mean()
        with open(os.path.join(TAB, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(f"{name}: {d.seed.nunique()} seeds, ~{n:,.0f} tracklets/run\n\n")
            f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
            for r in rows:
                f.write("| " + " | ".join(r) + " |\n")
        with open(os.path.join(TAB, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n")
            f.write(" & ".join(h.replace("³", "$^3$") for h in hdr) + " \\\\\n\\midrule\n")
            for r in rows:
                f.write(" & ".join(x.replace("±", "$\\pm$") for x in r) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        print("  table", name)


# ------------------------------------------------------------------------- E2
def e2():
    d = load("e2_incremental")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    line(ax, d, "n_tracklets", "update_ms_p50", "ours", "Ours, p50 per update")
    c, m, _, _ = STYLE["ours"]
    g = d[d.method == "ours"].groupby("n_tracklets")["update_ms_p99"].agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), color=c, marker=m, ls="--",
                mfc="white", capsize=2, elinewidth=0.8, label="Ours, p99 per update")
    line(ax, d, "n_tracklets", "update_ms_p50", "B1_full_recompute", "B1, one full recompute")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Tracklets processed (graph size)")
    ax.set_ylabel("Update latency (ms)")
    ax.legend(loc="upper left")
    save(fig, "fig_e2_incremental")


# ------------------------------------------------------------------------- E3
def e3():
    d = load("e3_corrections")
    files = sorted(glob.glob(os.path.join(R, "e3_ops_e3_corrections_*.parquet")))
    if d is None or not files:
        return
    ops = pd.concat([pd.read_parquet(f) for f in files])
    corr = ops[ops.kind.isin(["conflict_ok", "del_tree"])]
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.3))
    ax = axs[0]
    for m in ("B2_component", "ours"):
        s = corr[corr.method == m]
        s = s.sample(min(len(s), 4000), random_state=0)
        c, mk, _, lab = STYLE[m]
        ax.scatter(s.person_size, s.touched, s=7, color=c, marker=mk, alpha=0.45,
                   edgecolors="none", label=lab)
    lim = max(corr.person_size.max(), corr.touched.max())
    ax.plot([1, lim], [1, lim], color=MUTED, lw=0.8, ls=":", label="touched = person size")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Size of the person(s) involved")
    ax.set_ylabel("Vertices touched per correction")
    ax.legend(loc="upper left", markerscale=1.5)
    ax = axs[1]
    for m in ("ours", "B2_component"):
        s = np.sort(corr[corr.method == m].latency_us.values)
        c, _, ls, lab = STYLE[m]
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=c, ls=ls, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Correction latency (µs)")
    ax.set_ylabel("CDF")
    ax.legend(loc="upper left")
    save(fig, "fig_e3_corrections")
    cols = ["n_corrections", "n_conflict_ok", "n_conflict_rej", "n_deletions", "frac_del_hit_backup",
            "frac_treedel_repaired_by_backup", "touched_mean", "touched_over_size_max", "msgs_mean",
            "corr_us_p50", "corr_us_p99", "tree_depth_mean", "tree_depth_max", "pw_f1"]
    g = d.groupby("method")[cols].mean().T
    g.to_csv(os.path.join(TAB, "e3_summary.csv"))
    with open(os.path.join(TAB, "e3_summary.md"), "w", encoding="utf-8") as f:
        f.write(g.to_string(float_format=lambda v: f"{v:.4g}"))
    print("  table e3_summary")


# ------------------------------------------------------------------------- E4
def e4():
    for kind in ("weak", "strong"):
        d = load(f"e4_{kind}")
        if d is None:
            continue
        fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
        mx, fx = d[d.load == "max"], d[d.load == "fixed"]
        for m in ("ours", "B5_centralized"):
            line(axs[0], mx, "n_sites", "throughput_ev_s", m)
            line(axs[1], fx, "n_sites", "e2e_ms_p99", m)
            line(axs[2], mx, "n_sites", "hub_cpu_util", m)
        axs[0].set_ylabel("Max throughput (events/s)")
        axs[1].set_ylabel("End-to-end p99 latency (ms)\nat fixed load")
        axs[1].set_yscale("log")
        axs[2].set_ylabel("Hub CPU utilisation\nat max load")
        for ax in axs:
            ax.set_xlabel("Sites (processes)")
            ax.set_xticks(sorted(d.n_sites.unique()))
        h, l = axs[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
        save(fig, f"fig_e4_{kind}")


# ------------------------------------------------------------------------- E5
def e5():
    d = load("e5_bytes")
    if d is None:
        return
    types = [c for c in d.columns if c.startswith("bytes_")]
    g = d.groupby("method")[types + ["total_bytes", "staleness_s"]].mean()
    order = [m for m in ["ours", "B5_centralized", "B6_periodic_T1", "B6_periodic_T5", "B6_periodic_T30"] if m in g.index]
    g = g.loc[order]
    names = {"ours": "Ours\n(deltas)", "B5_centralized": "B5\ncentralized",
             "B6_periodic_T1": "B6\nT=1 s", "B6_periodic_T5": "B6\nT=5 s", "B6_periodic_T30": "B6\nT=30 s"}
    # message types: fixed colour per type (slot order), legend + table view in CSV
    tcol = {"bytes_BorderTracklet": "#2a78d6", "bytes_BorderRevise": "#eb6834",
            "bytes_TrackletRecord": "#1baf7a", "bytes_EdgeAdd": "#eda100", "bytes_EdgeDel": "#e87ba4",
            "bytes_FullTracklet": "#008300", "bytes_FullRevise": "#6250d6", "bytes_SiteTable": "#e34948",
            "bytes_SplitRequest": "#8a5a00"}
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw={"width_ratios": [2, 2]})
    ax = axs[0]
    # composition panel: ours, B5 and the cheapest B6; all B6 periods are on the right panel
    gb = g.loc[[m for m in ["ours", "B5_centralized", "B6_periodic_T30"] if m in g.index]]
    x = np.arange(len(gb))
    bottom = np.zeros(len(gb))
    for t in [t for t in tcol if t in gb.columns]:
        v = gb[t].fillna(0).values / 2**20
        if v.sum() == 0:
            continue
        ax.bar(x, v, bottom=bottom, color=tcol[t], width=0.62, edgecolor="white", linewidth=1,
               label=t.replace("bytes_", ""))
        bottom += v
    for i, tot in enumerate(bottom):
        ax.text(i, tot * 1.04, f"{tot:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x, [names[m] for m in gb.index])
    ax.set_ylabel("Data sent to hub (MiB)")
    ax.legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, bottom.max() * 1.15)
    ax = axs[1]
    b6 = g[g.index.str.startswith("B6")]
    ax.plot(b6.total_bytes / 2**20, b6.staleness_s, color="#6250d6", marker="o", label="B6 periodic sync")
    for m, r in b6.iterrows():
        ax.annotate(names[m].replace("\n", " "), (r.total_bytes / 2**20, r.staleness_s),
                    textcoords="offset points", xytext=(4, 3), fontsize=6, color=MUTED)
    st = g.loc["ours", "staleness_s"]
    e4s = load("e4_strong")            # ours: staleness = measured end-to-end delivery latency
    if e4s is not None:
        q = e4s[(e4s.method == "ours") & (e4s.load == "fixed")]
        if not q.empty:
            st = q.e2e_ms_p50.mean() / 1e3
    ax.scatter([g.loc["ours", "total_bytes"] / 2**20], [st],
               color=STYLE["ours"][0], marker="o", zorder=3, label="Ours (deltas)")
    ax.set_xscale("log")
    ax.set_xlabel("Data sent to hub (MiB, log)")
    ax.set_ylabel("Mean staleness at hub (s)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_e5_bytes")
    g.to_csv(os.path.join(TAB, "e5_bytes.csv"))


# ------------------------------------------------------------------------- E6
def e6():
    d = load("e6_noise")
    if d is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.2))
    for m in ("ours", "B3_no_correction", "B4_no_constraints"):
        line(axs[0], d, "sigma", "pw_f1", m)
    axs[0].set_xlabel("Re-ID noise σ"); axs[0].set_ylabel("Pairwise F1")
    axs[0].legend(loc="lower left")
    line(axs[1], d, "sigma", "corrections_per_min", "ours")
    axs[1].set_xlabel("Re-ID noise σ"); axs[1].set_ylabel("Corrections per minute (ours)")
    fig.tight_layout()
    save(fig, "fig_e6_noise")


# ------------------------------------------------------------------------- E7
def e7():
    d = load("e7_cross")
    if d is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.2))
    for m in ("ours_two_level", "ours_two_level_a"):
        line(axs[0], d, "p_cross", "agree_centralized", m)
    axs[0].set_ylabel("Agreement with centralized\n(pairwise F1)")
    for m in ("centralized", "ours_two_level", "ours_two_level_a"):
        line(axs[1], d, "p_cross", "pw_f1", m)
    axs[1].set_ylabel("Pairwise F1 vs ground truth")
    for ax in axs:
        ax.set_xlabel("Cross-site movement probability")
    h, l = axs[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig_e7_cross")


# ------------------------------------------------------------------------- E9
def e9():
    for name in ("e9_radius", "e9_radius_hard"):
        d = load(name)
        if d is not None:
            _e9(d, name)


def _e9(d, name):
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.75, 2.3))
    b = d[d.method == "ours_two_level_b"].groupby("radius")[["total_bytes", "pw_f1", "agree_centralized"]]
    g, e = b.mean(), b.std().fillna(0)
    c, mk, ls, lab = STYLE["ours_two_level_b"]
    for ax, y, yl in ((axs[0], "pw_f1", "Pairwise F1 vs ground truth"),
                      (axs[1], "agree_centralized", "Agreement with centralized")):
        ax.errorbar(g.total_bytes / 2**20, g[y], yerr=e[y], color=c, marker=mk, ls=ls, capsize=2,
                    elinewidth=0.8, label=lab + ", radius sweep")
        for r, row in g.iterrows():
            txt = "all" if r >= 1e8 else f"{r:g} s"
            ax.annotate(txt, (row.total_bytes / 2**20, row[y]), textcoords="offset points",
                        xytext=(3, -9), fontsize=6, color=MUTED)
        for m in ("ours_two_level_a", "centralized"):
            q = d[d.method == m]
            if q.empty or (m == "centralized" and y == "agree_centralized"):
                continue
            cc, mm, _, ll = STYLE[m]
            ax.errorbar([q.total_bytes.mean() / 2**20], [q[y].mean()], yerr=[q[y].std()],
                        color=cc, marker=mm, ls="none", capsize=2, label=ll + (" (B5 bytes)" if m == "centralized" else ""))
        ax.set_xlabel("Data sent site ↔ hub (MiB)")
        ax.set_ylabel(yl)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, f"fig_{name}")
    tab = d.groupby(["method", "radius"], dropna=False)[["vector_cams", "total_bytes", "pw_f1", "idf1",
                                                          "agree_centralized", "split_requests"]].mean()
    tab.to_csv(os.path.join(TAB, f"{name}.csv"))


# ------------------------------------------------------------------------- E8 / E4 / real-data tables
def _md(g, name):
    g.to_csv(os.path.join(TAB, f"{name}.csv"))
    with open(os.path.join(TAB, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(g.to_string(float_format=lambda v: f"{v:.4g}"))
    print("  table", name)


def e8():
    d = load("e8_neo4j")
    if d is None:
        return
    cols = ["n_deltas", "deltas_per_stream_s", "sink_deltas_per_s", "headroom",
            "flush_ms_p50", "flush_ms_p95", "flush_ms_p99", "graph_equals_hub"]
    g = d.groupby("batch")[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    _md(g, "e8_neo4j")


def e4_tab():
    for kind in ("weak", "strong"):
        d = load(f"e4_{kind}")
        if d is None:
            continue
        cols = ["throughput_ev_s", "e2e_ms_p50", "e2e_ms_p99", "hub_cpu_util", "hub_msgs", "split_requests"]
        _md(d.groupby(["load", "method", "n_sites"])[cols].mean(), f"e4_{kind}")


def real():
    d = load("real_wildtrack")
    if d is None:
        return
    cols = [c for c in ["pw_f1", "pw_p", "pw_r", "b3_f1", "idf1", "violations", "agree_centralized"] if c in d.columns]
    _md(d.groupby("method")[cols].mean(), "real_wildtrack")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for f in (e1, e2, e3, e4, e4_tab, e5, e6, e7, e8, e9, real):
        try:
            f()
        except Exception as ex:          # one broken figure should not block the others
            print(f"  {f.__name__} failed: {ex!r}")
