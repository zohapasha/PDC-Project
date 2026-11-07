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

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# validated categorical palette, fixed slot order, assigned per method (never cycled)
# method -> (colour, marker, line style, label)
METHOD_STYLE = {
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
INK_COLOR, MUTED_COLOR, GRID_COLOR = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED_COLOR, "axes.labelcolor": INK_COLOR, "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR, "axes.grid": True, "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
})
SINGLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH = 3.5, 7.16   # IEEE column widths (inches)
MIB = 2**20


def load_results(name):
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def plot_method_line(ax, results, x_column, y_column, method, label=None):
    color, marker, line_style, default_label = METHOD_STYLE[method]
    stats = results[results.method == method].groupby(x_column)[y_column].agg(["mean", "std"]).reset_index()
    if stats.empty:
        return
    ax.errorbar(stats[x_column], stats["mean"], yerr=stats["std"].fillna(0), color=color,
                marker=marker, ls=line_style, capsize=2, elinewidth=0.8, label=label or default_label)


def save_figure(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), dpi=200)
    plt.close(fig)
    print("  figure", name)


def mean_std(series):
    return f"{series.mean():.3f} ± {series.std():.3f}"


# ------------------------------------------------------------------------- E1 table
def e1():
    for name in ("e1_accuracy", "e1_accuracy_hard"):
        results = load_results(name)
        if results is None:
            continue
        method_order = ["ours", "ours_two_level", "ours_two_level_a", "B1_offline",
                        "B3_no_correction", "B4_no_constraints"]
        columns = [("pw_f1", "Pairwise F1"), ("b3_f1", "B³ F1"), ("idf1", "IDF1"),
                   ("violations", "Violations"), ("agree_offline", "Agreement w/ B1")]
        table_rows = []
        for method in method_order:
            method_results = results[results.method == method]
            if method_results.empty:
                continue
            cells = [METHOD_STYLE[method][3]]
            for column, _ in columns:
                values = method_results[column]
                cells.append(f"{values.mean():.1f} ± {values.std():.1f}" if column == "violations"
                             else mean_std(values))
            table_rows.append(cells)
        header = ["Method"] + [title for _, title in columns]
        mean_tracklets = results.n_tracklets.mean()
        with open(os.path.join(TABLES_DIR, f"{name}.md"), "w", encoding="utf-8") as file:
            file.write(f"{name}: {results.seed.nunique()} seeds, ~{mean_tracklets:,.0f} tracklets/run\n\n")
            file.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
            for cells in table_rows:
                file.write("| " + " | ".join(cells) + " |\n")
        with open(os.path.join(TABLES_DIR, f"{name}.tex"), "w", encoding="utf-8") as file:
            file.write("\\begin{tabular}{l" + "c" * len(columns) + "}\n\\toprule\n")
            file.write(" & ".join(title.replace("³", "$^3$") for title in header) + " \\\\\n\\midrule\n")
            for cells in table_rows:
                file.write(" & ".join(cell.replace("±", "$\\pm$") for cell in cells) + " \\\\\n")
            file.write("\\bottomrule\n\\end{tabular}\n")
        print("  table", name)


# ------------------------------------------------------------------------- E2
def e2():
    results = load_results("e2_incremental")
    if results is None:
        return
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.3))
    plot_method_line(ax, results, "n_tracklets", "update_ms_p50", "ours", "Ours, p50 per update")
    color, marker, _, _ = METHOD_STYLE["ours"]
    p99 = results[results.method == "ours"].groupby("n_tracklets")["update_ms_p99"].agg(["mean", "std"])
    ax.errorbar(p99.index, p99["mean"], yerr=p99["std"].fillna(0), color=color, marker=marker,
                ls="--", mfc="white", capsize=2, elinewidth=0.8, label="Ours, p99 per update")
    plot_method_line(ax, results, "n_tracklets", "update_ms_p50", "B1_full_recompute",
                     "B1, one full recompute")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tracklets processed (graph size)")
    ax.set_ylabel("Update latency (ms)")
    ax.legend(loc="upper left")
    save_figure(fig, "fig_e2_incremental")


# ------------------------------------------------------------------------- E3
def e3():
    summary = load_results("e3_corrections")
    op_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "e3_ops_e3_corrections_*.parquet")))
    if summary is None or not op_files:
        return
    ops = pd.concat([pd.read_parquet(path) for path in op_files])
    corrections = ops[ops.kind.isin(["conflict_ok", "del_tree"])]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH * 0.75, 2.3))
    ax = axes[0]
    for method in ("B2_component", "ours"):
        sample = corrections[corrections.method == method]
        sample = sample.sample(min(len(sample), 4000), random_state=0)
        color, marker, _, label = METHOD_STYLE[method]
        ax.scatter(sample.person_size, sample.touched, s=7, color=color, marker=marker, alpha=0.45,
                   edgecolors="none", label=label)
    axis_limit = max(corrections.person_size.max(), corrections.touched.max())
    ax.plot([1, axis_limit], [1, axis_limit], color=MUTED_COLOR, lw=0.8, ls=":",
            label="touched = person size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Size of the person(s) involved")
    ax.set_ylabel("Vertices touched per correction")
    ax.legend(loc="upper left", markerscale=1.5)
    ax = axes[1]
    for method in ("ours", "B2_component"):
        latencies = np.sort(corrections[corrections.method == method].latency_us.values)
        color, _, line_style, label = METHOD_STYLE[method]
        ax.plot(latencies, np.arange(1, len(latencies) + 1) / len(latencies), color=color,
                ls=line_style, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Correction latency (µs)")
    ax.set_ylabel("CDF")
    ax.legend(loc="upper left")
    save_figure(fig, "fig_e3_corrections")
    columns = ["n_corrections", "n_conflict_ok", "n_conflict_rej", "n_deletions", "frac_del_hit_backup",
               "frac_treedel_repaired_by_backup", "touched_mean", "touched_over_size_max", "msgs_mean",
               "corr_us_p50", "corr_us_p99", "tree_depth_mean", "tree_depth_max", "pw_f1"]
    means = summary.groupby("method")[columns].mean().T
    means.to_csv(os.path.join(TABLES_DIR, "e3_summary.csv"))
    with open(os.path.join(TABLES_DIR, "e3_summary.md"), "w", encoding="utf-8") as file:
        file.write(means.to_string(float_format=lambda value: f"{value:.4g}"))
    print("  table e3_summary")


# ------------------------------------------------------------------------- E4
def e4():
    for kind in ("weak", "strong"):
        results = load_results(f"e4_{kind}")
        if results is None:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN_WIDTH, 2.2))
        max_load, fixed_load = results[results.load == "max"], results[results.load == "fixed"]
        for method in ("ours", "B5_centralized"):
            plot_method_line(axes[0], max_load, "n_sites", "throughput_ev_s", method)
            plot_method_line(axes[1], fixed_load, "n_sites", "e2e_ms_p99", method)
            plot_method_line(axes[2], max_load, "n_sites", "hub_cpu_util", method)
        axes[0].set_ylabel("Max throughput (events/s)")
        axes[1].set_ylabel("End-to-end p99 latency (ms)\nat fixed load")
        axes[1].set_yscale("log")
        axes[2].set_ylabel("Hub CPU utilisation\nat max load")
        for ax in axes:
            ax.set_xlabel("Sites (processes)")
            ax.set_xticks(sorted(results.n_sites.unique()))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
        save_figure(fig, f"fig_e4_{kind}")


# ------------------------------------------------------------------------- E5
def e5():
    results = load_results("e5_bytes")
    if results is None:
        return
    byte_columns = [column for column in results.columns if column.startswith("bytes_")]
    means = results.groupby("method")[byte_columns + ["total_bytes", "staleness_s"]].mean()
    method_order = [method for method in ["ours", "B5_centralized", "B6_periodic_T1",
                                          "B6_periodic_T5", "B6_periodic_T30"] if method in means.index]
    means = means.loc[method_order]
    short_names = {"ours": "Ours\n(deltas)", "B5_centralized": "B5\ncentralized",
                   "B6_periodic_T1": "B6\nT=1 s", "B6_periodic_T5": "B6\nT=5 s",
                   "B6_periodic_T30": "B6\nT=30 s"}
    # message types: fixed colour per type (slot order), legend + table view in CSV
    message_type_color = {
        "bytes_BorderTracklet": "#2a78d6", "bytes_BorderRevise": "#eb6834",
        "bytes_TrackletRecord": "#1baf7a", "bytes_EdgeAdd": "#eda100", "bytes_EdgeDel": "#e87ba4",
        "bytes_FullTracklet": "#008300", "bytes_FullRevise": "#6250d6", "bytes_SiteTable": "#e34948",
        "bytes_SplitRequest": "#8a5a00"}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 2.3),
                             gridspec_kw={"width_ratios": [2, 2]})
    ax = axes[0]
    # composition panel: ours, B5 and the cheapest B6; all B6 periods are on the right panel
    composition = means.loc[[method for method in ["ours", "B5_centralized", "B6_periodic_T30"]
                             if method in means.index]]
    bar_positions = np.arange(len(composition))
    bar_bottom = np.zeros(len(composition))
    for column in [column for column in message_type_color if column in composition.columns]:
        mib = composition[column].fillna(0).values / MIB
        if mib.sum() == 0:
            continue
        ax.bar(bar_positions, mib, bottom=bar_bottom, color=message_type_color[column], width=0.62,
               edgecolor="white", linewidth=1, label=column.replace("bytes_", ""))
        bar_bottom += mib
    for position, total in enumerate(bar_bottom):
        ax.text(position, total * 1.04, f"{total:.1f}", ha="center", va="bottom", fontsize=6.5,
                color=INK_COLOR)
    ax.set_xticks(bar_positions, [short_names[method] for method in composition.index])
    ax.set_ylabel("Data sent to hub (MiB)")
    ax.legend(fontsize=6, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(0, bar_bottom.max() * 1.15)
    ax = axes[1]
    periodic = means[means.index.str.startswith("B6")]
    ax.plot(periodic.total_bytes / MIB, periodic.staleness_s, color="#6250d6", marker="o",
            label="B6 periodic sync")
    for method, row in periodic.iterrows():
        ax.annotate(short_names[method].replace("\n", " "), (row.total_bytes / MIB, row.staleness_s),
                    textcoords="offset points", xytext=(4, 3), fontsize=6, color=MUTED_COLOR)
    our_staleness = means.loc["ours", "staleness_s"]
    e4_strong = load_results("e4_strong")   # ours: staleness = measured end-to-end delivery latency
    if e4_strong is not None:
        fixed_load_ours = e4_strong[(e4_strong.method == "ours") & (e4_strong.load == "fixed")]
        if not fixed_load_ours.empty:
            our_staleness = fixed_load_ours.e2e_ms_p50.mean() / 1e3
    ax.scatter([means.loc["ours", "total_bytes"] / MIB], [our_staleness],
               color=METHOD_STYLE["ours"][0], marker="o", zorder=3, label="Ours (deltas)")
    ax.set_xscale("log")
    ax.set_xlabel("Data sent to hub (MiB, log)")
    ax.set_ylabel("Mean staleness at hub (s)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "fig_e5_bytes")
    means.to_csv(os.path.join(TABLES_DIR, "e5_bytes.csv"))


# ------------------------------------------------------------------------- E6
def e6():
    results = load_results("e6_noise")
    if results is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH * 0.75, 2.2))
    for method in ("ours", "B3_no_correction", "B4_no_constraints"):
        plot_method_line(axes[0], results, "sigma", "pw_f1", method)
    axes[0].set_xlabel("Re-ID noise σ")
    axes[0].set_ylabel("Pairwise F1")
    axes[0].legend(loc="lower left")
    plot_method_line(axes[1], results, "sigma", "corrections_per_min", "ours")
    axes[1].set_xlabel("Re-ID noise σ")
    axes[1].set_ylabel("Corrections per minute (ours)")
    fig.tight_layout()
    save_figure(fig, "fig_e6_noise")


# ------------------------------------------------------------------------- E7
def e7():
    results = load_results("e7_cross")
    if results is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH * 0.75, 2.2))
    for method in ("ours_two_level", "ours_two_level_a"):
        plot_method_line(axes[0], results, "p_cross", "agree_centralized", method)
    axes[0].set_ylabel("Agreement with centralized\n(pairwise F1)")
    for method in ("centralized", "ours_two_level", "ours_two_level_a"):
        plot_method_line(axes[1], results, "p_cross", "pw_f1", method)
    axes[1].set_ylabel("Pairwise F1 vs ground truth")
    for ax in axes:
        ax.set_xlabel("Cross-site movement probability")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save_figure(fig, "fig_e7_cross")


# ------------------------------------------------------------------------- E9
def e9():
    for name in ("e9_radius", "e9_radius_hard"):
        results = load_results(name)
        if results is not None:
            _plot_radius_sweep(results, name)


def _plot_radius_sweep(results, name):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH * 0.75, 2.3))
    by_radius = results[results.method == "ours_two_level_b"].groupby("radius")[
        ["total_bytes", "pw_f1", "agree_centralized"]]
    means, stds = by_radius.mean(), by_radius.std().fillna(0)
    color, marker, line_style, label = METHOD_STYLE["ours_two_level_b"]
    for ax, y_column, y_label in ((axes[0], "pw_f1", "Pairwise F1 vs ground truth"),
                                  (axes[1], "agree_centralized", "Agreement with centralized")):
        ax.errorbar(means.total_bytes / MIB, means[y_column], yerr=stds[y_column], color=color,
                    marker=marker, ls=line_style, capsize=2, elinewidth=0.8,
                    label=label + ", radius sweep")
        for radius, row in means.iterrows():
            radius_text = "all" if radius >= 1e8 else f"{radius:g} s"
            ax.annotate(radius_text, (row.total_bytes / MIB, row[y_column]),
                        textcoords="offset points", xytext=(3, -9), fontsize=6, color=MUTED_COLOR)
        for method in ("ours_two_level_a", "centralized"):
            reference = results[results.method == method]
            if reference.empty or (method == "centralized" and y_column == "agree_centralized"):
                continue
            ref_color, ref_marker, _, ref_label = METHOD_STYLE[method]
            ax.errorbar([reference.total_bytes.mean() / MIB], [reference[y_column].mean()],
                        yerr=[reference[y_column].std()], color=ref_color, marker=ref_marker,
                        ls="none", capsize=2,
                        label=ref_label + (" (B5 bytes)" if method == "centralized" else ""))
        ax.set_xlabel("Data sent site ↔ hub (MiB)")
        ax.set_ylabel(y_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    save_figure(fig, f"fig_{name}")
    table = results.groupby(["method", "radius"], dropna=False)[
        ["vector_cams", "total_bytes", "pw_f1", "idf1", "agree_centralized", "split_requests"]].mean()
    table.to_csv(os.path.join(TABLES_DIR, f"{name}.csv"))


# ------------------------------------------------------------------------- E8 / E4 / real-data tables
def _write_table(table, name):
    table.to_csv(os.path.join(TABLES_DIR, f"{name}.csv"))
    with open(os.path.join(TABLES_DIR, f"{name}.md"), "w", encoding="utf-8") as file:
        file.write(table.to_string(float_format=lambda value: f"{value:.4g}"))
    print("  table", name)


def e8():
    results = load_results("e8_neo4j")
    if results is None:
        return
    columns = ["n_deltas", "deltas_per_stream_s", "sink_deltas_per_s", "headroom",
               "flush_ms_p50", "flush_ms_p95", "flush_ms_p99", "graph_equals_hub"]
    table = results.groupby("batch")[columns].agg(["mean", "std"])
    table.columns = [f"{column}_{statistic}" for column, statistic in table.columns]
    _write_table(table, "e8_neo4j")


def e4_table():
    for kind in ("weak", "strong"):
        results = load_results(f"e4_{kind}")
        if results is None:
            continue
        columns = ["throughput_ev_s", "e2e_ms_p50", "e2e_ms_p99", "hub_cpu_util", "hub_msgs",
                   "split_requests"]
        _write_table(results.groupby(["load", "method", "n_sites"])[columns].mean(), f"e4_{kind}")


def real_data_table():
    results = load_results("real_wildtrack")
    if results is None:
        return
    columns = [column for column in ["pw_f1", "pw_p", "pw_r", "b3_f1", "idf1", "violations",
                                     "agree_centralized"] if column in results.columns]
    _write_table(results.groupby("method")[columns].mean(), "real_wildtrack")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for make in (e1, e2, e3, e4, e4_table, e5, e6, e7, e8, e9, real_data_table):
        try:
            make()
        except Exception as error:          # one broken figure should not block the others
            print(f"  {make.__name__} failed: {error!r}")
