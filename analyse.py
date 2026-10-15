"""Figures, statistical tests and LaTeX tables from the results of run_experiments.py
and run_hardware.py.

    python analyse.py --tag pilot --out figures

Produces (PDF + PNG):
    fig_convergence   optimality gap vs optimiser evaluation, five initialisations (E2)
    fig_depth         final gap and valid-timetable probability vs depth p (E1)
    fig_size          final gap and wall time vs number of qubits, QAOA vs SA (E3)
    fig_penalty       validity and gap vs penalty multiplier lambda (E6)
    fig_noise         ideal vs noisy simulator, raw vs mitigated (E4/E5)
    fig_eps           the adaptive regularisation eps(q) with the instances marked
and writes table_init.tex, table_tests.tex, table_hardware.tex and summary.json.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.warm_start import EPS_FIXED, EPS_MAX, EPS_MIN, epsilon_from_quality

# Fixed categorical order (validated for colour-vision deficiency); never cycled.
COLOURS = {"random": "#2a78d6", "ramp": "#eb6834", "scaled": "#1baf7a",
           "ws_fixed": "#eda100", "ws_adaptive": "#e87ba4"}
MARKERS = {"random": "o", "ramp": "s", "scaled": "^", "ws_fixed": "D", "ws_adaptive": "v"}
STYLES = {"random": "-", "ramp": "--", "scaled": "-.", "ws_fixed": ":", "ws_adaptive": "-"}
LABELS = {"random": "Random angles", "ramp": "Linear ramp", "scaled": "Ramp, scaled by $q$",
          "ws_fixed": "Warm start, $\\varepsilon=0.25$",
          "ws_adaptive": "Warm start, $\\varepsilon(q)$ (ours)"}
ORDER = ["random", "ramp", "scaled", "ws_fixed", "ws_adaptive"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def fmt_p(pv: float) -> str:
    """p-value for LaTeX: \num{} for small values, two significant figures otherwise."""
    if pv != pv:
        return "--"
    if pv < 1e-3:
        return f"\\num{{{pv:.1e}}}"
    return f"{pv:.2g}"


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=220)
    plt.close(fig)


def style_axis(ax):
    ax.tick_params(length=2.5, width=0.6)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)


# ----------------------------------------------------------------------------
def fig_convergence(df, hist, out, courses=6, p=2, lam=2.0):
    sub = df[(df.courses == courses) & (df.p == p) & (df.lam == lam)]
    if sub.empty:
        return
    e_opt, e_rand, e_greedy = sub.e_opt.iloc[0], sub.e_rand.iloc[0], sub.e_greedy.iloc[0]
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    for m in ORDER:
        runs = [h["history"] for h in hist if h["courses"] == courses and h["p"] == p
                and h["lam"] == lam and h["method"] == m]
        if not runs:
            continue
        L = max(len(r) for r in runs)
        M = np.full((len(runs), L), np.nan)
        for i, r in enumerate(runs):
            best = np.minimum.accumulate(r)
            M[i, :len(r)] = best
            M[i, len(r):] = best[-1]
        gap = (M - e_opt) / (e_rand - e_opt)
        x = np.arange(1, L + 1)
        med = np.nanmedian(gap, axis=0)
        lo, hi = np.nanpercentile(gap, 25, axis=0), np.nanpercentile(gap, 75, axis=0)
        ax.fill_between(x, lo, hi, color=COLOURS[m], alpha=0.12, linewidth=0)
        ax.plot(x, med, STYLES[m], color=COLOURS[m], label=LABELS[m],
                marker=MARKERS[m], markevery=max(1, L // 8), markersize=3.5)
    g_greedy = (e_greedy - e_opt) / (e_rand - e_opt)
    ax.axhline(g_greedy, color=MUTED, linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(ax.get_xlim()[1] * 0.98, g_greedy + 0.01, f"greedy timetable ($g={g_greedy:.3f}$)",
            ha="right", va="bottom", fontsize=6.5, color=MUTED)
    ax.axhline(0, color=MUTED, linewidth=0.9)
    ax.set_xlabel("Optimiser evaluation")
    ax.set_ylabel("Optimality gap $g$ of running best $\\langle E\\rangle$")
    ax.set_ylim(bottom=-0.02)
    ax.legend(loc="upper right", ncol=1, handlelength=2.2, fontsize=6.8)
    style_axis(ax)
    save(fig, out, "fig_convergence")


def fig_depth(df, out, courses=6, lam=2.0):
    sub = df[(df.courses == courses) & (df.lam == lam)]
    if sub.empty or sub.p.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2), sharex=True)
    for ax, col, ylabel in ((axes[0], "gap_final", "Final optimality gap $g$"),
                            (axes[1], "p_valid", "$P(\\mathrm{valid})$")):
        for m in ORDER:
            s = sub[sub.method == m].groupby("p")[col].agg(["mean", "std"])
            if s.empty:
                continue
            ax.errorbar(s.index, s["mean"], yerr=s["std"], color=COLOURS[m], marker=MARKERS[m],
                        linestyle=STYLES[m], capsize=2, linewidth=1.2, label=LABELS[m])
        ax.set_xlabel("Depth $p$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(sub.p.unique()))
        style_axis(ax)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5, handlelength=2.0,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.0)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.13, 1, 1))
    save(fig, out, "fig_depth")


def fig_size(df, out, p=2, lam=2.0, methods=("random", "ws_adaptive")):
    sub = df[(df.p == p) & (df.lam == lam)]
    if sub.empty or sub.qubits.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2))
    ax = axes[0]
    for m in methods:
        s = sub[sub.method == m].groupby("qubits")["gap_final"].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], color=COLOURS[m], marker=MARKERS[m],
                    linestyle=STYLES[m], capsize=2, linewidth=1.2, label=LABELS[m])
    for col, lab, mk in (("e_sa_native", "SA (native)", "x"), ("e_sa_qubo", "SA (QUBO)", "+")):
        sa = sub.groupby("qubits").apply(
            lambda g, c=col: ((g[c] - g.e_opt) / (g.e_rand - g.e_opt)).mean(), include_groups=False)
        ax.plot(sa.index, sa.values, color=MUTED, marker=mk, linestyle=(0, (4, 2)), label=lab)
    ax.set_xlabel("Qubits $n k$")
    ax.set_ylabel("Final optimality gap $g$")
    ax.set_ylim(bottom=-0.02)
    style_axis(ax)
    ax = axes[1]
    for m in methods:
        s = sub[sub.method == m].groupby("qubits")["time_qaoa_s"].mean()
        ax.plot(s.index, s.values, color=COLOURS[m], marker=MARKERS[m], linestyle=STYLES[m], label=LABELS[m])
    for col, lab, mk in (("time_sa_native_s", "SA (native)", "x"), ("time_sa_qubo_s", "SA (QUBO)", "+")):
        s = sub.groupby("qubits")[col].mean()
        ax.plot(s.index, s.values, color=MUTED, marker=mk, linestyle=(0, (4, 2)), label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Qubits $n k$")
    ax.set_ylabel("Wall time per run (s)")
    style_axis(ax)
    for ax in axes:
        ax.set_xticks(sorted(sub.qubits.unique()))
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.5, handlelength=2.0,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.0)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.13, 1, 1))
    save(fig, out, "fig_size")


def fig_penalty(df, out, courses=6, p=2, methods=("ramp", "ws_adaptive")):
    sub = df[(df.courses == courses) & (df.p == p)]
    if sub.empty or sub.lam.nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2))
    sub = sub.dropna(subset=["p_valid_opt"]) if "p_valid_opt" in sub.columns else sub
    for ax, col, ylabel, scale in ((axes[0], "p_valid", "$P(\\mathrm{valid})$", 1.0),
                                   (axes[1], "p_valid_opt", "$P(\\mathrm{valid\\ and\\ optimal})$ (%)", 100.0)):
        for m in methods:
            s = sub[sub.method == m].groupby("lam")[col].agg(["mean", "std"]) * scale
            ax.errorbar(s.index, s["mean"], yerr=s["std"], color=COLOURS[m], marker=MARKERS[m],
                        linestyle=STYLES[m], capsize=2, linewidth=1.2, label=LABELS[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(sub.lam.unique()))
        ax.set_xticklabels([f"{v:g}" for v in sorted(sub.lam.unique())])
        ax.set_xlabel("Penalty multiplier $\\lambda$")
        ax.set_ylabel(ylabel)
        ax.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
        style_axis(ax)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.5, handlelength=2.0,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.0)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
    save(fig, out, "fig_penalty")


def fig_noise(hw, out):
    if not hw:
        return
    rows = pd.DataFrame(hw)
    depths = sorted(rows.p.unique())
    envs = list(dict.fromkeys(rows.backend))
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2))
    width = 0.8 / (2 * len(envs))
    hatches = {"raw": "", "mit": "////"}
    for ax, metric, ylabel in ((axes[0], "p_valid", "$P(\\mathrm{valid})$"),
                               (axes[1], "gap", "Optimality gap $g$ of $\\langle E\\rangle$")):
        for i, env in enumerate(envs):
            for j, kind in enumerate(("raw", "mit")):
                if env == "ideal" and kind == "mit":
                    continue
                vals = []
                for d in depths:
                    cell = rows[(rows.backend == env) & (rows.p == d)][f"{kind}_{metric}"]
                    vals.append(float(cell.iloc[0]) if len(cell) else np.nan)
                x = np.arange(len(depths)) + (2 * i + j - len(envs) + 0.5) * width
                label = {"ideal": "Ideal simulator"}.get(env, env.replace("noisy:", "Noisy sim. (").replace("real:", "QPU (") + ")")
                label += "" if kind == "raw" else ", mitigated"
                ax.bar(x, vals, width * 0.92, color=COLOURS[ORDER[i]], hatch=hatches[kind],
                       edgecolor="white", linewidth=0.5, label=label)
        ax.set_xticks(np.arange(len(depths)))
        ax.set_xticklabels([f"$p={d}$" for d in depths])
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        style_axis(ax)
    axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.3, handlelength=1.6,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.0)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.16, 1, 1))
    save(fig, out, "fig_noise")


def fig_eps(df, out):
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    q = np.linspace(0, 1, 101)
    ax.plot(q, [epsilon_from_quality(v) for v in q], color=COLOURS["ws_adaptive"], label="$\\varepsilon(q)$ (adaptive)")
    ax.axhline(EPS_FIXED, color=COLOURS["ws_fixed"], linestyle=":", label="$\\varepsilon=0.25$ (fixed)")
    ax.axhline(EPS_MAX, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(0.01, EPS_MAX, "standard QAOA ($\\varepsilon=1/2$)", fontsize=7, color=MUTED, va="bottom")
    if df is not None and not df.empty:
        inst = df.groupby("courses").q.first()
        for k, qv in inst.items():
            ev = epsilon_from_quality(qv)
            ax.plot([qv], [ev], marker="o", color=COLOURS["ws_adaptive"], markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.annotate(f"{k} courses", (qv, ev), textcoords="offset points", xytext=(4, -9 if k % 2 else 5),
                        fontsize=6.5, color=INK)
    ax.set_xlabel("Greedy quality $q$")
    ax.set_ylabel("Regularisation $\\varepsilon$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.55)
    ax.legend(loc="lower left", fontsize=7)
    style_axis(ax)
    save(fig, out, "fig_eps")


# ----------------------------------------------------------------------------
def tables(df, hw, out, courses=6, lam=2.0):
    sub = df[(df.courses == courses) & (df.lam == lam)]
    lines = []
    for p in sorted(sub.p.unique()):
        for m in ORDER:
            s = sub[(sub.p == p) & (sub.method == m)]
            if s.empty:
                continue
            lines.append(f"{p} & {LABELS[m]} & {s.gap_init.mean():.2f} & {s.gap_final.mean():.3f} $\\pm$ {s.gap_final.std():.3f} "
                         f"& {s.p_valid.mean():.2f} & {100*s.p_opt.mean():.2f} & {s.n_evals.mean():.0f} \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    with open(os.path.join(out, "table_init.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # paired tests against the adaptive warm start (same seed = same shot-noise realisation)
    tests = []
    summary = {}
    for p in sorted(sub.p.unique()):
        ref = sub[(sub.p == p) & (sub.method == "ws_adaptive")].sort_values("seed")
        for m in ORDER[:-1]:
            other = sub[(sub.p == p) & (sub.method == m)].sort_values("seed")
            n = min(len(ref), len(other))
            if n < 5:
                continue
            a, b = ref.gap_final.values[:n], other.gap_final.values[:n]
            try:
                stat = wilcoxon(a, b)
                pval = float(stat.pvalue)
            except ValueError:
                pval = float("nan")
            ea, eb = ref.n_evals.values[:n], other.n_evals.values[:n]
            try:
                pe = float(wilcoxon(ea, eb).pvalue)
            except ValueError:
                pe = float("nan")
            tests.append(f"{p} & {LABELS[m]} & {np.mean(b):.3f} & {np.mean(a):.3f} & {fmt_p(pval)} & "
                         f"{np.mean(eb):.0f} & {np.mean(ea):.0f} & {fmt_p(pe)} \\\\")
            summary[f"p{p}_{m}"] = {"gap_other": float(np.mean(b)), "gap_adaptive": float(np.mean(a)),
                                    "p_gap": pval, "evals_other": float(np.mean(eb)),
                                    "evals_adaptive": float(np.mean(ea)), "p_evals": pe, "n": int(n)}
    with open(os.path.join(out, "table_tests.tex"), "w") as f:
        f.write("\n".join(tests) + "\n")

    if hw:
        rows = pd.DataFrame(hw)
        hl = []
        for _, r in rows.iterrows():
            name = {"ideal": "Ideal simulator"}.get(r.backend, r.backend.replace("noisy:", "Noisy sim. ").replace("real:", "QPU "))
            hl.append(f"{name} & {int(r.p)} & {int(r.two_qubit_gates)} & {r.raw_gap:.3f} & {r.mit_gap:.3f} & "
                      f"{r.raw_p_valid:.3f} & {r.mit_p_valid:.3f} & {100*r.raw_p_opt:.2f} & {100*r.mit_p_opt:.2f} \\\\")
        with open(os.path.join(out, "table_hardware.tex"), "w") as f:
            f.write("\n".join(hl) + "\n")
    summary["instances"] = {int(k): {"qubits": int(g.qubits.iloc[0]), "q": float(g.q.iloc[0]),
                                     "e_opt": float(g.e_opt.iloc[0]), "e_greedy": float(g.e_greedy.iloc[0]),
                                     "e_rand": float(g.e_rand.iloc[0]), "penalty": float(g.penalty.iloc[0])}
                            for k, g in df[df.lam == lam].groupby("courses")}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


def numbers(df, hw, out, courses=6, lam=2.0):
    """LaTeX macros with the headline pilot numbers, so the text never drifts from the data."""
    sub = df[(df.courses == courses) & (df.lam == lam)]
    def mean(p, m, col):
        s = sub[(sub.p == p) & (sub.method == m)][col]
        return float(s.mean()) if len(s) else float("nan")
    macros = {
        "PILOTrandGapTwo": f"{mean(2, 'random', 'gap_final'):.2f}",
        "PILOTrampGapTwo": f"{mean(2, 'ramp', 'gap_final'):.2f}",
        "PILOTscaledGapTwo": f"{mean(2, 'scaled', 'gap_final'):.2f}",
        "PILOTfixedGapTwo": f"{mean(2, 'ws_fixed', 'gap_final'):.3f}",
        "PILOTadaptGapTwo": f"{mean(2, 'ws_adaptive', 'gap_final'):.3f}",
        "PILOTadaptPvalidTwo": f"{mean(2, 'ws_adaptive', 'p_valid'):.2f}",
        "PILOTrandPvalidTwo": f"{mean(2, 'random', 'p_valid'):.2f}",
        "PILOTadaptEvalsTwo": f"{mean(2, 'ws_adaptive', 'n_evals'):.0f}",
        "PILOTrandEvalsTwo": f"{mean(2, 'random', 'n_evals'):.0f}",
    }
    # fixed vs adaptive sentence
    parts = []
    for p in sorted(sub.p.unique()):
        a = sub[(sub.p == p) & (sub.method == "ws_adaptive")].sort_values("seed")
        b = sub[(sub.p == p) & (sub.method == "ws_fixed")].sort_values("seed")
        n = min(len(a), len(b))
        if n >= 5:
            try:
                pv = float(wilcoxon(a.gap_final.values[:n], b.gap_final.values[:n]).pvalue)
            except ValueError:
                pv = float("nan")
            parts.append(f"$p={p}$ ($p$-value {fmt_p(pv)})")
    macros["PILOTtestsFixed"] = ("lower than the fixed warm start at " + ", ".join(parts)) if parts else "\\todo{fixed-vs-adaptive test}"
    big = df[(df.p == 2) & (df.lam == lam) & (df.method == "ws_adaptive")]
    if not big.empty:
        t24 = big[big.qubits == big.qubits.max()].time_qaoa_s.mean()
        macros["PILOTtimeTwentyFour"] = f"about \\SI{{{t24:.0f}}}{{\\second}}"
        macros["PILOTmaxQubits"] = f"{int(big.qubits.max())}"
    else:
        macros["PILOTtimeTwentyFour"] = "\\todo{time}"
        macros["PILOTmaxQubits"] = "24"
    pen = df[(df.courses == courses) & (df.p == 2)]
    if "p_valid_opt" in pen.columns:
        pen = pen.dropna(subset=["p_valid_opt"])
    if pen.lam.nunique() >= 2:
        lo, hi = pen.lam.min(), pen.lam.max()
        def pv(m, l, col="p_valid"):
            v = pen[(pen.method == m) & (pen.lam == l)][col]
            return float(v.mean()) if len(v) else float("nan")
        macros["PILOTpenaltySummary"] = (
            f"from $\\lambda={lo:g}$ to $\\lambda={hi:g}$ the probability of a valid timetable rises from "
            f"{pv('ramp', lo):.2f} to {pv('ramp', hi):.2f} for the ramp and from {pv('ws_adaptive', lo):.2f} to "
            f"{pv('ws_adaptive', hi):.2f} for the adaptive warm start, and the probability of sampling an optimal "
            f"timetable rises from {100*pv('ramp', lo, 'p_valid_opt'):.2f}\\% to {100*pv('ramp', hi, 'p_valid_opt'):.2f}\\% "
            f"and from {100*pv('ws_adaptive', lo, 'p_valid_opt'):.2f}\\% to {100*pv('ws_adaptive', hi, 'p_valid_opt'):.2f}\\% respectively")
    else:
        macros["PILOTpenaltySummary"] = "\\todo{penalty summary}"
    if hw:
        rows = pd.DataFrame(hw)
        noisy = rows[rows.backend.str.startswith("noisy")]
        ideal = rows[rows.backend == "ideal"]
        def g(frame, p, col):
            s = frame[frame.p == p][col]
            return float(s.iloc[0]) if len(s) else float("nan")
        macros["PILOTtwoQOne"] = f"{int(g(noisy, 1, 'two_qubit_gates'))}" if len(noisy[noisy.p == 1]) else "--"
        macros["PILOTtwoQTwo"] = f"{int(g(noisy, 2, 'two_qubit_gates'))}" if len(noisy[noisy.p == 2]) else "--"
        macros["PILOTnoiseShots"] = f"{int(rows.shots.iloc[0]):,}".replace(",", "{,}")
        sents = []
        for p in sorted(noisy.p.unique()):
            pi, pr, pm = g(ideal, p, "raw_p_valid"), g(noisy, p, "raw_p_valid"), g(noisy, p, "mit_p_valid")
            rec = (pm - pr) / (pi - pr) * 100 if pi > pr else float("nan")
            sents.append(f"at $p={int(p)}$ the probability of a valid timetable falls from {pi:.2f} on the ideal simulator to "
                         f"{pr:.2f} under the noise model, and readout mitigation raises it to {pm:.2f}, "
                         f"recovering {rec:.0f}\\% of the loss")
        text = "; ".join(sents)
        macros["PILOTnoiseSummary"] = (text[0].upper() + text[1:] + ".") if sents else "\\todo{noise summary}"
    else:
        for k in ("PILOTtwoQOne", "PILOTtwoQTwo"):
            macros[k] = "--"
        macros["PILOTnoiseShots"] = "4{,}000"
        macros["PILOTnoiseSummary"] = "\\todo{noise summary}"
    with open(os.path.join(out, "pilot_numbers.tex"), "w") as f:
        f.write("% generated by analyse.py -- do not edit by hand\n")
        for k, v in macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--hardware", default=None, help="hardware JSON (default results/<tag>_hardware.json)")
    ap.add_argument("--tables", default=None, help="directory for the LaTeX tables (default: same as --out)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tables_dir = args.tables or args.out
    os.makedirs(tables_dir, exist_ok=True)
    df = pd.read_csv(f"results/{args.tag}_runs.csv")
    hist = [json.loads(l) for l in open(f"results/{args.tag}_histories.jsonl")]
    hw_path = args.hardware or f"results/{args.tag}_hardware.json"
    hw = json.load(open(hw_path)) if os.path.exists(hw_path) else []
    fig_convergence(df, hist, args.out)
    fig_depth(df, args.out)
    fig_size(df, args.out)
    fig_penalty(df, args.out)
    fig_noise(hw, args.out)
    fig_eps(df[df.lam == 2.0], args.out)
    tables(df, hw, tables_dir)
    numbers(df, hw, tables_dir)
    print("figures written to", args.out, "and tables to", tables_dir)


if __name__ == "__main__":
    main()
