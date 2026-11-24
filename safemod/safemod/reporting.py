"""
reporting.py
============
Writes LaTeX tables (booktabs, IEEEtran-friendly) and figures (PNG + PDF).
All numbers in the tables are read from the result DataFrames: nothing typed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .aggregation import METHOD_ORDER, escalation_zone
from .scenarios import MAIN_SPEC, SUB_SPEC, MAIN_SUITE, SUB_SUITE, SIGNAL_CAP

PAL = {"Allow": "#4C9F70", "Review": "#E0874B", "Remove": "#C44E52"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.dpi": 150})


def _w(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save(fig, out: Path, name: str):
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / f"{name}.png", dpi=300)
    try:
        fig.savefig(out / "figures" / f"{name}.pdf")
    except Exception as exc:  # e.g. PDF backend blocked by a system policy
        print(f"    (PDF copy of {name} skipped: {type(exc).__name__}; PNG saved)")
    plt.close(fig)


def _p(x, d=2):
    return f"{x:.{d}f}"


def _ci(pct, lo, hi):
    return f"{pct:.2f} [{lo:.2f}, {hi:.2f}]"


def _esc(name: str) -> str:
    return name.replace("&", "\\&")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def latex_main_spec(out: Path, n_trials: int):
    p, t = C.REFERENCE_PARAMS, C.THRESHOLDS
    rows = "\n".join(f"{k} & {v[0]} & {v[1]} & {v[2]} \\\\" for k, v in MAIN_SPEC.items())
    _w(out / "latex" / "tab_main_spec.tex", f"""\\begin{{table*}}[!t]
\\caption{{Main Monte Carlo suite: generation of each score vector. {len(MAIN_SUITE)} scenarios $\\times$ {n_trials:,} trials;
one PCG64 generator, seed {C.SEED}, drawn in the row order shown; $\\theta_r={t.theta_r}$, $\\theta_m={t.theta_m}$;
operator at preset {C.REFERENCE_PRESET} midpoint ($f={p.f}$, $K={p.K}$, $\\tau={p.tau}$, $\\alpha={p.alpha}$).}}
\\label{{tab:mc-main-spec}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}llp{{6.8cm}}p{{4.6cm}}@{{}}}}
\\toprule
\\textbf{{Scenario}} & \\textbf{{$n$}} & \\textbf{{Generation}} & \\textbf{{Purpose}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


def latex_sub_spec(out: Path, n_trials: int):
    rows = "\n".join(f"{k} & {v[0]} & {v[1]} \\\\" for k, v in SUB_SPEC.items())
    _w(out / "latex" / "tab_sub_spec.tex", f"""\\begin{{table}}[!t]
\\caption{{Sub-threshold suite: generation of each score vector ({len(SUB_SUITE)} scenarios $\\times$ {n_trials:,} trials,
seed {C.SEED}, row order). Every score is clipped to $[0,{SIGNAL_CAP}]$, below $\\theta_m={C.THRESHOLDS.theta_m}$.}}
\\label{{tab:mc-sub-spec}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}llp{{4.6cm}}@{{}}}}
\\toprule
\\textbf{{Scenario}} & \\textbf{{$n$}} & \\textbf{{Generation}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def latex_overall(out: Path, dec: pd.DataFrame, name: str, label: str, caption: str):
    g = dec[dec.Scenario == "ALL"].set_index("Method").loc[METHOD_ORDER]
    rows = "\n".join(
        f"{m} & {_ci(r.Allow_pct, r.Allow_lo, r.Allow_hi)} & {_ci(r.Review_pct, r.Review_lo, r.Review_hi)} & "
        f"{_ci(r.Remove_pct, r.Remove_lo, r.Remove_hi)} \\\\" for m, r in g.iterrows())
    _w(out / "latex" / f"{name}.tex", f"""\\begin{{table}}[!t]
\\caption{{{caption} Percent of {int(g.N.iloc[0]):,} trials, 95\\% Wilson intervals.}}
\\label{{{label}}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
\\textbf{{Method}} & \\textbf{{Allow \\%}} & \\textbf{{Review \\%}} & \\textbf{{Remove \\%}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def latex_transitions(out: Path, tr: pd.DataFrame, name: str, label: str, caption: str, capture: bool):
    """Single-column per-scenario table: Review share under MAX and the proposed
    operator, and the Allow->Review count (= additional review volume) with CI."""
    body = []
    for _, r in tr.iterrows():
        if r.Scenario == "ALL":
            body.append("\\midrule")
            scen = "\\textbf{All}"
        else:
            scen = r.Scenario
        body.append(f"{scen} & {_p(r.MAX_Review_pct)} & {_p(r.Proposed_Review_pct)} & {int(r.AllowToReview_n):,} & "
                    f"{_p(r.AllowToReview_pct)} [{_p(r.AllowToReview_lo)}, {_p(r.AllowToReview_hi)}] \\\\")
    zeros = ""
    if capture:
        zeros = (f" Remove crossings: {int(tr[tr.Scenario != 'ALL'].RemoveCrossed_n.sum())}; "
                 f"scores below MAX: {int(tr[tr.Scenario != 'ALL'].Downgraded_n.sum())}.")
    _w(out / "latex" / f"{name}.tex", f"""\\begin{{table}}[!t]
\\caption{{{caption}{zeros}}}
\\label{{{label}}}
\\centering\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}lrrrc@{{}}}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{\\textbf{{Review \\%}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Allow$\\to$Review}}}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(l){{4-5}}
\\textbf{{Scenario}} & \\textbf{{MAX}} & \\textbf{{Prop.}} & \\textbf{{$n$}} & \\textbf{{\\% [95\\% CI]}} \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def latex_sensitivity(out: Path, pts: pd.DataFrame, fd: pd.DataFrame, n: int):
    sym = {"f": "$f$", "tau": "$\\tau$", "K": "$K$", "alpha": "$\\alpha$"}
    body = []
    for name in C.SENSITIVITY_POINTS:
        g = pts[pts.Parameter == name]
        for i, (_, r) in enumerate(g.iterrows()):
            first = f"\\multirow{{3}}{{*}}{{{sym[name]}}}" if i == 0 else ""
            v = int(r.Value) if name == "K" else f"{r.Value:.2f}"
            body.append(f"{first} & {r.Point} ({v}) & {_ci(r.Allow_pct, r.Allow_lo, r.Allow_hi)} & "
                        f"{_ci(r.Review_pct, r.Review_lo, r.Review_hi)} & {_p(r.AllowToReview_pct)} \\\\")
        body.append("\\midrule")
    body = body[:-1]
    _w(out / "latex" / "tab_sensitivity.tex", f"""\\begin{{table}}[!t]
\\caption{{Decision shares at the documented minimum, default and maximum of each parameter (sub-threshold suite,
{n:,} trials per point, identical draws for every point; Remove is 0\\% throughout). A$\\to$R: share of trials moved
from Allow under MAX to Review.}}
\\label{{tab:decision-sensitivity}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}llccr@{{}}}}
\\toprule
\\textbf{{Param.}} & \\textbf{{Point}} & \\textbf{{Allow \\% [CI]}} & \\textbf{{Review \\% [CI]}} & \\textbf{{A$\\to$R \\%}} \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")
    rows = "\n".join(
        f"{sym[r.Parameter]} & {('%d' % r.Min) if r.Parameter == 'K' else _p(r.Min)}--"
        f"{('%d' % r.Max) if r.Parameter == 'K' else _p(r.Max)} & {r.dMeanScore:+.4f} & {r.dAllow_pp:+.2f} & "
        f"{_p(r.AllowToReview_min_pct)} $\\to$ {_p(r.AllowToReview_max_pct)} \\\\" for _, r in fd.iterrows())
    _w(out / "latex" / "tab_finite_differences.tex", f"""\\begin{{table}}[!t]
\\caption{{Finite differences between the documented minimum and maximum of each parameter (others at default;
sub-threshold suite, {n:,} trials per point).}}
\\label{{tab:finite-differences}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}lcrrc@{{}}}}
\\toprule
\\textbf{{Param.}} & \\textbf{{Range}} & \\textbf{{$\\Delta$ mean $S_c$}} & \\textbf{{$\\Delta$ Allow (pp)}} & \\textbf{{A$\\to$R \\% (min$\\to$max)}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def latex_presets(out: Path, pr: pd.DataFrame):
    rows = "\n".join(
        f"{r.ID} & {_esc(r.Preset)} & {r.f:.3f} & {r.K} & {r.tau:.2f} & {r.alpha:.3f} & "
        f"[{r.ZoneLow:.3f}, {r.ZoneHigh:.2f}) & {_p(r.MAX_Allow_pct)} $\\to$ {_p(r.Allow_pct)} & {int(r.AllowToReview_n):,} \\\\"
        for _, r in pr.iterrows())
    _w(out / "latex" / "tab_presets.tex", f"""\\begin{{table*}}[!t]
\\caption{{Every preset at its range midpoint on the main suite ({int(pr.N.iloc[0]):,} trials, same draws).
Escalation zone: values of $M=\\max_i s_i$ for which Allow$\\to$Review is possible,
$[(\\theta_r-\\alpha\\theta_m)/(1-\\alpha),\\,\\theta_r)$. Remove shares equal MAX for every preset.}}
\\label{{tab:preset-robustness}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}llrrrrcrr@{{}}}}
\\toprule
\\textbf{{ID}} & \\textbf{{Preset}} & $f$ & $K$ & $\\tau$ & $\\alpha$ & \\textbf{{Escalation zone}} & \\textbf{{Allow \\% MAX$\\to$Prop.}} & \\textbf{{A$\\to$R}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


def latex_perturbation(out: Path, ps: pd.DataFrame, total: int):
    rows = "\n".join(
        f"{r.Subset} & {int(r.Trials):,} & {int(r.Decreases_n):,} ({r.Decreases_pct:.2f}\\%) & "
        f"{r.MaxDecrease:.4f} & {int(r.DecisionDowngrades_n):,} \\\\" for _, r in ps.iterrows())
    _w(out / "latex" / "tab_perturbation.tex", f"""\\begin{{table}}[!t]
\\caption{{Coordinate-wise perturbation study: {total:,} trials (15 presets, parameters drawn uniformly within each
preset's ranges; one coordinate raised by $\\delta\\sim U(0,{C.PERTURB_DELTA_MAX})$; seed {C.PERTURB_SEED}).
A decrease is $\\Delta S_c<0$ (exact, double precision).}}
\\label{{tab:perturbation}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
\\textbf{{Subset}} & \\textbf{{Trials}} & \\textbf{{Decreases}} & \\textbf{{Max decrease}} & \\textbf{{Decision lowered}} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_decisions(out: Path, dec: pd.DataFrame, name: str, title: str):
    g = dec[dec.Scenario == "ALL"].set_index("Method").loc[METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    bottom = np.zeros(len(g))
    for d in ["Allow", "Review", "Remove"]:
        vals = g[f"{d}_pct"].to_numpy()
        ax.bar(g.index, vals, bottom=bottom, color=PAL[d], label=d, width=0.65)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax.text(i, b + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white")
        bottom += vals
    ax.set_ylabel("% of trials")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, out, name)


def fig_scores(out: Path, df: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    data = [df[m].to_numpy() for m in METHOD_ORDER]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for b in parts["bodies"]:
        b.set_alpha(0.5)
    ax.axhline(C.THRESHOLDS.theta_r, ls="--", lw=0.8, color="grey")
    ax.axhline(C.THRESHOLDS.theta_m, ls=":", lw=0.8, color="grey")
    ax.set_xticks(range(1, len(METHOD_ORDER) + 1), METHOD_ORDER)
    ax.set_ylabel("Aggregated score")
    _save(fig, out, name)


def fig_review_by_scenario(out: Path, tr: pd.DataFrame, name: str):
    g = tr[tr.Scenario != "ALL"]
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.bar(x - 0.2, g.MAX_Review_pct, 0.4, label="MAX", color="#9AA5B1")
    ax.bar(x + 0.2, g.Proposed_Review_pct, 0.4, label="Proposed", color=PAL["Review"])
    ax.set_xticks(x, g.Scenario, rotation=30, ha="right")
    ax.set_ylabel("Review share (%)")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, out, name)


def fig_capture(out: Path, tr: pd.DataFrame, name: str):
    g = tr[tr.Scenario != "ALL"].sort_values("AllowToReview_pct", ascending=False)
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    err = np.clip(np.vstack([g.AllowToReview_pct - g.AllowToReview_lo,
                             g.AllowToReview_hi - g.AllowToReview_pct]), 0, None)
    ax.bar(x, g.AllowToReview_pct, 0.6, yerr=err, capsize=2, color=PAL["Review"], label="Allow $\\to$ Review")
    ax.set_xticks(x, g.Scenario, rotation=30, ha="right")
    ax.set_ylabel("% of trials")
    crossed, down = int(g.RemoveCrossed_n.sum()), int(g.Downgraded_n.sum())
    ax.set_title(f"Remove crossings: {crossed}; scores below MAX: {down}", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    _save(fig, out, name)


def fig_sensitivity(out: Path, curves: pd.DataFrame, name: str):
    sym = {"f": "$f$", "tau": "$\\tau$", "K": "$K$", "alpha": "$\\alpha$"}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True)
    for ax, p in zip(axes, C.SENSITIVITY_POINTS):
        g = curves[curves.Parameter == p]
        ax.plot(g.Value, g.AllowToReview_pct, marker="o", ms=3, color=PAL["Review"])
        ax.set_xlabel(sym[p])
    axes[0].set_ylabel("Allow$\\to$Review (%)")
    _save(fig, out, name)


def fig_pairwise(out: Path, grids: pd.DataFrame, name: str):
    sym = {"f": "$f$", "tau": "$\\tau$", "K": "$K$", "alpha": "$\\alpha$"}
    pairs = list(grids.Pair.unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(7.2, 2.5), constrained_layout=True)
    for ax, pair in zip(np.atleast_1d(axes), pairs):
        g = grids[grids.Pair == pair]
        piv = g.pivot(index="A", columns="B", values="AllowToReview_pct")
        im = ax.imshow(piv.to_numpy(), aspect="auto", origin="lower", cmap="Oranges")
        ax.set_xticks(range(piv.shape[1]), [f"{v:.3g}" for v in piv.columns], fontsize=6)
        ax.set_yticks(range(piv.shape[0]), [f"{v:.3g}" for v in piv.index], fontsize=6)
        ax.set_xlabel(sym[g.B_name.iloc[0]])
        ax.set_ylabel(sym[g.A_name.iloc[0]])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.iat[i, j]:.1f}", ha="center", va="center", fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, label="A$\\to$R (%)")
    _save(fig, out, name)
