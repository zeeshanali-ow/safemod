"""LaTeX tables and figures for the Jigsaw evaluation (numbers read from the results)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from safemod.aggregation import METHOD_ORDER
from safemod import config as C

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})


def _w(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _f(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def _ci(x, lo, hi):
    return "--" if pd.isna(x) else f"{x:.1f} [{lo:.1f}, {hi:.1f}]"


def _arrow(a, b, d=2):
    return f"{_f(a, d)} $\\to$ {_f(b, d)}"


def _esc(s):
    return s.replace("&", "\\&")


def tab_main(out, res, mapping, presets, policies, n, name, label, star=True):
    env = "table*" if star else "table"
    body = []
    for pid in presets:
        pname = _esc(C.PRESETS[pid][0])
        for i, pol in enumerate(policies):
            g = res[(res.Mapping == mapping) & (res.Preset == pid) & (res.Policy == pol)].set_index("Method")
            mx, pr = g.loc["MAX"], g.loc["Proposed"]
            first = f"\\multirow{{{len(policies)}}}{{*}}{{\\shortstack[l]{{{pid}\\\\{pname}}}}}" if i == 0 else ""
            body.append(f"{first} & {pol} & {pr.theta_r:.2f}/{pr.theta_m:.2f} & {_arrow(mx.Review_pct, pr.Review_pct)} & "
                        f"{int(pr.AllowToReview_n):,} & {int(pr.Escalations_true):,} & "
                        f"{_ci(pr.EscalationPrecision, pr.EscalationPrecision_lo, pr.EscalationPrecision_hi)} & "
                        f"{_f(pr.MatchedMAX_precision, 1)} & {_f(pr.Escalated_single_label_pct, 0)} & "
                        f"{_arrow(mx.FlagRecall, pr.FlagRecall)} & {_arrow(mx.FPR, pr.FPR)} \\\\")
        body.append("\\midrule")
    body = body[:-1]
    _w(out / "latex" / f"{name}.tex", f"""\\begin{{{env}}}[!t]
\\caption{{Jigsaw test set ({n:,} labelled comments), category ``{mapping}'': proposed operator versus MAX.
A$\\to$R: items moved from Allow (MAX) to Review; True: those that are labelled violations;
Esc.\\ precision: True / A$\\to$R with 95\\% Wilson interval. Matched: precision of MAX when $\\theta_r$ is lowered
just enough to add the same number of Review decisions. 1-label: share of escalated items with a single retained
label score. Recall and FPR treat Review or Remove as flagged.
Remove decisions are identical to MAX in every row.}}
\\label{{{label}}}
\\centering\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}lllcrrcrrcc@{{}}}}
\\toprule
\\textbf{{Preset}} & \\textbf{{Policy}} & $\\theta_r/\\theta_m$ & \\textbf{{Review \\%}} & \\textbf{{A$\\to$R}} & \\textbf{{True}} &
\\textbf{{Esc.\\ prec.\\ \\%}} & \\textbf{{Matched}} & \\textbf{{1-label}} & \\textbf{{Recall \\%}} & \\textbf{{FPR \\%}} \\\\
 & & & MAX$\\to$Prop. & & & [95\\% CI] & \\textbf{{\\%}} & \\textbf{{\\%}} & MAX$\\to$Prop. & MAX$\\to$Prop. \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{{env}}}
""")


def tab_methods(out, res, q, mapping, preset, policy, n):
    g = res[(res.Mapping == mapping) & (res.Preset == preset) & (res.Policy == policy)].set_index("Method")
    qq = q[q.Mapping == mapping]
    body = []
    for m in METHOD_ORDER:
        r = g.loc[m]
        qr = qq[(qq.Method == m) & ((qq.Preset == preset) if m == "Proposed" else True)].iloc[0]
        body.append(f"{m} & {_f(r.Allow_pct)} & {_f(r.Review_pct)} & {_f(r.Remove_pct)} & {_f(r.FlagPrecision, 1)} & "
                    f"{_f(r.FlagRecall, 1)} & {_f(r.FPR)} & {_f(r.RemovePrecision, 1)} & {_f(qr.AUC, 4)} & {_f(qr.ECE, 4)} \\\\")
    tr, tm = g.iloc[0].theta_r, g.iloc[0].theta_m
    _w(out / "latex" / "tab_jigsaw_methods.tex", f"""\\begin{{table*}}[!t]
\\caption{{Jigsaw test set ({n:,} comments), category ``{mapping}'', {policy} policy ($\\theta_r={tr}$, $\\theta_m={tm}$),
proposed operator at preset {preset} midpoint. Decision shares, flagged (Review or Remove) precision and recall,
false-positive rate, precision of automatic Remove, and threshold-free score quality (ROC AUC, expected calibration
error with 10 bins) against the ground-truth labels.}}
\\label{{tab:jigsaw-methods}}
\\centering\\scriptsize
\\begin{{tabular}}{{@{{}}lrrrrrrrrr@{{}}}}
\\toprule
\\textbf{{Method}} & \\textbf{{Allow\\%}} & \\textbf{{Review\\%}} & \\textbf{{Remove\\%}} & \\textbf{{Flag prec.\\%}} &
\\textbf{{Flag rec.\\%}} & \\textbf{{FPR\\%}} & \\textbf{{Rem.\\ prec.\\%}} & \\textbf{{AUC}} & \\textbf{{ECE}} \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


def tab_model(out, lab, n):
    body = "\n".join(f"{r.Label.replace('_', chr(92) + '_')} & {int(r.Positives):,} & {_f(r.Prevalence_pct)} & "
                     f"{_f(r.AUC, 4)} & {_f(r['F1_at_0.5'], 3)} \\\\" for _, r in lab.iterrows())
    _w(out / "latex" / "tab_jigsaw_model.tex", f"""\\begin{{table}}[!t]
\\caption{{Scoring model (Detoxify \\emph{{original}}, bert-base-uncased) on the Jigsaw test set ({n:,} comments):
per-label positives, prevalence, ROC AUC and F1 at a 0.5 cut-off.}}
\\label{{tab:jigsaw-model}}
\\centering\\footnotesize
\\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\\toprule
\\textbf{{Label}} & \\textbf{{Positives}} & \\textbf{{Prev.\\ \\%}} & \\textbf{{AUC}} & \\textbf{{F1@0.5}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_presets(out, pr, mapping, policy, n):
    g = pr[pr.Mapping == mapping]
    body = "\n".join(f"{r.ID} & {_esc(r.Preset)} & {r.alpha:.3f} & {r.ZoneLow:.3f} & {int(r.AllowToReview_n):,} & "
                     f"{int(r.Escalations_true):,} & {_f(r.EscalationPrecision, 1)} & {_arrow(r.FlagRecall_MAX, r.FlagRecall_Prop)} & "
                     f"{_arrow(r.FPR_MAX, r.FPR_Prop)} \\\\" for _, r in g.iterrows())
    _w(out / "latex" / "tab_jigsaw_presets.tex", f"""\\begin{{table*}}[!t]
\\caption{{Jigsaw test set ({n:,} comments), category ``{mapping}'', {policy} policy: every preset at its range midpoint.
Zone: lower edge of the escalation zone (upper edge $\\theta_r$).}}
\\label{{tab:jigsaw-presets}}
\\centering\\scriptsize
\\begin{{tabular}}{{@{{}}llrrrrrcc@{{}}}}
\\toprule
\\textbf{{ID}} & \\textbf{{Preset}} & $\\alpha$ & \\textbf{{Zone}} & \\textbf{{A$\\to$R}} & \\textbf{{True}} & \\textbf{{Esc.\\ prec.\\%}} &
\\textbf{{Recall \\% MAX$\\to$Prop.}} & \\textbf{{FPR \\% MAX$\\to$Prop.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


def fig_escalations(out, res, mapping, presets, policies):
    fig, axes = plt.subplots(1, len(presets), figsize=(7.0, 2.4), sharey=True, constrained_layout=True)
    for ax, pid in zip(np.atleast_1d(axes), presets):
        g = res[(res.Mapping == mapping) & (res.Preset == pid) & (res.Method == "Proposed")].set_index("Policy").loc[list(policies)]
        x = np.arange(len(g))
        ax.bar(x, g.Escalations_true, 0.6, color="#C44E52", label="labelled violation")
        ax.bar(x, g.Escalations_false, 0.6, bottom=g.Escalations_true, color="#9AA5B1", label="not a violation")
        for i, (t, f) in enumerate(zip(g.Escalations_true, g.Escalations_false)):
            ax.text(i, t + f, f"{int(t + f)}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [p.replace(" ", "\n") for p in g.index], fontsize=7)
        ax.set_title(f"Preset {pid} ({C.PRESETS[pid][0]})", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Allow$\\to$Review items")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / "fig_jigsaw_escalations.png", dpi=300)
    try:
        fig.savefig(out / "figures" / "fig_jigsaw_escalations.pdf")
    except Exception as exc:
        print(f"    (PDF copy skipped: {type(exc).__name__}; PNG saved)")
    plt.close(fig)


def write_all(out, res, q, pr, lab, n, primary, mappings, policies, presets):
    tab_main(out, res, primary, presets, policies, n, "tab_jigsaw_main", "tab:jigsaw-main")
    for m in mappings:
        if m != primary:
            tab_main(out, res, m, presets, policies, n, f"tab_jigsaw_{m.lower()}", f"tab:jigsaw-{m.lower()}")
    tab_methods(out, res, q, primary, presets[0], "Default", n)
    tab_model(out, lab, n)
    tab_presets(out, pr, primary, "Default", n)
    fig_escalations(out, res, primary, presets, policies)
