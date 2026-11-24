"""
eval_jigsaw.py  -  Step 2 (CPU): evaluate the aggregation methods on the scored
Jigsaw test set against its ground-truth labels.

Input : jigsaw_outputs/jigsaw_predictions.csv  (from score_jigsaw.py)
Output: jigsaw_outputs/eval/
  csv/jigsaw_decisions.csv       every mapping x preset x threshold policy x method:
                                 decision counts, flagged precision/recall/FPR/FNR,
                                 Remove precision/recall, Allow->Review count and
                                 its precision (share of escalations that are true
                                 violations), all with 95% Wilson intervals
  csv/jigsaw_score_quality.csv   AUC, Brier score and ECE of each aggregated score
  csv/jigsaw_model_labels.csv    per-label quality of the scoring model (disclosure)
  csv/jigsaw_presets.csv         all 15 presets at the Default policy
  latex/*.tex, figures/*.png|pdf, checks.csv, manifest.json

Usage:  python jigsaw/eval_jigsaw.py --pred jigsaw_outputs/jigsaw_predictions.csv
Everything is deterministic given the predictions file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safemod import config as C                                         # noqa: E402
from safemod.aggregation import (Thresholds, proposed_operator, decide,  # noqa: E402
                                 uplift_bound, escalation_zone, METHOD_ORDER)
from safemod.experiments import wilson                                   # noqa: E402

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ---------------------------------------------------------------------------
# Evaluation settings (all of them; change nothing elsewhere)
# ---------------------------------------------------------------------------
# Category mappings: model labels aggregated into one policy category, and the
# ground truth used for that category (an item is a violation if ANY listed
# ground-truth label is 1). Each label appears at most once per category.
MAPPINGS = {
    "Toxicity": {"labels": LABELS, "truth": LABELS},
    "Harassment": {"labels": ["toxic", "insult", "identity_hate", "threat"],
                   "truth": ["insult", "identity_hate", "threat"]},
}
PRIMARY_MAPPING = "Toxicity"

# Threshold policies (theta_r, theta_m), as in the previous revision's table
POLICIES = {
    "Highly Cautious": (0.20, 0.50),
    "Cautious": (0.35, 0.70),
    "Default": (0.40, 0.75),
    "Balanced": (0.50, 0.80),
    "Permissive": (0.55, 0.85),
}
MAIN_PRESETS = ["11", "3b"]          # General-Purpose Default, Zero-Tolerance Illegal (midpoints)
PRESET_SWEEP_POLICY = "Default"
ECE_BINS = 10
TOL = 0.0          # exact comparisons (no rounding anywhere)
BOUND_TOL = 1e-12  # floating-point tolerance for the uplift-bound check only


# ---------------------------------------------------------------------------
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties get half credit)."""
    r = pd.Series(score).rank(method="average").to_numpy()
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(score: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    idx = np.minimum((score * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(y[m].mean() - score[m].mean())
    return float(tot / len(y))


def pct_ci(k, n):
    lo, hi = wilson(int(k), int(n))
    return (100 * k / n if n else float("nan")), lo, hi


def aggregate(S: np.ndarray, preset: str, thr: Thresholds) -> dict:
    p = C.midpoint(preset)
    prop = np.array([proposed_operator(row, p, thr) for row in S])
    return {"MAX": S.max(axis=1), "Mean": S.mean(axis=1),
            "Noisy-OR": 1.0 - np.prod(1.0 - S, axis=1), "Proposed": prop}, p


def decisions(x: np.ndarray, thr: Thresholds) -> np.ndarray:
    return np.where(x >= thr.theta_m, 2, np.where(x >= thr.theta_r, 1, 0))  # 0 Allow, 1 Review, 2 Remove


def metrics_block(dec: np.ndarray, y: np.ndarray, dec_max: np.ndarray) -> dict:
    n = len(y)
    flag = dec >= 1
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    rem = dec == 2
    rtp, rfp = int((rem & (y == 1)).sum()), int((rem & (y == 0)).sum())
    a2r = (dec_max == 0) & (dec == 1)
    a2r_tp = int((a2r & (y == 1)).sum())
    out = {"N": n, "Positives": int(y.sum())}
    for name, code in (("Allow", 0), ("Review", 1), ("Remove", 2)):
        k = int((dec == code).sum())
        out[f"{name}_n"], out[f"{name}_pct"] = k, 100 * k / n
    for key, (k, m) in {"FlagPrecision": (tp, tp + fp), "FlagRecall": (tp, tp + fn),
                        "FPR": (fp, fp + tn), "FNR": (fn, fn + tp),
                        "RemovePrecision": (rtp, rtp + rfp), "RemoveRecall": (rtp, tp + fn)}.items():
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = pct_ci(k, m) if m else (float("nan"),) * 3
    out.update({"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "AllowToReview_n": int(a2r.sum()), "AllowToReview_pct": 100 * a2r.sum() / n,
                "Escalations_true": a2r_tp, "Escalations_false": int(a2r.sum()) - a2r_tp})
    e = pct_ci(a2r_tp, int(a2r.sum())) if a2r.any() else (float("nan"),) * 3
    out["EscalationPrecision"], out["EscalationPrecision_lo"], out["EscalationPrecision_hi"] = e
    return out


# ---------------------------------------------------------------------------
def run(pred_path: Path, out: Path):
    t0 = time.time()
    pred = pd.read_csv(pred_path)
    need = [f"p_{l}" for l in LABELS] + LABELS
    miss = [c for c in need if c not in pred.columns]
    if miss:
        sys.exit(f"Predictions file lacks columns: {miss}")
    (out / "csv").mkdir(parents=True, exist_ok=True)
    rows, quality, checks, preset_rows = [], [], [], []

    # Scoring-model disclosure: per-label AUC and F1 at 0.5
    lab = []
    for l in LABELS:
        y = pred[l].to_numpy(); s = pred[f"p_{l}"].to_numpy()
        yhat = s >= 0.5
        tp = int((yhat & (y == 1)).sum()); fp = int((yhat & (y == 0)).sum()); fn = int((~yhat & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        lab.append({"Label": l, "Positives": int(y.sum()), "Prevalence_pct": 100 * y.mean(),
                    "AUC": auc(s, y), "F1_at_0.5": f1})
    pd.DataFrame(lab).to_csv(out / "csv" / "jigsaw_model_labels.csv", index=False)

    for mname, mp in MAPPINGS.items():
        S = pred[[f"p_{l}" for l in mp["labels"]]].to_numpy(float)
        y = pred[mp["truth"]].max(axis=1).to_numpy(int)
        for preset in MAIN_PRESETS:
            for pol, (tr, tm) in POLICIES.items():
                thr = Thresholds(tr, tm)
                agg, p = aggregate(S, preset, thr)
                dmax = decisions(agg["MAX"], thr)
                for meth in METHOD_ORDER:
                    rec = {"Mapping": mname, "Preset": preset, "Policy": pol, "theta_r": tr, "theta_m": tm,
                           "Method": meth, **metrics_block(decisions(agg[meth], thr), y, dmax)}
                    if meth == "Proposed":
                        up = agg["Proposed"] - agg["MAX"]
                        rec["ScoresDiffer_pct"] = 100 * (up > TOL).mean()
                        rec["MeanUplift_when_differ"] = float(up[up > TOL].mean()) if (up > TOL).any() else 0.0
                        zlo, zhi = escalation_zone(p, thr)
                        rec["ZoneLow"], rec["ZoneHigh"] = zlo, zhi
                        rec["Items_in_zone"] = int(((agg["MAX"] >= zlo) & (agg["MAX"] < zhi)).sum())
                        # Matched baseline: MAX with theta_r lowered just enough to add the same
                        # number k of Review decisions (the k Allow items with the highest MAX score).
                        esc = (dmax == 0) & (decisions(agg["Proposed"], thr) == 1)
                        k = int(esc.sum())
                        allow_idx = np.where(dmax == 0)[0]
                        top = allow_idx[np.argsort(-agg["MAX"][allow_idx], kind="stable")[:k]]
                        mt = int(y[top].sum()) if k else 0
                        mp_, mlo, mhi = pct_ci(mt, k) if k else (float("nan"),) * 3
                        rec.update({"MatchedMAX_true": mt, "MatchedMAX_precision": mp_,
                                    "MatchedMAX_precision_lo": mlo, "MatchedMAX_precision_hi": mhi,
                                    "MatchedMAX_theta_r": float(agg["MAX"][top].min()) if k else float("nan")})
                        # How many labels each escalated item retains (>= rho): explains the mechanism
                        rho = p.f * tr
                        nret = (S[esc] >= rho).sum(axis=1) if k else np.array([])
                        rec["Escalated_single_label_pct"] = 100 * float((nret == 1).mean()) if k else float("nan")
                        rec["Escalated_median_MAX"] = float(np.median(agg["MAX"][esc])) if k else float("nan")
                        bnd = np.array([uplift_bound(m, p, thr) for m in agg["MAX"]])
                        dprop = decisions(agg["Proposed"], thr)
                        checks.append({"run": f"{mname}|{preset}|{pol}", "items": len(y),
                                       "S_c_below_MAX": int((-up > TOL).sum()),
                                       "S_c_above_bound": int((agg["Proposed"] - bnd > BOUND_TOL).sum()),
                                       "Remove_differs_from_MAX": int(((dprop == 2) != (dmax == 2)).sum()),
                                       "Review_to_Allow_vs_MAX": int(((dmax == 1) & (dprop == 0)).sum())})
                    rows.append(rec)
                # threshold-free score quality (depends on preset only through the proposed score)
                if pol == "Default":
                    for meth in METHOD_ORDER:
                        if meth != "Proposed" and preset != MAIN_PRESETS[0]:
                            continue
                        quality.append({"Mapping": mname, "Preset": preset if meth == "Proposed" else "-",
                                        "Method": meth, "AUC": auc(agg[meth], y),
                                        "Brier": float(np.mean((agg[meth] - y) ** 2)),
                                        "ECE": ece(agg[meth], y), "MeanScore": float(agg[meth].mean())})
        # all presets at one policy (supplementary)
        tr, tm = POLICIES[PRESET_SWEEP_POLICY]
        thr = Thresholds(tr, tm)
        for pid, (pname, *_r) in C.PRESETS.items():
            agg, p = aggregate(S, pid, thr)
            dmax = decisions(agg["MAX"], thr)
            m = metrics_block(decisions(agg["Proposed"], thr), y, dmax)
            mm = metrics_block(dmax, y, dmax)
            preset_rows.append({"Mapping": mname, "ID": pid, "Preset": pname, "f": p.f, "K": p.K, "tau": p.tau,
                                "alpha": p.alpha, "ZoneLow": escalation_zone(p, thr)[0],
                                "AllowToReview_n": m["AllowToReview_n"], "Escalations_true": m["Escalations_true"],
                                "EscalationPrecision": m["EscalationPrecision"],
                                "FlagRecall_MAX": mm["FlagRecall"], "FlagRecall_Prop": m["FlagRecall"],
                                "FPR_MAX": mm["FPR"], "FPR_Prop": m["FPR"]})
        print(f"[{time.time() - t0:5.0f} s] mapping '{mname}' done", flush=True)

    res = pd.DataFrame(rows); q = pd.DataFrame(quality); pr = pd.DataFrame(preset_rows); ch = pd.DataFrame(checks)
    res.to_csv(out / "csv" / "jigsaw_decisions.csv", index=False)
    q.to_csv(out / "csv" / "jigsaw_score_quality.csv", index=False)
    pr.to_csv(out / "csv" / "jigsaw_presets.csv", index=False)
    ch.to_csv(out / "checks.csv", index=False)

    import jigsaw_report as JR
    JR.write_all(out, res, q, pr, pd.DataFrame(lab), len(pred), PRIMARY_MAPPING, MAPPINGS, POLICIES, MAIN_PRESETS)

    bad = int(ch.drop(columns=["run", "items"]).to_numpy().sum())
    scoring_manifest = pred_path.parent / "scoring_manifest.json"
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "python": sys.version.split()[0],
                "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
                "predictions_file": pred_path.name, "predictions_sha256": sha256(pred_path),
                "scoring_manifest": json.loads(scoring_manifest.read_text()) if scoring_manifest.exists() else None,
                "items": len(pred), "mappings": MAPPINGS, "policies": POLICIES, "main_presets": MAIN_PRESETS,
                "preset_points": {k: C.midpoint(k).as_dict() for k in C.PRESETS},
                "seconds": round(time.time() - t0, 1),
                "outputs_sha256": {str(p.relative_to(out)): sha256(p) for p in files}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Structural checks: {'ALL PASSED' if bad == 0 else f'{bad} VIOLATIONS - see checks.csv'}")
    print(f"Done in {time.time() - t0:.0f} s. Outputs in {out.resolve()}")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", default="jigsaw_outputs/jigsaw_predictions.csv")
    ap.add_argument("--out", default=None, help="default: <pred folder>/eval")
    a = ap.parse_args()
    pred = Path(a.pred)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(run(pred, Path(a.out) if a.out else pred.parent / "eval"))


if __name__ == "__main__":
    main()
