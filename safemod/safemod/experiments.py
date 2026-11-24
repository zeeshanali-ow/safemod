"""
experiments.py
==============
All Monte Carlo experiments reported in the paper. Each function returns
pandas DataFrames; run_all.py writes them to disk and renders tables/figures.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import config as C
from .aggregation import (OperatorParams, Thresholds, BASELINES, METHOD_ORDER,
                          proposed_operator, decide, uplift_bound, escalation_zone)
from .scenarios import MAIN_SUITE, SUB_SUITE, draw

DECISIONS = ["Allow", "Review", "Remove"]
TOL = 0.0    # comparisons against MAX are exact (full precision, no rounding)
from .aggregation import BOUND_TOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, returned in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (centre - half), 100 * (centre + half))


def evaluate(draws: dict, params: OperatorParams, thr: Thresholds,
             baselines: bool = True) -> pd.DataFrame:
    """One row per (scenario, trial) with every method's score and decision."""
    rows = []
    for scen, vectors in draws.items():
        for t, s in enumerate(vectors, start=1):
            m = float(np.max(s))
            row = {"Scenario": scen, "Trial": t, "n": len(s), "MaxScore": m,
                   "Proposed": proposed_operator(s, params, thr)}
            if baselines:
                for name, fn in BASELINES.items():
                    row[name] = fn(s)
            rows.append(row)
    df = pd.DataFrame(rows)
    methods = METHOD_ORDER if baselines else ["Proposed"]
    for mth in methods:
        df[f"{mth}_dec"] = [decide(x, thr) for x in df[mth]]
    if not baselines:
        df["MAX"] = df["MaxScore"]
        df["MAX_dec"] = [decide(x, thr) for x in df["MAX"]]
    df["Bound"] = [uplift_bound(m, params, thr) for m in df["MaxScore"]]
    return df


def property_checks(df: pd.DataFrame, label: str) -> dict:
    """Structural guarantees that must hold on every trial."""
    return {
        "run": label,
        "trials": len(df),
        "S_c_below_MAX": int(((df["MAX"] - df["Proposed"]) > TOL).sum()),
        "S_c_above_bound": int(((df["Proposed"] - df["Bound"]) > BOUND_TOL).sum()),
        "Remove_differs_from_MAX": int(((df["Proposed_dec"] == "Remove") != (df["MAX_dec"] == "Remove")).sum()),
        "Review_to_Allow_vs_MAX": int(((df["MAX_dec"] == "Review") & (df["Proposed_dec"] == "Allow")).sum()),
    }


def decision_table(df: pd.DataFrame, methods=METHOD_ORDER, by_scenario=True) -> pd.DataFrame:
    """Allow/Review/Remove counts and shares (with Wilson CIs) per method."""
    groups = list(df.groupby("Scenario")) if by_scenario else []
    groups.append(("ALL", df))
    rows = []
    for scen, g in groups:
        n = len(g)
        for mth in methods:
            rec = {"Scenario": scen, "Method": mth, "N": n}
            for d in DECISIONS:
                k = int((g[f"{mth}_dec"] == d).sum())
                lo, hi = wilson(k, n)
                rec[f"{d}_n"] = k
                rec[f"{d}_pct"] = 100 * k / n
                rec[f"{d}_lo"], rec[f"{d}_hi"] = lo, hi
            rows.append(rec)
    return pd.DataFrame(rows)


def transitions_vs_max(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario differences between the proposed operator and MAX.

    Because S_c >= MAX and Remove sets coincide, the only possible decision
    change is Allow -> Review, so AllowToReview_n is exactly the extra
    human-review volume.
    """
    groups = list(df.groupby("Scenario")) + [("ALL", df)]
    rows = []
    for scen, g in groups:
        n = len(g)
        a2r = int(((g["MAX_dec"] == "Allow") & (g["Proposed_dec"] == "Review")).sum())
        lo, hi = wilson(a2r, n)
        up = g["Proposed"] - g["MAX"]
        diff = up > TOL
        rows.append({
            "Scenario": scen, "N": n,
            "MAX_Review_pct": 100 * (g["MAX_dec"] == "Review").mean(),
            "Proposed_Review_pct": 100 * (g["Proposed_dec"] == "Review").mean(),
            "AllowToReview_n": a2r,
            "AllowToReview_pct": 100 * a2r / n, "AllowToReview_lo": lo, "AllowToReview_hi": hi,
            "MAX_Allow_n": int((g["MAX_dec"] == "Allow").sum()),
            "RemoveCrossed_n": int(((g["MAX"] < C.THRESHOLDS.theta_m) & (g["Proposed"] >= C.THRESHOLDS.theta_m)).sum()),
            "Downgraded_n": int((-up > TOL).sum()),
            "ScoresDiffer_pct": 100 * diff.mean(),
            "MeanUplift_all": float(up.mean()),
            "MeanUplift_when_differ": float(up[diff].mean()) if diff.any() else 0.0,
            "MaxUplift": float(up.max()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E1. Main suite (8 scenarios)
# ---------------------------------------------------------------------------
def main_suite(n_trials=C.N_TRIALS, seed=C.SEED, params=C.REFERENCE_PARAMS, thr=C.THRESHOLDS):
    draws = draw(MAIN_SUITE, n_trials, seed, reset_per_scenario=False)
    df = evaluate(draws, params, thr)
    return df, decision_table(df), transitions_vs_max(df), property_checks(df, "main_suite"), draws


# ---------------------------------------------------------------------------
# E2. Sub-threshold suite (9 scenarios): capture rates + review volume
# ---------------------------------------------------------------------------
def sub_suite(n_trials=C.N_TRIALS, seed=C.SEED, params=C.REFERENCE_PARAMS, thr=C.THRESHOLDS):
    draws = draw(SUB_SUITE, n_trials, seed, reset_per_scenario=False)
    df = evaluate(draws, params, thr)
    return df, decision_table(df), transitions_vs_max(df), property_checks(df, "sub_suite")


# ---------------------------------------------------------------------------
# E3. One-at-a-time sensitivity on the sub-threshold suite
# ---------------------------------------------------------------------------
def _with(params: OperatorParams, name: str, value) -> OperatorParams:
    return replace(params, **{name: int(value) if name == "K" else float(value)})


def _summarise(draws, params, thr, label: dict) -> dict:
    df = evaluate(draws, params, thr, baselines=False)
    n = len(df)
    rec = dict(label)
    for d in DECISIONS:
        k = int((df["Proposed_dec"] == d).sum())
        lo, hi = wilson(k, n)
        rec.update({f"{d}_pct": 100 * k / n, f"{d}_lo": lo, f"{d}_hi": hi})
    a2r = int(((df["MAX_dec"] == "Allow") & (df["Proposed_dec"] == "Review")).sum())
    rec.update({"N": n, "MeanScore": float(df["Proposed"].mean()),
                "MeanUplift": float((df["Proposed"] - df["MAX"]).mean()),
                "AllowToReview_n": a2r, "AllowToReview_pct": 100 * a2r / n})
    return rec


def sensitivity(n_trials=C.N_TRIALS, seed=C.SEED, base=C.REFERENCE_PARAMS, thr=C.THRESHOLDS):
    """Decision shares at min/default/max of each parameter + finite differences
    + fine-grained curves. Same draws for every parameter value (generator reset
    per scenario)."""
    draws = draw(SUB_SUITE, n_trials, seed, reset_per_scenario=True)

    points = []
    for name, (lo, default, hi) in C.SENSITIVITY_POINTS.items():
        for tag, v in (("min", lo), ("default", default), ("max", hi)):
            points.append(_summarise(draws, _with(base, name, v), thr,
                                     {"Parameter": name, "Point": tag, "Value": v}))
    points = pd.DataFrame(points)

    fd = []
    for name in C.SENSITIVITY_POINTS:
        g = points[points.Parameter == name].set_index("Point")
        fd.append({"Parameter": name, "Min": g.loc["min", "Value"], "Max": g.loc["max", "Value"],
                   "dMeanScore": g.loc["max", "MeanScore"] - g.loc["min", "MeanScore"],
                   "dAllow_pp": g.loc["max", "Allow_pct"] - g.loc["min", "Allow_pct"],
                   "AllowToReview_min_pct": g.loc["min", "AllowToReview_pct"],
                   "AllowToReview_max_pct": g.loc["max", "AllowToReview_pct"]})
    fd = pd.DataFrame(fd)

    curves = []
    for name, (lo, _, hi) in C.SENSITIVITY_POINTS.items():
        values = range(int(lo), int(hi) + 1) if name == "K" else np.linspace(lo, hi, C.SENSITIVITY_CURVE_POINTS)
        for v in values:
            curves.append(_summarise(draws, _with(base, name, v), thr, {"Parameter": name, "Value": float(v)}))
    curves = pd.DataFrame(curves)

    grids = []
    for (pa, pb), (va, vb) in C.PAIRWISE_GRIDS.items():
        for a in va:
            for b in vb:
                p = _with(_with(base, pa, a), pb, b)
                rec = _summarise(draws, p, thr, {"Pair": f"{pa}-{pb}", "A_name": pa, "A": a, "B_name": pb, "B": b})
                grids.append(rec)
    grids = pd.DataFrame(grids)
    return points, fd, curves, grids


# ---------------------------------------------------------------------------
# E4. Every preset (midpoint) on the main suite
# ---------------------------------------------------------------------------
def preset_robustness(main_draws: dict, thr=C.THRESHOLDS):
    rows, checks = [], []
    for pid, (name, *_r) in C.PRESETS.items():
        p = C.midpoint(pid)
        df = evaluate(main_draws, p, thr, baselines=False)
        n = len(df)
        a2r = int(((df["MAX_dec"] == "Allow") & (df["Proposed_dec"] == "Review")).sum())
        zlo, zhi = escalation_zone(p, thr)
        rows.append({"ID": pid, "Preset": name, "f": p.f, "K": p.K, "tau": p.tau, "alpha": p.alpha,
                     "ZoneLow": zlo, "ZoneHigh": zhi, "N": n,
                     **{f"{d}_pct": 100 * (df["Proposed_dec"] == d).mean() for d in DECISIONS},
                     "MAX_Allow_pct": 100 * (df["MAX_dec"] == "Allow").mean(),
                     "AllowToReview_n": a2r, "AllowToReview_pct": 100 * a2r / n,
                     "MeanUplift": float((df["Proposed"] - df["MAX"]).mean())})
        checks.append(property_checks(df, f"preset_{pid}"))
    return pd.DataFrame(rows), checks


# ---------------------------------------------------------------------------
# E5. Coordinate-wise perturbation (monotonicity behaviour)
# ---------------------------------------------------------------------------
def perturbation(trials_per_preset=C.PERTURB_TRIALS_PER_PRESET, seed=C.PERTURB_SEED, thr=C.THRESHOLDS):
    """For each of the 15 presets: sample (f, K, tau, alpha) uniformly within the
    preset's documented ranges, pick one of the 17 scenarios uniformly, draw s,
    pick one coordinate j uniformly, raise it by delta ~ U(0, PERTURB_DELTA_MAX)
    (capped at 1), and record the change in S_c and in the decision."""
    rng = np.random.default_rng(seed)
    gens = list(MAIN_SUITE.values()) + list(SUB_SUITE.values())
    rows = []
    for pid, (_, fr, Kr, tr, ar) in C.PRESETS.items():
        for _ in range(trials_per_preset):
            p = OperatorParams(f=float(rng.uniform(*fr)), K=int(rng.integers(Kr[0], Kr[1] + 1)),
                               tau=float(rng.uniform(*tr)), alpha=float(rng.uniform(*ar)))
            s = gens[int(rng.integers(len(gens)))](rng)
            j = int(rng.integers(len(s)))
            delta = float(rng.uniform(0.0, C.PERTURB_DELTA_MAX))
            s2 = s.copy()
            s2[j] = min(1.0, s[j] + delta)
            S1, S2 = proposed_operator(s, p, thr), proposed_operator(s2, p, thr)
            d1, d2 = decide(S1, thr), decide(S2, thr)
            rows.append({"ID": pid, "is_max": bool(s[j] >= s.max()), "dS": S2 - S1,
                         "dec_before": d1, "dec_after": d2})
    df = pd.DataFrame(rows)
    order = {"Allow": 0, "Review": 1, "Remove": 2}

    def summ(g, label):
        dec = g["dS"] < -TOL
        down = g["dec_after"].map(order) < g["dec_before"].map(order)
        return {"Subset": label, "Trials": len(g), "Decreases_n": int(dec.sum()),
                "Decreases_pct": 100 * dec.mean(), "MaxDecrease": float(-g["dS"].min()) if dec.any() else 0.0,
                "DecisionDowngrades_n": int(down.sum())}

    summary = [summ(df, "All perturbations"), summ(df[df.is_max], "Raised coordinate is the maximum"),
               summ(df[~df.is_max], "Raised coordinate is not the maximum")]
    by_preset = [dict(summ(g, pid), ID=pid) for pid, g in df.groupby("ID", sort=False)]
    return pd.DataFrame(summary), pd.DataFrame(by_preset)
