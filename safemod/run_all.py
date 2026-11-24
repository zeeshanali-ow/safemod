"""
run_all.py
==========
Regenerates every Monte Carlo table, figure and CSV in the paper.

    python run_all.py              # full run (N = 10,000 per scenario, ~10-20 min)
    python run_all.py --quick      # smoke test (N = 300), for checking the setup only
    python run_all.py --only main sub   # run selected experiments

Outputs go to ./outputs/ :
    csv/        every result table as CSV
    latex/      booktabs tables ready to \\input{} in the manuscript
    figures/    PNG (300 dpi) and PDF
    checks.csv  structural guarantees on every trial (all counts must be 0)
    reproduction.csv  comparison with the numbers already published (full run only)
    manifest.json     versions, settings, timings and SHA-256 of every output
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

from safemod import config as C
from safemod import experiments as E
from safemod import reporting as R

EXPERIMENTS = ["main", "sub", "sensitivity", "presets", "perturbation"]

# Values already published in the manuscript (R2), used to confirm exact reproduction.
PUBLISHED = {
    "sub_AllowToReview_pct": {"DuplicateSubReview": 25.26, "TightClusterNearReview": 12.03,
                              "TightClusterSubReview": 9.73, "LooseSpreadSubReview": 2.16,
                              "LooseSpreadWide": 0.09, "UniformCapped": 0.07, "DilutedBorderline": 0.0,
                              "ManyWeak": 0.0, "TightClusterNearRemove": 0.0},
    "sensitivity_Allow_pct": {("f", "min"): 34.05, ("f", "default"): 34.05, ("f", "max"): 34.04,
                              ("tau", "min"): 33.81, ("tau", "max"): 34.18,
                              ("K", "min"): 31.63, ("K", "max"): 36.18,
                              ("alpha", "min"): 36.20, ("alpha", "max"): 28.94},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small N for a smoke test")
    ap.add_argument("--n-trials", type=int, default=None, help="trials per scenario (default 10,000)")
    ap.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=EXPERIMENTS)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    n = args.n_trials or (300 if args.quick else C.N_TRIALS)
    n_pert = 1_000 if args.quick else C.PERTURB_TRIALS_PER_PRESET
    out = Path(args.out)
    (out / "csv").mkdir(parents=True, exist_ok=True)
    timings, checks, repro = {}, [], []

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    main_draws = None
    if "main" in args.only or "presets" in args.only:
        t0 = time.time()
        log(f"E1 main suite: 8 scenarios x {n:,} trials")
        df, dec, tr, chk, main_draws = E.main_suite(n_trials=n)
        df.to_csv(out / "csv" / "main_trials.csv.gz", index=False)
        dec.to_csv(out / "csv" / "main_decisions.csv", index=False)
        tr.to_csv(out / "csv" / "main_vs_max.csv", index=False)
        checks.append(chk)
        R.latex_main_spec(out, n)
        R.latex_overall(out, dec, "tab_main_overall", "tab:mc-main-overall",
                        "Main suite: decision shares of each method, all scenarios pooled.")
        R.latex_transitions(out, tr, "tab_main_vs_max", "tab:mc-main-vs-max",
                            "Main suite: Review share under MAX and the proposed operator, and Allow$\\to$Review "
                            "transitions, which equal the additional human-review volume "
                            "(10,000 trials per scenario).", capture=False)
        R.fig_decisions(out, dec, "fig_main_decisions", "Main suite, all scenarios")
        R.fig_scores(out, df, "fig_main_scores")
        R.fig_review_by_scenario(out, tr, "fig_main_review_by_scenario")
        timings["main"] = time.time() - t0

    if "sub" in args.only:
        t0 = time.time()
        log(f"E2 sub-threshold suite: 9 scenarios x {n:,} trials")
        df, dec, tr, chk = E.sub_suite(n_trials=n)
        df.to_csv(out / "csv" / "sub_trials.csv.gz", index=False)
        dec.to_csv(out / "csv" / "sub_decisions.csv", index=False)
        tr.to_csv(out / "csv" / "sub_vs_max.csv", index=False)
        checks.append(chk)
        R.latex_sub_spec(out, n)
        R.latex_transitions(out, tr, "tab_sub_vs_max", "tab:mc-sub-vs-max",
                            "Sub-threshold suite: Review share under MAX and the proposed operator, and "
                            "Allow$\\to$Review transitions, which equal the additional human-review volume "
                            "(10,000 trials per scenario).", capture=True)
        R.fig_capture(out, tr, "fig_sub_capture")
        R.fig_review_by_scenario(out, tr, "fig_sub_review_by_scenario")
        if n == C.N_TRIALS:
            for scen, pub in PUBLISHED["sub_AllowToReview_pct"].items():
                got = round(float(tr.set_index("Scenario").loc[scen, "AllowToReview_pct"]), 2)
                repro.append({"Item": f"Fig. 5 A->R {scen}", "Published": pub, "Reproduced": got, "Match": got == pub})
        timings["sub"] = time.time() - t0

    if "sensitivity" in args.only:
        t0 = time.time()
        log(f"E3 sensitivity: points, finite differences, curves, pairwise grids ({n:,} per scenario)")
        pts, fd, curves, grids = E.sensitivity(n_trials=n)
        pts.to_csv(out / "csv" / "sensitivity_points.csv", index=False)
        fd.to_csv(out / "csv" / "sensitivity_finite_differences.csv", index=False)
        curves.to_csv(out / "csv" / "sensitivity_curves.csv", index=False)
        grids.to_csv(out / "csv" / "sensitivity_pairwise.csv", index=False)
        R.latex_sensitivity(out, pts, fd, int(pts.N.iloc[0]))
        R.fig_sensitivity(out, curves, "fig_sensitivity_curves")
        R.fig_pairwise(out, grids, "fig_pairwise")
        if n == C.N_TRIALS:
            idx = pts.set_index(["Parameter", "Point"])
            for key, pub in PUBLISHED["sensitivity_Allow_pct"].items():
                got = round(float(idx.loc[key, "Allow_pct"]), 2)
                repro.append({"Item": f"Table IV Allow {key[0]} {key[1]}", "Published": pub,
                              "Reproduced": got, "Match": got == pub})
        timings["sensitivity"] = time.time() - t0

    if "presets" in args.only:
        t0 = time.time()
        log("E4 all 15 presets on the main suite")
        pr, pchk = E.preset_robustness(main_draws)
        pr.to_csv(out / "csv" / "presets_main_suite.csv", index=False)
        checks.extend(pchk)
        R.latex_presets(out, pr)
        timings["presets"] = time.time() - t0

    if "perturbation" in args.only:
        t0 = time.time()
        log(f"E5 perturbation: 15 presets x {n_pert:,} trials")
        ps, pp = E.perturbation(trials_per_preset=n_pert)
        ps.to_csv(out / "csv" / "perturbation_summary.csv", index=False)
        pp.to_csv(out / "csv" / "perturbation_by_preset.csv", index=False)
        R.latex_perturbation(out, ps, 15 * n_pert)
        timings["perturbation"] = time.time() - t0

    chk = pd.DataFrame(checks)
    chk.to_csv(out / "checks.csv", index=False)
    bad = chk.drop(columns=["run", "trials"]).to_numpy().sum() if len(chk) else 0
    if repro:
        pd.DataFrame(repro).to_csv(out / "reproduction.csv", index=False)

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "seed": C.SEED, "perturb_seed": C.PERTURB_SEED, "n_trials_per_scenario": n,
        "perturb_trials_per_preset": n_pert, "numerics": "IEEE-754 double precision throughout; no rounding inside the operator; uplift bound checked to 1e-12",
        "thresholds": vars(C.THRESHOLDS) if hasattr(C.THRESHOLDS, "__dict__") else str(C.THRESHOLDS),
        "reference_preset": C.REFERENCE_PRESET, "reference_params": C.REFERENCE_PARAMS.as_dict(),
        "experiments": args.only, "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "outputs_sha256": {str(p.relative_to(out)): sha256(p) for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    log(f"Structural checks: {'ALL PASSED' if bad == 0 else f'{bad} VIOLATIONS - see checks.csv'}")
    if repro:
        r = pd.DataFrame(repro)
        log(f"Reproduction of published numbers: {int(r.Match.sum())}/{len(r)} exact matches (reproduction.csv)")
    log(f"Done. Outputs in {out.resolve()}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
