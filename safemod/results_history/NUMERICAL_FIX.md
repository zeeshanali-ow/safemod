# Numerical-precision correction (19 Nov 2026)

## What was wrong
1. `proposed_operator` rounded S_c to 6 decimals before the decision. Rounding could
   (a) move a score within 5e-7 of a threshold across it, and (b) put S_c up to 5e-7 below
   max_i s_i, contradicting the exact MAX lower bound.
2. Independently of rounding, S_base = sum_i w_i u_i can exceed u_max by one floating-point
   step (e.g. five equal scores 0.69999999999999998 sum to exactly 0.7 with theta_m = 0.7),
   which can produce a spurious Remove.
3. The structural checks used a tolerance of 1e-6, which hid (1b).

## Fix
- No rounding inside the operator; decisions and checks use full double precision.
- S_base is evaluated as u_max - sum_i w_i (u_max - u_i), algebraically identical to
  sum_i w_i u_i, so S_base <= u_max holds exactly in floating point.
- Checks are exact (tolerance 0) for S_c >= max_i s_i, Remove equivalence and no
  Review->Allow; the uplift bound is checked to 1e-12 (floating-point arithmetic).
- New unit tests: scores one floating-point step below theta_m and theta_r for all
  combinations of theta, alpha, K, f, n; 50,000 random near-threshold Remove-equivalence cases.

## Effect on results (old = rounded code, new = corrected code)
| Item | Old | New |
|---|---|---|
| Monte Carlo trials with S_c < max_i s_i (exact check) | 35,712 (all by < 5e-7) | 0 |
| Jigsaw item evaluations with S_c < max_i s_i (exact check) | 94,904 of 1,279,560 (all by < 5e-7) | 0 |
| Decisions changed, main and sub-threshold suites | - | 0 |
| Decisions changed, Jigsaw (all 20 configurations) | - | 0 |
| Sensitivity, alpha = 0.15: Allow->Review | 3.39% | 3.38% (one rounding-induced escalation removed) |
| Pairwise grid cells with a changed count | - | 7 cells, 1 trial each |
| Perturbation decreases (Table S3) | 7,786 | 7,788 (1.73% / 2.42% unchanged) |
| Published values reproduced (Fig. 5, Table IV) | 18/18 | 18/18 |
| Every other count, decision and table | - | identical |

Full file-by-file comparisons: `compare_montecarlo_v1_vs_v2.txt`, `compare_jigsaw_v1_vs_v2.txt`.
Previous manifests are kept as `manifest_*_v1_rounded.json`.
