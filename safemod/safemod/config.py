"""
config.py
=========
Every number that defines an experiment lives here. Change nothing elsewhere.
"""

from .aggregation import OperatorParams, Thresholds

SEED = 42
N_TRIALS = 10_000                    # per scenario, every suite
THRESHOLDS = Thresholds(theta_r=0.30, theta_m=0.70)

# ---------------------------------------------------------------------------
# Policy presets: parameter RANGES exactly as in Table III of the manuscript.
# (f range), (K range), (tau range), (alpha range)
# ---------------------------------------------------------------------------
PRESETS = {
    "1":  ("Ultra-Strict Auto-Remove",        (0.75, 0.90), (3, 3), (0.08, 0.12), (0.05, 0.10)),
    "2a": ("Review-First",                    (0.45, 0.60), (5, 5), (0.16, 0.25), (0.10, 0.18)),
    "2b": ("High Recall to Review",           (0.25, 0.45), (5, 5), (0.12, 0.18), (0.12, 0.22)),
    "3a": ("Safety-First (Youth)",            (0.20, 0.40), (5, 5), (0.08, 0.12), (0.20, 0.30)),
    "3b": ("Zero-Tolerance Illegal",          (0.10, 0.30), (5, 5), (0.08, 0.12), (0.15, 0.25)),
    "4a": ("Marketplace Anti-Scam",           (0.55, 0.75), (3, 3), (0.10, 0.14), (0.05, 0.10)),
    "4b": ("Spam/SEO",                        (0.70, 0.90), (3, 3), (0.12, 0.18), (0.00, 0.08)),
    "4c": ("Fake Reviews",                    (0.60, 0.80), (3, 5), (0.12, 0.18), (0.05, 0.12)),
    "5":  ("Audit-First",                     (0.60, 0.80), (3, 3), (0.10, 0.14), (0.05, 0.12)),
    "6":  ("Automation-First",                (0.40, 0.60), (5, 5), (0.10, 0.14), (0.15, 0.25)),
    "7":  ("False-Positive Shield",           (0.80, 0.95), (3, 3), (0.12, 0.18), (0.00, 0.08)),
    "8":  ("Crisis-Sensitive",                (0.20, 0.35), (5, 5), (0.08, 0.12), (0.20, 0.30)),
    "9":  ("NSFW-Strict",                     (0.45, 0.65), (5, 5), (0.10, 0.14), (0.10, 0.20)),
    "10": ("Phishing & Impersonation Guard",  (0.55, 0.75), (3, 5), (0.10, 0.14), (0.05, 0.12)),
    "11": ("General-Purpose Default",         (0.55, 0.70), (3, 5), (0.12, 0.18), (0.08, 0.15)),
}


def midpoint(preset_id: str) -> OperatorParams:
    """Point value of a preset = midpoint of each range (K rounded down)."""
    _, f, K, tau, a = PRESETS[preset_id]
    return OperatorParams(f=round((f[0] + f[1]) / 2, 4), K=int((K[0] + K[1]) // 2),
                          tau=round((tau[0] + tau[1]) / 2, 4), alpha=round((a[0] + a[1]) / 2, 4))


# Reference configuration used by both Monte Carlo suites and the sensitivity
# study: preset 3b (Zero-Tolerance Illegal) at its midpoint.
REFERENCE_PRESET = "3b"
REFERENCE_PARAMS = midpoint(REFERENCE_PRESET)        # f=0.20, K=5, tau=0.10, alpha=0.20

# ---------------------------------------------------------------------------
# One-at-a-time sensitivity: documented min / default / max of preset 3b,
# with K and alpha extended to the extremes used elsewhere in Table III.
# ---------------------------------------------------------------------------
SENSITIVITY_POINTS = {
    "f":     (0.10, REFERENCE_PARAMS.f, 0.30),
    "tau":   (0.08, REFERENCE_PARAMS.tau, 0.12),
    "K":     (3, REFERENCE_PARAMS.K, 8),
    "alpha": (0.15, REFERENCE_PARAMS.alpha, 0.30),
}
SENSITIVITY_CURVE_POINTS = 9          # evenly spaced points for f, tau, alpha curves

# Pairwise interaction grids (decision-relevant metric: Allow->Review rate)
PAIRWISE_GRIDS = {
    ("K", "alpha"):   ([3, 4, 5, 6, 7, 8], [0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]),
    ("tau", "alpha"): ([0.08, 0.09, 0.10, 0.11, 0.12], [0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]),
}

# ---------------------------------------------------------------------------
# Perturbation study (coordinate-wise response)
# ---------------------------------------------------------------------------
PERTURB_TRIALS_PER_PRESET = 30_000    # x 15 presets = 450,000 trials
PERTURB_DELTA_MAX = 0.10              # increase delta ~ U(0, 0.10), result capped at 1
PERTURB_SEED = 2026
