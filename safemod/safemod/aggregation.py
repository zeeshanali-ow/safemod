"""
aggregation.py
==============
Reference implementation of the aggregation methods compared in the paper.

The proposed operator implements, line for line, the equations of the
manuscript (Section III, Stages 2-5) and Algorithm 1:

    rho      = f * theta_r                                  (noise floor)
    E        = { s_i : s_i >= rho }                         (retained set)
    if E is empty:   S_c = max_i s_i                        (MAX fallback)
    u        = TopK(E, K)                                   (Top-K, descending)
    w_i      = exp((u_i - u_max)/tau) / sum_j exp((u_j - u_max)/tau)
    S_base   = sum_i w_i u_i                                (stable softmax pooling)
    N_R      = |{u_i >= theta_r}|,  N_M = |{rho <= u_i < theta_r}|
    A        = min{1, (N_R + 0.5 N_M) / K}                  (divides by configured K)
    Delta    = alpha * A * max{0, theta_m - S_base}
    S_pooled = min{1, S_base + Delta}
    S_c      = max{S_pooled, max_i s_i}                     (raw-maximum floor)

Decision:  S_c >= theta_m -> Remove;  theta_r <= S_c < theta_m -> Review;  else Allow.

Numerical note
--------------
All computation, every decision and every structural check use full IEEE-754
double precision. No value is rounded inside the operator (an earlier version
rounded S_c to 6 decimals, which could in principle move a score across a
threshold or a hair below max_i s_i; it was removed). Rounding is applied only
when numbers are printed in tables. Because S_c = max(S_pooled, max_i s_i) is a
floating-point max, S_c >= max_i s_i holds exactly. S_base is evaluated as
u_max - sum_i w_i (u_max - u_i) (identical to sum_i w_i u_i in exact arithmetic) so
that S_base <= u_max also holds exactly in floating point. The uplift bound is
checked to within BOUND_TOL = 1e-12 to allow for floating-point arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

ROUND_DECIMALS = None   # kept for manifest compatibility; no rounding is applied
BOUND_TOL = 1e-12       # floating-point tolerance used only for the uplift-bound check


@dataclass(frozen=True)
class OperatorParams:
    """Parameters of the proposed operator for one policy category."""
    f: float          # noise-floor factor, rho = f * theta_r
    K: int            # Top-K cap (also the denominator of A)
    tau: float        # softmax temperature (> 0)
    alpha: float      # bounded adjustment strength, domain [0, 0.30]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Thresholds:
    theta_r: float    # review threshold
    theta_m: float    # removal threshold, theta_r < theta_m <= 1


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def max_pool(scores, **_) -> float:
    return float(np.max(np.asarray(scores, dtype=float)))


def mean_pool(scores, **_) -> float:
    return float(np.mean(np.asarray(scores, dtype=float)))


def noisy_or(scores, **_) -> float:
    s = np.asarray(scores, dtype=float)
    return float(1.0 - np.prod(1.0 - s))


# ---------------------------------------------------------------------------
# Proposed operator
# ---------------------------------------------------------------------------
def proposed_operator(scores, params: OperatorParams, thr: Thresholds,
                      return_parts: bool = False):
    """Aggregate one category's mapped score multiset into S_c.

    If return_parts is True, a dict with the intermediate quantities is
    returned instead of the float (used for logging / audit examples).
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        out = {"S_c": 0.0, "branch": "empty"}
        return out if return_parts else 0.0

    s_max = float(np.max(s))
    rho = params.f * thr.theta_r
    retained = s[s >= rho]

    if retained.size == 0:
        S_c = s_max
        out ={"S_c": float(S_c), "branch": "max_fallback", "rho": rho, "s_max": s_max}
        return out if return_parts else float(S_c)

    u = np.sort(retained)[::-1][: params.K]
    u_max = float(u.max())
    z = np.exp((u - u_max) / params.tau)
    w = z / z.sum()
    # S_base = sum_i w_i u_i, evaluated in the algebraically identical form
    # u_max - sum_i w_i (u_max - u_i). The subtracted sum is non-negative in
    # floating point, so S_base <= u_max holds exactly; the direct weighted sum
    # can exceed u_max by one rounding step (e.g. five equal scores just below
    # theta_m), which would otherwise allow a spurious Remove.
    S_base = u_max - float(np.sum(w * (u_max - u)))

    N_R = int(np.sum(u >= thr.theta_r))
    N_M = int(np.sum((u >= rho) & (u < thr.theta_r)))
    A = min(1.0, (N_R + 0.5 * N_M) / params.K)
    delta = params.alpha * A * max(0.0, thr.theta_m - S_base)
    S_pooled = min(1.0, S_base + delta)
    S_c = max(S_pooled, s_max)          # full precision; no rounding

    if return_parts:
        return {"S_c": float(S_c), "branch": "pooled", "rho": rho, "u": u.tolist(),
                "weights": w.tolist(), "S_base": S_base, "N_R": N_R, "N_M": N_M,
                "A": A, "Delta": delta, "S_pooled": S_pooled, "s_max": s_max}
    return float(S_c)


def decide(score: float, thr: Thresholds) -> str:
    if score >= thr.theta_m:
        return "Remove"
    if score >= thr.theta_r:
        return "Review"
    return "Allow"


def uplift_bound(s_max: float, params: OperatorParams, thr: Thresholds) -> float:
    """Upper bound of the Proposition: S_c <= M + alpha * (theta_m - M)^+ ."""
    return s_max + params.alpha * max(0.0, thr.theta_m - s_max)


def escalation_zone(params: OperatorParams, thr: Thresholds) -> tuple[float, float]:
    """Interval of M = max_i s_i in which Allow -> Review escalation is possible."""
    lo = (thr.theta_r - params.alpha * thr.theta_m) / (1.0 - params.alpha)
    return max(0.0, lo), thr.theta_r


BASELINES = {"MAX": max_pool, "Mean": mean_pool, "Noisy-OR": noisy_or}
METHOD_ORDER = ["MAX", "Mean", "Noisy-OR", "Proposed"]


def aggregate_all(scores, params: OperatorParams, thr: Thresholds) -> dict:
    out = {name: fn(scores) for name, fn in BASELINES.items()}
    out["Proposed"] = proposed_operator(scores, params, thr)
    return out
