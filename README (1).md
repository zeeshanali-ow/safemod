# SafeMod: Policy-Governed Evidence Fusion for Deterministic Content Moderation

This repository contains the reference implementation, experiments, data and generated results for the paper:

> **Policy-Governed Evidence Fusion for Deterministic Content Moderation: A Hybrid Aggregation Framework for Multi-Label Moderation Signals**
> IEEE Transactions on Artificial Intelligence (under review, manuscript TAI-2026-Mar-A-00466).

Every Monte Carlo table and figure in the paper can be regenerated with one command. The real-data evaluation on the Jigsaw benchmark can be repeated from the included predictions file without a GPU.

---

## 1. What SafeMod does

A moderation model typically returns a confidence score for each of several policy labels (for example `toxic`, `insult`, `threat`). An organization groups these labels into policy categories, and each category needs one decision: **Allow**, **Review** (send to a human) or **Remove**.

SafeMod is a deterministic, policy-conditioned operator that turns the label scores that **one selected moderation model** produces for **one content item** into a category-level decision:

| Stage | Operation |
|---|---|
| 1. Category mapping | Each mapped label contributes at most one score in [0, 1] to its category. |
| 2. Noise floor | Scores below ρ = f · θr are discarded. If none remain, the operator falls back to MAX. |
| 3. Top-K | At most K retained scores are pooled. |
| 4. Softmax pooling | Temperature-τ softmax-weighted mean S_base of the retained scores. |
| 5. Bounded adjustment | A = min{1, (N_R + 0.5·N_M)/K}; S_pooled = min{1, S_base + α·A·max{0, θm − S_base}}. |
| Final score | S_c = max{S_pooled, max_i s_i}: never below the strongest input. |
| Decision | Remove if S_c ≥ θm; Review if S_c ≥ θr; otherwise Allow. |

The parameters (f, K, τ, α, θr, θm) are fixed, interpretable presets that encode moderation postures (15 presets in the paper). Nothing is trained or calibrated at run time.

**Guarantees proven in the paper and checked on every trial in this repository:**
- **MAX lower bound:** S_c ≥ max_i s_i.
- **No aggregation-only removal:** Remove occurs exactly when an individual score reaches θm, so Remove decisions are identical to MAX.
- **Bounded uplift (Proposition 1):** max_i s_i ≤ S_c ≤ max_i s_i + α(θm − max_i s_i)⁺. Relative to MAX, the only possible decision change is Allow → Review.

**What SafeMod does not claim.** On the Jigsaw benchmark, SafeMod does not detect violations better than simply lowering the MAX review threshold to send the same number of items to review. Its contribution is bounded, deterministic and auditable behavior, not higher accuracy.

The count factor A measures label-count support. Reading it as corroboration from distinct evidence requires the labels mapped to a category to be non-redundant. The operator does not test this, and it does not hold for the six closely related Jigsaw labels.

---

## 2. Repository layout

```
.
├── README.md                      this file
├── safemod/                       code (start with safemod/README.md for full details)
│   ├── safemod/aggregation.py     MAX, mean, Noisy-OR and the SafeMod operator; decision rule; bounds
│   ├── safemod/scenarios.py       all 17 synthetic score generators
│   ├── safemod/config.py          seeds, trial counts, thresholds, the 15 presets
│   ├── safemod/experiments.py     experiments E1-E5 and the structural checks
│   ├── safemod/reporting.py       LaTeX tables and figures
│   ├── run_all.py                 one command for all Monte Carlo experiments
│   ├── tests/test_operator.py     9 unit tests (worked examples, bounds, threshold boundaries)
│   ├── jigsaw/score_jigsaw.py     scores the Jigsaw test set with Detoxify (GPU)
│   ├── jigsaw/eval_jigsaw.py      evaluates all methods from the predictions file (CPU)
│   ├── tools/                     output comparison and the alpha > 1 check
│   ├── results_history/           record of the numerical-precision correction
│   ├── SafeMod_MonteCarlo_Colab.ipynb
│   ├── SafeMod_Jigsaw_Colab.ipynb
│   └── requirements.txt
└── outputs/
    ├── montecarlo/                all Monte Carlo CSVs, LaTeX tables, figures, checks, manifest
    ├── jigsaw_scoring/            jigsaw_predictions.csv, scoring_manifest.json, DATA_NOTICE.md
    └── jigsaw_eval/               all Jigsaw CSVs, LaTeX tables, figures, checks, manifest
```

---

## 3. Quick start

Requirements: Python 3.10 or later, with numpy, pandas and matplotlib. Tested with Python 3.13, numpy 2.5.3, pandas 3.0.5 and matplotlib 3.11.2.

```bash
cd safemod
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python tests/test_operator.py                      # 9 unit tests, ~15 s
python run_all.py --quick --out outputs_quick      # smoke test, ~15 s
python run_all.py --out outputs_regenerated        # full Monte Carlo run, ~10-20 min on a laptop CPU
python tools/compare_outputs.py ../outputs/montecarlo outputs_regenerated   # compare with the published results
```

All commands are run from the `safemod/` folder. `--out` sets the output folder (default `outputs`, created inside `safemod/`); the published results in `outputs/montecarlo/` at the repository root are not overwritten.

**Repeat the Jigsaw evaluation from the included predictions (no GPU, no Kaggle download):**
```bash
python jigsaw/eval_jigsaw.py --pred ../outputs/jigsaw_scoring/jigsaw_predictions.csv --out jigsaw_eval_regenerated
python tools/compare_outputs.py ../outputs/jigsaw_eval jigsaw_eval_regenerated
```

**Re-score Jigsaw from scratch (GPU):** open `safemod/SafeMod_Jigsaw_Colab.ipynb` in Google Colab with a T4 GPU and follow path B. It installs Detoxify 0.5.4 and downloads the data through the Kaggle API, after you have accepted the competition rules once on Kaggle.

**Google Colab:** both notebooks (`safemod/SafeMod_MonteCarlo_Colab.ipynb`, `safemod/SafeMod_Jigsaw_Colab.ipynb`) clone this repository in their first cell; nothing needs to be uploaded.

---

## 4. Experiments and where they appear in the paper

| ID | Experiment | Size | Paper |
|---|---|---|---|
| E1 | Main suite: 8 scenarios; MAX, Mean, Noisy-OR and SafeMod on identical draws | 8 × 10,000 = 80,000 | Tables IV, VII; Figs. 2-3 |
| E2 | Sub-threshold suite: 9 scenarios, every score below θm | 9 × 10,000 = 90,000 | Tables V, VIII; Fig. 4 |
| E3 | Sensitivity: min/default/max of f, K, τ, α; finite differences; K-α and τ-α grids | 90,000 per point | Tables IX-X; Fig. 5 |
| E4 | All 15 presets (range midpoints) on the main suite | 15 × 80,000 | Suppl. Table S1 |
| E5 | Coordinate-wise perturbation: one score raised by δ ~ U(0, 0.10) | 15 × 30,000 = 450,000 | Section III-C; Suppl. Table S3 |
| J1 | Jigsaw test set: 4 methods × presets 11 and 3b × 5 threshold policies, plus a review-volume-matched MAX baseline | 63,978 comments | Tables XI-XII; Suppl. Tables S4-S6, Fig. S1 |

Fixed Monte Carlo settings (`safemod/safemod/config.py`):
- Seed 42; the perturbation study uses seed 2026.
- θr = 0.30 and θm = 0.70.
- Reference preset 3b at its range midpoint: f = 0.20, K = 5, τ = 0.10, α = 0.20.
- 95% Wilson intervals throughout.

---

## 5. Synthetic evaluation data

### Why synthetic data?

The synthetic data is not meant to reproduce the Jigsaw dataset. It provides controlled score scenarios that isolate specific properties of the aggregation operator, so that each behavior can be tested where it matters. This includes behavior near the review threshold, with repeated or correlated scores, and with many weak scores.

Each trial draws a vector of normalized scores in [0, 1], representing the label scores that one selected model assigns to the labels mapped to one policy category. The same vectors are aggregated by every method, so differences between methods come only from the aggregation.

### Main suite (8 scenarios, scores in [0, 1])

| Scenario | Scores per trial | Score distribution | Purpose |
|---|---|---|---|
| ManyWeak | 6-12 | U(0.05, 0.25) i.i.d. | many weak scores; none reaches θr |
| OneStrongManyWeak | 4-11 weak + 1 | weak U(0.05, 0.25); one strong U(0.75, 0.95) | one decisive score among weak ones |
| MediumCorroboration | 2-12 | centre c ~ U(0.35, 0.55), noise N(0, 0.05) | moderate, clustered scores |
| Correlated | 2-12 | base b ~ U(0.1, 0.9), noise N(0, 0.03) | near-identical scores at any level |
| Duplicate | 5-12 | one value v ~ U(0.1, 0.4) repeated | exact repetition of one value |
| HeavyTailed | 2-12 | 0.15 × Pareto-II(3), clipped | mostly low scores, occasional outlier |
| Uniform | 2-12 | U(0, 1) i.i.d. | no structure |
| RandomBeta | 2-12 | Beta(a, b), a, b ~ U(0.5, 5) | random unimodal or U-shaped shapes |

### Sub-threshold suite (9 scenarios, all scores below θm, capped at 0.68)

This suite concentrates trials just below the review and removal thresholds. It tests whether several individually non-removal scores can cause escalation, and confirms that they never cause removal. Its escalation rates are deliberately higher than the main suite's and are not population estimates.

| Scenario | Scores per trial | Score distribution |
|---|---|---|
| TightClusterSubReview | 3-6 | centre U(0.10, 0.32), noise N(0, 0.02) |
| TightClusterNearReview | 3-6 | centre U(0.22, 0.38), noise N(0, 0.02) |
| TightClusterNearRemove | 3-6 | centre U(0.50, 0.66), noise N(0, 0.015) |
| LooseSpreadSubReview | 3-6 | U(0.05, 0.35) i.i.d. |
| LooseSpreadWide | 2-12 | U(0.05, 0.68) i.i.d. |
| DilutedBorderline | 2-6 weak + 1 | weak U(0.02, 0.10); one U(0.30, 0.40) |
| ManyWeak | 6-12 | U(0.05, 0.20) i.i.d. |
| UniformCapped | 2-12 | U(0, 0.68) i.i.d. |
| DuplicateSubReview | 4-12 | one value v ~ U(0.15, 0.32) repeated |

The exact generators are in `safemod/safemod/scenarios.py`. The specification tables in the paper are generated from the same code.

---

## 6. Real-data evaluation: Jigsaw

### Relationship between the synthetic data and Jigsaw

- **Synthetic data:** controlled score distributions, used for mechanism-level stress testing of the operator.
- **Jigsaw:** real comments, real labels and a public model's real outputs, used to measure what the operator does in practice, against ground truth.

The two complement each other. The synthetic suites show *when* the operator escalates. Jigsaw shows whether those escalations are actual violations, and how the operator compares with simply lowering the MAX review threshold.

### Setup
- **Data:** test set of the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) (`test.csv` with `test_labels.csv`). Rows labeled −1, which were not used for scoring in the challenge, are removed, leaving **63,978 comments**, of which 6,243 (9.76%) have at least one positive label.
- **Model:** [Detoxify](https://github.com/unitaryai/detoxify) `original` (bert-base-uncased), version 0.5.4, checkpoint `toxic_original-c1212f89`. It was trained only on the challenge's training split, so the test comments are unseen. One model supplies all six label scores.
- **Category mappings:**
  - *Toxicity:* all six labels; ground truth positive if any label is 1.
  - *Harassment:* {toxic, insult, identity_hate, threat}; ground truth is insult, identity_hate or threat.
  - Each label appears at most once per category.
- **Threshold policies (θr / θm):** Highly Cautious 0.20/0.50; Cautious 0.35/0.70; Default 0.40/0.75; Balanced 0.50/0.80; Permissive 0.55/0.85.
- **Baselines:** MAX, Mean, Noisy-OR, and a **review-volume-matched MAX** that sends to review the same number of highest-scoring items that MAX allowed (stable ordering for ties).

### Predictions file and licence
`outputs/jigsaw_scoring/jigsaw_predictions.csv` contains comment identifiers, the six Detoxify scores and the six test labels for all 63,978 comments. It contains **no comment text**.

The Jigsaw dataset is released under **CC0**; the underlying comment text is governed by Wikipedia's CC-SA-3.0 and is therefore not included. See `outputs/jigsaw_scoring/DATA_NOTICE.md`.

SHA-256: `0e9522cc56e283ed24c56dd82957605fcea7a1a1ea8038ba8a7c236941512a71`. An independent re-scoring in a fresh environment produced a byte-identical file.

---

## 7. Main results

- **Structural guarantees:** no violation of the MAX lower bound, the uplift bound, Remove equivalence with MAX or the absence of Review → Allow changes, in 1.37 million Monte Carlo evaluations and 1.28 million Jigsaw evaluations (`checks.csv`, all zeros).
- **Main suite:** 1,680 of 80,000 trials (2.10%) move from Allow to Review relative to MAX, mostly in the Duplicate (14.34%) and Correlated (2.25%) scenarios. Because no other decision change is possible, this is exactly the additional human-review volume.
- **Sub-threshold suite:** 4,934 of 90,000 trials (5.48%) move from Allow to Review; no trial is removed.
- **Perturbation study (450,000 trials):**
  - Raising the maximum score never lowered S_c (0 of 128,719 trials).
  - Raising a non-maximum score lowered S_c in 2.42% of such trials (largest decrease 0.0201).
- **Jigsaw:**
  - SafeMod preserves every Remove decision of MAX and adds at most 0.09% Review decisions (29-57 comments per configuration).
  - Its escalation precision never exceeds that of the review-volume-matched MAX baseline.
  - Most escalated comments have a single retained label score.

---

## 8. Reproducibility and verification

- **Determinism:** the implementation is deterministic and all random draws use fixed seeds (42 for the Monte Carlo suites, 2026 for the perturbation study) with NumPy's PCG64 generator. Each run writes `manifest.json` with the Python and package versions, platform, settings, timings and the SHA-256 of every output file, so a rerun can be compared with the published outputs (`tools/compare_outputs.py`). Results obtained with other software versions or on other platforms should be compared with the published outputs rather than assumed to be identical.
- **Numerics:** all computation and decisions use full double precision, with no rounding inside the operator. The softmax-weighted mean is evaluated as u_max − Σ wᵢ(u_max − uᵢ), which is algebraically identical to Σ wᵢuᵢ, so it cannot exceed the largest retained score in floating point. Checks are exact, except the uplift bound (tolerance 1e-12).
- **Precision correction:** an earlier version rounded S_c to 6 decimals before the decision. The correction changed no decision. Details and before/after comparisons are in `safemod/results_history/NUMERICAL_FIX.md`.
- **Supporting diagnostic (outside the paper's preset ranges):** Theorem 3 of the paper assumes α ≤ 1, and every documented preset uses α ≤ 0.30. `python tools/alpha_above_one_check.py` illustrates why the assumption matters: in random trials, maximum-score monotonicity held for α = 0.30 and 1.0 and failed from α = 1.2.
