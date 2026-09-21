# SafeMod Synthetic Evaluation Data

This repository contains the synthetic datasets used to validate the **SafeMod: Policy-Governed Evidence Fusion for Deterministic Content Moderation** framework.

## Motivation

SafeMod is designed for multi-label content moderation, where multiple moderation signals may be generated for the same piece of content. The aggregation problem is therefore similar in structure to datasets such as the **Jigsaw Toxic Comment Classification Challenge**, where a single comment can receive multiple toxicity-related labels.

The original Jigsaw dataset contains Wikipedia comments annotated for multiple toxicity categories, including `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

**Jigsaw dataset:**
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

## Why Synthetic Data?

The purpose of the synthetic data is **not to reproduce the Jigsaw dataset**. Instead, it provides controlled moderation-score scenarios that allow specific properties of the SafeMod aggregation operator to be isolated and tested.

The synthetic signals represent hypothetical outputs from multiple moderation detectors for the same content. Different scenarios model situations such as:

* tightly clustered moderation signals,
* weak or sub-threshold signals,
* widely dispersed signals,
* duplicated signals,
* diluted borderline evidence,
* many weak signals, and
* uniformly distributed scores.

All generated scores are normalized to the `[0,1]` range and constrained below the removal threshold where required, allowing the experiments to specifically examine whether **multiple individually non-removal signals can collectively cause inappropriate escalation**.

## Relationship to Jigsaw

Jigsaw provides the **real-world moderation context and multi-label motivation**, while the synthetic dataset provides **controlled numerical evidence** for testing the aggregation mechanism.

In other words:

**Jigsaw:** real moderation data → realistic classifier outputs
**Synthetic data:** controlled classifier-score distributions → mechanism-level stress testing

The synthetic experiments therefore complement, rather than replace, evaluation on publicly available moderation data.

## Reference

If you use these materials, please cite the associated SafeMod paper.

## Data Source

The real-world benchmark motivating the moderation setting is:

**Jigsaw Toxic Comment Classification Challenge**
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

The Jigsaw dataset is a multi-label toxicity classification benchmark based on human-annotated Wikipedia comments, making it relevant for evaluating moderation systems that must combine multiple category-level signals.
