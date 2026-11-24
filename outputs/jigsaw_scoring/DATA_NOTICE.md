# Jigsaw predictions file

`jigsaw_predictions.csv` contains, for the 63,978 scored comments of the Jigsaw Toxic Comment
Classification Challenge test set (rows with label -1 removed):

- `id`: the comment identifier from Kaggle `test.csv`;
- `p_*`: Detoxify 0.5.4 `original` scores (checkpoint `toxic_original-c1212f89`);
- six label columns from Kaggle `test_labels.csv`.

It contains no comment text.

Labels come from the Jigsaw Toxic Comment Classification Challenge (Kaggle), whose dataset is
released under CC0; the underlying comment text is governed by Wikipedia's CC-SA-3.0 and is
therefore not included here.

SHA-256: 0e9522cc56e283ed24c56dd82957605fcea7a1a1ea8038ba8a7c236941512a71
(identical on an independent re-scoring run, 23 Nov 2026; see `scoring_manifest.json`).
