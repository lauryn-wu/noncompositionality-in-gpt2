# Small usage-probe demo

This four-token usage-probe run completed on September 18, 2026. It checks that
the code runs on real data. With only six test examples, it is too small to
estimate accuracy or reproduce the thesis results. See
[VERIFICATION.md](../../VERIFICATION.md) for the remaining checks.

- `metrics.csv`: all 12 layers and three feature groups, including selected C,
  development accuracy, test accuracy, and feature counts.
- `run.json`: configuration, seed, selected record IDs, counts, dataset
  and implementation hashes, environment versions, and warnings.

The run used six examples per original split (three per class), seed 1, and the
cached, pinned GPT-2 model. There were no convergence warnings. The run recorded
deprecation warnings for NumPy matrices and scikit-learn's penalty argument.
Raw examples, model files, prediction arrays, and the 37 generated figures are
not included here; a local run saves detailed results under `results/`.

From the repository root with dependencies installed:

```sh
python scripts/run_pie_demo.py \
  --data "/path/to/MAGPIE_filtered_split_typebased.jsonl" \
  --limit-per-split 6 --seed 1 --offline \
  --output-dir results/probe-sample-readability
```

Use your local dataset path. Omit `--offline` to download the model if it is not
cached. Choose a new output directory if the example has already been run.
