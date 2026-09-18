# Small real-data demonstration

These are results from a successful run of the four-token code sample.
They demonstrate execution, **not reproduction of the thesis results** or a
reliable accuracy estimate: there are only six test examples.

The run completed on September 18, 2026. `run.json` records its configuration,
execution environment, and implementation hash. See
[validation details](../../VERIFICATION.md) for the checks and remaining limitations.

- `metrics.csv`: all 12 layers and three feature groups, including selected C,
  development accuracy, test accuracy, and feature counts.
- `run.json`: actual configuration, seed, selected record IDs, counts, dataset
  and implementation hashes, environment versions, and warnings.

The run used six examples per original split (three per class), seed 1, and the
cached, pinned GPT-2 model. No convergence warnings were recorded. NumPy matrix
and scikit-learn penalty-argument deprecation warnings are recorded explicitly.
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
