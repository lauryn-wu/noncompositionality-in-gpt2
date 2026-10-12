# How and What Does GPT Capture About Non-Compositionality?

Code from Lauryn Wu's 2023 Harvard senior thesis, supervised by Martin
Wattenberg. The thesis studies how GPT-2 represents idioms and other expressions
whose meanings cannot be inferred from their individual words.

[Read the thesis on Harvard DASH](https://dash.harvard.edu/entities/publication/6346cebc-6c17-4dc8-add8-37011a4f1c07).

The thesis reports differences in embeddings and attention patterns between
compositional and non-compositional phrases, especially in upper layers. Linear
probes recovered idiom paraphrases and distinguished literal from idiomatic
usage with high accuracy on the evaluated tasks. Usage-probe performance
remained high after directly ablating more than 40% of attention heads. The
thesis contains the full experiments and results.

## Literal versus figurative usage

The cleaned-up example in this repository asks where literal-versus-figurative
usage can be recovered across GPT-2's layers, and whether attention features
add information beyond embeddings. The four-token experiment compares embeddings
(768 features), attention (120), and their combination (888) across all 12 layers.
Scaling uses training data only; regularization is selected on development data,
with accuracy and ROC curves reported on the test split.

Start with [run_pie_demo.py](scripts/run_pie_demo.py), which imports
[pie_usage_probe.py](scripts/pie_usage_probe.py). The same definitions are in the
[notebook](notebooks/pie_usage_probe.ipynb).

The current code has been checked on an 18-example run with six test examples.
That verifies execution but is too small to estimate accuracy or replicate the
thesis. Full-data inference and the controlled head-masking comparison remain
unverified; see [VERIFICATION.md](VERIFICATION.md). Probe accuracy measures what
a trained classifier can recover from the representations. Establishing whether
GPT-2 itself uses that information requires further evidence.

## Run the small demo

From the repository root, using Python 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-smoke-tested.txt
python -m unittest discover -s tests -v
python scripts/run_pie_demo.py \
  --data "/path/to/MAGPIE_filtered_split_typebased.jsonl" \
  --limit-per-split 6 --seed 1 --offline --output-dir results/probe-sample
```

Follow [RUNNING.md](RUNNING.md#2-get-the-datasets) to get the dataset. This command
selects up to six examples per split, balanced between the two classes, before
running GPT-2. It uses cached model files; omit `--offline` on the first run if
they need to be downloaded. The seed controls sampling, Torch/NumPy randomness,
and probe training.

The run saves `metrics.csv`, a `report.json` with configuration, counts,
environment details and warnings, an execution log, and 37 figures.

A small [example result](examples/baseline-demo/README.md) from a successful
real-data run is included. Datasets, model files, and generated plots are not.

[RUNNING.md](RUNNING.md) also covers notebook setup and the other experiments.

### Code structure

Importing the implementation only defines functions and classes; it does not load
data or GPT-2, or train a probe. `run_sample()` calls these stages explicitly:

1. `prepare_examples()` and `select_examples()`: preprocessing and split
   boundaries, followed by seeded balancing and a small cap.
2. `extract_features()`: `run_model_inference()` obtains GPT-2 outputs;
   `extract_embedding_features()` and `extract_attention_features()` compute
   each expression's features; `assemble_feature_matrix()` returns matrices and labels.
3. `train_and_evaluate()`: training-only scaling, development-based C selection,
   and separate test results for EMB, ATTN, and EMB+ATTN, using
   `scale_features()` and `fit_and_evaluate_probe()`.
4. `save_results()`: metrics, provenance, warnings, and figures.

The notebook follows the same section order as the Python file.

## Other thesis experiments

Each experiment has a notebook and matching Python export.

| Notebook / Python export stem | Analysis |
| --- | --- |
| `Noun_Compounds` | Noun-compound representations |
| `NC_Attention` | Noun-compound attention |
| `NC_Paraphrase` | Noun-compound paraphrase probing |
| `PIE` | Potentially idiomatic expression analysis |
| `PIE_masked_attn` | Masked-attention expression analysis |
| `Probing_Classifier_PIE` | Another saved usage-probing variant |
| `Probing_PIE-Masked_Attn` | Usage probing with attention-head masking |
