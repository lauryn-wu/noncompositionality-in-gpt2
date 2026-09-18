# How and What Does GPT Capture About Non-Compositionality?

This repository contains code from Lauryn Wu's 2023 Harvard senior thesis,
supervised by Martin Wattenberg. The research asks: how does GPT-2 represent
expressions whose meaning is not simply the combination of their individual words?

The original thesis work examines contextual embeddings and attention patterns,
implements linear probes for paraphrase identification and literal/figurative
usage, and investigates attention-head ablation.

## Start here: literal versus figurative usage

Start with [run_pie_demo.py](scripts/run_pie_demo.py), a small entry point that
imports the [selected implementation](scripts/pie_usage_probe.py).
The same definitions are available in the
[notebook](notebooks/pie_usage_probe.ipynb). The four-token
experiment compares embeddings (768 features), attention (120), and their
combination (888) across GPT-2's 12 layers. Regularization is selected on
development data; test accuracy and ROC curves are reported for every case.

The selected sample has since been cleaned up and corrected for sharing.

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

Get the dataset using [RUNNING.md](RUNNING.md#2-get-the-datasets). The demo uses
at most six examples per split, balanced between the two classes: 18 examples
with the verified dataset. The cap is applied before GPT-2 inference. The seed
controls sampling, Torch/NumPy randomness, and probe training. Results include
`metrics.csv`, configuration/counts/environment/warnings in `report.json`, an
execution log, and the existing 37 figures. This verified command uses cached
GPT-2 files; omit `--offline` on the first run if they need to be downloaded.

A small [example result](examples/baseline-demo/README.md) from a successful
real-data run is included. Datasets, model files, and generated plots are not.

[RUNNING.md](RUNNING.md) covers notebook setup and the other experiments;
[VERIFICATION.md](VERIFICATION.md) summarizes current validation and limitations.

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

These functions take explicit inputs and return their outputs. Start with
`run_sample()` for the data flow, then read the feature and fitting functions.
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
