# Running the experiments

Each notebook is a separate experiment with a corresponding Python export in
`scripts/`. None requires output from another notebook. The selected four-token
sample is import-safe and runs only on an explicit function call; the other saved
experiments execute their cells in order in a fresh kernel.

## 1. Set up Python

These terminal commands are for macOS or Linux. Start in the repository root,
the directory containing `README.md`, `scripts/`, and `notebooks/`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-smoke-tested.txt
```

Activate the environment again when opening a new terminal. The subset execution
checks used Python 3.12.14; [requirements-smoke-tested.txt](requirements-smoke-tested.txt)
lists the direct-package versions observed in that environment.
`requirements.txt` lists the same direct dependencies without version pins.

The code loads pretrained `gpt2` weights and its tokenizer. The first load needs
internet access unless those files are already cached. The experiments use CPU
inference.

## Recommended: bounded four-token demo

After obtaining the MAGPIE file below, run:

```sh
python -m unittest discover -s tests -v
python scripts/run_pie_demo.py \
  --data "/path/to/MAGPIE_filtered_split_typebased.jsonl" \
  --limit-per-split 6 --seed 1 --output-dir results/probe-sample
```

The runner imports `pie_usage_probe.py` and calls its `main()`;
there is one implementation of the experiment. It scans the full input file for
filtering and seeded class balancing, then caps each original split before
loading GPT-2. The default cap gives at most six training, six development, and
six test examples. Odd limits leave one slot unused to preserve 50/50 balance.
If fewer examples are available, actual counts are smaller and printed before
inference. A split missing either class fails clearly, as does a limit below two.
No records are moved between splits. Overlapping retained idiom types are rejected.
Small-demo scores are execution evidence, not research-performance estimates.

The demo uses CPU inference without gradients, two CPU threads, seed 1 for the
sampling, Torch, NumPy, and the classifier, and GPT-2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. It retains outputs as float32 arrays
and converts selected features to float64 for the original calculations.

Options:

- `--offline`: require cached model/tokenizer files (the usual Hugging Face cache,
  or a custom directory set through `HF_HOME`).
- `--limit-per-split`: maximum total per split, not per class (default 6). Larger
  values increase resource use; the tested command is the bounded 18-example run.
- `--seed`: deterministic sampling and model/probe randomness (default 1).
- `--output-dir results/probe-sample-repeat`: use a new output directory for another
  run. Existing directories are never overwritten.

Each successful run writes `metrics.csv`, `execution.log`, `report.json`, and
37 PNG figures. The report includes the configuration, dataset hash, selected
example IDs, actual class counts, environment versions, per-case results,
warnings, and pipeline time (excluding figure export). Module runs also record
the implementation hash. Python warnings during preparation/inference/fitting
are captured and counted, not discarded. `results/` is excluded from Git.

For the same bounded run in the selected notebook, run its definition cells
first, then explicitly call the entry point in a new cell:

```python
main([
    "--data", "/path/to/MAGPIE_filtered_split_typebased.jsonl",
    "--limit-per-split", "6", "--seed", "1",
    "--output-dir", "results/probe-sample-notebook", "--offline",
])
```

Omit `--offline` if the pinned model is not yet cached. There is no automatic
data/model loading while you read or execute the definition cells.

## 2. Get the datasets

Datasets are not included in this repository. Use the authors' releases and follow
their citation and license instructions.

### Noun compounds

Get the English task-independent data from the
[AStitchInLanguageModels dataset directory](https://github.com/H-TayyarMadabushi/AStitchInLanguageModels/tree/main/Dataset/TaskIndependentData).
Place it in the repository root as `en_TaskIndependentData.json`.

The code expects one JSON object containing the original splits, including
`train_zero_shot`, `dev`, and `test`. It reads the nested example arrays directly;
do not convert them to CSV or flatten the groups. This file is used by
`Noun_Compounds`, `NC_Attention`, and `NC_Paraphrase`.

### Potentially idiomatic expressions

Get `MAGPIE_filtered_split_typebased.jsonl` from the
[MAGPIE authors' repository](https://github.com/hslh/magpie-corpus).
Use the filtered, type-based split, not the random split or unfiltered corpus.
Save it in the repository root under the filename the code expects:
`MAGPIE_filtered_split_typebased.json`.

Despite the local `.json` extension, keep the file in JSON Lines format: one JSON
object per line, not a JSON array. Preserve the original fields, including `id`,
`context`, `idiom`, `confidence`, `label_distribution`, and `split`, with split
values `training`, `development`, and `test`. The selected probe finds phrase
indices from tokenized text; it does not use the annotated character offsets.
This file is used by the five `PIE` and usage-probing experiments. The demo accepts
an explicit path and does not require renaming or copying the dataset.

The resulting layout is:

```text
gpt2-noncompositionality/
├── README.md
├── RUNNING.md
├── requirements.txt
├── en_TaskIndependentData.json
├── MAGPIE_filtered_split_typebased.json
├── notebooks/
└── scripts/
```

You only need the dataset for the experiment you choose. Both expected dataset
filenames are excluded by `.gitignore`; keep the data out of commits.

## 3. Check memory before running

The corrected four-token probe retains attention matrices and embeddings as
compact NumPy arrays. Start with its bounded demo above; full-data inference has
not been checked. The other saved experiments retain large nested Python lists.
The historical storage estimates below came from data preparation, not inference:

| Experiment | Selected examples | Estimated attention + embedding storage |
| --- | ---: | ---: |
| `Probing_Classifier_PIE-Copy1` (now `pie_usage_probe`), before compact-array storage | 764 | 54.9 GiB |
| `Probing_PIE-Masked_Attn` | 6,504 | 212.3 GiB |

These are estimates for those structures alone, not peak-memory measurements.
Model weights, temporary tensors, data objects, and classifier fitting need more
memory. Use a large-memory machine for these full runs.

The direct experiment commands below start full selected-data runs unless an
explicit limit is set. Use the bounded demo for the recommended coding sample.
Supply complete input files with their original split structure in either case.

## 4. Choose a notebook or script

### Notebooks

With the environment active, start JupyterLab from the repository root:

```sh
python -m pip install jupyterlab
jupyter lab
```

Open the `.ipynb` file matching an experiment in the table below. The
[JupyterLab documentation](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html)
describes launching the notebook interface.

Input paths are relative to the kernel's working directory, which may be
`notebooks/`. Before the existing cells, run this setup cell to use the datasets
in the repository root:

```python
from pathlib import Path
import os
import sys

repo_root = Path.cwd()
if repo_root.name == "notebooks":
    repo_root = repo_root.parent
assert (repo_root / "scripts").is_dir() and (repo_root / "notebooks").is_dir()
os.chdir(repo_root)
print("Working directory:", Path.cwd())
print("Python:", sys.executable)
```

Check that the Python path points into this repository's `.venv`. Skip existing
`!pip install transformers` cells after installing dependencies, so they do not
change the environment. Use a fresh kernel for each experiment, then execute
its cells in order. Check the memory requirements before starting inference.

### Python exports

Run from the repository root with the environment active and the required data
in place. Choose **one** command; these are separate experiments, not sequential
pipeline stages. The matching notebook has the same filename stem and an
`.ipynb` extension.

| Command | Main displayed output |
| --- | --- |
| `python scripts/Noun_Compounds.py` | Layerwise probe accuracies and coefficient plots for noun-compound representations. |
| `python scripts/NC_Attention.py` | Attention heatmaps and boxplots of attention and embedding statistics. |
| `python scripts/NC_Paraphrase.py` | Paraphrase accuracy, cosine-similarity, and probe plots. |
| `python scripts/PIE.py` | Literal/figurative attention heatmaps and embedding-statistic boxplots. |
| `python scripts/PIE_masked_attn.py` | Corresponding descriptive plots with attention-head masking. |
| `python scripts/Probing_Classifier_PIE.py` | Layerwise literal/figurative probe scores and ROC plots from this saved variant. |
| `python scripts/run_pie_demo.py --data MAGPIE_filtered_split_typebased.json --limit-per-split 6 --seed 1 --output-dir results/probe-sample` | Bounded four-token usage probe; metrics and plots are saved. |
| `python scripts/Probing_PIE-Masked_Attn.py` | Masked usage-probe scores and an accuracy plot; coefficient and ROC sections are inactive in the saved code. |

`pie_usage_probe` (formerly `Probing_Classifier_PIE-Copy1`) and
`Probing_Classifier_PIE` have different saved configurations. The four-token and masked probes also use
different data settings; they are not a matched baseline/ablation comparison.
See [Technical notes](VERIFICATION.md) for details.

### Where outputs go

Progress, counts, and scores appear in the terminal or notebook output cells.
Figures appear inline in notebooks or in Matplotlib windows when running scripts
with a graphical backend. For blocking interactive plots, close each figure
window to continue execution. The notebook interface also displays the values of
bare expressions; terminal output comes from explicit print calls.

Results are displayed during execution; saving files is manual. Save notebook
outputs with the notebook, or use the save control in an interactive plot window
to export an image. Headless figure export requires explicit Matplotlib saving
commands.

The bounded demo runner is the exception: it saves every figure and its report
automatically, without opening plot windows.

## Execution checks

Subset checks cover both main usage probes and the paraphrase calculation and
plotting section, using real examples and pretrained GPT-2. Import, regression,
and notebook/export consistency checks also passed. Sample counts and the test
environment are recorded in the [technical notes](VERIFICATION.md).

After editing the recommended Python export, synchronize its notebook with
`python tools/sync_pie_notebook.py`, then rerun the focused tests. The sync tool
touches only that notebook; it follows the Python file's named section order and
clears saved outputs.
