# Validation

The checked sample is [pie_usage_probe.py](scripts/pie_usage_probe.py), with its
[matching notebook](notebooks/pie_usage_probe.ipynb), launched through
[run_pie_demo.py](scripts/run_pie_demo.py). The thesis research is Lauryn Wu's
original work.

All 28 tests passed with `python -m unittest discover -s tests -v`. They cover
phrase indexing and cropping, feature construction, scaling, development-based
model selection, sampling, import safety, and notebook/export consistency.
Feature values, selected IDs, regularization choices, and demo metrics were
also compared with the saved pre-refactor baseline and were unchanged.

The September 18, 2026 demo used Python 3.12.14, CPU inference, and the cached,
pinned GPT-2 model. With `--limit-per-split 6 --seed 1 --offline`, it ran on
18 MAGPIE examples: three literal and three figurative examples in each original
split. It produced 36 layer/feature-group metric rows and 37 figures.

[Example results](examples/baseline-demo/metrics.csv) and
[run metadata](examples/baseline-demo/run.json) record the scores, configuration,
selected IDs, split counts, dataset and implementation hashes, environment
versions, timing, and warnings. See [RUNNING.md](RUNNING.md) for setup.

The small demo verifies execution, not reproduction of the thesis results or
reliable accuracy estimates. Full-data inference has not been validated.
Phrase matching retains five plural-prefix mismatches among 1,315 eligible
examples. NumPy and scikit-learn deprecation warnings were recorded; no
convergence warnings occurred in the demo. A controlled baseline/masked
comparison and larger-scale memory optimization remain future work. These
checks do not establish the other experiments' scientific results.
