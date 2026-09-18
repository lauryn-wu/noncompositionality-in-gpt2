# Exported from notebooks/pie_usage_probe.ipynb.
# Setup and execution notes: README.md.

# %% Imports and settings
"""Four-token literal/figurative probe from Lauryn Wu's senior thesis.

Definitions are safe to import. Use run_pie_demo.py or call run_sample explicitly.
Validation details are in VERIFICATION.md.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import sys
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import torch
from transformers import GPT2Model, GPT2Tokenizer

MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
REGULARIZATION_CANDIDATES = (
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    5e-6,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    0.001,
    0.01,
    0.05,
    1,
)
CASE_NAMES = {1: "EMB", 4: "ATTN", 5: "EMB+ATTN"}
SPLITS = ("train", "dev", "test")


# %% Preprocessing and sampling
def idiom_in_context(context, idiom):
    """Check the case-sensitive substring in any original context sentence."""
    for sentence in context:
        if idiom in sentence:
            return True
    return False


def apply_contraction_change(text):
    """Apply the original contraction and punctuation-spacing substitutions."""
    text = text.replace(" n't", "n't")
    text = text.replace("\n", "")
    text = text.replace(" ‘", "‘")
    text = text.replace(" ’", "’")
    text = text.replace(" '", "'")
    text = text.replace(" , ", ", ")
    text = text.replace(" .", ". ")
    text = text.replace(" ?", "? ")
    text = text.replace(" !", "! ")
    text = text.replace(" - ", "-")
    text = text.replace(" %", "%")
    return text


def context_to_text(context):
    """Concatenate normalized sentences without adding new separators."""
    output = ""
    for sentence in context:
        output += apply_contraction_change(sentence)
    return output.strip()


def find_phrase_indices(tokens, idiom):
    """Return original-token positions for the first normalized prefix match.

    Quote-only tokens are skipped for matching, not removed from the model input.
    Returned positions may therefore be noncontiguous. Prefix matching retains
    the existing behavior for plural endings.
    """
    idiom_words = idiom.lower().split()
    idiom_text = "".join(idiom_words)
    normalized, original_positions = [], []
    for position, token in enumerate(tokens):
        token = token.lower()
        token = token[1:] if token.startswith("ġ") else token
        token = token.replace("'", "").replace('"', "")
        if token:
            normalized.append(token)
            original_positions.append(position)
    tokens = normalized
    for i, token in enumerate(tokens):
        if idiom_words[0].startswith(token):
            for j in range(i + 1, len(tokens) + 1):
                candidate_text = "".join(tokens[i:j])
                if candidate_text.startswith(idiom_text):
                    # Quote-only tokens stay in the model input, but not the phrase.
                    return original_positions[i:j]
                if len(candidate_text) > len(idiom_text) + 2:
                    break
    return None


def crop_phrase_context(tokens, indices, *, tokens_max=120):
    """Retain the original crop rule and rebase original-token phrase positions."""
    if tokens_max < 2:
        raise ValueError("tokens_max must be at least 2 to retain the complete phrase")
    if (
        not indices
        or indices != sorted(set(indices))
        or any(i < 0 or i >= len(tokens) for i in indices)
    ):
        raise ValueError(
            "Phrase indices must be increasing, unique, and within the tokens"
        )
    phrase_tokens = [tokens[i] for i in indices]
    if len(tokens) > tokens_max:
        # Original crop: end is exclusive; 120 is not a hard length cap.
        start = max(0, indices[0] - int(tokens_max / 2))
        end = min(len(tokens), indices[-1] + int(tokens_max / 2))
        tokens = tokens[start:end]
        indices = [i - start for i in indices]
    if (
        any(i < 0 or i >= len(tokens) for i in indices)
        or [tokens[i] for i in indices] != phrase_tokens
    ):
        raise ValueError("Cropping did not preserve the phrase indices and tokens")
    return tokens, indices


def prepare_examples(
    data_path, tokenizer, *, tokens_max=120, confidence_threshold=1, phrase_length=4
):
    """Apply the original context, span, four-token, and within-split type filters."""
    if phrase_length != 4:
        raise ValueError("This sample supports four-token expressions only")
    split_names = {"training": "train", "development": "dev", "test": "test"}
    groups = {split: {"idiom": [], "literal": []} for split in SPLITS}
    raw_count = 0
    with Path(data_path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                example = json.loads(line)
                raw_count += 1
                if example["split"] not in split_names:
                    raise ValueError(f"Unknown split: {example['split']!r}")
                if example[
                    "confidence"
                ] != confidence_threshold or not idiom_in_context(
                    example["context"], example["idiom"]
                ):
                    continue
                tokens = tokenizer.tokenize(context_to_text(example["context"]))
                indices = find_phrase_indices(tokens, example["idiom"])
                if not indices:
                    continue
                tokens, indices = crop_phrase_context(
                    tokens, indices, tokens_max=tokens_max
                )
                if len(indices) != phrase_length:
                    continue
                label = (
                    "idiom"
                    if example["label_distribution"]["i"] > 0.9
                    else "literal"
                    if example["label_distribution"]["l"] > 0.9
                    else None
                )
                if label is not None:
                    groups[split_names[example["split"]]][label].append(
                        {
                            "id": example["id"],
                            "idiom": example["idiom"],
                            "tokens": tokens,
                            "phrase_indices": indices,
                            "split": example["split"],
                        }
                    )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid MAGPIE record at line {line_number}: {exc}"
                ) from exc

    for split in SPLITS:
        types = [
            {example["idiom"] for example in groups[split][label]}
            for label in ("idiom", "literal")
        ]
        common = types[0] & types[1]
        for label in ("idiom", "literal"):
            groups[split][label] = select_shared_idiom_types(
                groups[split][label], common
            )
    return groups, raw_count


def select_shared_idiom_types(examples, shared_types):
    """Keep phrase types represented in both classes of the same split."""
    return [example for example in examples if example["idiom"] in shared_types]


def select_examples(groups, *, limit_per_split=6, seed=1):
    """Seeded class balancing, then a cap within each original split.

    Odd limits leave one slot unused to retain the baseline's 50/50 balance.
    For the original MAGPIE data, selection matches the previous seed-1 demo.
    """
    if limit_per_split < 2:
        raise ValueError("limit_per_split must be at least 2 to retain both classes")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 2**32-1")
    rng = random.Random(seed)
    selected = {}
    for split in SPLITS:
        figurative, literal = groups[split]["idiom"], groups[split]["literal"]
        if not figurative or not literal:
            raise ValueError(
                f"{split}: need at least one literal and one figurative "
                f"example after filtering; got {len(literal)} literal, "
                f"{len(figurative)} figurative"
            )
        # Original data has more figurative examples. Also handle a smaller
        # figurative class without sampling beyond the available population.
        balanced_count = min(len(figurative), len(literal))
        figurative = rng.sample(figurative, balanced_count)
        count = min(limit_per_split // 2, balanced_count)
        selected[split] = {"idiom": figurative[:count], "literal": literal[:count]}
    split_types = {
        split: {
            example["idiom"] for rows in selected[split].values() for example in rows
        }
        for split in SPLITS
    }
    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if split_types[first] & split_types[second]:
            raise ValueError(
                f"Idiom types overlap between {first} and {second}; use the type-based MAGPIE split"
            )
    return selected


def example_counts(examples):
    return {
        split: {label: len(rows) for label, rows in examples[split].items()}
        for split in SPLITS
    }


# %% Model inference and feature construction
@torch.inference_mode()
def run_model_inference(examples, model, tokenizer):
    """Return examples with per-example float32 GPT-2 outputs (batch axis removed).

    Hidden states contain 13 arrays of shape (tokens, 768); attentions contain
    12 arrays of shape (12 heads, query tokens, key tokens).
    """
    model.eval()
    outputs = []
    for count, example in enumerate(examples, 1):
        token_ids = tokenizer.convert_tokens_to_ids(example["tokens"])
        token_tensor = torch.tensor(token_ids).unsqueeze(0)
        output = model(token_tensor, output_hidden_states=True)
        outputs.append(
            {
                **example,
                "attentions": [
                    attention[0].cpu().numpy().copy() for attention in output[-1]
                ],
                "hidden_states": [
                    state[0].cpu().numpy().copy() for state in output.hidden_states
                ],
            }
        )
        if count % 500 == 0:
            print(f"Processed {count} examples")
    return outputs


def pairwise_cosine_summary(phrase_embeddings):
    """Return mean, maximum, and minimum pairwise cosines for matrix rows."""
    similarities = []
    for first in range(len(phrase_embeddings) - 1):
        for second in range(first + 1, len(phrase_embeddings)):
            first_vector = phrase_embeddings[first].tolist()[0]
            second_vector = phrase_embeddings[second].tolist()[0]
            similarity = np.dot(first_vector, second_vector) / (
                np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
            )
            similarities.append(similarity)
    return [np.mean(similarities), max(similarities), min(similarities)]


def extract_embedding_features(hidden_states, phrase_indices, *, layer):
    """Return 772 values: mean embedding (768), its norm, then three cosines.

    Each hidden state has shape (tokens, 768). The zero-based transformer layer
    uses hidden_states[layer + 1]: hidden_states[0] is the embedding input.
    Indices refer to the original tokens in the supplied (possibly cropped) input.
    """
    # Float64 retains the calculation precision of the baseline's matrix API.
    phrase_embeddings = np.matrix(
        [hidden_states[layer + 1][index] for index in phrase_indices], dtype=float
    )
    features = phrase_embeddings.mean(0).tolist()[0]
    features.append(np.linalg.norm(features))
    features.extend(pairwise_cosine_summary(phrase_embeddings))
    return features


def extract_attention_features(attentions, phrase_indices, *, layer):
    """Return 120 features for four phrase tokens, in head/key/query order.

    attentions[layer] has shape (12 heads, tokens, tokens), indexed as query, key.
    For each head and phrase key, include that token and later phrase queries:
    (0,0), (1,0), (2,0), (3,0), (1,1), ... (3,3), written as (query,key).
    This is the causal triangle, including self-attention. Each weight is
    multiplied by its query's zero-based position in the full supplied context,
    not its position within the phrase. No additional normalization is applied.
    """
    features = []
    for head in range(12):
        adjusted_attention = np.array(attentions[layer][head], dtype=float)
        for query in range(adjusted_attention.shape[0]):
            for key in range(adjusted_attention.shape[1]):
                adjusted_attention[query, key] *= query
        for key_offset in range(len(phrase_indices)):
            for query_offset in range(key_offset, len(phrase_indices)):
                features.append(
                    float(
                        adjusted_attention[
                            phrase_indices[query_offset], phrase_indices[key_offset]
                        ]
                    )
                )
    return features


def assemble_feature_matrix(figurative_examples, literal_examples, *, layer):
    """Return X (examples, 892) and y (examples,), figurative rows first.

    Raw columns are embedding 0:768, norm/cosine summaries 768:772, and
    attention 772:892. Labels are 1 for figurative and 0 for literal examples.
    All examples must already have four phrase indices and model outputs.
    """
    rows = []
    for example in [*figurative_examples, *literal_examples]:
        phrase_indices = example["phrase_indices"]
        if len(phrase_indices) != 4:
            raise ValueError("This sample supports four-token expressions only")
        row = extract_embedding_features(
            example["hidden_states"], phrase_indices, layer=layer
        )
        row.extend(
            extract_attention_features(
                example["attentions"], phrase_indices, layer=layer
            )
        )
        rows.append(row)
    X = np.asarray(rows, dtype=float).reshape(-1, 892)
    y = np.array([1] * len(figurative_examples) + [0] * len(literal_examples))
    return X, y


def extract_features(examples, model, tokenizer):
    """Return per-layer, per-split (X, y) pairs and retained model-output bytes."""
    outputs = {
        split: {
            label: run_model_inference(rows, model, tokenizer)
            for label, rows in examples[split].items()
        }
        for split in SPLITS
    }
    features = {
        layer: {
            split: assemble_feature_matrix(
                outputs[split]["idiom"], outputs[split]["literal"], layer=layer
            )
            for split in SPLITS
        }
        for layer in range(12)
    }
    retained_bytes = sum(
        array.nbytes
        for split in outputs.values()
        for rows in split.values()
        for example in rows
        for key in ("attentions", "hidden_states")
        for array in example[key]
    )
    return features, retained_bytes


# %% Probe fitting
def feature_columns(case):
    """Return raw-column indices for EMB (1), ATTN (4), or EMB+ATTN (5).

    The four norm/cosine summary columns are excluded from all active cases.
    """
    embedding = list(range(768))
    attention = list(range(772, 892))
    return {1: embedding, 4: attention, 5: embedding + attention}[case]


def scale_features(X_train, X_dev, X_test, *, case):
    """Select a feature group, fit on training only, and return arrays + scaler."""
    columns = feature_columns(case)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train[:, columns])
    X_dev = scaler.transform(X_dev[:, columns])
    X_test = scaler.transform(X_test[:, columns])
    return X_train, X_dev, X_test, scaler


def fit_and_evaluate_probe(
    X_train,
    y_train,
    X_dev,
    y_dev,
    X_test,
    y_test,
    *,
    case,
    regularization_candidates=REGULARIZATION_CANDIDATES,
    seed=1,
):
    """Return (result, fitted_model, fitted_scaler) for one layer/feature group.

    Inputs use the raw 892-column layout. Scaling is fit only on training rows.
    C is selected by development accuracy, keeping the first candidate on ties.
    Test labels are used only after selection; result keys match the saved report.
    """
    if not regularization_candidates or any(C <= 0 for C in regularization_candidates):
        raise ValueError(
            "regularization_candidates must contain positive regularization values"
        )
    X_train, X_dev, X_test, scaler = scale_features(X_train, X_dev, X_test, case=case)
    best_dev_accuracy = -1
    best_C = regularization_candidates[0]
    for C in regularization_candidates:
        model = LogisticRegression(
            solver="saga", tol=1e-3, max_iter=200, random_state=seed, penalty="l2", C=C
        )
        model.fit(X_train, y_train)
        dev_accuracy = model.score(X_dev, y_dev)
        if dev_accuracy > best_dev_accuracy:
            best_model = model
            best_dev_accuracy = dev_accuracy
            best_C = C
    result = {
        "test_accuracy": best_model.score(X_test, y_test),
        "dev_accuracy": best_dev_accuracy,
        "best_C": best_C,
        "Y_pred": best_model.predict(X_test),
        "Y_test": y_test,
        "Y_pred_prob": best_model.predict_proba(X_test)[:, 1],
    }
    return result, best_model, scaler


def train_and_evaluate(
    features, *, seed=1, regularization_candidates=REGULARIZATION_CANDIDATES
):
    """Return results for all layer/case pairs and the attention coefficients."""
    if not regularization_candidates or any(C <= 0 for C in regularization_candidates):
        raise ValueError(
            "regularization_candidates must contain positive regularization values"
        )
    results, coefficients = {}, []
    for layer, splits in features.items():
        X_train, y_train = splits["train"]
        X_dev, y_dev = splits["dev"]
        X_test, y_test = splits["test"]
        print("layer", layer + 1)
        results[layer] = {}
        for case, name in CASE_NAMES.items():
            result, model, _ = fit_and_evaluate_probe(
                X_train,
                y_train,
                X_dev,
                y_dev,
                X_test,
                y_test,
                case=case,
                regularization_candidates=regularization_candidates,
                seed=seed,
            )
            results[layer][case] = result
            if case == 4:
                coefficients.append(model.coef_[0])
            print(
                f"{name}: accuracy {result['test_accuracy']:.5f}, best C {result['best_C']:g}"
            )
    return results, coefficients


# %% Plotting and saving
def metrics_rows(results):
    rows = []
    for layer, cases in results.items():
        for case, result in cases.items():
            rows.append(
                {
                    "layer": layer + 1,
                    "feature_group": CASE_NAMES[case],
                    "n_features": {1: 768, 4: 120, 5: 888}[case],
                    "selected_C": result["best_C"],
                    "dev_accuracy": result["dev_accuracy"],
                    "test_accuracy": result["test_accuracy"],
                }
            )
    return rows


def plot_coefficient_exponentials(coefficients):
    """Yield each layer's exponentiated attention coefficients and head sums."""
    import matplotlib.pyplot as plt

    for layer_coefficients in coefficients:
        exponentials = np.exp(layer_coefficients)
        pair_values = exponentials[-120:]
        by_head_and_pair = pair_values.reshape(12, 10)
        head_sum = by_head_and_pair.sum(axis=0)
        figure, axes = plt.subplots(1, 2, figsize=(16, 4), tight_layout=True)
        axes[1].bar(range(len(pair_values)), pair_values)
        axes[0].bar(range(len(head_sum)), head_sum)
        yield figure


def plot_coefficients(coefficients):
    """Yield raw attention coefficients and their per-pair/per-head means."""
    import matplotlib.pyplot as plt

    for layer_coefficients in coefficients:
        by_head_and_pair = layer_coefficients.reshape(12, 10)
        pair_means = np.mean(by_head_and_pair, axis=0)
        head_means = np.mean(by_head_and_pair, axis=1)
        figure, axes = plt.subplots(1, 3, figsize=(16, 4), tight_layout=True)
        axes[0].bar(range(len(layer_coefficients)), layer_coefficients)
        axes[1].bar(range(len(pair_means)), pair_means)
        axes[2].bar(range(len(head_means)), head_means)
        yield figure


def plot_roc(y_test, y_pred_prob, label):
    """Add one feature group's ROC curve to the current axes."""
    import matplotlib.pyplot as plt

    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, y_pred_prob)
    plt.plot(false_positive_rate, true_positive_rate, label=label)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")


def plot_results(results, coefficients):
    """Yield the existing accuracy, coefficient, and per-case ROC figures."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 5))
    for case, name in CASE_NAMES.items():
        plt.plot(
            [layer + 1 for layer in results],
            [cases[case]["test_accuracy"] for cases in results.values()],
            label=name,
            marker="o",
        )
    plt.xlim(0.5, 12.5)
    plt.locator_params(axis="x", nbins=12)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("PIE Usage Task Accuracy - Token Length 4")
    plt.legend()
    # Keep only one figure open at a time when the caller saves the generator.
    yield fig
    yield from plot_coefficient_exponentials(coefficients)
    yield from plot_coefficients(coefficients)
    for layer, cases in results.items():
        fig = plt.figure(figsize=(7, 5))
        for case, result in cases.items():
            plot_roc(result["Y_test"], result["Y_pred_prob"], CASE_NAMES[case])
        plt.title(f"PIE Usage ROC - Layer {layer + 1}")
        plt.legend()
        yield fig


def save_results(output_dir, results, coefficients, report):
    """Write a compact metrics table, complete run metadata, and existing figures."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    rows = metrics_rows(results)
    if rows:
        with (output_dir / "metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report["results"] = [
        {
            "layer": layer + 1,
            "feature_group": CASE_NAMES[case],
            **{
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in result.items()
            },
        }
        for layer, cases in results.items()
        for case, result in cases.items()
    ]
    report["figures"] = []
    if results:
        for number, figure in enumerate(plot_results(results, coefficients), 1):
            name = f"figure-{number:02d}.png"
            figure.savefig(output_dir / name, dpi=120)
            plt.close(figure)
            report["figures"].append(name)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


# %% Orchestration and CLI
class Tee:
    """Send progress to the terminal and the saved execution log."""

    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_sample(
    data_path,
    output_dir,
    *,
    limit_per_split=6,
    seed=1,
    offline=False,
    threads=2,
    model_revision=MODEL_REVISION,
    tokens_max=120,
    regularization_candidates=REGULARIZATION_CANDIDATES,
):
    """Coordinate explicit preparation, extraction, training, and saving stages."""
    import contextlib
    import io
    import os

    if limit_per_split < 2 or not 0 <= seed < 2**32 or threads < 1:
        raise ValueError(
            "Use limit_per_split >= 2, seed in [0, 2**32), and threads >= 1"
        )
    data_path, output_dir = Path(data_path), Path(output_dir)
    if not data_path.is_file():
        raise ValueError(f"Dataset not found: {data_path}")
    # Never overwrite an earlier result or silently reuse a partially completed run.
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault(
        "MPLCONFIGDIR", str((output_dir / "matplotlib-cache").resolve())
    )
    import matplotlib

    matplotlib.use("Agg")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    report = {
        "kind": "demonstration_not_thesis_reproduction",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "config": {
            "dataset": data_path.name,
            "dataset_sha256": digest(data_path),
            "limit_per_split": limit_per_split,
            "seed": seed,
            "threads": threads,
            "model": "gpt2",
            "model_revision": model_revision,
            "tokens_max": tokens_max,
            "phrase_length": 4,
            "confidence_threshold": 1,
            "solver": "saga",
            "penalty": "l2",
            "tolerance": 0.001,
            "max_iterations": 200,
            "reg_cs": list(regularization_candidates),
            "offline": offline,
        },
        "environment": {
            "python": platform.python_version(),
            "packages": {
                package: version(package)
                for package in (
                    "torch",
                    "transformers",
                    "numpy",
                    "scikit-learn",
                    "pandas",
                    "scipy",
                    "seaborn",
                    "matplotlib",
                )
            },
        },
        "limitations": [
            "Small execution demonstration, not a reproduction of the thesis or an accuracy estimate.",
            "Original first-prefix span matching and position-adjusted attention are retained; quote normalization preserves original token positions.",
            "A controlled baseline/masked comparison and larger-scale memory optimization are future work.",
        ],
    }
    # __file__ is available for script/module use, but not necessarily a notebook.
    if "__file__" in globals():
        report["implementation_sha256"] = digest(__file__)
    start = time.monotonic()
    results, coefficients, caught = {}, [], []
    log = io.StringIO()
    error = None
    try:
        with (
            warnings.catch_warnings(record=True) as caught,
            contextlib.redirect_stdout(Tee(sys.stdout, log)),
        ):
            warnings.simplefilter("always")
            tokenizer = GPT2Tokenizer.from_pretrained(
                "gpt2", revision=model_revision, local_files_only=offline
            )
            groups, raw_count = prepare_examples(
                data_path, tokenizer, tokens_max=tokens_max
            )
            examples = select_examples(
                groups, limit_per_split=limit_per_split, seed=seed
            )
            report["raw_records"] = raw_count
            report["eligible_counts"] = example_counts(groups)
            report["counts"] = example_counts(examples)
            report["selected_ids"] = {
                split: {
                    label: [example["id"] for example in rows]
                    for label, rows in examples[split].items()
                }
                for split in SPLITS
            }
            print("Actual selected counts:", report["counts"])
            model = GPT2Model.from_pretrained(
                "gpt2",
                revision=model_revision,
                local_files_only=offline,
                output_attentions=True,
                attn_implementation="eager",
            )
            features, retained_bytes = extract_features(examples, model, tokenizer)
            report["retained_array_bytes"] = retained_bytes
            results, coefficients = train_and_evaluate(
                features, seed=seed, regularization_candidates=regularization_candidates
            )
            report["candidate_fits"] = (
                len(features) * len(CASE_NAMES) * len(regularization_candidates)
            )
            report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        error = exc
    finally:
        report["warnings"] = [
            {"category": category, "message": message, "count": count}
            for (category, message), count in Counter(
                (warning.category.__name__, str(warning.message)) for warning in caught
            ).items()
        ]
        report["pipeline_seconds"] = round(time.monotonic() - start, 3)
        (output_dir / "execution.log").write_text(log.getvalue(), encoding="utf-8")
        save_results(output_dir, results, coefficients, report)
    if error is not None:
        raise error
    print(
        f"Saved {len(metrics_rows(results))} metric rows to {output_dir / 'metrics.csv'}"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the bounded four-token thesis probe sample."
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Filtered, type-based MAGPIE JSONL"
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=6,
        help="Maximum examples per original split; balanced, so odd caps round down (default: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Sampling, Torch, NumPy, and probe seed (default: 1)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/probe-sample"))
    parser.add_argument(
        "--offline", action="store_true", help="Require cached GPT-2 files"
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        run_sample(
            args.data,
            args.output_dir,
            limit_per_split=args.limit_per_split,
            seed=args.seed,
            offline=args.offline,
            threads=args.threads,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Sample failed: {exc}\n")
    return 0


# Notebook users call main([...]) explicitly after loading the definition cells.
if __name__ == "__main__" and "__file__" in globals():
    raise SystemExit(main())
