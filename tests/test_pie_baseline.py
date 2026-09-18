"""Focused checks; no dataset, model download, or full experiment required."""

import ast
import contextlib
import csv
import hashlib
import io
import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

ROOT = Path(__file__).resolve().parents[1]
STEM = "pie_usage_probe"
SCRIPT = ROOT / "scripts" / f"{STEM}.py"


def definitions():
    """Import the same implementation used by the entry point, without AST surgery."""
    spec = importlib.util.spec_from_file_location(
        "sample_entry", ROOT / "scripts" / "run_pie_demo.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return vars(entry.load_baseline())


class BaselineTests(unittest.TestCase):
    def test_exact_feature_membership_and_no_duplicates(self):
        feature_columns = definitions()["feature_columns"]
        expected = {
            1: list(range(768)),
            4: list(range(772, 892)),
            5: list(range(768)) + list(range(772, 892)),
        }
        for case, columns in expected.items():
            with self.subTest(case=case):
                actual = feature_columns(case)
                self.assertEqual(actual, columns)
                self.assertEqual(len(actual), len(set(actual)))
                self.assertFalse(set(actual) & set(range(768, 772)))
        self.assertEqual([len(feature_columns(c)) for c in (1, 4, 5)], [768, 120, 888])

    def test_scaler_uses_only_training_statistics_for_every_case(self):
        scale_features = definitions()["scale_features"]
        X_train = np.tile([[0.0], [2.0]], (1, 892))
        X_dev = np.tile([[100.0], [102.0]], (1, 892))
        X_test = np.tile([[200.0], [202.0]], (1, 892))
        for case in (1, 4, 5):
            train, dev, test, scaler = scale_features(X_train, X_dev, X_test, case=case)
            np.testing.assert_array_equal(train[:, 0], [-1, 1])
            np.testing.assert_array_equal(dev[:, 0], [99, 101])
            np.testing.assert_array_equal(test[:, 0], [199, 201])
            np.testing.assert_array_equal(scaler.mean_, np.ones(train.shape[1]))
            self.assertEqual(scaler.n_samples_seen_, 2)

    def test_auxiliary_columns_cannot_affect_active_inputs(self):
        scale_features = definitions()["scale_features"]
        rng = np.random.default_rng(7)
        matrices = [rng.normal(size=(6, 892)) for _ in range(3)]
        before = {case: scale_features(*matrices, case=case)[:3] for case in (1, 4, 5)}
        for matrix in matrices:
            matrix[:, 768:772] = rng.normal(size=(6, 4)) * 1e6
        for case in (1, 4, 5):
            for old, new in zip(before[case], scale_features(*matrices, case=case)[:3]):
                np.testing.assert_array_equal(old, new)

    def test_test_features_cannot_change_training_or_dev_scaling(self):
        scale_features = definitions()["scale_features"]
        rng = np.random.default_rng(8)
        matrices = [rng.normal(size=(6, 892)) for _ in range(3)]
        before = scale_features(*matrices, case=5)
        matrices[2] *= 10000
        after = scale_features(*matrices, case=5)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])

    def test_regularization_uses_development_score_and_first_tie(self):
        namespace = definitions()
        for scores, expected in [
            ({0.01: 0, 1: 0, 10: 0}, 0.01),
            ({0.01: 0.2, 1: 0.8, 10: 0.8}, 1.0),
        ]:

            class Classifier:
                def __init__(self, C, **kwargs):
                    self.C = C

                def fit(self, x, y):
                    self.coef_ = np.zeros((1, x.shape[1]))

                def predict(self, x):
                    return np.array([0, 1])

                def predict_proba(self, x):
                    return np.array([[0.8, 0.2], [0.3, 0.7]])

                def score(self, x, y):
                    return scores[self.C]

            namespace["LogisticRegression"] = Classifier
            x, y = np.tile([[0.0], [1.0]], (1, 892)), np.array([0, 1])
            result, model, scaler = namespace["fit_and_evaluate_probe"](
                x,
                y,
                x,
                y,
                x,
                y,
                case=1,
                regularization_candidates=[0.01, 1.0, 10.0],
                seed=0,
            )
            self.assertEqual(result["best_C"], expected)
            self.assertEqual(model.C, expected)
            self.assertEqual(scaler.n_samples_seen_, 2)
            np.testing.assert_array_equal(result["Y_pred_prob"], [0.2, 0.7])

    def test_test_labels_cannot_change_models_or_selected_C(self):
        def evaluate(test_labels):
            namespace = definitions()
            rng = np.random.default_rng(11)
            matrices = [rng.normal(size=(count, 892)) for count in (12, 6, 6)]
            labels = [np.arange(len(matrix)) % 2 for matrix in matrices]
            labels[2] = test_labels
            return {
                case: namespace["fit_and_evaluate_probe"](
                    matrices[0],
                    labels[0],
                    matrices[1],
                    labels[1],
                    matrices[2],
                    labels[2],
                    case=case,
                    regularization_candidates=[0.01, 1.0, 10.0],
                    seed=0,
                )
                for case in (1, 4, 5)
            }

        first = evaluate(np.zeros(6, dtype=int))
        second = evaluate(np.ones(6, dtype=int))
        self.assertEqual(set(first), {1, 4, 5})
        self.assertNotIn("bestCase", first)
        self.assertNotIn("maxScore", first)
        for case in first:
            a, first_model, _ = first[case]
            b, second_model, _ = second[case]
            self.assertEqual(first_model.C, second_model.C)
            np.testing.assert_array_equal(first_model.coef_, second_model.coef_)
            np.testing.assert_array_equal(
                first_model.intercept_, second_model.intercept_
            )
            self.assertEqual(a["best_C"], b["best_C"])
            self.assertEqual(a["dev_accuracy"], b["dev_accuracy"])
            np.testing.assert_array_equal(a["Y_pred_prob"], b["Y_pred_prob"])
            np.testing.assert_array_equal(a["Y_pred"], b["Y_pred"])
            self.assertAlmostEqual(a["test_accuracy"] + b["test_accuracy"], 1.0)

    def test_roc_cell_plots_every_case_even_at_zero_accuracy(self):
        result = {
            "Y_test": np.array([0, 1]),
            "Y_pred_prob": np.array([0.8, 0.2]),
            "test_accuracy": 0.0,
        }
        results = {i: {c: result for c in (1, 4, 5)} for i in range(12)}
        with patch.object(plt, "plot", wraps=plt.plot) as plotted:
            for figure in definitions()["plot_results"](results, []):
                plt.close(figure)
        self.assertEqual(
            plotted.call_count, 39
        )  # Three accuracy curves plus 36 ROC curves.
        self.assertEqual(
            [call.kwargs["label"] for call in plotted.call_args_list],
            ["EMB", "ATTN", "EMB+ATTN"] * 13,
        )

    def test_inference_disables_gradients_and_stores_compact_arrays(self):
        class Model(torch.nn.Module):
            def forward(self, tokens, output_hidden_states):
                assert not torch.is_grad_enabled() and not self.training
                assert torch.is_inference_mode_enabled()
                hidden = (torch.ones(1, tokens.shape[1], 768),) * 13
                attn = (torch.ones(1, 12, tokens.shape[1], tokens.shape[1]),) * 12

                class Output:
                    hidden_states = hidden

                    def __getitem__(self, index):
                        return attn

                return Output()

        rows = [{"tokens": ["a", "b", "c", "d"]}]
        tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda tokens: list(range(len(tokens)))
        )
        result = definitions()["run_model_inference"](rows, Model(), tokenizer)
        self.assertEqual(rows, [{"tokens": ["a", "b", "c", "d"]}])
        for key, count, shape in (
            ("attentions", 12, (12, 4, 4)),
            ("hidden_states", 13, (4, 768)),
        ):
            self.assertEqual(len(result[0][key]), count)
            for array in result[0][key]:
                self.assertEqual(array.dtype, np.float32)
                self.assertEqual(array.shape, shape)

    def test_compact_storage_preserves_feature_values_and_attention_order(self):
        namespace = definitions()
        rng = np.random.default_rng(13)
        emb = [rng.normal(size=(6, 768)).astype(np.float32) for _ in range(13)]
        attn = [rng.random(size=(12, 6, 6)).astype(np.float32) for _ in range(12)]
        indices = [1, 2, 3, 4]
        embedding_features = namespace["extract_embedding_features"]
        attention_features = namespace["extract_attention_features"]
        np.testing.assert_array_equal(
            embedding_features(emb, indices, layer=0),
            embedding_features([a.tolist() for a in emb], indices, layer=0),
        )
        features = attention_features(attn, indices, layer=0)
        np.testing.assert_array_equal(
            features, attention_features([a.tolist() for a in attn], indices, layer=0)
        )
        expected = [
            float(attn[0][h, indices[k], indices[j]]) * indices[k]
            for h in range(12)
            for j in range(4)
            for k in range(j, 4)
        ]
        np.testing.assert_array_equal(features, expected)

    def test_all_notebooks_match_exports_and_have_cleared_outputs(self):
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text())
                sources = []
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    sources.append(
                        "".join(cell["source"]).replace(
                            "!pip install transformers", "# !pip install transformers"
                        )
                    )
                expected = ast.dump(ast.parse("\n\n".join(sources)))
                export = (ROOT / "scripts" / f"{path.stem}.py").read_text()
                self.assertEqual(
                    export.splitlines()[0], f"# Exported from notebooks/{path.name}."
                )
                actual = ast.dump(ast.parse(export))
                self.assertEqual(expected, actual)

    def test_notebook_sync_follows_sections_and_is_idempotent(self):
        spec = importlib.util.spec_from_file_location(
            "notebook_sync", ROOT / "tools" / "sync_pie_notebook.py"
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        source = ROOT / "notebooks" / f"{STEM}.ipynb"
        with tempfile.TemporaryDirectory() as directory:
            notebook_path = Path(directory) / source.name
            notebook_path.write_bytes(source.read_bytes())
            original = json.loads(source.read_text())
            self.assertEqual(helper.sync_notebook(SCRIPT, notebook_path), 6)
            first = notebook_path.read_bytes()
            helper.sync_notebook(SCRIPT, notebook_path)
            self.assertEqual(first, notebook_path.read_bytes())
            synced = json.loads(first)
            self.assertEqual(synced["metadata"], original["metadata"])
            self.assertEqual(
                [cell["id"] for cell in synced["cells"]],
                [
                    "imports-and-settings",
                    "preprocessing-and-sampling",
                    "model-inference-and-feature-construction",
                    "probe-fitting",
                    "plotting-and-saving",
                    "orchestration-and-cli",
                ],
            )

    def test_import_does_not_load_data_models_or_train(self):
        from transformers import GPT2Model, GPT2Tokenizer

        with (
            patch.object(GPT2Model, "from_pretrained") as model,
            patch.object(GPT2Tokenizer, "from_pretrained") as tokenizer,
            patch.object(LogisticRegression, "fit") as fit,
            patch.object(
                Path, "open", side_effect=AssertionError("Unexpected data access")
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            namespace = definitions()
            self.assertNotIn("Probe", namespace)
            self.assertTrue(callable(namespace["fit_and_evaluate_probe"]))
        model.assert_not_called()
        tokenizer.assert_not_called()
        fit.assert_not_called()
        self.assertEqual(output.getvalue(), "")

    def test_cli_passes_explicit_settings_and_has_small_defaults(self):
        namespace = definitions()
        with patch.dict(namespace, run_sample=unittest.mock.Mock()) as patched:
            namespace["main"](["--data", "data.jsonl"])
            self.assertEqual(
                patched["run_sample"].call_args.kwargs["limit_per_split"], 6
            )
            self.assertEqual(patched["run_sample"].call_args.kwargs["seed"], 1)
            namespace["main"](
                [
                    "--data",
                    "data.jsonl",
                    "--limit-per-split",
                    "8",
                    "--seed",
                    "17",
                    "--output-dir",
                    "output",
                    "--offline",
                ]
            )
            self.assertEqual(
                patched["run_sample"].call_args.kwargs["limit_per_split"], 8
            )
            self.assertEqual(patched["run_sample"].call_args.kwargs["seed"], 17)
            self.assertEqual(
                patched["run_sample"].call_args.args,
                (Path("data.jsonl"), Path("output")),
            )

    def test_notebook_definition_cells_do_not_start_experiment(self):
        from transformers import GPT2Model, GPT2Tokenizer

        notebook = json.loads((ROOT / "notebooks" / f"{STEM}.ipynb").read_text())
        namespace = {"__name__": "__main__"}  # Notebook has no __file__.
        with (
            patch.object(GPT2Model, "from_pretrained") as model,
            patch.object(GPT2Tokenizer, "from_pretrained") as tokenizer,
            patch.object(LogisticRegression, "fit") as fit,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(
                        compile("".join(cell["source"]), "notebook-cell", "exec"),
                        namespace,
                    )
        model.assert_not_called()
        tokenizer.assert_not_called()
        fit.assert_not_called()
        self.assertEqual(output.getvalue(), "")
        self.assertTrue(callable(namespace["run_sample"]))

    def test_requested_seed_reaches_actual_classifier(self):
        namespace = definitions()
        rng = np.random.default_rng(19)
        features = {}
        for split in ("train", "dev", "test"):
            features[split] = (rng.normal(size=(6, 892)), np.array([0, 1, 0, 1, 0, 1]))
        classifiers = []

        def classifier(**kwargs):
            model = LogisticRegression(**kwargs)
            classifiers.append(model)
            return model

        with (
            patch.dict(namespace, LogisticRegression=classifier),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            results, coefficients = namespace["train_and_evaluate"](
                {0: features}, seed=17, regularization_candidates=(0.01,)
            )
        self.assertEqual(len(classifiers), 3)
        self.assertTrue(all(model.random_state == 17 for model in classifiers))
        self.assertEqual(set(results[0]), {1, 4, 5})
        self.assertEqual(len(coefficients), 1)
        np.testing.assert_array_equal(coefficients[0], classifiers[1].coef_[0])

    def test_included_example_is_small_and_consistent(self):
        directory = ROOT / "examples" / "baseline-demo"
        report = json.loads((directory / "run.json").read_text())
        with (directory / "metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["kind"], "demonstration_not_thesis_reproduction")
        self.assertEqual(
            report["implementation_sha256"],
            hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        )
        self.assertEqual(report["config"]["seed"], 1)
        self.assertEqual(
            report["counts"],
            {s: {"idiom": 3, "literal": 3} for s in ("train", "dev", "test")},
        )
        self.assertEqual(len(rows), 36)
        self.assertEqual(
            {(int(r["layer"]), r["feature_group"]) for r in rows},
            {
                (layer, group)
                for layer in range(1, 13)
                for group in ("EMB", "ATTN", "EMB+ATTN")
            },
        )
        for row in rows:
            self.assertEqual(
                int(row["n_features"]),
                {"EMB": 768, "ATTN": 120, "EMB+ATTN": 888}[row["feature_group"]],
            )
            self.assertIn(float(row["selected_C"]), report["config"]["reg_cs"])
            self.assertTrue(0 <= float(row["test_accuracy"]) <= 1)


class PhraseIndexTests(unittest.TestCase):
    def setUp(self):
        self.namespace = definitions()
        self.phrase = "go through the motions"

    def assert_span(self, tokens, expected_indices, expected_tokens):
        indices = self.namespace["find_phrase_indices"](tokens, self.phrase)
        self.assertEqual(indices, expected_indices)
        self.assertEqual([tokens[i] for i in indices], expected_tokens)

    def test_quotes_before_phrase_keep_original_positions(self):
        tokens = [
            "He",
            "Ġsaid",
            'Ġ"',
            "hello",
            '"',
            "Ġgo",
            "Ġthrough",
            "Ġthe",
            "Ġmotions",
            ".",
        ]
        self.assert_span(tokens, [5, 6, 7, 8], ["Ġgo", "Ġthrough", "Ġthe", "Ġmotions"])

    def test_quotes_within_phrase_keep_original_positions(self):
        for quote in ('"', "'"):
            with self.subTest(quote=quote):
                tokens = [
                    "They",
                    "Ġgo",
                    "Ġ" + quote,
                    "through",
                    quote,
                    "Ġthe",
                    "Ġmotions",
                    ".",
                ]
                self.assert_span(
                    tokens, [1, 3, 5, 6], ["Ġgo", "through", "Ġthe", "Ġmotions"]
                )

    def test_unquoted_phrase_keeps_original_positions(self):
        tokens = ["They", "Ġgo", "Ġthrough", "Ġthe", "Ġmotions", "."]
        self.assert_span(tokens, [1, 2, 3, 4], ["Ġgo", "Ġthrough", "Ġthe", "Ġmotions"])

    def test_prepare_examples_crops_corrected_noncontiguous_span(self):
        tokens = (
            ["Ġlead"] * 80
            + ["Ġgo", 'Ġ"', "through", '"', "Ġthe", "Ġmotions"]
            + ["Ġtail"] * 80
        )
        self.assert_span(
            tokens, [80, 82, 84, 85], ["Ġgo", "through", "Ġthe", "Ġmotions"]
        )
        records = [
            {
                "id": i,
                "idiom": self.phrase,
                "context": [self.phrase],
                "confidence": 1,
                "split": "training",
                "label_distribution": {"i": i, "l": 1 - i},
            }
            for i in (0, 1)
        ]
        tokenizer = SimpleNamespace(tokenize=lambda text: tokens.copy())
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "fixture.jsonl"
            data.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            groups, count = self.namespace["prepare_examples"](data, tokenizer)
        self.assertEqual(count, 2)
        for rows in groups["train"].values():
            (row,) = rows
            self.assertEqual(
                row["tokens"], tokens[20:145]
            )  # Original exclusive crop endpoint.
            self.assertEqual(row["phrase_indices"], [60, 62, 64, 65])
            self.assertEqual(
                [row["tokens"][i] for i in row["phrase_indices"]],
                ["Ġgo", "through", "Ġthe", "Ġmotions"],
            )
            self.assertTrue(
                all(0 <= i < len(row["tokens"]) for i in row["phrase_indices"])
            )

    def test_crop_rejects_invalid_indices_and_preserves_edge_spans(self):
        crop = self.namespace["crop_phrase_context"]
        tokens = list(range(150))
        for indices in ([], [-1, 0, 1, 2], [147, 148, 149, 150], [2, 1], [1, 1]):
            with (
                self.subTest(indices=indices),
                self.assertRaisesRegex(ValueError, "indices"),
            ):
                crop(tokens, indices)
        for indices in ([0, 1, 2, 3], [146, 147, 148, 149]):
            with self.subTest(indices=indices):
                cropped, retained = crop(tokens, indices)
                self.assertEqual(
                    [cropped[i] for i in retained], [tokens[i] for i in indices]
                )
                self.assertTrue(all(0 <= i < len(cropped) for i in retained))
        short = ["a", "b", "c", "d"]
        self.assertEqual(crop(short, [0, 1, 2, 3]), (short, [0, 1, 2, 3]))
        with self.assertRaisesRegex(ValueError, "tokens_max"):
            crop(tokens, [80, 81, 82, 83], tokens_max=1)

    def test_corrected_spans_feed_exact_features_at_every_layer(self):
        fixtures = [
            ["They", "Ġgo", "Ġthrough", "Ġthe", "Ġmotions", "."],
            [
                "He",
                "Ġsaid",
                'Ġ"',
                "hello",
                '"',
                "Ġgo",
                "Ġthrough",
                "Ġthe",
                "Ġmotions",
                ".",
            ],
            ["They", "Ġgo", 'Ġ"', "through", '"', "Ġthe", "Ġmotions", "."],
            ["Ġlead"] * 80
            + ["Ġgo", 'Ġ"', "through", '"', "Ġthe", "Ġmotions"]
            + ["Ġtail"] * 80,
        ]
        rng = np.random.default_rng(29)
        for tokens in fixtures:
            indices = self.namespace["find_phrase_indices"](tokens, self.phrase)
            tokens, indices = self.namespace["crop_phrase_context"](tokens, indices)
            hidden_states = rng.normal(size=(13, len(tokens), 768)).astype(np.float32)
            attentions = rng.random(size=(12, 12, len(tokens), len(tokens))).astype(
                np.float32
            )
            for layer in range(12):
                with self.subTest(tokens=len(tokens), layer=layer):
                    selected = hidden_states[layer + 1, indices].astype(float)
                    mean = selected.mean(axis=0)
                    cosines = [
                        np.dot(selected[i], selected[j])
                        / (np.linalg.norm(selected[i]) * np.linalg.norm(selected[j]))
                        for i in range(3)
                        for j in range(i + 1, 4)
                    ]
                    expected_embedding = [
                        *mean,
                        np.linalg.norm(mean),
                        np.mean(cosines),
                        max(cosines),
                        min(cosines),
                    ]
                    actual_embedding = self.namespace["extract_embedding_features"](
                        hidden_states, indices, layer=layer
                    )
                    np.testing.assert_array_equal(actual_embedding, expected_embedding)
                    expected_attention = [
                        float(attentions[layer, head, indices[query], indices[key]])
                        * indices[query]
                        for head in range(12)
                        for key in range(4)
                        for query in range(key, 4)
                    ]
                    actual_attention = self.namespace["extract_attention_features"](
                        attentions, indices, layer=layer
                    )
                    np.testing.assert_array_equal(actual_attention, expected_attention)

    def test_feature_matrix_keeps_row_labels_and_raw_column_order(self):
        rng = np.random.default_rng(31)
        examples = [
            {
                "phrase_indices": indices,
                "hidden_states": rng.normal(size=(13, 8, 768)).astype(np.float32),
                "attentions": rng.random(size=(12, 12, 8, 8)).astype(np.float32),
            }
            for indices in ([1, 3, 5, 6], [1, 2, 3, 4], [3, 4, 5, 6])
        ]
        for layer in (0, 11):
            X, y = self.namespace["assemble_feature_matrix"](
                examples[:2], examples[2:], layer=layer
            )
            self.assertEqual(X.shape, (3, 892))
            self.assertEqual(X.dtype, np.float64)
            np.testing.assert_array_equal(y, [1, 1, 0])
            for row, example in zip(X, examples):
                np.testing.assert_array_equal(
                    row[:772],
                    self.namespace["extract_embedding_features"](
                        example["hidden_states"], example["phrase_indices"], layer=layer
                    ),
                )
                np.testing.assert_array_equal(
                    row[772:],
                    self.namespace["extract_attention_features"](
                        example["attentions"], example["phrase_indices"], layer=layer
                    ),
                )


def fixture_records():
    records = []
    for split in ("training", "development", "test"):
        phrase = f"{split} one two three"
        for label in ("i", "l"):
            for index in range(8):
                records.append(
                    {
                        "id": len(records),
                        "idiom": phrase,
                        "context": [phrase + " tail"],
                        "confidence": 1,
                        "label_distribution": {
                            "i": int(label == "i"),
                            "l": int(label == "l"),
                        },
                        "split": split,
                    }
                )
    return records


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.namespace = definitions()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "fixture.jsonl"
        self.data.write_text("\n".join(json.dumps(r) for r in fixture_records()) + "\n")
        self.tokenizer = SimpleNamespace(tokenize=lambda text: text.split())
        self.groups, self.raw_count = self.namespace["prepare_examples"](
            self.data, self.tokenizer
        )

    def test_cap_seed_classes_and_original_split_boundaries(self):
        select = self.namespace["select_examples"]
        first = select(self.groups, limit_per_split=7, seed=1)
        self.assertEqual(first, select(self.groups, limit_per_split=7, seed=1))
        self.assertNotEqual(first, select(self.groups, limit_per_split=7, seed=2))
        for split, original in (
            ("train", "training"),
            ("dev", "development"),
            ("test", "test"),
        ):
            self.assertEqual(
                [len(first[split][label]) for label in ("idiom", "literal")], [3, 3]
            )
            self.assertTrue(
                all(
                    example["split"] == original
                    for rows in first[split].values()
                    for example in rows
                )
            )
        self.assertEqual(self.raw_count, 48)

    def test_insufficient_class_and_invalid_cap_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            self.namespace["select_examples"](self.groups, limit_per_split=1)
        self.groups["dev"]["literal"] = []
        with self.assertRaisesRegex(ValueError, "dev: need at least one"):
            self.namespace["select_examples"](self.groups)

    def test_overlap_between_splits_is_rejected(self):
        for rows in self.groups["test"].values():
            for example in rows:
                example["idiom"] = self.groups["train"]["literal"][0]["idiom"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.namespace["select_examples"](self.groups)

    def test_cap_is_applied_before_model_inference(self):
        namespace = self.namespace
        seen = []

        def extraction(examples, model, tokenizer):
            seen.append(namespace["example_counts"](examples))
            return {}, 0

        with (
            patch.object(
                namespace["GPT2Tokenizer"],
                "from_pretrained",
                return_value=self.tokenizer,
            ),
            patch.object(namespace["GPT2Model"], "from_pretrained"),
            patch.dict(namespace, extract_features=extraction),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            report = namespace["run_sample"](
                self.data,
                Path(self.temp.name) / "run",
                limit_per_split=4,
                seed=17,
                offline=True,
            )
        self.assertEqual(
            seen, [{s: {"idiom": 2, "literal": 2} for s in ("train", "dev", "test")}]
        )
        self.assertEqual(report["config"]["seed"], 17)

    def test_insufficient_examples_fail_before_model_load(self):
        self.data.write_text(
            "\n".join(json.dumps(r) for r in fixture_records() if r["split"] != "test")
            + "\n"
        )
        namespace = self.namespace
        with (
            patch.object(
                namespace["GPT2Tokenizer"],
                "from_pretrained",
                return_value=self.tokenizer,
            ),
            patch.object(namespace["GPT2Model"], "from_pretrained") as model,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "test: need at least one"),
        ):
            namespace["run_sample"](
                self.data, Path(self.temp.name) / "invalid", offline=True
            )
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
