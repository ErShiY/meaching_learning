from contextlib import contextmanager
import csv
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from ablation_report import (
    THRESHOLDS,
    _cache_metadata,
    calibrate_threshold,
    multilabel_metrics,
    write_report,
)


@contextmanager
def temporary_workspace_directory():
    parent = Path(__file__).resolve().parent
    directory = parent / f"ablation_report_fixture_{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent or not resolved.name.startswith("ablation_report_fixture_"):
            raise RuntimeError("Unexpected test fixture cleanup path.")
        shutil.rmtree(resolved)


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def predictions():
    targets = np.zeros((4, 28), dtype=np.uint8)
    targets[np.arange(4), [0, 1, 0, 1]] = 1
    probabilities = np.full((4, 28), 0.01, dtype=np.float32)
    probabilities[:, :2] = 0.6
    probabilities[targets.astype(bool)] = 0.8
    return targets, probabilities


def make_suite(directory):
    directory = Path(directory)
    run_dir = directory / "reference"
    run_dir.mkdir()
    recipe = {
        "epochs": 2, "image_size": 64, "batch_size": 2,
        "learning_rate": 0.0001, "optimizer": "adamw", "weight_decay": 0.0001,
        "scheduler": "warmup_cosine", "warmup_epochs": 1, "min_lr": 0.000001,
        "augmentation": "microscopy", "seed": 42, "threshold": 0.5,
        "asl_gamma_pos": 0, "asl_gamma_neg": 4, "asl_clip": 0.05,
    }
    config = dict(recipe, model_name="densenet_attention", loss="asl")
    splits = {
        "schema_version": 1, "seed": 42, "method": "stratified",
        "splits": {"train": ["training"], "val": ["a", "b", "c", "d"],
                   "holdout": ["h1", "h2"]},
    }
    write_json(run_dir / "split_manifest.json", splits)
    write_json(run_dir / "run_config.json", config)
    write_json(run_dir / "metrics_summary.json", {
        "best_epoch": 2, "holdout_macro_f1": 0.1, "holdout_micro_f1": 0.2,
    })
    targets, probabilities = predictions()
    np.savez_compressed(
        run_dir / "val_predictions.npz", ids=np.asarray(["a", "b", "c", "d"]),
        targets=targets, probabilities=probabilities,
    )
    suite = {
        "schema_version": 1, "recipe": recipe,
        "split_file": str(run_dir / "split_manifest.json"),
        "cells": {
            "attention_asl": {"model_name": "densenet_attention", "loss": "asl",
                              "status": "reused", "run_dir": str(run_dir)},
            "baseline_asl": {"model_name": "baseline", "loss": "asl",
                             "status": "pending", "run_dir": None},
        },
    }
    write_json(directory / "suite.json", suite)
    return suite, run_dir, targets, probabilities


class ThresholdCalibrationTests(unittest.TestCase):
    def test_grid_tie_rule_and_absent_classes(self):
        self.assertEqual(len(THRESHOLDS), 190)
        self.assertEqual(THRESHOLDS[0], 0.05)
        self.assertEqual(THRESHOLDS[-1], 0.995)
        targets, probabilities = predictions()
        best, rows = calibrate_threshold(targets, probabilities)
        self.assertEqual(best["threshold"], 0.605)
        self.assertAlmostEqual(best["macro_f1"], 2 / 28)
        self.assertAlmostEqual(best["micro_f1"], 1)
        self.assertEqual(len(rows), 190)
        self.assertGreater(
            best["macro_f1"], multilabel_metrics(targets, probabilities, 0.5)["macro_f1"]
        )

    def test_empty_predictions_are_not_replaced_by_argmax(self):
        targets = np.asarray([[1, 0], [0, 1]])
        probabilities = np.asarray([[0.4, 0.3], [0.2, 0.4]])
        metrics = multilabel_metrics(targets, probabilities, 0.5)
        self.assertEqual(metrics["macro_f1"], 0)
        self.assertEqual(metrics["micro_f1"], 0)
        np.testing.assert_array_equal(metrics["fn"], [1, 1])

    def test_greater_equal_boundary_and_invalid_inputs(self):
        metrics = multilabel_metrics([[1, 0]], [[0.5, 0.4]], 0.5)
        self.assertEqual(metrics["micro_f1"], 1)
        self.assertEqual(metrics["macro_f1"], 0.5)
        for targets, probabilities in (
            ([], []), ([[1]], [[float("nan")]]), ([[1]], [[1.1]]),
            ([[0.2]], [[0.2]]), ([[1]], [[0.2, 0.3]]),
        ):
            with self.assertRaises(ValueError):
                calibrate_threshold(targets, probabilities)
        for threshold in (-1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                multilabel_metrics([[1]], [[0.5]], threshold)


class AblationReportTests(unittest.TestCase):
    def test_partial_report_writes_scores_and_class_supports_without_inference(self):
        with temporary_workspace_directory() as directory:
            _, run_dir, _, _ = make_suite(directory)
            original_files = {path.name for path in run_dir.iterdir()}
            with patch("ablation_report._holdout_predictions") as collect:
                rows = write_report(directory)
                collect.assert_not_called()
            self.assertEqual(rows[0]["calibrated_threshold"], 0.605)
            self.assertIsNone(rows[0]["holdout_macro_f1_calibrated"])
            self.assertEqual(rows[0]["holdout_macro_f1_fixed"], 0.1)
            self.assertEqual(rows[1]["status"], "pending")
            self.assertIsNone(rows[1]["val_macro_f1_fixed"])
            self.assertEqual(original_files, {path.name for path in run_dir.iterdir()})
            for name in ("summary.csv", "summary.json", "summary.md"):
                self.assertTrue((directory / name).is_file())
            report_dir = directory / "reports" / "attention_asl"
            with (report_dir / "class_metrics.csv").open(encoding="utf-8-sig") as file:
                classes = list(csv.DictReader(file))
            self.assertEqual(len(classes), 56)
            self.assertEqual(classes[0]["support"], "2")
            self.assertEqual(classes[2]["support"], "0")

    def test_calibration_uses_validation_and_applies_frozen_threshold_to_holdout(self):
        with temporary_workspace_directory() as directory:
            make_suite(directory)
            h_targets = np.zeros((2, 28), dtype=np.uint8)
            h_targets[0, 0] = h_targets[1, 1] = 1
            h_probabilities = np.full((2, 28), 0.01, dtype=np.float32)
            h_probabilities[h_targets.astype(bool)] = 0.55
            # Holdout alone prefers 0.05; it must not move the validation optimum 0.605.
            with patch(
                "ablation_report._holdout_predictions",
                return_value=(h_targets, h_probabilities),
            ):
                row = write_report(directory, evaluate_holdout=True)[0]
            self.assertEqual(row["calibrated_threshold"], 0.605)
            self.assertEqual(row["holdout_macro_f1_calibrated"], 0)
            self.assertEqual(row["holdout_micro_f1_calibrated"], 0)

    def test_prediction_id_order_is_aligned_but_unknown_ids_are_rejected(self):
        with temporary_workspace_directory() as directory:
            _, run_dir, targets, probabilities = make_suite(directory)
            np.savez_compressed(
                run_dir / "val_predictions.npz", ids=np.asarray(["d", "c", "b", "a"]),
                targets=targets[::-1], probabilities=probabilities[::-1],
            )
            self.assertEqual(write_report(directory)[0]["calibrated_threshold"], 0.605)
            np.savez_compressed(
                run_dir / "val_predictions.npz", ids=np.asarray(["a", "b", "c", "unknown"]),
                targets=targets, probabilities=probabilities,
            )
            with self.assertRaisesRegex(ValueError, "Prediction IDs"):
                write_report(directory)

    def test_valid_holdout_cache_survives_report_and_stale_cache_never_triggers_inference(self):
        with temporary_workspace_directory() as directory:
            suite, run_dir, _, _ = make_suite(directory)
            config_path = run_dir / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            data_dir = directory / "data"
            data_dir.mkdir()
            (data_dir / "train.csv").write_text("Id,Target\nh1,0\nh2,1\n")
            config["data_dir"] = str(data_dir)
            write_json(config_path, config)
            checkpoint = run_dir / "best_model.pth"
            checkpoint.write_bytes(b"fake checkpoint: cache-only test")
            report_dir = directory / "reports" / "attention_asl"
            report_dir.mkdir(parents=True)
            h_targets = np.zeros((2, 28), dtype=np.uint8)
            h_targets[0, 0] = h_targets[1, 1] = 1
            h_probabilities = np.full((2, 28), 0.01, dtype=np.float32)
            h_probabilities[h_targets.astype(bool)] = 0.8
            metadata = _cache_metadata(run_dir, config, Path(suite["split_file"]))
            np.savez_compressed(
                report_dir / "holdout_predictions.npz", ids=np.asarray(["h1", "h2"]),
                targets=h_targets, probabilities=h_probabilities,
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            # No real torch checkpoint exists: this can pass only by reusing cache.
            row = write_report(directory)[0]
            self.assertAlmostEqual(row["holdout_macro_f1_calibrated"], 2 / 28)
            self.assertIn("without inference", row["notes"])
            self.assertEqual(write_report(directory)[0], row)
            checkpoint.write_bytes(b"checkpoint changed; old cache is stale")
            row = write_report(directory)[0]
            self.assertIsNone(row["holdout_macro_f1_calibrated"])

    def test_wrong_suite_split_hash_is_rejected(self):
        with temporary_workspace_directory() as directory:
            suite, _, _, _ = make_suite(directory)
            suite["split_sha256"] = "wrong"
            write_json(directory / "suite.json", suite)
            with self.assertRaisesRegex(ValueError, "split_sha256"):
                write_report(directory)

    def test_mismatched_recipe_split_and_targets_are_rejected(self):
        with temporary_workspace_directory() as directory:
            suite, run_dir, targets, probabilities = make_suite(directory)
            config_path = run_dir / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["epochs"] += 1
            write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, "shared recipe mismatch"):
                write_report(directory)
            config["epochs"] -= 1
            write_json(config_path, config)
            other_run = directory / "other_run"
            shutil.copytree(run_dir, other_run)
            config["model_name"] = "baseline"
            write_json(other_run / "run_config.json", config)
            suite["cells"]["baseline_asl"].update(
                status="completed", run_dir=str(other_run)
            )
            write_json(directory / "suite.json", suite)
            altered_targets = targets.copy()
            altered_targets[0, 1] = 1
            np.savez_compressed(
                other_run / "val_predictions.npz", ids=np.asarray(["a", "b", "c", "d"]),
                targets=altered_targets, probabilities=probabilities,
            )
            with self.assertRaisesRegex(ValueError, "targets differ"):
                write_report(directory)
            np.savez_compressed(
                other_run / "val_predictions.npz", ids=np.asarray(["a", "b", "c", "d"]),
                targets=targets, probabilities=probabilities,
            )
            split = json.loads((other_run / "split_manifest.json").read_text())
            split["splits"]["train"] = ["other_training"]
            write_json(other_run / "split_manifest.json", split)
            with self.assertRaisesRegex(ValueError, "split differs"):
                write_report(directory)


if __name__ == "__main__":
    unittest.main()
