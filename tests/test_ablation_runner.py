"""Runner checks use synthetic run artifacts and never start training."""

from contextlib import contextmanager, redirect_stdout
import csv
import io
from pathlib import Path
import shutil
import subprocess
import sys
import types
import unittest
from unittest import mock
import uuid

import ablation


@contextmanager
def workspace_fixture():
    parent = Path(__file__).resolve().parent
    directory = parent / f"ablation_fixture_{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent or not resolved.name.startswith("ablation_fixture_"):
            raise RuntimeError("Unexpected test fixture cleanup path.")
        shutil.rmtree(resolved)


def make_recipe(directory):
    data = directory / "data with spaces"
    data.mkdir()
    (data / "train").mkdir()
    (data / "test").mkdir()
    (data / "train.csv").write_text("Id,Target\na,0\nb,1\nc,0\n", encoding="utf-8")
    (data / "sample_submission.csv").write_text("Id,Predicted\nt,\n", encoding="utf-8")
    weights = directory / "imagenet weights.pth"
    weights.write_bytes(b"synthetic weights; not loaded")
    return {
        "image_size": 512, "batch_size": 8, "epochs": 3,
        "learning_rate": 1e-4, "asl_gamma_pos": 0.0, "asl_gamma_neg": 4.0,
        "asl_clip": 0.05, "optimizer": "adamw", "weight_decay": 1e-4,
        "scheduler": "warmup_cosine", "warmup_epochs": 2.0, "min_lr": 1e-6,
        "split_method": "stratified", "augmentation": "microscopy",
        "num_workers": 4, "threshold": 0.5, "seed": 42, "device": "cuda",
        "data_dir": str(data), "pretrained_weights": str(weights),
    }


def make_run(run_dir, recipe, model_name="densenet_attention", loss="asl", split=None):
    run_dir.mkdir(parents=True)
    manifest = split or {
        "schema_version": 1, "seed": 42, "method": "stratified",
        "splits": {"train": ["a"], "val": ["b"], "holdout": ["c"]},
    }
    ablation.write_json(run_dir / "split_manifest.json", manifest)
    checksum = ablation.file_sha256(run_dir / "split_manifest.json")
    config = dict(recipe, model_name=model_name, loss=loss, debug=False,
                  smoke_test=False, split_sha256=checksum)
    ablation.write_json(run_dir / "run_config.json", config)
    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "val_macro_f1", "is_best"])
        writer.writeheader()
        for epoch in range(1, recipe["epochs"] + 1):
            writer.writerow({"epoch": epoch, "val_macro_f1": epoch / 10, "is_best": 1})
    ablation.write_json(run_dir / "metrics_summary.json", {
        "model_name": model_name, "loss": loss, "split_sha256": checksum,
        "best_epoch": recipe["epochs"], "best_val_macro_f1": recipe["epochs"] / 10,
        "holdout_macro_f1": 0.25, "holdout_micro_f1": 0.5,
    })
    (run_dir / "best_model.pth").write_bytes(b"synthetic checkpoint; not loaded")
    (run_dir / "val_predictions.npz").write_bytes(b"synthetic predictions; report mocked")
    return run_dir


def make_suite(directory, experiments):
    recipe = make_recipe(directory)
    reference = make_run(directory / "reference", recipe)
    suite = ablation.create_plan(reference, experiments)
    suite_dir = directory / "suite"
    suite_dir.mkdir()
    frozen = suite_dir / "split_manifest.json"
    frozen.write_bytes(Path(suite["split_file"]).read_bytes())
    suite["split_file"] = str(frozen)
    suite["split_sha256"] = ablation.file_sha256(frozen)
    ablation.write_json(suite_dir / "suite.json", suite)
    return suite_dir, suite


def report_stub():
    module = types.ModuleType("ablation_report")
    module.write_report = mock.Mock(return_value=[])
    return module


class AblationRunnerTests(unittest.TestCase):
    def test_reference_reuse_and_commands_freeze_the_training_recipe(self):
        with workspace_fixture() as directory:
            recipe = make_recipe(directory)
            reference = make_run(directory / "reference", recipe)
            suite = ablation.create_plan(reference, ablation.DEFAULT_EXPERIMENTS)
            self.assertEqual(suite["cells"]["attention_asl"]["status"], "reused")
            self.assertEqual(suite["cells"]["attention_asl"]["run_dir"], str(reference))
            command = ablation.build_command(suite, "baseline_asl", directory / "suite")
            self.assertEqual(command[:3], [sys.executable, "-u", str(ablation.ROOT / "train.py")])
            flags = dict(zip(command[3:-1:2], command[4:-1:2]))
            self.assertEqual(flags["--model"], "baseline")
            self.assertEqual(flags["--loss"], "asl")
            self.assertEqual(flags["--data-dir"], str(directory / "data with spaces"))
            self.assertEqual(flags["--split-file"], str(reference / "split_manifest.json"))
            self.assertEqual(flags["--output-dir"], str(directory / "suite" / "baseline_asl"))
            for key, value in recipe.items():
                self.assertEqual(flags["--" + key.replace("_", "-")], str(value))
            self.assertEqual(command[-1], "--skip-submission")
            retrain = ablation.create_plan(reference, ["attention_asl"])
            self.assertEqual(retrain["cells"]["attention_asl"]["status"], "pending")
            self.assertEqual(retrain["cells"]["baseline_bce"]["status"], "not_selected")

    def test_completed_run_rejects_unmatched_recipe_unfinished_epochs_and_wrong_split(self):
        with workspace_fixture() as directory:
            recipe = make_recipe(directory)
            run = make_run(directory / "run", recipe)
            config = ablation.read_json(run / "run_config.json")
            ablation.validate_completed_run(run, "densenet_attention", "asl", recipe)
            altered = dict(recipe, optimizer="adam")
            with self.assertRaisesRegex(ValueError, "Recipe mismatch for optimizer"):
                ablation.validate_completed_run(run, "densenet_attention", "asl", altered)
            config["epochs"] += 1
            ablation.write_json(run / "run_config.json", config)
            with self.assertRaisesRegex(ValueError, "finished all"):
                ablation.validate_completed_run(run, "densenet_attention", "asl")
            config["epochs"] -= 1
            ablation.write_json(run / "run_config.json", config)
            wrong = directory / "wrong_split.json"
            manifest = ablation.read_json(run / "split_manifest.json")
            manifest["splits"]["train"], manifest["splits"]["val"] = ["b"], ["a"]
            ablation.write_json(wrong, manifest)
            with self.assertRaisesRegex(ValueError, "Different train/validation/holdout"):
                ablation.validate_completed_run(run, "densenet_attention", "asl", split_file=wrong)

    def test_rare_augmentation_is_forwarded_and_checked_for_legacy_recipes(self):
        with workspace_fixture() as directory:
            legacy_recipe = make_recipe(directory)
            augmented_recipe = dict(
                legacy_recipe, rare_augment_max_count=100, rare_augment_extra_copies=3,
            )
            reference = make_run(directory / "reference", augmented_recipe)
            suite = ablation.create_plan(reference, ["attention_bce"])
            for recipe, expected in ((suite["recipe"], (100, 3)), (legacy_recipe, (0, 2))):
                command = ablation.build_command(dict(suite, recipe=recipe), "attention_bce", directory)
                flags = dict(zip(command[3:-1:2], command[4:-1:2]))
                self.assertEqual(flags["--rare-augment-max-count"], str(expected[0]))
                self.assertEqual(flags["--rare-augment-extra-copies"], str(expected[1]))
            ablation.validate_completed_run(reference, "densenet_attention", "asl", suite["recipe"])
            with self.assertRaisesRegex(ValueError, "recipe mismatch for rare augmentation"):
                ablation.validate_completed_run(reference, "densenet_attention", "asl", legacy_recipe)
            legacy_run = make_run(directory / "legacy_run", legacy_recipe)
            disabled_recipe = dict(legacy_recipe, rare_augment_max_count=100, rare_augment_extra_copies=0)
            ablation.validate_completed_run(legacy_run, "densenet_attention", "asl", disabled_recipe)

    def test_dry_run_checks_real_reference_without_writing_a_suite_or_launching_training(self):
        with workspace_fixture() as directory:
            reference = make_run(directory / "reference", make_recipe(directory))
            suite_dir = directory / "not_created"
            with mock.patch.object(ablation.subprocess, "run") as launch, redirect_stdout(io.StringIO()):
                ablation.main([
                    "run", "--reference-run", str(reference), "--suite-dir", str(suite_dir),
                    "--experiments", "baseline_asl", "attention_bce", "--dry-run",
                ])
            self.assertFalse(suite_dir.exists())
            launch.assert_not_called()

    def test_sequential_cells_finish_once_and_completed_suite_is_skipped(self):
        with workspace_fixture() as directory:
            suite_dir, suite = make_suite(directory, ["baseline_asl", "attention_bce"])
            launches = []
            report = report_stub()

            def train(command, *, cwd, check):
                self.assertEqual(cwd, ablation.ROOT)
                self.assertTrue(check)
                options = dict(zip(command[3:-1:2], command[4:-1:2]))
                launches.append((options["--model"], options["--loss"]))
                make_run(Path(options["--output-dir"]) / "finished", suite["recipe"],
                         options["--model"], options["--loss"])

            with mock.patch.dict(sys.modules, {"ablation_report": report}), \
                    mock.patch.object(ablation.subprocess, "run", side_effect=train), \
                    redirect_stdout(io.StringIO()):
                ablation.run_suite(suite, suite_dir)
                ablation.run_suite(ablation.load_suite(suite_dir), suite_dir)
            self.assertEqual(launches, [("baseline", "asl"), ("densenet_attention", "bce")])
            saved = ablation.load_suite(suite_dir)
            for cell in ("baseline_asl", "attention_bce"):
                self.assertEqual(saved["cells"][cell]["status"], "completed")
                self.assertTrue(Path(saved["cells"][cell]["run_dir"]).is_dir())
            self.assertEqual(saved["cells"]["baseline_bce"]["status"], "not_selected")
            self.assertGreaterEqual(report.write_report.call_count, 2)

    def test_recovery_finds_cell_that_finished_before_parent_wrote_completion(self):
        with workspace_fixture() as directory:
            suite_dir, suite = make_suite(directory, ["baseline_asl"])
            cell = suite["cells"]["baseline_asl"]
            cell["status"] = "running"
            recovered = make_run(suite_dir / "baseline_asl" / "finished", suite["recipe"],
                                 "baseline", "asl")
            with mock.patch.dict(sys.modules, {"ablation_report": report_stub()}), \
                    mock.patch.object(ablation.subprocess, "run") as launch, \
                    redirect_stdout(io.StringIO()):
                ablation.run_suite(suite, suite_dir)
            launch.assert_not_called()
            saved = ablation.load_suite(suite_dir)["cells"]["baseline_asl"]
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["run_dir"], str(recovered))

    def test_failure_is_recorded_and_later_cell_is_not_started(self):
        with workspace_fixture() as directory:
            suite_dir, suite = make_suite(directory, ["baseline_asl", "attention_bce"])
            failure = subprocess.CalledProcessError(2, ["synthetic trainer"])
            with mock.patch.object(ablation.subprocess, "run", side_effect=failure) as launch, \
                    redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                ablation.run_suite(suite, suite_dir)
            launch.assert_called_once()
            saved = ablation.load_suite(suite_dir)
            self.assertEqual(saved["cells"]["baseline_asl"]["status"], "failed")
            self.assertIn("CalledProcessError", saved["cells"]["baseline_asl"]["error"])
            self.assertEqual(saved["cells"]["attention_bce"]["status"], "pending")

    def test_preflight_detects_training_csv_or_weight_changes_between_attempts(self):
        with workspace_fixture() as directory:
            suite_dir, suite = make_suite(directory, ["baseline_asl"])
            ablation.preflight(suite)
            csv_path = Path(suite["recipe"]["data_dir"]) / "train.csv"
            original = csv_path.read_bytes()
            csv_path.write_bytes(original + b"d,1\n")
            with self.assertRaisesRegex(ValueError, "train.csv changed"):
                ablation.preflight(suite)
            csv_path.write_bytes(original)
            Path(suite["recipe"]["pretrained_weights"]).write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "Pretrained weights changed"):
                ablation.preflight(suite)


if __name__ == "__main__":
    unittest.main()
