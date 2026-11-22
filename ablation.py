"""Run a controlled DenseNet head/loss ablation using the existing trainer."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "outputs/densenet_attention/20261117_110115"
CELLS = {
    "baseline_bce": ("baseline", "bce"),
    "baseline_asl": ("baseline", "asl"),
    "attention_bce": ("densenet_attention", "bce"),
    "attention_asl": ("densenet_attention", "asl"),
}
DEFAULT_EXPERIMENTS = tuple(name for name in CELLS if name != "attention_asl")
RECIPE_KEYS = (
    "image_size", "batch_size", "epochs", "learning_rate", "asl_gamma_pos",
    "asl_gamma_neg", "asl_clip", "optimizer", "weight_decay", "scheduler",
    "warmup_epochs", "min_lr", "split_method", "augmentation", "num_workers",
    "threshold", "seed", "device", "data_dir", "pretrained_weights",
)
RARE_AUGMENT_DEFAULTS = {
    "rare_augment_max_count": 0,
    "rare_augment_extra_copies": 2,
}
# Device changes are recorded separately; all training hyperparameters stay fixed.
CHECK_KEYS = tuple(key for key in RECIPE_KEYS if key != "device")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def project_path(value):
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rare_augmentation_settings(config):
    max_count, extra_copies = (
        config.get(key, default) for key, default in RARE_AUGMENT_DEFAULTS.items()
    )
    if max_count > 0 and extra_copies > 0:
        return max_count, extra_copies
    return None


def validate_rare_augmentation_recipe(config, recipe, run_dir):
    actual = _rare_augmentation_settings(config)
    expected = _rare_augmentation_settings(recipe)
    if actual != expected:
        raise ValueError(
            f"{run_dir}: shared recipe mismatch for rare augmentation "
            f"(max count, extra copies): {actual!r} != {expected!r}."
        )


def validate_completed_run(run_dir, model_name, loss, recipe=None, split_file=None):
    """Never count a smoke test or a partially written run as an experiment."""
    run_dir = Path(run_dir)
    for name in (
        "run_config.json", "training_log.csv", "metrics_summary.json",
        "best_model.pth", "val_predictions.npz", "split_manifest.json",
    ):
        if not (run_dir / name).is_file():
            raise ValueError(f"Incomplete run: missing {run_dir / name}")
    config = read_json(run_dir / "run_config.json")
    if config.get("model_name") != model_name or config.get("loss") != loss:
        raise ValueError(f"Wrong model/loss in {run_dir}")
    if config.get("debug") or config.get("smoke_test"):
        raise ValueError(f"Debug/smoke run cannot be used for ablation: {run_dir}")
    if recipe is not None:
        for key in CHECK_KEYS:
            actual, expected = config.get(key), recipe[key]
            if key in ("data_dir", "pretrained_weights"):
                actual = project_path(actual) if actual else actual
                expected = project_path(expected) if expected else expected
            if actual != expected:
                raise ValueError(f"Recipe mismatch for {key} in {run_dir}: {actual} != {expected}")
        validate_rare_augmentation_recipe(config, recipe, run_dir)
    with (run_dir / "training_log.csv").open(newline="", encoding="utf-8-sig") as file:
        history = list(csv.DictReader(file))
    if [int(row["epoch"]) for row in history] != list(range(1, config["epochs"] + 1)):
        raise ValueError(f"Run has not finished all {config['epochs']} epochs: {run_dir}")
    metrics = read_json(run_dir / "metrics_summary.json")
    for key in ("model_name", "loss", "split_sha256"):
        if metrics.get(key) != config.get(key):
            raise ValueError(f"Metrics/config mismatch for {key} in {run_dir}")
    # The log rounds F1 to eight decimals; is_best preserves the trainer's
    # decisions even when two distinct scores round to the same value.
    best_rows = [row for row in history if int(row["is_best"]) == 1]
    if not best_rows:
        raise ValueError(f"No selected checkpoint in training log: {run_dir}")
    best = best_rows[-1]
    if int(best["epoch"]) != metrics["best_epoch"]:
        raise ValueError(f"Checkpoint/log selection mismatch in {run_dir}")
    if abs(float(best["val_macro_f1"]) - metrics["best_val_macro_f1"]) > 1e-7:
        raise ValueError(f"Checkpoint/log F1 mismatch in {run_dir}")
    own_split = run_dir / "split_manifest.json"
    if config.get("split_sha256") != file_sha256(own_split):
        raise ValueError(f"Split manifest was changed after training: {run_dir}")
    if split_file is not None:
        if read_json(own_split)["splits"] != read_json(split_file)["splits"]:
            raise ValueError(f"Different train/validation/holdout IDs or order: {run_dir}")
    return config


def create_plan(reference_run, experiments, device=None):
    reference_run = project_path(reference_run)
    config = validate_completed_run(reference_run, "densenet_attention", "asl")
    missing = set(RECIPE_KEYS) - set(config)
    if missing:
        raise ValueError(f"Reference configuration lacks: {sorted(missing)}")
    recipe = {key: config[key] for key in RECIPE_KEYS}
    recipe.update({key: config.get(key, default) for key, default in RARE_AUGMENT_DEFAULTS.items()})
    for key in ("data_dir", "pretrained_weights"):
        if not recipe[key]:
            raise ValueError(f"Reference must specify {key} explicitly.")
        recipe[key] = str(project_path(recipe[key]))
    if recipe["threshold"] != 0.5:
        raise ValueError("This ablation keeps checkpoint selection at threshold 0.5.")
    if device is not None:
        recipe["device"] = device
    cells = {}
    for name, (model_name, loss) in CELLS.items():
        reused = name == "attention_asl" and name not in experiments
        cells[name] = {
            "model_name": model_name, "loss": loss,
            "status": "reused" if reused else "pending" if name in experiments else "not_selected",
            "run_dir": str(reference_run) if reused else None,
        }
    return {
        "schema_version": 1,
        "reference_run": str(reference_run),
        "recipe": recipe,
        "split_file": str(reference_run / "split_manifest.json"),
        "experiments": list(experiments),
        "checkpoint_selection": "highest validation Macro-F1 at threshold 0.5; earliest tie",
        "cells": cells,
    }


def build_command(suite, name, suite_dir):
    recipe = suite["recipe"]
    cell = suite["cells"][name]
    command = [sys.executable, "-u", str(ROOT / "train.py")]
    for key in RECIPE_KEYS:
        command.extend(["--" + key.replace("_", "-"), str(recipe[key])])
    for key, default in RARE_AUGMENT_DEFAULTS.items():
        command.extend(["--" + key.replace("_", "-"), str(recipe.get(key, default))])
    command.extend([
        "--model", cell["model_name"], "--loss", cell["loss"],
        "--split-file", str(suite["split_file"]),
        "--output-dir", str(Path(suite_dir) / name), "--skip-submission",
    ])
    return command


def preflight(suite):
    data_dir = Path(suite["recipe"]["data_dir"])
    # These are also required by train.py when --skip-submission is set.
    for path in (
        data_dir / "train.csv", data_dir / "sample_submission.csv",
        data_dir / "train", data_dir / "test",
        Path(suite["recipe"]["pretrained_weights"]), Path(suite["split_file"]),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    current_hash = file_sha256(data_dir / "train.csv")
    if suite.get("train_csv_sha256", current_hash) != current_hash:
        raise ValueError("train.csv changed since this ablation suite was created.")
    suite["train_csv_sha256"] = current_hash
    weights_hash = file_sha256(suite["recipe"]["pretrained_weights"])
    if suite.get("pretrained_sha256", weights_hash) != weights_hash:
        raise ValueError("Pretrained weights changed since this ablation suite was created.")
    suite["pretrained_sha256"] = weights_hash


def load_suite(suite_dir):
    suite = read_json(Path(suite_dir) / "suite.json")
    if suite.get("schema_version") != 1:
        raise ValueError("Unsupported ablation suite schema.")
    if set(suite["cells"]) != set(CELLS):
        raise ValueError("Unexpected ablation cells.")
    if file_sha256(suite["split_file"]) != suite["split_sha256"]:
        raise ValueError("The frozen suite split was modified.")
    return suite


def print_plan(suite, suite_dir):
    print(f"Ablation suite: {suite_dir}")
    print(f"Reference: {suite['reference_run']}")
    print("Shared recipe:", json.dumps(suite["recipe"], ensure_ascii=False))
    for name, cell in suite["cells"].items():
        print(f"\n[{name}] {cell['status']}")
        if cell["status"] in ("reused", "completed"):
            print(f"  {cell['run_dir']}")
        elif name in suite["experiments"]:
            print("  " + subprocess.list2cmdline(build_command(suite, name, suite_dir)))


def run_suite(suite, suite_dir):
    preflight(suite)
    manifest_path = Path(suite_dir) / "suite.json"
    for name, cell in suite["cells"].items():
        if cell["status"] in ("reused", "completed"):
            validate_completed_run(
                cell["run_dir"], cell["model_name"], cell["loss"],
                suite["recipe"], suite["split_file"],
            )
    for name in suite["experiments"]:
        cell = suite["cells"][name]
        if cell["status"] in ("completed", "reused"):
            continue
        output_base = Path(suite_dir) / name
        # Recover a cell that finished just before the parent was interrupted.
        recovered = []
        if output_base.exists():
            for candidate in sorted(output_base.iterdir()):
                if not candidate.is_dir():
                    continue
                try:
                    validate_completed_run(candidate, cell["model_name"], cell["loss"],
                                           suite["recipe"], suite["split_file"])
                except (ValueError, KeyError, OSError):
                    continue
                recovered.append(candidate)
        if recovered:
            cell.update(status="completed", run_dir=str(recovered[-1].resolve()))
            write_json(manifest_path, suite)
            continue
        before = set(output_base.iterdir()) if output_base.exists() else set()
        cell.update(status="running", error=None)
        command = build_command(suite, name, suite_dir)
        cell["command"] = command
        cell["started_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        write_json(manifest_path, suite)
        print(f"\nStarting {name} ({suite['recipe']['epochs']} epochs)...", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            created = [path for path in output_base.iterdir() if path not in before and path.is_dir()]
            if len(created) != 1:
                raise ValueError(f"Expected one new training run for {name}, found {len(created)}")
            run_dir = created[0].resolve()
            validate_completed_run(run_dir, cell["model_name"], cell["loss"],
                                   suite["recipe"], suite["split_file"])
            cell.update(status="completed", run_dir=str(run_dir))
        except (Exception, KeyboardInterrupt) as error:
            cell.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(manifest_path, suite)
            raise
        cell["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        write_json(manifest_path, suite)
        from ablation_report import write_report
        write_report(suite_dir)
    from ablation_report import write_report
    write_report(suite_dir)
    print(f"\nFinished. Comparison: {Path(suite_dir) / 'summary.md'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run", help="Run missing cells sequentially; reuse the completed reference.")
    run.add_argument("--reference-run", default=None, help=f"Default: {DEFAULT_REFERENCE}")
    run.add_argument("--suite-dir", help="New suite directory, or an existing suite to continue.")
    run.add_argument("--experiments", nargs="+", choices=tuple(CELLS), default=None,
                     help="Default: baseline_bce baseline_asl attention_bce. Include attention_asl to retrain it.")
    run.add_argument("--device", default=None, help="Optional device override, recorded in suite recipe.")
    run.add_argument("--dry-run", action="store_true", help="Print the plan; do not create files or train.")
    report = commands.add_parser("report", help="Compare completed cells and calibrate thresholds on validation only.")
    report.add_argument("--suite-dir", required=True)
    report.add_argument("--evaluate-holdout", action="store_true",
                        help="Run/cache holdout predictions; apply validation-selected thresholds.")
    report.add_argument("--device", default="auto")
    report.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.action == "report":
        suite_dir = project_path(args.suite_dir)
        load_suite(suite_dir)
        from ablation_report import write_report
        write_report(suite_dir, evaluate_holdout=args.evaluate_holdout,
                     device=args.device, num_workers=args.num_workers)
        return
    suite_dir = project_path(args.suite_dir) if args.suite_dir else (
        ROOT / "outputs/ablation" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    if (suite_dir / "suite.json").exists():
        if args.reference_run is not None or args.experiments is not None or args.device is not None:
            raise ValueError("When continuing a suite, use only --suite-dir (and optionally --dry-run).")
        suite = load_suite(suite_dir)
    else:
        if suite_dir.exists() and any(suite_dir.iterdir()):
            raise ValueError(f"New suite directory must be empty: {suite_dir}")
        experiments = args.experiments if args.experiments is not None else DEFAULT_EXPERIMENTS
        if len(experiments) != len(set(experiments)):
            raise ValueError("Experiments must not be repeated.")
        suite = create_plan(args.reference_run or DEFAULT_REFERENCE, experiments, args.device)
    preflight(suite)
    print_plan(suite, suite_dir)
    if args.dry_run:
        return
    if not (suite_dir / "suite.json").exists():
        suite_dir.mkdir(parents=True, exist_ok=True)
        frozen_split = suite_dir / "split_manifest.json"
        frozen_split.write_bytes(Path(suite["split_file"]).read_bytes())
        suite["split_file"] = str(frozen_split.resolve())
        suite["split_sha256"] = file_sha256(frozen_split)
        suite["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        write_json(suite_dir / "suite.json", suite)
    run_suite(suite, suite_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Continue with: python ablation.py run --suite-dir <printed suite directory>", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Ablation stopped: {error}", file=sys.stderr)
        sys.exit(1)
