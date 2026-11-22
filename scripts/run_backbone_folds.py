"""Finish the DenseNet121 and SE-ResNeXt50 comparison on folds 2 through 5."""

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
import sys

from run_fold1_experiments import (
    ROOT,
    check_environment,
    find_checkpoint,
    print_dry_run,
    read_json,
    run_suite,
    sha256,
    suite_lock,
)


FOLD1_DIR = ROOT / "outputs" / "fold1_kaggle"
FOLD1_EXPERIMENTS = {
    "densenet121": "02_densenet121_bce",
    "se_resnext50": "03_se_resnext50_bce",
}
REMAINING_FOLDS = (2, 3, 4, 5)


def load_reference():
    source = read_json(FOLD1_DIR / "suite.json")
    configurations = {}
    checkpoints = {}
    input_hashes = {}
    for model, name in FOLD1_EXPERIMENTS.items():
        experiment = source["experiments"][name]
        config = experiment["train"]
        if config["fold"] != 1 or experiment["thresholds"] != ["0.50"]:
            raise ValueError(f"Unexpected fold or submission threshold in fold 1 experiment {name}.")
        fold_hash = source["input_sha256"][config["fold_manifest"]]
        checkpoint = find_checkpoint(FOLD1_DIR / name, config, fold_hash)
        if checkpoint is None:
            raise FileNotFoundError(f"No completed fold 1 checkpoint for {name}.")
        configurations[model] = config
        checkpoints[model] = checkpoint

        data_dir = Path(config["data_dir"])
        for path in (
            data_dir / "train.csv", data_dir / "sample_submission.csv",
            Path(config["fold_manifest"]), Path(config["pretrained_weights"]),
        ):
            key = str(path)
            if key not in input_hashes:
                current_hash = sha256(path)
                if current_hash != source["input_sha256"][key]:
                    raise ValueError(f"Input changed since fold 1: {path}")
                input_hashes[key] = current_hash
    return configurations, checkpoints, input_hashes


def make_plan(configurations):
    return {
        f"fold_{fold}_{model}_bce": {
            "train": dict(config, fold=fold),
            "thresholds": ["0.50"],
        }
        for fold in REMAINING_FOLDS
        for model, config in configurations.items()
    }


def write_summary(suite_dir, configurations, fold1_checkpoints, input_hashes):
    columns = (
        "fold",
        "densenet121_macro_f1", "se_resnext50_macro_f1", "se_minus_dense_macro_f1",
        "densenet121_micro_f1", "se_resnext50_micro_f1",
    )
    rows = []
    for fold in (1, *REMAINING_FOLDS):
        row = {"fold": fold}
        for model, config in configurations.items():
            if fold == 1:
                checkpoint = fold1_checkpoints[model]
            else:
                checkpoint = find_checkpoint(
                    suite_dir / f"fold_{fold}_{model}_bce", dict(config, fold=fold),
                    input_hashes[config["fold_manifest"]],
                )
            if checkpoint is not None:
                metrics = read_json(checkpoint.parent / "metrics_summary.json")
                row[f"{model}_macro_f1"] = metrics["holdout_macro_f1"]
                row[f"{model}_micro_f1"] = metrics["holdout_micro_f1"]
        if "densenet121_macro_f1" in row and "se_resnext50_macro_f1" in row:
            row["se_minus_dense_macro_f1"] = row["se_resnext50_macro_f1"] - row["densenet121_macro_f1"]
        rows.append(row)

    if all(all(column in row for column in columns) for row in rows):
        aggregates = []
        for label, statistic in (("mean", mean), ("sample_std", stdev)):
            aggregate = {column: statistic(row[column] for row in rows) for column in columns[1:]}
            aggregates.append(dict(aggregate, fold=label))
        rows.extend(aggregates)

    summary_path = suite_dir / "five_fold_local_summary.csv"
    temporary_path = summary_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value for key, value in row.items()})
    temporary_path.replace(summary_path)
    print(f"Five-fold local results: {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=Path("outputs/backbone_folds_2_to_5"))
    parser.add_argument("--dry-run", action="store_true", help="Verify the original runs and print all commands.")
    args = parser.parse_args()

    suite_dir = (ROOT / args.suite_dir).resolve()
    try:
        configurations, fold1_checkpoints, input_hashes = load_reference()
        plan = make_plan(configurations)
        print(f"Folds 2-5: {len(plan)} training runs using the fold 1 configurations\nSuite: {suite_dir}", flush=True)
        if args.dry_run:
            print_dry_run(suite_dir, plan)
            return 0

        print("Checking the GPU environment, image files, and local weights...", flush=True)
        check_environment({config["model_name"]: config["pretrained_weights"] for config in configurations.values()})
        with suite_lock(suite_dir):
            result = run_suite(suite_dir, plan, input_hashes)
            write_summary(suite_dir, configurations, fold1_checkpoints, input_hashes)
        return result
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to continue from completed runs.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
