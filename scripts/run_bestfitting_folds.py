"""Train the final Bestfitting ASL model on folds 2–5 and export Kaggle CSVs."""

import argparse
from pathlib import Path
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
FOLD1_EXPERIMENT = "06_bestfitting_asl"
REMAINING_FOLDS = (2, 3, 4, 5)
SUBMISSION_THRESHOLD = "0.70"


def load_reference():
    suite = read_json(FOLD1_DIR / "suite.json")
    config = suite["experiments"][FOLD1_EXPERIMENT]["train"]
    if (
        config["fold"] != 1
        or config["model_name"] != "bestfitting_densenet121"
        or config["loss"] != "asl"
        or config["rare_augment_max_count"] != 0
    ):
        raise ValueError(f"Unexpected training settings for {FOLD1_EXPERIMENT}.")

    data_dir = Path(config["data_dir"])
    input_hashes = {}
    for path in (
        data_dir / "train.csv",
        data_dir / "sample_submission.csv",
        Path(config["fold_manifest"]),
        Path(config["pretrained_weights"]),
    ):
        digest = sha256(path)
        if digest != suite["input_sha256"][str(path)]:
            raise ValueError(f"Input changed since fold 1: {path}")
        input_hashes[str(path)] = digest

    fold_hash = input_hashes[config["fold_manifest"]]
    if find_checkpoint(FOLD1_DIR / FOLD1_EXPERIMENT, config, fold_hash) is None:
        raise FileNotFoundError(f"No completed fold 1 checkpoint for {FOLD1_EXPERIMENT}.")
    return config, input_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-dir", type=Path, default=Path("outputs/bestfitting_asl_folds_2_to_5"),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Verify fold 1 and print commands without training.",
    )
    args = parser.parse_args()
    suite_dir = (ROOT / args.suite_dir).resolve()

    try:
        config, input_hashes = load_reference()
        plan = {
            f"fold_{fold}_bestfitting_asl": {
                "train": dict(config, fold=fold),
                "thresholds": [SUBMISSION_THRESHOLD],
            }
            for fold in REMAINING_FOLDS
        }
        print(
            f"Folds 2-5: {len(plan)} training runs x {config['epochs']} epochs"
            f"\nKaggle threshold: {SUBMISSION_THRESHOLD}\nSuite: {suite_dir}",
            flush=True,
        )
        if args.dry_run:
            print_dry_run(suite_dir, plan)
            return 0

        print("Checking the GPU environment, image files, and local weights...", flush=True)
        check_environment({config["model_name"]: config["pretrained_weights"]})
        with suite_lock(suite_dir):
            return run_suite(suite_dir, plan, input_hashes)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to continue from completed folds.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
