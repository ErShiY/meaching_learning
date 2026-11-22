"""Export the additional Kaggle submissions from the two trained fold 1 ASL models."""

import argparse
import subprocess
import sys

from run_fold1_experiments import ROOT, find_checkpoint, inference_command, read_json


SUITE_DIR = ROOT / "outputs" / "fold1_kaggle"
THRESHOLDS = {
    "06_bestfitting_asl": ("0.70", "0.80"),
    "07_bestfitting_asl_rare_aug": ("0.60", "0.70", "0.80"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check checkpoints and print commands.")
    args = parser.parse_args()

    try:
        suite = read_json(SUITE_DIR / "suite.json")
        commands = []
        for name, thresholds in THRESHOLDS.items():
            config = suite["experiments"][name]["train"]
            fold_hash = suite["input_sha256"][config["fold_manifest"]]
            checkpoint = find_checkpoint(SUITE_DIR / name, config, fold_hash)
            if checkpoint is None:
                raise FileNotFoundError(f"No completed fold 1 checkpoint found for {name}.")
            submission_dir = SUITE_DIR / "submissions" / name
            commands.append(inference_command(config, thresholds, checkpoint, submission_dir))

        for command in commands:
            if args.dry_run:
                print(subprocess.list2cmdline(command), flush=True)
            else:
                subprocess.run(command, cwd=ROOT, check=True)
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1

    if not args.dry_run:
        print(f"Five submission files exported to {SUITE_DIR / 'submissions'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
