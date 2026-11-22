"""Train the fold 1 HPA experiments in order and export Kaggle submissions."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
WEIGHTS = ROOT / "load_checkpoint"
MODEL_WEIGHTS = {
    "resnet18": WEIGHTS / "resnet18-f37072fd.pth",
    "baseline": WEIGHTS / "densenet121-a639ec97.pth",
    "se_resnext50": WEIGHTS / "SE" / "pytorch_model.bin",
    "bestfitting_densenet121": WEIGHTS / "bestfitting-densenet121.pth",
}
DEFAULT_THRESHOLDS = ["0.50"]
ASL_THRESHOLDS = ["0.30", "0.40", "0.50", "0.60"]
EPOCH_SUMMARY = re.compile(r"^Epoch \[(\d+)/\d+\] Train Loss:")


def make_plan(args):
    common = {
        "data_dir": str(DATA),
        "fold_manifest": str(DATA / "fold_manifest.json"),
        "fold": 1,
        "epochs": args.epochs,
        "image_size": 512,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": "cuda",
        "seed": 42,
        "optimizer": "adam",
        "scheduler": "none",
        "augmentation": "basic",
        "threshold": 0.50,
    }
    experiments = [
        ("01_resnet18_bce", "resnet18", "bce"),
        ("02_densenet121_bce", "baseline", "bce"),
        ("03_se_resnext50_bce", "se_resnext50", "bce"),
        ("04_bestfitting_bce", "bestfitting_densenet121", "bce"),
        ("05_bestfitting_focal", "bestfitting_densenet121", "focal"),
        ("06_bestfitting_asl", "bestfitting_densenet121", "asl"),
        ("07_bestfitting_asl_rare_aug", "bestfitting_densenet121", "asl"),
    ]
    plan = {}
    for name, model, loss in experiments:
        rare_aug = name == "07_bestfitting_asl_rare_aug"
        train = dict(
            common,
            model_name=model,
            loss=loss,
            learning_rate=1e-5 if model == "bestfitting_densenet121" else 1e-4,
            pretrained_weights=str(MODEL_WEIGHTS[model]),
            rare_augment_max_count=40 if rare_aug else 0,
            rare_augment_extra_copies=2,
        )
        if loss == "focal":
            train["focal_gamma"] = 2.0
        if loss == "asl":
            train.update(asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_clip=0.05)
        thresholds = ASL_THRESHOLDS if name == "06_bestfitting_asl" else DEFAULT_THRESHOLDS
        plan[name] = {"train": train, "thresholds": thresholds}
    return plan


def input_files():
    return (
        DATA / "train.csv",
        DATA / "fold_manifest.json",
        DATA / "sample_submission.csv",
        *MODEL_WEIGHTS.values(),
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_environment(model_weights=MODEL_WEIGHTS):
    sys.path.insert(0, str(ROOT))
    try:
        import torch
        import matplotlib.pyplot
        import timm
        import torchvision
    except ImportError as exc:
        raise RuntimeError(
            "Use the Python environment with PyTorch, torchvision and timm. "
            f"Current interpreter: {sys.executable}. Import failed: {exc}"
        ) from exc

    from dataset import CHANNELS, NUM_CLASSES, read_test_records, read_train_records
    from fold_manifest import load_fold_manifest
    from model import build_model

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in {sys.executable}; use the GPU training environment.")

    train_records = read_train_records(DATA / "train.csv")
    load_fold_manifest(DATA / "fold_manifest.json", train_records)
    test_records = read_test_records(DATA / "sample_submission.csv")
    for split, records in (("train", train_records), ("test", test_records)):
        directory = DATA / split
        with os.scandir(directory) as entries:
            files = {entry.name for entry in entries if entry.is_file()}
        for record in records:
            for channel in CHANNELS:
                name = f"{record['id']}_{channel}.png"
                if name not in files:
                    raise FileNotFoundError(f"Missing image: {directory / name}")

    for model, weights in model_weights.items():
        build_model(model, NUM_CLASSES, pretrained_weights=str(weights))


def train_command(config, output_base):
    command = [sys.executable, "-u", str(ROOT / "train.py"), "--output-dir", str(output_base)]
    for key, value in config.items():
        option = "model" if key == "model_name" else key.replace("_", "-")
        command.extend((f"--{option}", str(value)))
    return command


def inference_command(config, thresholds, checkpoint, submission_dir):
    return [
        sys.executable, "-u", str(ROOT / "inference.py"),
        "--checkpoint", str(checkpoint), "--submission-dir", str(submission_dir),
        "--data-dir", config["data_dir"], "--device", config["device"],
        "--batch-size", str(config["batch_size"]), "--num-workers", str(config["num_workers"]),
        "--thresholds", *thresholds,
    ]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_checkpoint(output_base, config, fold_hash):
    fold_dir = output_base / "cross_validation" / f"fold_{config['fold']}"
    for path in sorted(fold_dir.glob("*/metrics_summary.json"), reverse=True):
        checkpoint = path.parent / "final_model.pth"
        try:
            if checkpoint.stat().st_size == 0:
                continue
            metrics = read_json(path)
            actual = read_json(path.parent / "run_config.json")
        except (ValueError, OSError):
            continue
        if (
            not actual.get("smoke_test") and not actual.get("debug")
            and all(actual.get(key) == value for key, value in config.items())
            and metrics.get("fold") == config["fold"]
            and metrics.get("epoch") == config["epochs"]
            and metrics.get("fold_manifest_sha256") == fold_hash
        ):
            return checkpoint
    return None


@contextmanager
def suite_lock(suite_dir):
    suite_dir.mkdir(parents=True, exist_ok=True)
    with (suite_dir / ".running").open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another runner is using {suite_dir}.") from exc
        yield


def log_command(command, log_path, *, epochs=None):
    from tqdm import tqdm

    print(f"  Log: {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with log_path.open("a", encoding="utf-8", newline="") as log:
        log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {subprocess.list2cmdline(command)}\n")
        log.flush()
        with subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        ) as process, tqdm(
            total=epochs, desc="Training", unit="epoch", mininterval=0,
            disable=epochs is None,
        ) as progress:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if epochs is not None and (summary := EPOCH_SUMMARY.match(line)):
                        progress.update(int(summary[1]) - progress.n)
                returncode = process.wait()
            except BaseException:
                process.kill()
                raise
    if returncode:
        raise RuntimeError(f"Command exited with code {returncode}. See {log_path}")


def run_suite(suite_dir, plan, input_hashes):
    recipe = {"experiments": plan, "input_sha256": input_hashes}
    recipe_path = suite_dir / "suite.json"
    if recipe_path.is_file():
        if read_json(recipe_path) != recipe:
            raise ValueError("Settings or input files changed. Use a new --suite-dir for a different experiment.")
    else:
        temporary = recipe_path.with_suffix(".tmp")
        if any(path.name not in (".running", temporary.name) for path in suite_dir.iterdir()):
            raise ValueError(f"Choose an empty suite directory: {suite_dir}")
        temporary.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")
        temporary.replace(recipe_path)

    errors = []
    for index, (name, experiment) in enumerate(plan.items(), start=1):
        print(f"\n[{datetime.now():%H:%M:%S}] [{index}/{len(plan)}] {name}", flush=True)
        config, thresholds = experiment["train"], experiment["thresholds"]
        fold_hash = input_hashes[config["fold_manifest"]]
        output_base = suite_dir / name
        log_dir = suite_dir / "logs"
        submission_dir = suite_dir / "submissions" / name
        files = [submission_dir / f"submission_t{threshold}.csv" for threshold in thresholds]
        try:
            checkpoint = find_checkpoint(output_base, config, fold_hash)
            if checkpoint is None:
                log_command(
                    train_command(config, output_base), log_dir / f"{name}_train.log",
                    epochs=config["epochs"],
                )
                checkpoint = find_checkpoint(output_base, config, fold_hash)
                if checkpoint is None:
                    raise RuntimeError(f"Trainer did not produce a complete fold {config['fold']} checkpoint in {output_base}")
            else:
                print(f"  Reusing completed training: {checkpoint}", flush=True)

            checkpoint_time = checkpoint.stat().st_mtime_ns
            if not all(
                file.is_file() and file.stat().st_size > 0 and file.stat().st_mtime_ns >= checkpoint_time
                for file in files
            ):
                print(f"  Exporting {len(files)} submission CSV(s)", flush=True)
                log_command(
                    inference_command(config, thresholds, checkpoint, submission_dir),
                    log_dir / f"{name}_submission.log",
                )
                if not all(file.is_file() and file.stat().st_size > 0 for file in files):
                    raise RuntimeError(f"Missing expected submission files in {submission_dir}")
            print(f"  Ready to upload: {submission_dir}", flush=True)
        except (RuntimeError, OSError) as exc:
            print(f"  Failed: {exc}", file=sys.stderr, flush=True)
            errors.append(name)

    print(f"\nSubmissions: {suite_dir / 'submissions'}", flush=True)
    if errors:
        print(f"Failed experiments: {', '.join(errors)}. Rerun the same command to retry.", file=sys.stderr)
        return 1
    total_submissions = sum(len(experiment["thresholds"]) for experiment in plan.values())
    print(f"All {len(plan)} training runs and {total_submissions} submission files are ready.")
    return 0


def print_dry_run(suite_dir, plan):
    for name, experiment in plan.items():
        config = experiment["train"]
        output_base = suite_dir / name
        fold_dir = output_base / "cross_validation" / f"fold_{config['fold']}"
        checkpoint = fold_dir / "RUN_TIMESTAMP" / "final_model.pth"
        print(subprocess.list2cmdline(train_command(config, output_base)))
        print(subprocess.list2cmdline(inference_command(
            config, experiment["thresholds"], checkpoint, suite_dir / "submissions" / name,
        )))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20, help="Full epochs per experiment (default: 20).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--suite-dir", type=Path, default=Path("outputs/fold1_kaggle"))
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without writing or training.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch size must be positive, and workers must be nonnegative")
    suite_dir = (ROOT / args.suite_dir).resolve()
    plan = make_plan(args)
    print(f"Fold 1: {len(plan)} training runs x {args.epochs} epochs\nSuite: {suite_dir}", flush=True)
    if args.dry_run:
        print_dry_run(suite_dir, plan)
        return 0

    try:
        input_hashes = {str(path): sha256(path) for path in input_files()}
        print("Checking the GPU environment, image files, and local weights...", flush=True)
        check_environment()
        with suite_lock(suite_dir):
            return run_suite(suite_dir, plan, input_hashes)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to continue from completed runs.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
