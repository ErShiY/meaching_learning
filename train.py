import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn as nn

from dataset import (
    CHANNELS,
    HPAImageDataset,
    expand_rare_training_indices,
    iterative_multilabel_split,
    load_split_manifest,
    NUM_CLASSES,
    make_loader,
    read_test_records,
    read_train_records,
    sample_records,
    save_label_distribution,
    save_split_manifest,
    split_indices,
)
from engine import (
    resolve_device,
    run_epoch,
    run_smoke_test,
    save_prediction_artifacts,
    save_training_log,
    save_validation_artifacts,
    set_seed,
)
from fold_manifest import (
    FOLD_COUNT,
    FOLD_MANIFEST_FILENAME,
    fold_indices,
    load_fold_manifest,
)
from inference import run_inference
from losses import AsymmetricLoss, SigmoidFocalLoss
from training_utils import WarmupCosineScheduler, build_optimizer
from model import (
    DEFAULT_MODEL_NAME,
    available_models,
    build_model,
    default_pretrained_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a model on the Human Protein Atlas dataset."
    )

    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument(
        "--model",
        dest="model_name",
        type=str,
        choices=available_models(),
        default=DEFAULT_MODEL_NAME,
        help="Model architecture to train.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("bce", "focal", "asl"), default="bce")
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss focusing exponent.",
    )
    parser.add_argument(
        "--focal-alpha", type=float, default=None,
        help="Optional focal loss weights: alpha for positives, 1-alpha for negatives.",
    )
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="Defaults to 0 for Adam and 1e-4 for AdamW.",
    )
    parser.add_argument(
        "--scheduler", choices=("none", "warmup_cosine"), default="none"
    )
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--split-method", choices=("random", "stratified"), default=None
    )
    parser.add_argument(
        "--split-file", type=str, default=None,
        help="Reuse an existing split_manifest.json by image ID.",
    )
    parser.add_argument(
        "--fold",
        choices=tuple(str(number) for number in range(1, FOLD_COUNT + 1)) + ("all",),
        help="Evaluate on the selected fixed fold after the last epoch. "
        "'all' runs and summarizes five independent trainings.",
    )
    parser.add_argument(
        "--fold-manifest", type=str,
        help=f"Fixed five-fold manifest (default: <data-dir>/{FOLD_MANIFEST_FILENAME}).",
    )
    parser.add_argument(
        "--augmentation", choices=("basic", "microscopy"), default="basic"
    )
    parser.add_argument(
        "--rare-augment-max-count", type=int, default=0,
        help="Add augmented training draws for images with a label at or below "
        "this many training positives. 0 disables this.",
    )
    parser.add_argument(
        "--rare-augment-extra-copies", type=int, default=2,
        help="Additional draws per eligible training image per epoch.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default=None,
        help=(
            "Local pretrained weights. By default, a model-specific file in "
            "load_checkpoint/ is used. No weights are downloaded automatically."
        ),
    )
    parser.add_argument(
        "--skip-submission",
        action="store_true",
        help="Skip prediction on the Kaggle test images.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only one train batch and one validation batch.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Train on a small labeled subset and skip Kaggle submission.",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=1000,
        help="Number of labeled images used when --debug is enabled.",
    )

    args = parser.parse_args()
    if args.fold and (args.split_file or args.split_method or args.debug):
        parser.error(
            "--fold selects a fixed split and cannot be combined with "
            "--split-file, --split-method or --debug. Use --smoke-test for fold checks."
        )
    if args.fold_manifest and not args.fold:
        parser.error("--fold-manifest requires --fold")
    if args.split_method is None:
        args.split_method = "random"
    if (
        args.epochs < 1 or args.batch_size < 1
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("epochs, batch-size and learning-rate must be positive")
    if args.image_size < 1 or args.num_workers < 0:
        parser.error("image-size must be positive and num-workers nonnegative")
    if args.rare_augment_max_count < 0 or args.rare_augment_extra_copies < 0:
        parser.error("rare augmentation counts must be nonnegative")
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    if args.weight_decay is not None and (
        not math.isfinite(args.weight_decay) or args.weight_decay < 0
    ):
        parser.error("weight-decay must be finite and nonnegative")
    if not math.isfinite(args.focal_gamma) or args.focal_gamma < 0:
        parser.error("focal-gamma must be finite and nonnegative")
    if args.focal_alpha is not None and not 0 <= args.focal_alpha <= 1:
        parser.error("focal-alpha must be in [0, 1]")
    if (
        not math.isfinite(args.asl_gamma_pos) or args.asl_gamma_pos < 0
        or not math.isfinite(args.asl_gamma_neg) or args.asl_gamma_neg < 0
        or not 0 <= args.asl_clip < 1
    ):
        parser.error("ASL gammas must be finite/nonnegative and clip in [0, 1)")
    if args.scheduler == "warmup_cosine":
        if not 0 <= args.warmup_epochs < args.epochs:
            parser.error("warmup-epochs must be in [0, epochs)")
        if not 0 <= args.min_lr <= args.learning_rate:
            parser.error("min-lr must be between 0 and learning-rate")
    return args


def create_run_output_dir(base_output_dir):
    base_output_dir = Path(base_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 0
    while True:
        run_name = f"{timestamp}_{suffix:02d}" if suffix else timestamp
        output_dir = base_output_dir / run_name
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            return output_dir
        except FileExistsError:
            suffix += 1


def main():
    args = parse_args()
    if args.pretrained_weights is None:
        args.pretrained_weights = default_pretrained_weights(args.model_name)
    if args.weight_decay is None:
        args.weight_decay = 1e-4 if args.optimizer == "adamw" else 0.0
    if args.output_dir == "outputs/baseline":
        if args.debug:
            if args.model_name == DEFAULT_MODEL_NAME:
                args.output_dir = "outputs/debug"
            else:
                args.output_dir = f"outputs/debug/{args.model_name}"
            print(f"Debug output directory changed to: {args.output_dir}")
        elif args.model_name != DEFAULT_MODEL_NAME:
            args.output_dir = f"outputs/{args.model_name}"
            print(f"Model output directory changed to: {args.output_dir}")

    data_dir = Path(args.data_dir)
    train_records = read_train_records(data_dir / "train.csv")
    if args.debug:
        train_records = sample_records(train_records, args.debug_samples, args.seed)
        print(f"Debug mode: using {len(train_records)} labeled images.")
    if args.fold or args.debug:
        args.skip_submission = True
    if not args.fold:
        fit(args, train_records)
        return

    manifest_path = (
        Path(args.fold_manifest) if args.fold_manifest else data_dir / FOLD_MANIFEST_FILENAME
    )
    manifest = load_fold_manifest(manifest_path, train_records)
    args.fold_manifest = str(manifest_path.resolve())
    args.fold_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    args.split_method = manifest["method"]
    print(f"Loaded fixed folds from: {args.fold_manifest}")

    if args.fold == "all":
        suite_dir = create_run_output_dir(Path(args.output_dir) / "cross_validation")
        fold_numbers = range(1, manifest["fold_count"] + 1)
        fold_results = []
        for number in fold_numbers:
            fold_args = argparse.Namespace(**vars(args))
            fold_args.fold = number
            output_dir = suite_dir / f"fold_{number}"
            result = fit(fold_args, train_records, manifest, output_dir)
            if result is not None:
                fold_results.append({"run_directory": output_dir.name, **result})
        if not args.smoke_test:
            save_cv_summary(suite_dir, args, fold_results)
    else:
        args.fold = int(args.fold)
        fit(args, train_records, manifest)


def save_cv_summary(output_dir, args, results):
    summary = {
        "model_name": args.model_name,
        "loss": args.loss,
        "epochs": args.epochs,
        "threshold": args.threshold,
        "fold_manifest_sha256": args.fold_manifest_sha256,
        "folds": results,
    }
    for metric in ("holdout_loss", "holdout_macro_f1", "holdout_micro_f1"):
        scores = [result[metric] for result in results]
        summary[f"{metric}_mean"] = mean(scores)
        summary[f"{metric}_std"] = stdev(scores)
    path = output_dir / "cv_summary.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(
        f"Cross-validation Macro-F1: {summary['holdout_macro_f1_mean']:.4f} "
        f"+/- {summary['holdout_macro_f1_std']:.4f} (sample standard deviation)"
    )
    print(f"Cross-validation results saved to: {path}")


def fit(args, train_records, fold_manifest=None, output_dir=None):
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    sample_submission_path = data_dir / "sample_submission.csv"
    train_image_dir = data_dir / "train"
    kaggle_test_image_dir = data_dir / "test"
    if not train_image_dir.is_dir():
        raise FileNotFoundError(f"Required dataset path not found: {train_image_dir}")
    kaggle_test_records = None
    if not args.skip_submission and not args.smoke_test:
        if not kaggle_test_image_dir.is_dir():
            raise FileNotFoundError(f"Required dataset path not found: {kaggle_test_image_dir}")
        kaggle_test_records = read_test_records(sample_submission_path)

    if output_dir is None:
        base_dir = Path(args.output_dir)
        if args.fold:
            base_dir = base_dir / "cross_validation" / f"fold_{args.fold}"
        output_dir = create_run_output_dir(base_dir)
    else:
        output_dir.mkdir()
    print(f"Run output directory: {output_dir.resolve()}")

    best_model_path = output_dir / "best_model.pth"
    training_log_path = output_dir / "training_log.csv"
    label_distribution_path = output_dir / "label_distribution.csv"
    submission_path = output_dir / "submission.csv"

    split_seed = args.seed
    if fold_manifest is not None:
        train_indices, test_indices = fold_indices(fold_manifest, train_records, args.fold)
        val_indices = []
        split_seed = fold_manifest["seed"]
        print(f"Fold {args.fold}: evaluating after epoch {args.epochs} at threshold {args.threshold}")
    elif args.split_file:
        train_indices, val_indices, test_indices = load_split_manifest(
            args.split_file, train_records
        )
        with Path(args.split_file).open("r", encoding="utf-8") as file:
            split_metadata = json.load(file)
        args.split_method = split_metadata.get("method", "loaded")
        split_seed = split_metadata.get("seed", args.seed)
        print(f"Loaded fixed split from: {args.split_file}")
    elif args.split_method == "stratified":
        train_indices, val_indices, test_indices = iterative_multilabel_split(
            train_records, args.seed
        )
    else:
        train_indices, val_indices, test_indices = split_indices(
            len(train_records), args.seed
        )
    if not train_indices or not test_indices or (fold_manifest is None and not val_indices):
        raise ValueError("Train, validation and holdout splits must be nonempty.")
    split_manifest_path = output_dir / "split_manifest.json"
    save_split_manifest(
        split_manifest_path, train_records, train_indices, val_indices,
        test_indices, split_seed, args.split_method,
    )
    save_label_distribution(
        label_distribution_path,
        train_records,
        train_indices,
        val_indices,
        test_indices,
    )
    print(f"Label distribution saved to: {label_distribution_path}")
    print(f"Split manifest saved to: {split_manifest_path}")

    train_loader_indices = expand_rare_training_indices(
        train_records, train_indices,
        args.rare_augment_max_count, args.rare_augment_extra_copies,
    )
    extra_train_samples = len(train_loader_indices) - len(train_indices)

    run_config_path = output_dir / "run_config.json"
    run_config = vars(args).copy()
    run_config.update(
        run_output_dir=str(output_dir.resolve()),
        started_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        split_seed=split_seed,
        split_manifest=str(split_manifest_path.resolve()),
        split_sha256=hashlib.sha256(split_manifest_path.read_bytes()).hexdigest(),
        rare_augment_extra_samples=extra_train_samples,
        train_samples_per_epoch=len(train_loader_indices),
    )
    with run_config_path.open("w", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)
    print(f"Run configuration saved to: {run_config_path}")

    print(f"Data directory: {data_dir.resolve()}")
    print(f"Channel order: {CHANNELS}")
    print(f"Total labeled images: {len(train_records)}")
    split_sizes = f"train: {len(train_indices)}, holdout test: {len(test_indices)}"
    if val_indices:
        split_sizes += f", val: {len(val_indices)}"
    print(f"Split sizes - {split_sizes}")
    if args.rare_augment_max_count and args.rare_augment_extra_copies:
        print(
            f"Rare-class augmentation adds {extra_train_samples} training draws "
            f"per epoch ({len(train_loader_indices)} total)."
        )
    if kaggle_test_records is not None:
        print(f"Kaggle test images: {len(kaggle_test_records)}")

    train_dataset = HPAImageDataset(
        train_records,
        train_image_dir,
        args.image_size,
        augment=True,
        augmentation=args.augmentation,
    )
    eval_dataset = HPAImageDataset(
        train_records,
        train_image_dir,
        args.image_size,
        augment=False,
    )
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    print(f"Device: {device}")

    train_loader = make_loader(
        train_dataset,
        train_loader_indices,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    val_loader = None
    if val_indices:
        val_loader = make_loader(
            eval_dataset,
            val_indices,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        )
    test_loader = make_loader(
        eval_dataset,
        test_indices,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    sample = eval_dataset[train_indices[0]]
    print(f"One input image shape: {tuple(sample['image'].shape)}")
    print(f"One target shape: {tuple(sample['target'].shape)}")

    print(f"Selected model: {args.model_name}")
    model = build_model(
        model_name=args.model_name,
        num_classes=NUM_CLASSES,
        pretrained_weights=args.pretrained_weights,
    ).to(device)
    if args.loss == "focal":
        criterion = SigmoidFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    elif args.loss == "asl":
        criterion = AsymmetricLoss(
            gamma_pos=args.asl_gamma_pos,
            gamma_neg=args.asl_gamma_neg,
            clip=args.asl_clip,
        )
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(
        model, name=args.optimizer, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "warmup_cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            total_steps=args.epochs * len(train_loader),
            warmup_steps=int(args.warmup_epochs * len(train_loader)),
            min_lr=args.min_lr,
        )
    print(
        f"Training: loss={args.loss}, optimizer={args.optimizer}, "
        f"scheduler={args.scheduler}, split={args.split_method}, "
        f"augmentation={args.augmentation}"
    )

    if args.smoke_test:
        run_smoke_test(
            model,
            train_loader,
            val_loader if val_loader is not None else test_loader,
            criterion,
            optimizer,
            device,
            scheduler=scheduler,
            eval_label="Holdout" if fold_manifest is not None else "Val",
        )
        return

    best_f1 = -1.0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args.threshold,
            optimizer=optimizer,
            progress_desc=f"Epoch {epoch}/{args.epochs} [Train]",
            scheduler=scheduler,
            global_step_start=(epoch - 1) * len(train_loader),
        )
        history_row = {
            "epoch": epoch,
            "learning_rate": f"{optimizer.param_groups[0]['lr']:.8g}",
            "train_loss": f"{train_metrics['loss']:.8f}",
            "train_macro_f1": f"{train_metrics['macro_f1']:.8f}",
            "train_micro_f1": f"{train_metrics['micro_f1']:.8f}",
        }
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Train Macro-F1: {train_metrics['macro_f1']:.4f}"
        )

        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                args.threshold,
                optimizer=None,
                progress_desc=f"Epoch {epoch}/{args.epochs} [Val]",
            )
            print(
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f} "
                f"Val Micro-F1: {val_metrics['micro_f1']:.4f}"
            )
            is_best = val_metrics["macro_f1"] > best_f1
            if is_best:
                best_f1 = val_metrics["macro_f1"]
                torch.save(
                    {
                        "epoch": epoch,
                        "best_f1": best_f1,
                        "model_name": args.model_name,
                        "run_config": run_config,
                        "model_state_dict": model.state_dict(),
                    },
                    best_model_path,
                )
                print(f"Best model saved to: {best_model_path}")
            history_row.update(
                val_loss=f"{val_metrics['loss']:.8f}",
                val_macro_f1=f"{val_metrics['macro_f1']:.8f}",
                val_micro_f1=f"{val_metrics['micro_f1']:.8f}",
                is_best=int(is_best),
            )
        history_rows.append(history_row)
        save_training_log(training_log_path, history_rows)

    if val_loader is not None:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model from epoch {checkpoint['epoch']} "
            f"with Val Macro-F1: {checkpoint['best_f1']:.4f}"
        )
        save_validation_artifacts(model, val_loader, device, args.threshold, output_dir)

    holdout_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        args.threshold,
        optimizer=None,
        progress_desc="Holdout [Eval]",
        collect_outputs=fold_manifest is not None,
    )
    print(
        f"Holdout Test Loss: {holdout_metrics['loss']:.4f} "
        f"Macro-F1: {holdout_metrics['macro_f1']:.4f} "
        f"Micro-F1: {holdout_metrics['micro_f1']:.4f}"
    )

    metrics_summary_path = output_dir / "metrics_summary.json"
    metrics_summary = {
        "model_name": args.model_name,
        "loss": args.loss,
        "split_sha256": run_config["split_sha256"],
        "holdout_loss": holdout_metrics["loss"],
        "holdout_macro_f1": holdout_metrics["macro_f1"],
        "holdout_micro_f1": holdout_metrics["micro_f1"],
    }
    if fold_manifest is not None:
        final_model_path = output_dir / "final_model.pth"
        torch.save(
            {
                "epoch": args.epochs,
                "holdout_macro_f1": holdout_metrics["macro_f1"],
                "model_name": args.model_name,
                "run_config": run_config,
                "model_state_dict": model.state_dict(),
            },
            final_model_path,
        )
        print(f"Final model saved to: {final_model_path}")
        save_prediction_artifacts(
            holdout_metrics["predictions"], args.threshold, output_dir, split="holdout"
        )
        metrics_summary.update(
            fold=args.fold,
            epoch=args.epochs,
            threshold=args.threshold,
            holdout_size=len(test_indices),
            fold_manifest_sha256=args.fold_manifest_sha256,
        )
    else:
        metrics_summary.update(
            best_epoch=checkpoint["epoch"],
            best_val_macro_f1=checkpoint["best_f1"],
        )
    with metrics_summary_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_summary, file, ensure_ascii=False, indent=2)
    print(f"Metric summary saved to: {metrics_summary_path}")

    if kaggle_test_records is not None:
        kaggle_test_dataset = HPAImageDataset(
            kaggle_test_records, kaggle_test_image_dir, args.image_size, augment=False,
        )
        kaggle_test_loader = make_loader(
            kaggle_test_dataset,
            indices=None,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        )
        run_inference(
            model,
            kaggle_test_loader,
            device,
            args.threshold,
            sample_submission_path,
            submission_path,
        )
    return metrics_summary


if __name__ == "__main__":
    main()
