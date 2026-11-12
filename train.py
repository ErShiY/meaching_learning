import argparse
from pathlib import Path

import torch
import torch.nn as nn

from dataset import (
    CHANNELS,
    HPAImageDataset,
    NUM_CLASSES,
    make_loader,
    read_test_records,
    read_train_records,
    sample_records,
    save_label_distribution,
    split_indices,
)
from engine import (
    resolve_device,
    run_epoch,
    run_smoke_test,
    save_training_log,
    save_validation_artifacts,
    set_seed,
)
from inference import run_inference
from model import DenseNet121Baseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="DenseNet121 baseline for Human Protein Atlas"
    )

    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="load_checkpoint/densenet121-a639ec97.pth",
        help="Local ImageNet DenseNet121 .pth file. No weights are downloaded automatically.",
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

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.debug and args.output_dir == "outputs/baseline":
        args.output_dir = "outputs/debug"
        print("Debug output directory changed to: outputs/debug")

    data_dir = Path(args.data_dir)
    train_csv_path = data_dir / "train.csv"
    sample_submission_path = data_dir / "sample_submission.csv"
    train_image_dir = data_dir / "train"
    kaggle_test_image_dir = data_dir / "test"

    required_paths = [
        train_csv_path,
        sample_submission_path,
        train_image_dir,
        kaggle_test_image_dir,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path not found: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pth"
    training_log_path = output_dir / "training_log.csv"
    label_distribution_path = output_dir / "label_distribution.csv"
    submission_path = output_dir / "submission.csv"

    train_records = read_train_records(train_csv_path)
    kaggle_test_records = read_test_records(sample_submission_path)

    if args.debug:
        train_records = sample_records(
            train_records, args.debug_samples, args.seed
        )
        args.skip_submission = True
        print(
            f"Debug mode: using {len(train_records)} labeled images "
            "and skipping Kaggle test inference."
        )

    train_indices, val_indices, test_indices = split_indices(
        len(train_records), args.seed
    )
    save_label_distribution(
        label_distribution_path,
        train_records,
        train_indices,
        val_indices,
        test_indices,
    )
    print(f"Label distribution saved to: {label_distribution_path}")

    print(f"Data directory: {data_dir.resolve()}")
    print(f"Channel order: {CHANNELS}")
    print(f"Total labeled images: {len(train_records)}")
    print(
        f"Split sizes - train: {len(train_indices)}, "
        f"val: {len(val_indices)}, holdout test: {len(test_indices)}"
    )
    print(f"Kaggle test images: {len(kaggle_test_records)}")

    train_dataset = HPAImageDataset(
        train_records,
        train_image_dir,
        args.image_size,
        augment=True,
    )
    eval_dataset = HPAImageDataset(
        train_records,
        train_image_dir,
        args.image_size,
        augment=False,
    )
    kaggle_test_dataset = HPAImageDataset(
        kaggle_test_records,
        kaggle_test_image_dir,
        args.image_size,
        augment=False,
    )

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    print(f"Device: {device}")

    train_loader = make_loader(
        train_dataset,
        train_indices,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
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
    kaggle_test_loader = make_loader(
        kaggle_test_dataset,
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    sample = eval_dataset[val_indices[0]]
    print(f"One input image shape: {tuple(sample['image'].shape)}")
    print(f"One target shape: {tuple(sample['target'].shape)}")

    model = DenseNet121Baseline(
        num_classes=NUM_CLASSES,
        pretrained_weights=args.pretrained_weights,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate
    )

    if args.smoke_test:
        run_smoke_test(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
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
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            args.threshold,
            optimizer=None,
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Train Macro-F1: {train_metrics['macro_f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} "
            f"Val Micro-F1: {val_metrics['micro_f1']:.4f}"
        )

        # Kaggle uses Macro-F1, so validation Macro-F1 selects the checkpoint.
        is_best = val_metrics["macro_f1"] > best_f1
        if is_best:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "model_state_dict": model.state_dict(),
                },
                best_model_path,
            )
            print(f"Best model saved to: {best_model_path}")

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.8f}",
                "train_macro_f1": f"{train_metrics['macro_f1']:.8f}",
                "train_micro_f1": f"{train_metrics['micro_f1']:.8f}",
                "val_loss": f"{val_metrics['loss']:.8f}",
                "val_macro_f1": f"{val_metrics['macro_f1']:.8f}",
                "val_micro_f1": f"{val_metrics['micro_f1']:.8f}",
                "is_best": int(is_best),
            }
        )
        save_training_log(training_log_path, history_rows)

    checkpoint = torch.load(
        best_model_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded best model from epoch {checkpoint['epoch']} "
        f"with Val Macro-F1: {checkpoint['best_f1']:.4f}"
    )

    save_validation_artifacts(
        model,
        val_loader,
        device,
        args.threshold,
        output_dir,
    )

    holdout_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        args.threshold,
        optimizer=None,
    )
    print(
        f"Holdout Test Loss: {holdout_metrics['loss']:.4f} "
        f"Macro-F1: {holdout_metrics['macro_f1']:.4f} "
        f"Micro-F1: {holdout_metrics['micro_f1']:.4f}"
    )

    if not args.skip_submission:
        run_inference(
            model,
            kaggle_test_loader,
            device,
            args.threshold,
            sample_submission_path,
            submission_path,
        )


if __name__ == "__main__":
    main()
