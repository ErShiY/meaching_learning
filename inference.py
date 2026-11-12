import argparse
import csv
from pathlib import Path

import torch

from dataset import HPAImageDataset, NUM_CLASSES, make_loader, read_test_records
from engine import resolve_device, set_seed
from model import DenseNet121Baseline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for Human Protein Atlas baseline"
    )

    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/baseline/best_model.pth",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default="outputs/baseline/submission.csv",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


@torch.no_grad()
def predict_test(model, test_loader, device, threshold):
    model.eval()
    predictions = {}

    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits).cpu()

        for image_id, probability in zip(batch["id"], probabilities):
            labels = torch.where(probability >= threshold)[0].tolist()
            if not labels:
                labels = [int(torch.argmax(probability).item())]
            predictions[image_id] = " ".join(str(label) for label in labels)

    return predictions


def write_submission(sample_submission_path, submission_path, predictions):
    with sample_submission_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as source_file, submission_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        reader = csv.DictReader(source_file)
        writer = csv.DictWriter(output_file, fieldnames=["Id", "Predicted"])
        writer.writeheader()

        for row in reader:
            image_id = row["Id"]
            if image_id not in predictions:
                raise KeyError(f"No prediction was produced for {image_id}")
            writer.writerow(
                {"Id": image_id, "Predicted": predictions[image_id]}
            )


def run_inference(
    model,
    test_loader,
    device,
    threshold,
    sample_submission_path,
    submission_path,
):
    predictions = predict_test(model, test_loader, device, threshold)
    write_submission(sample_submission_path, submission_path, predictions)
    print(f"Submission saved to: {submission_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    sample_submission_path = data_dir / "sample_submission.csv"
    test_image_dir = data_dir / "test"
    checkpoint_path = Path(args.checkpoint)
    submission_path = Path(args.submission_path)

    for path in (sample_submission_path, test_image_dir, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    print(f"Device: {device}")

    test_records = read_test_records(sample_submission_path)
    test_dataset = HPAImageDataset(
        test_records,
        test_image_dir,
        args.image_size,
        augment=False,
    )
    test_loader = make_loader(
        test_dataset,
        indices=None,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model = DenseNet121Baseline(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"with Val Macro-F1: {checkpoint['best_f1']:.4f}"
    )

    run_inference(
        model,
        test_loader,
        device,
        args.threshold,
        sample_submission_path,
        submission_path,
    )


if __name__ == "__main__":
    main()
