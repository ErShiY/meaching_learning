import csv
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU.")
    return device


def calculate_f1(logits, targets, threshold):
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5

    predictions = predictions.float()
    targets = targets.float()

    true_positive = (predictions * targets).sum(dim=0)
    false_positive = (predictions * (1.0 - targets)).sum(dim=0)
    false_negative = ((1.0 - predictions) * targets).sum(dim=0)

    denominator = 2.0 * true_positive + false_positive + false_negative
    class_f1 = torch.where(
        denominator > 0,
        2.0 * true_positive / denominator,
        torch.zeros_like(denominator),
    )
    macro_f1 = class_f1.mean().item()

    total_true_positive = true_positive.sum()
    total_false_positive = false_positive.sum()
    total_false_negative = false_negative.sum()
    micro_denominator = (
        2.0 * total_true_positive
        + total_false_positive
        + total_false_negative
    )
    if micro_denominator.item() > 0:
        micro_f1 = (2.0 * total_true_positive / micro_denominator).item()
    else:
        micro_f1 = 0.0

    return macro_f1, micro_f1


def run_epoch(
    model,
    data_loader,
    criterion,
    device,
    threshold,
    optimizer=None,
    progress_desc=None,
    scheduler=None,
    global_step_start=0,
    collect_outputs=False,
):
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_ids = []
    all_logits = []
    all_targets = []

    if progress_desc is None:
        progress_desc = "Train" if is_training else "Evaluate"

    progress_bar = tqdm(
        data_loader,
        desc=progress_desc,
        total=len(data_loader),
        unit="batch",
        dynamic_ncols=True,
    )

    for batch_index, batch in enumerate(progress_bar):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        if is_training:
            if scheduler is not None:
                scheduler.set_learning_rate(global_step_start + batch_index)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        if collect_outputs:
            all_ids.extend(batch["id"])

        progress_metrics = {
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / total_samples:.4f}",
        }
        if is_training:
            progress_metrics["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
        if device.type == "cuda":
            gpu_memory_gb = torch.cuda.memory_allocated(device) / (1024**3)
            progress_metrics["gpu_mem"] = f"{gpu_memory_gb:.1f}G"
        progress_bar.set_postfix(progress_metrics)

    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty data loader.")

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    macro_f1, micro_f1 = calculate_f1(
        all_logits, all_targets, threshold
    )

    metrics = {
        "loss": total_loss / total_samples,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }
    if collect_outputs:
        metrics["predictions"] = (
            np.asarray(all_ids), all_targets.numpy(), torch.sigmoid(all_logits).numpy(),
        )
    return metrics


def save_training_log(log_path, history_rows):
    if not history_rows:
        return

    log_path = Path(log_path)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)


@torch.no_grad()
def collect_predictions(
    model,
    data_loader,
    device,
    progress_desc="Validation reports",
):
    model.eval()
    all_ids = []
    all_targets = []
    all_probabilities = []

    progress_bar = tqdm(
        data_loader,
        desc=progress_desc,
        total=len(data_loader),
        unit="batch",
        dynamic_ncols=True,
    )
    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits)

        all_ids.extend(batch["id"])
        all_targets.append(batch["target"].cpu())
        all_probabilities.append(probabilities.cpu())

    if not all_targets:
        raise ValueError("Cannot collect predictions from an empty data loader.")

    targets = torch.cat(all_targets, dim=0).numpy()
    probabilities = torch.cat(all_probabilities, dim=0).numpy()
    return np.asarray(all_ids), targets, probabilities


def calculate_per_class_metrics(targets, probabilities, threshold):
    targets = targets >= 0.5
    predictions = probabilities >= threshold

    true_positive = np.sum(predictions & targets, axis=0).astype(np.int64)
    false_positive = np.sum(predictions & ~targets, axis=0).astype(np.int64)
    false_negative = np.sum(~predictions & targets, axis=0).astype(np.int64)
    true_negative = np.sum(~predictions & ~targets, axis=0).astype(np.int64)

    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=(true_positive + false_positive) != 0,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=(true_positive + false_negative) != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    support = true_positive + false_negative

    return {
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def save_validation_artifacts(
    model,
    val_loader,
    device,
    threshold,
    output_dir,
):
    predictions = collect_predictions(model, val_loader, device)
    save_prediction_artifacts(predictions, threshold, output_dir)


def save_prediction_artifacts(predictions, threshold, output_dir, split="validation"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids, targets, probabilities = predictions
    metrics = calculate_per_class_metrics(targets, probabilities, threshold)

    prediction_prefix = "val" if split == "validation" else split
    np.savez_compressed(
        output_dir / f"{prediction_prefix}_predictions.npz",
        ids=image_ids,
        targets=targets.astype(np.uint8),
        probabilities=probabilities.astype(np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
    )

    report_path = output_dir / "classification_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["class", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for class_index in range(targets.shape[1]):
            writer.writerow(
                {
                    "class": class_index,
                    "precision": f"{metrics['precision'][class_index]:.6f}",
                    "recall": f"{metrics['recall'][class_index]:.6f}",
                    "f1": f"{metrics['f1'][class_index]:.6f}",
                    "support": int(metrics["support"][class_index]),
                }
            )

    matrix_csv_path = output_dir / "confusion_matrix_per_class.csv"
    with matrix_csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["class", "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for class_index in range(targets.shape[1]):
            writer.writerow(
                {
                    "class": class_index,
                    "tn": int(metrics["tn"][class_index]),
                    "fp": int(metrics["fp"][class_index]),
                    "fn": int(metrics["fn"][class_index]),
                    "tp": int(metrics["tp"][class_index]),
                }
            )

    num_classes = targets.shape[1]
    num_columns = 7
    num_rows = int(np.ceil(num_classes / num_columns))
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(3.0 * num_columns, 2.8 * num_rows),
    )
    axes = np.asarray(axes).reshape(-1)

    for class_index in range(num_classes):
        axis = axes[class_index]
        matrix = np.asarray(
            [
                [metrics["tn"][class_index], metrics["fp"][class_index]],
                [metrics["fn"][class_index], metrics["tp"][class_index]],
            ],
            dtype=np.int64,
        )
        maximum = max(int(matrix.max()), 1)
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set_title(f"Class {class_index}")
        axis.set_xticks([0, 1], ["0", "1"])
        axis.set_yticks([0, 1], ["0", "1"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")

        cell_names = (("TN", "FP"), ("FN", "TP"))
        for row_index in range(2):
            for column_index in range(2):
                value = int(matrix[row_index, column_index])
                text_color = "white" if value > maximum / 2 else "black"
                axis.text(
                    column_index,
                    row_index,
                    f"{cell_names[row_index][column_index]}\n{value}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    for axis in axes[num_classes:]:
        axis.axis("off")

    figure.suptitle(
        f"{split.capitalize()} one-vs-rest confusion matrices (threshold={threshold:.3f})",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(
        output_dir / "confusion_matrices.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"{split.capitalize()} reports saved to: {output_dir}")


def run_smoke_test(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    eval_label="Val",
):
    print("Running one-batch smoke test...")

    model.train()
    train_batch = next(iter(train_loader))
    train_images = train_batch["image"].to(device)
    train_targets = train_batch["target"].to(device)

    if scheduler is not None:
        scheduler.set_learning_rate(0)
    optimizer.zero_grad(set_to_none=True)
    train_logits = model(train_images)
    train_loss = criterion(train_logits, train_targets)
    train_loss.backward()
    optimizer.step()

    model.eval()
    val_batch = next(iter(val_loader))
    with torch.no_grad():
        val_images = val_batch["image"].to(device)
        val_targets = val_batch["target"].to(device)
        val_logits = model(val_images)
        val_loss = criterion(val_logits, val_targets)

    print(f"Train image shape: {tuple(train_images.shape)}")
    print(f"Train logits shape: {tuple(train_logits.shape)}")
    print(f"Train loss: {train_loss.item():.4f}")
    print(f"{eval_label} loss: {val_loss.item():.4f}")
    print("Smoke test passed: data, forward, loss and backward are working.")
