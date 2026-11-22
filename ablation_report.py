"""Validation-only threshold calibration and reports for DenseNet ablations.

Importing this module needs NumPy, but not PyTorch. Holdout inference is lazy
and opt-in; no training run or checkpoint is modified by the reporting code.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ablation import validate_rare_augmentation_recipe


PROJECT_ROOT = Path(__file__).resolve().parent
THRESHOLDS = tuple(index / 200 for index in range(10, 200))
SHARED_RECIPE_KEYS = (
    "epochs", "image_size", "batch_size", "learning_rate", "optimizer",
    "weight_decay", "scheduler", "warmup_epochs", "min_lr", "augmentation",
    "seed", "threshold", "asl_gamma_pos", "asl_gamma_neg", "asl_clip", "num_workers",
)
SUMMARY_FIELDS = (
    "experiment", "model_name", "loss", "status", "best_epoch",
    "fixed_threshold", "val_macro_f1_fixed", "val_micro_f1_fixed",
    "calibrated_threshold", "val_macro_f1_calibrated",
    "val_micro_f1_calibrated", "holdout_macro_f1_fixed",
    "holdout_micro_f1_fixed", "holdout_macro_f1_calibrated",
    "holdout_micro_f1_calibrated", "run_dir", "notes",
)


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_arrays(targets, probabilities):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if (
        targets.ndim != 2 or targets.shape != probabilities.shape
        or not all(targets.shape)
    ):
        raise ValueError("Targets/probabilities must have the same nonempty [N,C] shape.")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must contain only 0 and 1.")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Probabilities must be finite and in [0,1].")
    return targets.astype(bool), probabilities


def _metrics_unchecked(targets, probabilities, threshold):
    predictions = probabilities >= threshold
    tp = np.sum(predictions & targets, axis=0, dtype=np.int64)
    fp = np.sum(predictions & ~targets, axis=0, dtype=np.int64)
    fn = np.sum(~predictions & targets, axis=0, dtype=np.int64)
    denominator = 2 * tp + fp + fn
    class_f1 = np.divide(
        2 * tp, denominator, out=np.zeros(tp.shape, dtype=np.float64),
        where=denominator != 0,
    )
    micro_denominator = int(denominator.sum())
    return {
        "macro_f1": float(class_f1.mean()),
        "micro_f1": float(2 * tp.sum() / micro_denominator)
        if micro_denominator else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "f1": class_f1,
    }


def multilabel_metrics(targets, probabilities, threshold):
    """Match engine.py: independent >= decisions and zero F1 for absent classes.

    There is deliberately no forced argmax for rows without positive labels.
    That Kaggle submission policy is separate from these local F1 reports.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0,1].")
    targets, probabilities = _validate_arrays(targets, probabilities)
    return _metrics_unchecked(targets, probabilities, threshold)


def calibrate_threshold(targets, probabilities):
    """Select a single global threshold using validation Macro-F1 only.

    Search 0.050..0.995 at increments of 0.005. Equal optima choose the lowest
    threshold, so repeated calls have a documented deterministic tie rule.
    """
    targets, probabilities = _validate_arrays(targets, probabilities)
    rows = []
    for threshold in THRESHOLDS:
        metrics = _metrics_unchecked(targets, probabilities, threshold)
        rows.append({
            "threshold": threshold,
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
        })
    best = max(rows, key=lambda row: row["macro_f1"])
    return best.copy(), rows


def _load_predictions(path, expected_ids):
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["ids"]).astype(str)
        targets = np.asarray(data["targets"])
        probabilities = np.asarray(data["probabilities"])
    _validate_arrays(targets, probabilities)
    if ids.ndim != 1 or len(ids) != len(targets) or len(set(ids)) != len(ids):
        raise ValueError(f"Invalid or duplicate prediction IDs: {path}")
    if targets.shape[1] != 28:
        raise ValueError(f"Expected 28 labels in prediction cache: {path}")
    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ValueError(f"Prediction IDs differ from the fixed split: {path}")
    indices = {image_id: index for index, image_id in enumerate(ids)}
    order = [indices[image_id] for image_id in expected_ids]
    return targets[order], probabilities[order]


def _validate_run(run_dir, cell, recipe, expected_splits):
    config = _read_json(run_dir / "run_config.json")
    for key in ("model_name", "loss"):
        if config.get(key) != cell[key]:
            raise ValueError(f"{run_dir}: expected {key}={cell[key]!r}.")
    for key in SHARED_RECIPE_KEYS:
        if key in recipe and config.get(key) != recipe[key]:
            raise ValueError(
                f"{run_dir}: shared recipe mismatch for {key}: "
                f"{config.get(key)!r} != {recipe[key]!r}."
            )
    validate_rare_augmentation_recipe(config, recipe, run_dir)
    for key in ("pretrained_weights", "data_dir"):
        if key in recipe and _resolve(config.get(key, "")) != _resolve(recipe[key]):
            raise ValueError(f"{run_dir}: shared recipe path differs for {key}.")
    actual_splits = _read_json(run_dir / "split_manifest.json")["splits"]
    if actual_splits != expected_splits:
        raise ValueError(f"{run_dir}: train/val/holdout split differs from suite.")
    summary = _read_json(run_dir / "metrics_summary.json")
    return config, summary


def _cache_metadata(run_dir, config, split_file):
    checkpoint = run_dir / "best_model.pth"
    stat = checkpoint.stat()
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "image_size": config["image_size"],
        "train_csv_sha256": hashlib.sha256(
            (_resolve(config["data_dir"]) / "train.csv").read_bytes()
        ).hexdigest(),
    }


def _holdout_predictions(
    run_dir, config, split_file, report_dir, device, num_workers, allow_inference=True,
):
    """Cache probabilities once; threshold selection never sees holdout labels."""
    cache_path = report_dir / "holdout_predictions.npz"
    expected_ids = _read_json(split_file)["splits"]["holdout"]
    if not cache_path.is_file() and not allow_inference:
        return None
    try:
        metadata = _cache_metadata(run_dir, config, split_file)
    except FileNotFoundError:
        if not allow_inference:
            return None
        raise
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                cached_metadata = json.loads(str(data["metadata_json"].item()))
            if cached_metadata == metadata:
                return _load_predictions(cache_path, expected_ids)
        except (KeyError, ValueError, OSError):
            # Corrupt or obsolete caches can be regenerated, but never trigger
            # GPU work unless the caller explicitly opted into inference.
            pass
    if not allow_inference:
        return None

    # Keep torch/model/dataset imports out of CPU-only cached report generation.
    import torch
    from dataset import HPAImageDataset, load_split_manifest, make_loader, read_train_records
    from engine import collect_predictions, resolve_device, set_seed
    from model import build_model

    set_seed(config["seed"])
    resolved_device = resolve_device(device)
    records = read_train_records(_resolve(config["data_dir"]) / "train.csv")
    _, _, indices = load_split_manifest(split_file, records)
    dataset = HPAImageDataset(
        records, _resolve(config["data_dir"]) / "train", config["image_size"],
        augment=False,
    )
    loader = make_loader(
        dataset, indices, config["batch_size"], shuffle=False,
        num_workers=num_workers, seed=config["seed"],
        pin_memory=resolved_device.type == "cuda",
    )
    checkpoint = torch.load(
        run_dir / "best_model.pth", map_location="cpu", weights_only=True
    )
    if checkpoint.get("model_name") != config["model_name"]:
        raise ValueError(f"Checkpoint/model mismatch: {run_dir}")
    model = build_model(
        model_name=config["model_name"], num_classes=28, pretrained_weights=""
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    ids, targets, probabilities = collect_predictions(
        model, loader, resolved_device, progress_desc="Ablation holdout evaluation"
    )
    np.savez_compressed(
        cache_path, ids=ids, targets=targets.astype(np.uint8),
        probabilities=probabilities.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    del model, checkpoint
    return _load_predictions(cache_path, expected_ids)


def _class_rows(split, targets, probabilities, threshold):
    metrics = multilabel_metrics(targets, probabilities, threshold)
    rows = []
    for index in range(targets.shape[1]):
        tp, fp, fn = (int(metrics[key][index]) for key in ("tp", "fp", "fn"))
        rows.append({
            "split": split, "threshold": threshold, "class": index,
            "support": tp + fn, "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f1": float(metrics["f1"][index]), "tp": tp, "fp": fp, "fn": fn,
        })
    return rows


def write_report(suite_dir, evaluate_holdout=False, device="auto", num_workers=0):
    """Write a partial or complete suite report and return the summary rows.

    By default saved validation predictions, fixed-threshold holdout metrics,
    and any valid existing holdout caches are read. ``evaluate_holdout=True``
    permits checkpoint inference if a current holdout cache is unavailable.
    """
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")
    suite_dir = _resolve(suite_dir)
    suite = _read_json(suite_dir / "suite.json")
    if suite.get("schema_version") != 1:
        raise ValueError("Unsupported ablation suite schema.")
    recipe = suite["recipe"]
    fixed_threshold = recipe.get("threshold", 0.5)
    split_file = _resolve(suite["split_file"])
    split_sha = hashlib.sha256(split_file.read_bytes()).hexdigest()
    if suite.get("split_sha256", split_sha) != split_sha:
        raise ValueError("Suite split_sha256 does not match its fixed split file.")
    expected_splits = _read_json(split_file)["splits"]
    rows = []
    canonical_targets = None
    holdout_targets = None

    # Validate all completed cells before running any expensive holdout inference.
    prepared = {}
    for name, cell in suite["cells"].items():
        if cell["status"] not in ("completed", "reused"):
            continue
        run_dir = _resolve(cell["run_dir"])
        config, summary = _validate_run(run_dir, cell, recipe, expected_splits)
        targets, probabilities = _load_predictions(
            run_dir / "val_predictions.npz", expected_splits["val"]
        )
        if canonical_targets is None:
            canonical_targets = targets
        elif not np.array_equal(targets, canonical_targets):
            raise ValueError("Validation targets differ across ablation cells.")
        prepared[name] = (run_dir, config, summary, targets, probabilities)

    for name, cell in suite["cells"].items():
        row = dict.fromkeys(SUMMARY_FIELDS)
        row.update(
            experiment=name, model_name=cell["model_name"], loss=cell["loss"],
            status=cell["status"], run_dir=cell.get("run_dir"),
            fixed_threshold=fixed_threshold,
        )
        if name not in prepared:
            row["notes"] = "No completed training artifacts; metrics unavailable."
            rows.append(row)
            continue
        run_dir, config, summary, targets, probabilities = prepared[name]
        fixed = multilabel_metrics(targets, probabilities, fixed_threshold)
        best, scan_rows = calibrate_threshold(targets, probabilities)
        row.update(
            best_epoch=summary["best_epoch"],
            val_macro_f1_fixed=fixed["macro_f1"],
            val_micro_f1_fixed=fixed["micro_f1"],
            calibrated_threshold=best["threshold"],
            val_macro_f1_calibrated=best["macro_f1"],
            val_micro_f1_calibrated=best["micro_f1"],
            holdout_macro_f1_fixed=summary.get("holdout_macro_f1"),
            holdout_micro_f1_fixed=summary.get("holdout_micro_f1"),
        )
        report_dir = suite_dir / "reports" / name
        report_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            report_dir / "threshold_scan.csv", scan_rows,
            ("threshold", "macro_f1", "micro_f1"),
        )
        class_rows = _class_rows("val_fixed", targets, probabilities, fixed_threshold)
        class_rows += _class_rows(
            "val_calibrated", targets, probabilities, best["threshold"]
        )
        holdout_predictions = None
        if evaluate_holdout or (report_dir / "holdout_predictions.npz").is_file():
            holdout_predictions = _holdout_predictions(
                run_dir, config, split_file, report_dir, device, num_workers,
                allow_inference=evaluate_holdout,
            )
        if holdout_predictions is not None:
            h_targets, h_probabilities = holdout_predictions
            if holdout_targets is None:
                holdout_targets = h_targets
            elif not np.array_equal(holdout_targets, h_targets):
                raise ValueError("Holdout targets differ across ablation cells.")
            holdout = multilabel_metrics(h_targets, h_probabilities, best["threshold"])
            row.update(
                holdout_macro_f1_calibrated=holdout["macro_f1"],
                holdout_micro_f1_calibrated=holdout["micro_f1"],
            )
            class_rows += _class_rows(
                "holdout_calibrated", h_targets, h_probabilities, best["threshold"]
            )
            row["notes"] = (
                "Threshold selected on validation only; applied to holdout. "
                + ("Inference allowed; valid caches reused." if evaluate_holdout
                   else "Reused a valid holdout cache without inference.")
            )
        else:
            row["notes"] = (
                "Calibrated holdout not evaluated; enable --evaluate-holdout."
            )
        _write_csv(
            report_dir / "class_metrics.csv", class_rows,
            ("split", "threshold", "class", "support", "precision", "recall",
             "f1", "tp", "fp", "fn"),
        )
        rows.append(row)

    suite_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(suite_dir / "summary.csv", rows, SUMMARY_FIELDS)
    with (suite_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({
            "protocol": {
                "checkpoint_selection": "validation Macro-F1 at fixed training threshold",
                "threshold_selection": "validation Macro-F1 only; lowest threshold on ties",
                "threshold_grid": {"start": 0.05, "stop": 0.995, "step": 0.005},
                "absent_class_f1": 0.0,
                "force_at_least_one_label": False,
                "evaluate_holdout": evaluate_holdout,
            },
            "results": rows,
        }, file, ensure_ascii=False, indent=2, allow_nan=False)
    _write_markdown(suite_dir / "summary.md", rows)
    print(f"Ablation summary saved to: {suite_dir / 'summary.csv'}")
    return rows


def _write_markdown(path, rows):
    def number(value):
        return "—" if value is None else f"{value:.6f}"

    lines = [
        "# DenseNet head / loss ablation", "",
        "Each completed cell uses the same split and training recipe. Checkpoints are",
        "selected by validation Macro-F1 at the fixed training threshold. A single",
        "global threshold is then selected on validation only (0.050–0.995, step 0.005).",
        "Calibrated validation scores are tuning scores; use the holdout for evaluation.",
        "Local metrics permit empty predictions, matching the existing training metrics.",
        "Kaggle inference additionally forces an argmax label for empty predictions.", "",
        "| Experiment | Status | Epoch | Val Macro fixed | Threshold | Val Macro calibrated | Holdout Macro fixed | Holdout Macro calibrated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['status']} | {row['best_epoch'] or '—'} | "
            f"{number(row['val_macro_f1_fixed'])} | {number(row['calibrated_threshold'])} | "
            f"{number(row['val_macro_f1_calibrated'])} | "
            f"{number(row['holdout_macro_f1_fixed'])} | "
            f"{number(row['holdout_macro_f1_calibrated'])} |"
        )
    lines += [
        "", "The new head is the entire multi-scale label-attention + GeM package.",
        "Its comparison with GAP + Linear does not isolate individual head components.",
        "Rare-class F1 can be unstable; class_metrics.csv includes each class support.",
        "Do not compare the magnitudes of BCE and ASL loss values.", "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
