import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from dataset import (
    NUM_CLASSES,
    _record_id_map,
    iterative_multilabel_split,
    read_train_records,
)


FOLD_COUNT = 5
FOLD_MANIFEST_FILENAME = "fold_manifest.json"
FOLD_METHOD = "iterative_multilabel"


def _canonical_labels(records):
    _record_id_map(records)
    return {record["id"]: sorted(set(record["labels"])) for record in records}


def _dataset_sha256(labels_by_id):
    content = json.dumps(labels_by_id, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _label_counts(folds, labels_by_id):
    counts_by_fold = []
    for fold in folds:
        counts = [0] * NUM_CLASSES
        for image_id in fold:
            for label in labels_by_id[image_id]:
                counts[label] += 1
        counts_by_fold.append(counts)
    return counts_by_fold


def create_fold_manifest(records, seed=42):
    """Build five fixed folds by image ID and record the label support per fold."""
    labels_by_id = _canonical_labels(records)
    if len(records) < FOLD_COUNT:
        raise ValueError(f"At least {FOLD_COUNT} labeled records are required.")

    ordered_records = sorted(records, key=lambda record: record["id"])
    base_size, remainder = divmod(len(ordered_records), FOLD_COUNT)
    sizes = [base_size + (index < remainder) for index in range(FOLD_COUNT)]
    partition_indices = iterative_multilabel_split(ordered_records, seed, sizes=sizes)
    folds = [
        [ordered_records[index]["id"] for index in indices]
        for indices in partition_indices
    ]
    return {
        "version": 1,
        "fold_count": FOLD_COUNT,
        "seed": int(seed),
        "method": FOLD_METHOD,
        "dataset_sha256": _dataset_sha256(labels_by_id),
        "label_counts": _label_counts(folds, labels_by_id),
        "folds": folds,
    }


def load_fold_manifest(path, records):
    """Restore a fold manifest and check it against the current training labels."""
    with Path(path).open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("fold_count") != FOLD_COUNT
    ):
        raise ValueError(f"Expected a version 1 manifest with {FOLD_COUNT} folds.")
    seed = manifest.get("seed")
    if type(seed) is not int or seed < 0 or manifest.get("method") != FOLD_METHOD:
        raise ValueError("Fold manifest has invalid stratification metadata.")

    folds = manifest.get("folds")
    if (
        not isinstance(folds, list)
        or len(folds) != FOLD_COUNT
        or any(not isinstance(fold, list) or not fold for fold in folds)
    ):
        raise ValueError(f"Fold manifest must contain {FOLD_COUNT} nonempty image ID lists.")
    fold_sizes = [len(fold) for fold in folds]
    if max(fold_sizes) - min(fold_sizes) > 1:
        raise ValueError("Sizes of folds must differ by at most one record.")

    all_ids = [image_id for fold in folds for image_id in fold]
    if any(not isinstance(image_id, str) for image_id in all_ids):
        raise ValueError("Fold manifest image IDs must be strings.")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Fold manifest contains duplicate image IDs.")

    labels_by_id = _canonical_labels(records)
    if set(all_ids) != set(labels_by_id):
        raise ValueError("Fold manifest image IDs differ from the current training CSV.")
    if manifest.get("dataset_sha256") != _dataset_sha256(labels_by_id):
        raise ValueError("Fold manifest labels differ from the current training CSV.")
    if manifest.get("label_counts") != _label_counts(folds, labels_by_id):
        raise ValueError("Fold manifest support counts do not match its assignments.")
    return manifest


def fold_indices(manifest, records, fold):
    """Return training and held-out indices for a validated 1-based fold number."""
    folds = manifest["folds"]
    if type(fold) is not int or not 1 <= fold <= len(folds):
        raise ValueError(f"Fold must be an integer from 1 to {len(folds)}.")
    index_by_id = _record_id_map(records)
    train_indices = [
        index_by_id[image_id]
        for fold_number, image_ids in enumerate(folds, start=1)
        if fold_number != fold
        for image_id in image_ids
    ]
    heldout_indices = [index_by_id[image_id] for image_id in folds[fold - 1]]
    return train_indices, heldout_indices


def main():
    parser = argparse.ArgumentParser(
        description="Save a fixed multilabel-stratified five-fold manifest."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = read_train_records(args.data_dir / "train.csv")
    manifest = create_fold_manifest(records, args.seed)
    output = args.output if args.output is not None else args.data_dir / FOLD_MANIFEST_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")
        load_fold_manifest(temporary_path, records)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Saved fold manifest to: {output}")
    summaries = zip(manifest["folds"], manifest["label_counts"])
    for index, (fold, counts) in enumerate(summaries, start=1):
        present = sum(count > 0 for count in counts)
        print(f"Fold {index}: {len(fold)} images, {present}/{NUM_CLASSES} labels")


if __name__ == "__main__":
    main()
