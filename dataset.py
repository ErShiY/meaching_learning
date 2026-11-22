from collections import Counter
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


CHANNELS = ("blue", "green", "red", "yellow")
NUM_CLASSES = 28


def read_train_records(csv_path):
    records = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["Id", "Target"]:
            raise ValueError(
                f"Unexpected train.csv columns: {reader.fieldnames}. "
                "Expected ['Id', 'Target']."
            )

        for row in reader:
            labels = [int(value) for value in row["Target"].split()]
            if any(label < 0 or label >= NUM_CLASSES for label in labels):
                raise ValueError(f"Invalid labels for image {row['Id']}: {labels}")

            records.append({"id": row["Id"], "labels": labels})

    return records


def read_test_records(sample_submission_path):
    records = []

    with sample_submission_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError(
                f"Unexpected sample_submission.csv columns: {reader.fieldnames}. "
                "Expected ['Id', 'Predicted']."
            )

        for row in reader:
            records.append({"id": row["Id"]})

    return records


def augment_image(image, policy="basic"):
    """Apply spatial transforms jointly to all fluorescence channels.

    Inputs are float images in [0, 1] before normalization. ``basic`` preserves
    the original two random draws and horizontal/vertical flip behavior.
    ``microscopy`` adds 90-degree rotations and mild shared intensity changes;
    it never crops away a region that may be the only evidence for a label.
    """
    if policy not in ("basic", "microscopy"):
        raise ValueError(f"Unknown augmentation policy: {policy}")
    if random.random() < 0.5:
        image = torch.flip(image, dims=(2,))
    if random.random() < 0.5:
        image = torch.flip(image, dims=(1,))
    if policy == "microscopy":
        image = torch.rot90(image, random.randrange(4), dims=(1, 2))
        gain = random.uniform(0.9, 1.1)
        gamma = random.uniform(0.9, 1.1)
        image = (image.clamp(0.0, 1.0).pow(gamma) * gain).clamp(0.0, 1.0)
    return image


class HPAImageDataset(Dataset):
    def __init__(
        self, records, image_dir, image_size, augment=False,
        augmentation="basic",
    ):
        if augmentation not in ("basic", "microscopy"):
            raise ValueError(f"Unknown augmentation policy: {augmentation}")
        self.records = records
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment
        self.augmentation = augmentation

    def __len__(self):
        return len(self.records)

    def load_image(self, image_id):
        channel_images = []
        original_shape = None

        for channel in CHANNELS:
            image_path = self.image_dir / f"{image_id}_{channel}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image channel: {image_path}")

            with Image.open(image_path) as image:
                channel_image = np.asarray(image.convert("L"), dtype=np.float32)

            if original_shape is None:
                original_shape = channel_image.shape
            elif channel_image.shape != original_shape:
                raise ValueError(
                    f"Four channels of {image_id} do not have the same size: "
                    f"{original_shape} and {channel_image.shape}"
                )

            channel_images.append(channel_image)

        # The fixed channel order is Blue, Green, Red, Yellow.
        image = np.stack(channel_images, axis=0)
        image = torch.from_numpy(image).float() / 255.0

        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.augment:
            image = augment_image(image, self.augmentation)

        image = (image - 0.5) / 0.5
        return image

    def __getitem__(self, index):
        record = self.records[index]
        image = self.load_image(record["id"])

        sample = {
            "id": record["id"],
            "image": image,
        }

        if "labels" in record:
            target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            target[record["labels"]] = 1.0
            sample["target"] = target

        return sample


def split_indices(num_samples, seed):
    indices = list(range(num_samples))
    random_generator = random.Random(seed)
    random_generator.shuffle(indices)

    train_size = int(num_samples * 0.70)
    val_size = int(num_samples * 0.20)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    return train_indices, val_indices, test_indices


def iterative_multilabel_split(records, seed, *, sizes=None):
    """Use exact partition sizes and approximate label stratification.

    The default sizes are 70/20/10.
    Assign rare labels first. For each labeled sample, prefer the partition
    with the greatest remaining desired count for that label, break ties by
    available sample capacity, then use the seeded RNG. Account for every label
    carried by that sample before moving on. Sparse/co-occurring labels can
    make exact marginal proportions impossible; this is not a group split and
    cannot remove biological-sample leakage when group metadata is unavailable.
    """
    num_samples = len(records)
    if sizes is None:
        train_size = int(num_samples * 0.70)
        val_size = int(num_samples * 0.20)
        sizes = [train_size, val_size, num_samples - train_size - val_size]
    else:
        sizes = tuple(sizes)
        if not sizes or any(
            not isinstance(size, (int, np.integer))
            or isinstance(size, bool) or size < 0
            for size in sizes
        ) or sum(sizes) != num_samples:
            raise ValueError(
                "Partition sizes must be nonnegative integers that cover all records."
            )
    capacities = np.array(sizes, dtype=np.int64)
    partitions = [[] for _ in capacities]
    if num_samples == 0:
        return tuple(partitions)

    random_generator = np.random.default_rng(seed)
    labels_per_sample = []
    label_samples = [[] for _ in range(NUM_CLASSES)]
    for sample_index, record in enumerate(records):
        labels = sorted(set(record.get("labels", [])))
        if any(
            not isinstance(label, (int, np.integer))
            or label < 0 or label >= NUM_CLASSES
            for label in labels
        ):
            raise ValueError(f"Invalid labels for record {record.get('id')}: {labels}")
        labels_per_sample.append(labels)
        for label in labels:
            label_samples[label].append(sample_index)

    remaining_label_counts = np.asarray(
        [len(indices) for indices in label_samples], dtype=np.int64
    )
    desired_label_counts = np.outer(
        capacities / num_samples, remaining_label_counts
    )
    assigned = np.zeros(num_samples, dtype=bool)

    def choose_partition(label=None):
        candidates = np.flatnonzero(capacities > 0)
        if label is not None:
            demand = desired_label_counts[candidates, label]
            candidates = candidates[demand == demand.max()]
        room = capacities[candidates]
        candidates = candidates[room == room.max()]
        return int(random_generator.choice(candidates))

    def assign_sample(sample_index, partition_index):
        partitions[partition_index].append(int(sample_index))
        capacities[partition_index] -= 1
        assigned[sample_index] = True
        labels = labels_per_sample[sample_index]
        remaining_label_counts[labels] -= 1
        desired_label_counts[partition_index, labels] -= 1.0

    while np.any(remaining_label_counts > 0):
        present_labels = np.flatnonzero(remaining_label_counts > 0)
        rare_count = remaining_label_counts[present_labels].min()
        rare_labels = present_labels[
            remaining_label_counts[present_labels] == rare_count
        ]
        label = int(random_generator.choice(rare_labels))
        candidates = [
            index for index in label_samples[label] if not assigned[index]
        ]
        random_generator.shuffle(candidates)
        for sample_index in candidates:
            assign_sample(sample_index, choose_partition(label))

    # Empty-label samples do not guide stratification but still fill exact sizes.
    unlabeled_indices = np.flatnonzero(~assigned)
    random_generator.shuffle(unlabeled_indices)
    for sample_index in unlabeled_indices:
        assign_sample(sample_index, choose_partition())

    for partition in partitions:
        random_generator.shuffle(partition)
    return tuple(partitions)


def _record_id_map(records):
    record_ids = [record.get("id") for record in records]
    if any(not isinstance(image_id, str) or not image_id for image_id in record_ids):
        raise ValueError("All records must have a nonempty string id.")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("Dataset contains duplicate record IDs.")
    return {image_id: index for index, image_id in enumerate(record_ids)}


def _validate_split_indices(partitions, num_samples):
    flattened = [index for partition in partitions for index in partition]
    if any(
        not isinstance(index, (int, np.integer))
        or isinstance(index, bool) or index < 0 or index >= num_samples
        for index in flattened
    ):
        raise ValueError("Split contains an invalid record index.")
    if len(flattened) != len(set(flattened)):
        raise ValueError("Split partitions contain duplicate or overlapping records.")
    if len(flattened) != num_samples:
        raise ValueError("Split partitions must cover every dataset record exactly once.")


def save_split_manifest(
    path, records, train_indices, val_indices, holdout_indices, seed, method,
):
    """Save exact image IDs so experiment splits can be reused and audited."""
    _record_id_map(records)
    partitions = (train_indices, val_indices, holdout_indices)
    _validate_split_indices(partitions, len(records))
    manifest = {
        "schema_version": 1,
        "seed": int(seed),
        "method": method,
        "splits": {
            name: [records[index]["id"] for index in indices]
            for name, indices in zip(("train", "val", "holdout"), partitions)
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def load_split_manifest(path, records):
    """Load a manifest by ID, allowing CSV reordering but rejecting set changes."""
    id_map = _record_id_map(records)
    with Path(path).open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported split manifest schema; expected version 1.")
    splits = manifest.get("splits")
    names = ("train", "val", "holdout")
    if not isinstance(splits, dict) or set(splits) != set(names):
        raise ValueError("Split manifest must contain train, val and holdout partitions.")
    if any(not isinstance(splits[name], list) for name in names):
        raise ValueError("Split manifest partitions must be lists of image IDs.")
    all_ids = [image_id for name in names for image_id in splits[name]]
    if any(not isinstance(image_id, str) for image_id in all_ids):
        raise ValueError("Split manifest IDs must be strings.")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Split manifest has duplicate or overlapping image IDs.")
    unknown_ids = set(all_ids) - set(id_map)
    missing_ids = set(id_map) - set(all_ids)
    if unknown_ids or missing_ids:
        raise ValueError(
            "Split manifest does not match the current dataset: "
            f"{len(unknown_ids)} unknown IDs and {len(missing_ids)} missing IDs."
        )
    return tuple([id_map[image_id] for image_id in splits[name]] for name in names)


def expand_rare_training_indices(records, train_indices, max_count, extra_copies):
    """Add per-image training draws based on positive counts in the original split."""
    if max_count == 0 or extra_copies == 0:
        return train_indices

    label_counts = Counter(
        label for index in train_indices for label in set(records[index]["labels"])
    )
    rare_indices = [
        index for index in train_indices
        if any(label_counts[label] <= max_count for label in records[index]["labels"])
    ]
    # Subset uses the original record, so every draw keeps all its labels and
    # HPAImageDataset samples a fresh synchronized transform when reading it.
    return list(train_indices) + rare_indices * extra_copies


def save_label_distribution(
    csv_path,
    records,
    train_indices,
    val_indices,
    holdout_indices,
):
    def count_labels(indices):
        counts = [0] * NUM_CLASSES
        for index in indices:
            for label in records[index]["labels"]:
                counts[label] += 1
        return counts

    all_indices = range(len(records))
    total_counts = count_labels(all_indices)
    train_counts = count_labels(train_indices)
    val_counts = count_labels(val_indices)
    holdout_counts = count_labels(holdout_indices)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Class", "Total", "Train", "Val", "Holdout"],
        )
        writer.writeheader()
        for class_index in range(NUM_CLASSES):
            writer.writerow(
                {
                    "Class": class_index,
                    "Total": total_counts[class_index],
                    "Train": train_counts[class_index],
                    "Val": val_counts[class_index],
                    "Holdout": holdout_counts[class_index],
                }
            )


def sample_records(records, num_samples, seed):
    if num_samples <= 0 or num_samples >= len(records):
        return records
    if num_samples < 10:
        raise ValueError("Debug mode requires at least 10 labeled images.")

    random_generator = random.Random(seed)
    selected_indices = random_generator.sample(
        range(len(records)), num_samples
    )
    return [records[index] for index in selected_indices]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset,
    indices,
    batch_size,
    shuffle,
    num_workers,
    seed,
    pin_memory,
):
    if indices is not None:
        dataset = Subset(dataset, indices)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
