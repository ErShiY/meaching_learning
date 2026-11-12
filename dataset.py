import csv
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


class HPAImageDataset(Dataset):
    def __init__(self, records, image_dir, image_size, augment=False):
        self.records = records
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment

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

        # Flips are applied after stacking, so all four channels stay aligned.
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=(2,))
        if self.augment and random.random() < 0.5:
            image = torch.flip(image, dims=(1,))

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
