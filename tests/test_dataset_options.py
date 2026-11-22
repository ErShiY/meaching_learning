from contextlib import contextmanager
import json
from pathlib import Path
import random
import shutil
import unittest
import uuid

import numpy as np
from PIL import Image
import torch

from dataset import (
    CHANNELS,
    HPAImageDataset,
    augment_image,
    expand_rare_training_indices,
    iterative_multilabel_split,
    load_split_manifest,
    save_split_manifest,
    split_indices,
)


@contextmanager
def temporary_workspace_directory():
    # Windows restricted tokens may not reopen tempfile's mode-0700 directories.
    # Use an inherited-permission directory inside the known writable workspace.
    parent = Path(__file__).resolve().parent
    directory = parent / f"dataset_fixture_{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        if resolved.parent != parent or not resolved.name.startswith("dataset_fixture_"):
            raise RuntimeError("Unexpected test fixture cleanup path.")
        shutil.rmtree(resolved)


class DatasetAugmentationTests(unittest.TestCase):
    def test_basic_matches_legacy_output_and_rng_state(self):
        image = torch.arange(4 * 8 * 8).reshape(4, 8, 8).float() / 255
        for seed in range(10):
            random.seed(seed)
            expected = image
            if random.random() < 0.5:
                expected = torch.flip(expected, dims=(2,))
            if random.random() < 0.5:
                expected = torch.flip(expected, dims=(1,))
            expected_state = random.getstate()
            random.seed(seed)
            actual = augment_image(image, "basic")
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(random.getstate(), expected_state)

    def test_microscopy_preserves_channel_alignment_and_range(self):
        channel = torch.linspace(0, 1, 64).reshape(8, 8)
        image = channel.unsqueeze(0).repeat(4, 1, 1)
        random.seed(42)
        actual = augment_image(image, "microscopy")
        self.assertEqual(tuple(actual.shape), (4, 8, 8))
        self.assertGreaterEqual(actual.min(), 0)
        self.assertLessEqual(actual.max(), 1)
        self.assertTrue(torch.isfinite(actual).all())
        for channel_index in range(1, 4):
            torch.testing.assert_close(actual[0], actual[channel_index], atol=0, rtol=0)

    def test_validation_ignores_augmentation_policy_and_consumes_no_rng(self):
        with temporary_workspace_directory() as temporary_dir:
            pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
            for channel in CHANNELS:
                Image.fromarray(pixels).save(Path(temporary_dir) / f"sample_{channel}.png")
            dataset = HPAImageDataset(
                [{"id": "sample", "labels": [1]}], temporary_dir, 8,
                augment=False, augmentation="microscopy",
            )
            random.seed(9)
            expected_state = random.getstate()
            actual = dataset[0]
            self.assertEqual(random.getstate(), expected_state)
            expected = (torch.from_numpy(pixels.copy()).float() / 255 - 0.5) / 0.5
            torch.testing.assert_close(actual["image"][0], expected, atol=0, rtol=0)
            self.assertEqual(actual["target"][1].item(), 1)

    def test_rejects_unknown_policy(self):
        with self.assertRaisesRegex(ValueError, "Unknown augmentation"):
            augment_image(torch.zeros(4, 8, 8), "crop_everything")


class RareTrainingExpansionTests(unittest.TestCase):
    def test_counts_only_training_positives_and_copies_each_image_once_per_repeat(self):
        records = [
            {"id": "common-a", "labels": [0]},
            {"id": "rare-a", "labels": [0, 1, 2]},
            {"id": "val", "labels": [1]},
            {"id": "common-b", "labels": [0]},
            {"id": "holdout", "labels": [1, 2]},
            {"id": "rare-b", "labels": [0, 1]},
        ]
        train_indices = [3, 0, 5, 1]

        expanded = list(expand_rare_training_indices(records, train_indices, 2, 2))

        # Class 1 appears in two training images and two evaluation images.
        # The image with two rare labels still gets only two extra copies.
        self.assertEqual(train_indices, [3, 0, 5, 1])
        self.assertEqual(expanded[:len(train_indices)], train_indices)
        self.assertCountEqual(expanded[len(train_indices):], [1, 1, 5, 5])
        self.assertEqual(records[1]["labels"], [0, 1, 2])
        self.assertEqual(records[5]["labels"], [0, 1])

    def test_either_zero_option_disables_expansion(self):
        records = [{"id": "rare", "labels": [1]}]
        for max_count, extra_copies in ((0, 2), (1, 0)):
            with self.subTest(max_count=max_count, extra_copies=extra_copies):
                expanded = expand_rare_training_indices(
                    records, [0], max_count, extra_copies
                )
                self.assertEqual(list(expanded), [0])


class MultiLabelSplitTests(unittest.TestCase):
    @staticmethod
    def records(count=1000):
        return [
            {
                "id": str(index),
                "labels": [0] + ([1] if index % 10 == 0 else [])
                + ([2] if index < 11 else []),
            }
            for index in range(count)
        ]

    def assert_partition_coverage(self, partitions, total):
        flattened = [index for partition in partitions for index in partition]
        self.assertEqual(len(flattened), total)
        self.assertEqual(len(set(flattened)), total)
        self.assertEqual(sorted(flattened), list(range(total)))
        self.assertEqual(
            list(map(len, partitions)),
            [int(total * 0.7), int(total * 0.2), total - int(total * 0.7) - int(total * 0.2)],
        )

    def test_seeded_stratification_is_repeatable_and_proportional(self):
        records = self.records()
        partitions = iterative_multilabel_split(records, 42)
        self.assertEqual(partitions, iterative_multilabel_split(records, 42))
        self.assertNotEqual(partitions, iterative_multilabel_split(records, 43))
        self.assert_partition_coverage(partitions, len(records))
        for label, total in ((0, 1000), (1, 100), (2, 11)):
            for indices, fraction in zip(partitions, (0.7, 0.2, 0.1)):
                count = sum(label in records[index]["labels"] for index in indices)
                self.assertLessEqual(abs(count - total * fraction), 1)

    def test_empty_unlabeled_and_tiny_inputs(self):
        for count in (0, 1, 2, 3, 9, 10, 19):
            records = [{"id": str(index), "labels": []} for index in range(count)]
            self.assert_partition_coverage(iterative_multilabel_split(records, 1), count)
            if records:
                records[-1]["labels"] = [27]
                self.assert_partition_coverage(iterative_multilabel_split(records, 1), count)

    def test_multilabel_cooccurrence_and_capacity_conflicts(self):
        records = [
            {"id": str(index), "labels": list(range(index % 28))}
            for index in range(125)
        ]
        partitions = iterative_multilabel_split(records, 7)
        self.assert_partition_coverage(partitions, len(records))

    def test_legacy_random_split_is_unchanged(self):
        indices = list(range(1000))
        random.Random(42).shuffle(indices)
        self.assertEqual(split_indices(1000, 42), (indices[:700], indices[700:900], indices[900:]))

    def test_rejects_out_of_range_labels(self):
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            iterative_multilabel_split([{"id": "one", "labels": [28]}], 42)

    def test_manifest_roundtrip_survives_csv_reordering(self):
        records = self.records(100)
        partitions = iterative_multilabel_split(records, 42)
        with temporary_workspace_directory() as temporary_dir:
            path = Path(temporary_dir) / "splits.json"
            save_split_manifest(path, records, *partitions, seed=42, method="stratified")
            self.assertEqual(load_split_manifest(path, records), partitions)
            shuffled_records = list(reversed(records))
            restored = load_split_manifest(path, shuffled_records)
            for previous_indices, restored_indices in zip(partitions, restored):
                self.assertEqual(
                    [records[index]["id"] for index in previous_indices],
                    [shuffled_records[index]["id"] for index in restored_indices],
                )

    def test_manifest_rejects_duplicate_missing_and_changed_ids(self):
        records = self.records(100)
        partitions = iterative_multilabel_split(records, 42)
        with temporary_workspace_directory() as temporary_dir:
            path = Path(temporary_dir) / "splits.json"
            save_split_manifest(path, records, *partitions, seed=42, method="stratified")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_split_manifest(path, records[:-1])
            changed = records[:-1] + [{"id": "new_image", "labels": [1]}]
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_split_manifest(path, changed)
            with path.open(encoding="utf-8") as file:
                manifest = json.load(file)
            manifest["splits"]["val"].append(manifest["splits"]["train"][0])
            with path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file)
            with self.assertRaisesRegex(ValueError, "duplicate or overlapping"):
                load_split_manifest(path, records)
            with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
                save_split_manifest(path, records + records[:1], *partitions, seed=42, method="random")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                save_split_manifest(path, records, partitions[0] + partitions[0][:1], *partitions[1:], seed=42, method="random")


if __name__ == "__main__":
    unittest.main()
