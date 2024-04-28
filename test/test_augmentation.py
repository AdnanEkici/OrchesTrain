from __future__ import annotations

import random
import unittest

import albumentations as A
import cv2
import numpy as np
import yaml

from orchestrain import augments
from orchestrain import dataset


class TestAugmentations(unittest.TestCase):
    class FakeDataset(dataset.Dataset):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.input_image = np.random.randint(255, size=(1080, 1920), dtype=np.uint8)
            self.prev_image = np.random.randint(255, size=(1080, 1920), dtype=np.uint8)
            self.target_image = np.random.randint(255, size=(1080, 1920), dtype=np.uint8)
            self.target_image[self.target_image > 125] = 255
            self.target_image[self.target_image <= 125] = 0

        def __getitem__(self, idx):
            transformed = self.apply_augmentations(images=[self.input_image, self.prev_image], masks=[self.target_image], zoom_level="880")
            [input_image, prev_image] = self.images_to_tensors(transformed["images"])
            target_image = self.images_to_tensors(transformed["masks"])[0]

            return input_image, prev_image, target_image

    def test_custom_resize(self):
        config = yaml.safe_load(
            """
augment:
  CustomResize:
    zoom_levels:
      "587": [1920, 1072]
      "880": [1280, 720]
      "880x2": [640, 360]
""",
        )

        random.seed(41)
        dataset = self.FakeDataset(**config)
        input_transformed_w_dataset, prev_transformed_w_dataset, target_transformed_w_dataset = dataset.__getitem__(0)
        input_transformed_w_dataset = input_transformed_w_dataset.numpy().astype(np.uint8)[0]
        prev_transformed_w_dataset = prev_transformed_w_dataset.numpy().astype(np.uint8)[0]
        target_transformed_w_dataset = target_transformed_w_dataset.numpy().astype(np.uint8)[0]

        input_image = dataset.input_image
        prev_image = dataset.prev_image
        target_image = dataset.target_image

        transform = augments.CustomResize({"587": [1920, 1072], "880": [1280, 720], "880x2": [640, 360]})
        random.seed(41)
        data = transform(image=input_image, image1=prev_image, mask=target_image, zoom_level="880")

        input_diff = cv2.subtract(input_transformed_w_dataset, data["image"]).max()
        prev_diff = cv2.subtract(prev_transformed_w_dataset, data["image1"]).max()
        target_diff = cv2.subtract(target_transformed_w_dataset, data["mask"]).max()

        self.assertLess(input_diff, 10)
        self.assertLess(prev_diff, 10)
        self.assertLess(target_diff, 10)

    def test_albumentations_segmentation(self):
        config = yaml.safe_load(
            """
augment:
  A.Rotate:
    limit: [-45, 45]
    p: 1.0
  A.Perspective:
    p: 1.0
""",
        )
        random.seed(5)
        dataset = self.FakeDataset(**config)
        input_transformed_w_dataset, prev_transformed_w_dataset, target_transformed_w_dataset = dataset.__getitem__(0)
        input_transformed_w_dataset = input_transformed_w_dataset.numpy().astype(np.uint8)[0]
        prev_transformed_w_dataset = prev_transformed_w_dataset.numpy().astype(np.uint8)[0]
        target_transformed_w_dataset = target_transformed_w_dataset.numpy().astype(np.uint8)[0]

        transform = A.Compose(
            [A.Rotate(limit=(-45, 45), p=1.0), A.Perspective(p=1.0)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            additional_targets={"image1": "image"},
        )

        input_image = dataset.input_image
        prev_image = dataset.prev_image
        target_image = dataset.target_image

        random.seed(5)
        transformed = transform(image=input_image, image1=prev_image, mask=target_image, bboxes=[], class_labels=[])
        input_transformed_directly, prev_transformed_directly, target_transformed_directly = (
            transformed["image"],
            transformed["image1"],
            transformed["mask"],
        )

        input_diff = cv2.subtract(input_transformed_w_dataset, input_transformed_directly).max()
        prev_diff = cv2.subtract(prev_transformed_w_dataset, prev_transformed_directly).max()
        target_diff = cv2.subtract(target_transformed_w_dataset, target_transformed_directly).max()

        self.assertLess(input_diff, 10)
        self.assertLess(prev_diff, 10)
        self.assertLess(target_diff, 10)

    def test_custom_augmentation_with_albumentation(self):
        config = yaml.safe_load(
            """
augment:
  CustomResize:
    zoom_levels:
      "587": [1920, 1072]
      "880": [1280, 720]
      "880x2": [640, 360]
  A.Rotate:
    limit: [-12, -12]
    p: 1.0
""",
        )

        random.seed(41)
        dataset = self.FakeDataset(**config)
        input_transformed_w_dataset, prev_transformed_w_dataset, target_transformed_w_dataset = dataset.__getitem__(0)
        input_transformed_w_dataset = input_transformed_w_dataset.numpy().astype(np.uint8)[0]
        prev_transformed_w_dataset = prev_transformed_w_dataset.numpy().astype(np.uint8)[0]
        target_transformed_w_dataset = target_transformed_w_dataset.numpy().astype(np.uint8)[0]

        input_image = dataset.input_image
        prev_image = dataset.prev_image
        target_image = dataset.target_image

        transform = augments.CustomResize({"587": [1920, 1072], "880": [1280, 720], "880x2": [640, 360]})
        random.seed(41)
        transformed = transform(image=input_image, image1=prev_image, mask=target_image, zoom_level="880")
        a_transform = A.Compose(
            [A.Rotate(limit=(-12, -12), p=1.0)],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            additional_targets={"image1": "image"},
        )
        transformed = a_transform(
            image=transformed["image"],
            image1=transformed["image1"],
            mask=transformed["mask"],
            bboxes=[],
            class_labels=[],
        )

        input_transformed_directly, prev_transformed_directly, target_transformed_directly = (
            transformed["image"],
            transformed["image1"],
            transformed["mask"],
        )

        input_diff = cv2.subtract(input_transformed_w_dataset, input_transformed_directly).max()
        prev_diff = cv2.subtract(prev_transformed_w_dataset, prev_transformed_directly).max()
        target_diff = cv2.subtract(target_transformed_w_dataset, target_transformed_directly).max()

        self.assertLess(input_diff, 10)
        self.assertLess(prev_diff, 10)
        self.assertLess(target_diff, 10)

    # To Do test keypoint and bbox augmentation
