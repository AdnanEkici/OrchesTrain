from __future__ import annotations

import random
import unittest

import albumentations as A
import cv2
import numpy as np


class TestAlbumentations(unittest.TestCase):
    # Voc format is suitable for cv2 rectangle
    def yolo_to_voc_format(self, bbox, img_w, img_h):
        x0 = int((bbox[0] - bbox[2] / 2) * img_w)
        y0 = int((bbox[1] - bbox[3] / 2) * img_h)
        x1 = int((bbox[0] + bbox[2] / 2) * img_w)
        y1 = int((bbox[1] + bbox[3] / 2) * img_h)

        return x0, y0, x1, y1

    def load_sample_yolo_image(self):
        img = np.random.randint(255, size=(1080, 1920), dtype=np.uint8)
        bbox = [0.3, 0.4, 0.1, 0.1]

        return img, bbox

    def test_reproducable_augmentation_with_seed(self):
        seed = random.randint(1, 9999)

        img, bbox = self.load_sample_yolo_image()

        transform = A.Compose(
            [
                A.Rotate(limit=(-90, 90), p=1),
                A.HorizontalFlip(p=1),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

        # Albumentations uses python random, so we will set random.seed
        random.seed(seed)
        transformed1 = transform(image=img, bboxes=[bbox], class_labels=[0])

        random.seed(seed)
        transformed2 = transform(image=img, bboxes=[bbox], class_labels=[0])

        error = cv2.subtract(transformed1["image"], transformed2["image"]).sum()

        self.assertEqual(error, 0)
        self.assertListEqual(transformed1["bboxes"], transformed2["bboxes"])

    def test_compare_rotation_with_opencv(self):
        img, bbox = self.load_sample_yolo_image()
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        bbox_voc = self.yolo_to_voc_format(bbox, w, h)

        cv2.rectangle(mask, (bbox_voc[0], bbox_voc[1]), (bbox_voc[2], bbox_voc[3]), (255, 255, 255), -1)

        transform = A.Compose(
            [
                A.Rotate(limit=(-37, -37), p=1),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

        transformed = transform(image=img, mask=mask, bboxes=[bbox], class_labels=[0])

        M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), -37, 1)
        rotated_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        img_diff = cv2.subtract(transformed["image"], rotated_img).max()
        mask_diff = cv2.subtract(transformed["mask"], rotated_mask).max()

        self.assertLess(img_diff, 5)
        self.assertLess(mask_diff, 5)
