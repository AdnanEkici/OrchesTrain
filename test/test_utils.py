from __future__ import annotations

import unittest

import cv2
import numpy as np
from parameterized import parameterized

from orchestrain.utils import calculate_precision_recall
from orchestrain.utils import mask_2_bbox


class TestUtils(unittest.TestCase):
    test_cases = [
        # single rectangle
        ({"inputs": [(100, 100, 250, 250)], "outputs": [(106, 106, 344, 344)]},),
        # two rectangle shares top left corner
        ({"inputs": [(100, 100, 250, 250), (100, 100, 200, 200)], "outputs": [(106, 106, 294, 294)]},),
        # single circle
        ({"inputs": [(500, 500, 30)], "outputs": [(478, 478, 523, 523)]},),
        # two circles with equal radius intersecting each other at their centers.
        ({"inputs": [(500, 500, 30), (470, 500, 30)], "outputs": [[478, 487, 493, 514], [504, 481, 523, 520], [448, 481, 467, 520]]},),
        # a circle and a rectangle, half of the circle intersects with the rectangle
        ({"inputs": [(500, 500, 30), (470, 500, 300, 300)], "outputs": [[482, 506, 519, 523], [482, 478, 519, 495]]},),
    ]

    @parameterized.expand(test_cases)
    def test_mask_2_bbox(self, test):
        inputs, outputs = test.values()
        white = (255, 255, 255)
        image = np.zeros((1000, 1000), dtype=np.uint8)
        for input in inputs:
            cv2.rectangle(image, input, white, 1) if len(input) == 4 else cv2.circle(image, (input[0], input[1]), input[2], white, 1)
        calculated_bboxes = mask_2_bbox([image])[0]
        for i, bbox in enumerate(outputs):
            for j, expected in enumerate(bbox):
                self.assertEqual(expected, calculated_bboxes[i][j])

    precision_recall_parameters = [
        # Perfect match 00
        (
            {
                "bbox_true": [(0, 0, 2, 2), (3, 3, 5, 5)],
                "bbox_pred": [(0, 0, 2, 2), (3, 3, 5, 5)],
                "iou_threshold": 0.5,
                "precision_recall": (1, 1),
                "message": "There should be a perfect match",
            },
        ),
        # Negative groundtruth boxes 01
        (
            {
                "bbox_true": [(-2, -2, 2, 2)],
                "bbox_pred": [(0, 0, 3, 3)],
                "iou_threshold": 0.5,
                "precision_recall": (1, 1),
                "message": "Negative values will be set to zero, there should be a match",
            },
        ),
        # Negative groundtruth boxes higher iou threshold 02
        (
            {
                "bbox_true": [(-2, -2, 2, 2)],
                "bbox_pred": [(0, 0, 3, 3)],
                "iou_threshold": 0.95,
                "precision_recall": (0, 0),
                "message": "Negative values will be set to zero, there should not be any match",
            },
        ),
        # Negative prediction boxes 03
        (
            {
                "bbox_true": [(2, 2, 2, 2)],
                "bbox_pred": [(0, 0, -3, -3)],
                "iou_threshold": 0.5,
                "precision_recall": (0, 0),
                "message": "Metric should be zero for negative boxes",
            },
        ),
        # Prediction zero 04
        (
            {
                "bbox_true": [(0, 0, 2, 2)],
                "bbox_pred": [(0, 0, 0, 0)],
                "iou_threshold": 0.5,
                "precision_recall": (0, 0),
                "message": "Value should be zero for zero boxes",
            },
        ),
        # Groundtruth zero 05
        (
            {
                "bbox_true": [(0, 0, 0, 0)],
                "bbox_pred": [(0, 0, 2, 2)],
                "iou_threshold": 0.5,
                "precision_recall": (0, 0),
                "message": "Value should be zero for negative boxes",
            },
        ),
        # Multiple false positives 06
        (
            {
                "bbox_true": [(3, 3, 5, 5)],
                "bbox_pred": [(3, 3, 5, 5), (6, 6, 8, 8), (9, 9, 11, 11)],
                "iou_threshold": 0.5,
                "precision_recall": (0.33, 1),
                "message": "One of the predictions is correct",
            },
        ),
        # Single prediction, multiple groundtruths 07
        (
            {
                "bbox_true": [(2, 0, 5, 3), (0, 0, 3, 3), (1, 2, 4, 5)],
                "bbox_pred": [(1, 1, 3, 3), (2, 2, 3, 3)],
                "iou_threshold": 0.5,
                "precision_recall": (0.5, 0.33),
                "message": "Prediction is correct for one of the groundtruths",
            },
        ),
        # Single prediction, multiple groundtruths lower iou threshold 08
        (
            {
                "bbox_true": [(2, 0, 5, 3), (0, 0, 3, 3), (1, 2, 4, 5)],
                "bbox_pred": [(1, 1, 3, 3), (2, 2, 3, 3)],
                "iou_threshold": 0.25,
                "precision_recall": (1.0, 0.6666666666666666),
                "message": "Prediction is correct for one of the groundtruths",
            },
        ),
        # Single prediction, multiple groundtruths higher iou threshold 09
        (
            {
                "bbox_true": [(2, 0, 5, 3), (0, 0, 3, 3), (1, 2, 4, 5)],
                "bbox_pred": [(1, 1, 3, 3), (2, 2, 3, 3)],
                "iou_threshold": 0.75,
                "precision_recall": (0.0, 0.0),
                "message": "Prediction is correct for one of the groundtruths",
            },
        ),
        # Empty groundtruth 10
        (
            {
                "bbox_true": [],
                "bbox_pred": [(0, 0, 2, 2), (3, 3, 5, 5)],
                "iou_threshold": 0.5,
                "precision_recall": (0, -1),
                "message": "There is no groundtruth, recall must be -1",
            },
        ),
        # Empty prediction 11
        (
            {
                "bbox_true": [(0, 0, 2, 2), (3, 3, 5, 5)],
                "bbox_pred": [],
                "iou_threshold": 0.5,
                "precision_recall": (-1, 0),
                "message": "There is no prediction, precision must be -1",
            },
        ),
        # Empty bboxes 12
        (
            {
                "bbox_true": [],
                "bbox_pred": [],
                "iou_threshold": 0.5,
                "precision_recall": (-1, -1),
                "message": "There is no prediction and  groundtruth, result must be -1",
            },
        ),
    ]

    @parameterized.expand(precision_recall_parameters)
    def test_calculate_precision_recall(self, parameters):
        bbox_true, bbox_pred, iou_threshold, precision_recall, message = parameters.values()
        precision, recall = calculate_precision_recall(bbox_true, bbox_pred, iou_threshold)
        expected_precision, expected_recall = precision_recall
        self.assertAlmostEqual(precision, expected_precision, places=2, msg=message)
        self.assertAlmostEqual(recall, expected_recall, places=2, msg=message)
