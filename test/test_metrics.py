from __future__ import annotations

import math
import unittest

import numpy as np
import torch
from parameterized import parameterized

import orchestrain.loss.metrics as metrics


def randomLabel(n_classes, n_samples):
    return (torch.rand(n_samples) * n_classes).int()


def randomPrediction(n_classes, n_samples):
    predictions = torch.rand((n_samples, n_classes))
    predictions = torch.nn.functional.softmax(predictions, dim=0)
    return predictions


def confusion_matrix(y_true, y_pred, N):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y = N * y_true + y_pred
    y = np.bincount(y, minlength=N * N)
    y = y.reshape(N, N)
    return y


class TestParameters:
    classification_parameters = [
        ({"n_samples": 1}, {"n_classes": 1}),
        ({"n_samples": 2}, {"n_classes": 2}),
        ({"n_samples": 10}, {"n_classes": 10}),
        ({"n_samples": 100}, {"n_classes": 10}),
        ({"n_samples": 10}, {"n_classes": 100}),
        ({"n_samples": 83}, {"n_classes": 7}),
        ({"n_samples": 3}, {"n_classes": 5}),
    ]
    segmentation_parameters = [
        [
            {"prediction": torch.tensor([[[0.8, 0.6, 0.4, 0.2], [0.0, 0.0, 1.0, 1.0]]])},
            {"label": torch.tensor([[0, 0, 1, 1]])},
            {"accuracy": 1, "iou": 1},
        ],
        [
            {"prediction": torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])},
            {"label": torch.tensor([[1, 1, 0, 0]])},
            {"accuracy": 0, "iou": 0},
        ],
        [
            {"prediction": torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])},
            {"label": torch.tensor([[1, 1, 0, 0]])},
            {"accuracy": 0.5, "iou": 1 / 3},
        ],
        [
            {"prediction": torch.tensor([[[0.5, 0.3, 0.9, 0.0], [0.3, 0.3, 0.1, 0.8], [0.2, 0.4, 0.0, 0.2]]])},
            {"label": torch.tensor([[0, 2, 0, 1]])},
            {"accuracy": 1, "iou": 1},
        ],
        [
            {"prediction": torch.tensor([[[0.5, 0.3, 0.9, 0.0], [0.3, 0.3, 0.1, 0.8], [0.2, 0.4, 0.0, 0.2]]])},
            {"label": torch.tensor([[0, 1, 0, 1]])},
            {"accuracy": 0.75, "iou": 0.5},
        ],
        [
            {"prediction": torch.tensor([[[0.5, 0.3, 0.9, 0.0, 0.3], [0.3, 0.3, 0.1, 0.8, 0.2], [0.2, 0.4, 0.0, 0.2, 0.5]]])},
            {"label": torch.tensor([[1, -1, -1, 1, 2]])},
            {"accuracy": 2 / 3, "iou": 1 / 3},
        ],
        [
            {"prediction": torch.tensor([[[0.5, 0.3, 0.9, 0.0, 0.3], [0.3, 0.3, 0.1, 0.8, 0.2], [0.2, 0.4, 0.0, 0.2, 0.5]]])},
            {"label": torch.tensor([[-1, -1, -1, -1, -1]])},
            {"accuracy": 0, "iou": 0},
        ],
        [
            {"prediction": torch.tensor([[[0.5, 0.3, 0.9, 0.0, 0.3], [0.3, 0.2, 0.1, 0.8, 0.1], [0.2, 0.4, 0.0, 0.2, 0.5], [0, 0, 0.1, 0, 0.1]]])},
            {"label": torch.tensor([[1, -1, -1, 1, 2]])},
            {"accuracy": 2 / 3, "iou": 1 / 3},
        ],
    ]


class TestMetrics(unittest.TestCase):
    @parameterized.expand(TestParameters.classification_parameters)
    def test_accuracy_calculation(self, n_samples, n_classes):
        n_samples = n_samples["n_samples"]
        n_classes = n_classes["n_classes"]
        labels = randomLabel(n_classes=n_classes, n_samples=n_samples)
        predictions = randomPrediction(n_classes=n_classes, n_samples=n_samples)

        # Calculate accuracy using metrics
        accuracy_metric = metrics.Accuracy(device_fn=None, log_image_fn=None, log_fn=lambda a, b, sync_dist: True, log_prefix_fn=lambda: "")
        accuracy_calculated = accuracy_metric(predictions=predictions, labels=labels)

        predictions_np = predictions.numpy()
        labels_np = labels.numpy()

        predicted_labels = np.argmax(predictions_np, axis=1)

        accuracy_expected = np.sum(predicted_labels == labels_np) / len(labels_np)

        self.assertTrue(math.isclose(accuracy_calculated, accuracy_expected), msg="Calculated accuracy  is not close enough to expected.")

    @parameterized.expand(TestParameters.classification_parameters)
    def test_f1_calculation(self, n_samples, n_classes):
        n_samples = n_samples["n_samples"]
        n_classes = n_classes["n_classes"]
        labels = randomLabel(n_classes=n_classes, n_samples=n_samples)
        predictions = randomPrediction(n_classes=n_classes, n_samples=n_samples)

        # Calculate f1 score using metrics
        f1_metric = metrics.F1(device_fn=None, log_image_fn=None, log_fn=lambda a, b, sync_dist: True, log_prefix_fn=lambda: "", n_classes=n_classes)
        f1_calculated = f1_metric(labels=labels, predictions=predictions)

        predictions_np = predictions.numpy()
        labels_np = labels.numpy()

        predicted_labels = np.argmax(predictions_np, axis=1)

        conf_matrix = confusion_matrix(labels_np, predicted_labels, N=n_classes)
        weights = np.array([((labels_np == i).sum() / n_samples) for i in range(n_classes)])

        f1_expected = 0
        for i in range(n_classes):
            recall_denominator = conf_matrix[i].sum()
            precision_denominator = conf_matrix[:, i].sum()
            precision = conf_matrix[i][i] / precision_denominator if precision_denominator > 0 else 0
            recall = conf_matrix[i][i] / recall_denominator if recall_denominator > 0 else 0
            f1_expected += weights[i] * 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        self.assertTrue(math.isclose(f1_calculated, f1_expected), msg="Calculated f1 score  is not close enough to expected.")

    @parameterized.expand(TestParameters.classification_parameters)
    def test_precision_calculation(self, n_samples, n_classes):
        n_samples = n_samples["n_samples"]
        n_classes = n_classes["n_classes"]
        labels = randomLabel(n_classes=n_classes, n_samples=n_samples)
        predictions = randomPrediction(n_classes=n_classes, n_samples=n_samples)

        precision_metric = metrics.Precision(
            device_fn=None, log_image_fn=None, log_fn=lambda a, b, sync_dist: True, log_prefix_fn=lambda: "", n_classes=n_classes
        )
        precision_calculated = precision_metric(labels=labels, predictions=predictions)

        predictions_np = predictions.numpy()
        labels_np = labels.numpy()

        predicted_labels = np.argmax(predictions_np, axis=1)
        conf_matrix = confusion_matrix(labels_np, predicted_labels, n_classes)

        weights = np.array([((labels_np == i).sum() / n_samples) for i in range(n_classes)])

        precision_expected = 0
        for i in range(n_classes):
            denominator = conf_matrix[:, i].sum()
            precision_expected += weights[i] * (conf_matrix[i][i] / denominator) if denominator > 0 else 0
        self.assertTrue(math.isclose(precision_calculated, precision_expected), msg="Calculated precision  is not close enough to expected.")

    @parameterized.expand(TestParameters.classification_parameters)
    def test_recall_calculation(self, n_samples, n_classes):
        n_samples = n_samples["n_samples"]
        n_classes = n_classes["n_classes"]
        labels = randomLabel(n_classes=n_classes, n_samples=n_samples)
        predictions = randomPrediction(n_classes=n_classes, n_samples=n_samples)

        recall_metric = metrics.Recall(
            device_fn=None, log_image_fn=None, log_fn=lambda a, b, sync_dist: True, log_prefix_fn=lambda: "", n_classes=n_classes
        )
        recall_calculated = recall_metric(labels=labels, predictions=predictions)

        predictions_np = predictions.numpy()
        labels_np = labels.numpy()

        predicted_labels = np.argmax(predictions_np, axis=1)
        conf_matrix = confusion_matrix(labels_np, predicted_labels, n_classes)

        weights = np.array([((labels_np == i).sum() / n_samples) for i in range(n_classes)])

        recall_expected = 0
        for i in range(n_classes):
            denominator = conf_matrix[i].sum()
            recall_expected += weights[i] * (conf_matrix[i][i] / denominator) if denominator > 0 else 0
        self.assertTrue(math.isclose(recall_calculated, recall_expected), msg="Calculated recall  is not close enough to expected.")

    @parameterized.expand(TestParameters.segmentation_parameters)
    def test_pixel_accuracy_calculation(self, predictions, labels, expected_loss):
        pixel_accuracy_metric = metrics.PixelAccuracy(
            device_fn=None, log_image_fn=None, log_fn=lambda a, b, sync_dist: True, log_prefix_fn=lambda: ""
        )
        pixel_accuracy_calculated = pixel_accuracy_metric(predictions["prediction"], labels["label"])
        self.assertAlmostEqual(
            pixel_accuracy_calculated,
            expected_loss["accuracy"],
            msg="Calculated pixel accuracy is not close enough to expected pixel accuracy.",
        )

    @parameterized.expand(TestParameters.segmentation_parameters)
    def test_iou_calculation(self, predictions, labels, expected_loss):
        iou_metric = metrics.IoU(device_fn=None, log_image_fn=None, log_fn=lambda a, b, sync_dist: True, log_prefix_fn=lambda: "")
        iou_calculated = iou_metric(predictions["prediction"], labels["label"])
        self.assertAlmostEqual(iou_calculated, expected_loss["iou"], msg="Calculated IoU is not close enough to expected IoU.")
