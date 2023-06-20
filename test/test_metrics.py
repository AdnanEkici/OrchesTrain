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


def prefix_fn():
    return "prefix_"


def log_fn(log_header, value, sync_dist=True):
    print(log_header, value)


class TestMetrics(unittest.TestCase):
    test_parameters = [(1, 1), (2, 2), (3, 3), (5, 5), (7, 5), (9, 5), (11, 5), (100, 5)]

    @parameterized.expand(test_parameters)
    def test_accuracy(self, n_samples, n_classes):
        labels = randomLabel(n_classes=n_classes, n_samples=n_samples)
        predictions = randomPrediction(n_classes=n_classes, n_samples=n_samples)

        # Calculate accuracy using metrics
        accuracy_metric = metrics.Accuracy(device_fn=None, log_image_fn=None, log_fn=log_fn, log_prefix_fn=prefix_fn)
        accuracy_calculated = accuracy_metric(predictions=predictions, labels=labels)

        predictions_np = predictions.numpy()
        labels_np = labels.numpy()

        predicted_labels = np.argmax(predictions_np, axis=1)

        accuracy_expected = np.sum(predicted_labels == labels_np) / len(labels_np)

        self.assertTrue(math.isclose(accuracy_calculated, accuracy_expected))

    @parameterized.expand(test_parameters)
    def test_F1(self, n_samples, n_classes):
        labels = randomLabel(n_classes=n_classes, n_samples=n_samples)
        predictions = randomPrediction(n_classes=n_classes, n_samples=n_samples)

        # Calculate f1 score using metrics
        f1_metric = metrics.F1(device_fn=None, log_image_fn=None, log_fn=log_fn, log_prefix_fn=prefix_fn)
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
        self.assertTrue(math.isclose(f1_calculated, f1_expected))

    @parameterized.expand(test_parameters)
    def test_Precision(self, n_samples, n_classes):
        labels = randomLabel(n_classes=n_classes, n_samples=n_samples)
        predictions = randomPrediction(n_classes=n_classes, n_samples=n_samples)

        precision_metric = metrics.Precision(device_fn=None, log_image_fn=None, log_fn=log_fn, log_prefix_fn=prefix_fn)
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
        self.assertTrue(math.isclose(precision_calculated, precision_expected))

    @parameterized.expand(test_parameters)
    def test_Recall(self, n_samples, n_classes):
        labels = randomLabel(n_classes=n_classes, n_samples=n_samples)
        predictions = randomPrediction(n_classes=n_classes, n_samples=n_samples)

        recall_metric = metrics.Recall(device_fn=None, log_image_fn=None, log_fn=log_fn, log_prefix_fn=prefix_fn)
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
        self.assertTrue(math.isclose(recall_calculated, recall_expected))
