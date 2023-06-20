from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class BaseMetric:
    """The BaseMetric class is designed to be inherited from and extended to create custom metric functions."""

    def __init__(
        self,
        device_fn: Callable,
        log_fn: Callable,
        log_image_fn: Callable,
        log_prefix_fn: Callable,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.device_fn = device_fn
        self.log = log_fn
        self.log_image = log_image_fn
        self.log_prefix_fn = log_prefix_fn

    @property
    def log_prefix(self):
        return self.log_prefix_fn()


class Accuracy(BaseMetric):
    """Custom metric class for calculating accuracy in classification tasks.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Inherits
    --------
    BaseMetric
        Base metric class for custom metric classes.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor):
        predictions = predictions.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        predictions_labeled = np.argmax(predictions, axis=1)
        accuracy = np.sum(predictions_labeled == labels) / predictions.shape[0]

        return accuracy


class F1(BaseMetric):
    """Custom metric class for calculating f1 score in classification tasks.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Inherits
    --------
    BaseMetric
        Base metric class for custom metric classes.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor):
        """Calculates weighted, micro, and macro f1 scores for the given predictions and labels. Returns weighted F1 score
        and logs the others.

        Parameters
        ----------
        predictions: torch.tensor
            Predicted labels tensor.
        labels: torch.tensor
            Ground truth labels tensor.

        Returns
        -------
        float
            Weighted F1 score.

        Notes
        -----
        The `predictions` tensor should be an array of probabilities.
        The `labels` tensor should contain the ground truth labels.

        """

        labels_np = labels.cpu().numpy()
        predictions_labeled = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        if labels_np.shape != predictions_labeled.shape:
            raise RuntimeError("Dimensions of prediction and labels are not the same.")
        f1_weighted = f1_score(y_true=labels_np, y_pred=predictions_labeled, average="weighted")
        f1_micro = f1_score(y_true=labels_np, y_pred=predictions_labeled, average="micro")
        f1_macro = f1_score(y_true=labels_np, y_pred=predictions_labeled, average="macro")
        self.log(self.log_prefix + "F1_micro", f1_micro, sync_dist=True)
        self.log(self.log_prefix + "F1_macro", f1_macro, sync_dist=True)
        return f1_weighted


class Precision(BaseMetric):
    """Custom metric class for calculating precision score in classification tasks.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Inherits
    --------
    BaseMetric
        Base metric class for custom metric classes.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor):
        """Calculates weighted, micro, and macro precision scores for the given predictions and labels. Returns weighted precision score
        and logs the others.

        Parameters
        ----------
        predictions: torch.tensor
            Predicted labels tensor.
        labels: torch.tensor
            Ground truth labels tensor.

        Returns
        -------
        float
            Weighted precision score.

        Notes
        -----
        The `predictions` tensor should be an array of probabilities.
        The `labels` tensor should contain the ground truth labels.

        """

        labels_np = labels.cpu().numpy()
        predictions_labeled = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        if labels_np.shape != predictions_labeled.shape:
            raise RuntimeError("Dimensions of prediction and labels are not the same.")
        precision_weighted = precision_score(y_true=labels_np, y_pred=predictions_labeled, average="weighted")
        precision_micro = precision_score(y_true=labels_np, y_pred=predictions_labeled, average="micro")
        precision_macro = precision_score(y_true=labels_np, y_pred=predictions_labeled, average="macro")
        self.log(self.log_prefix + "Precision_micro", precision_micro, sync_dist=True)
        self.log(self.log_prefix + "Precision_macro", precision_macro, sync_dist=True)
        return precision_weighted


class Recall(BaseMetric):
    """Custom metric class for calculating recall score in classification tasks.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Inherits
    --------
    BaseMetric
        Base metric class for custom metric classes.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor):
        """Calculates weighted, micro, and macro recall scores for the given predictions and labels. Returns weighted recall score
        and logs the others.
        Parameters
        ----------
        predictions: torch.tensor
            Predicted labels tensor.
        labels: torch.tensor
            Ground truth labels tensor.

        Returns
        -------
        float
            Weighted recall score.

        Notes
        -----
        The `predictions` tensor should be an array of probabilities.
        The `labels` tensor should contain the ground truth labels.

        """

        labels_np = labels.cpu().numpy()
        predictions_labeled = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        if labels_np.shape != predictions_labeled.shape:
            raise RuntimeError("Dimensions of prediction and labels are not the same.")
        recall_weighted = recall_score(y_true=labels_np, y_pred=predictions_labeled, average="weighted")
        recall_micro = recall_score(y_true=labels_np, y_pred=predictions_labeled, average="micro")
        recall_macro = recall_score(y_true=labels_np, y_pred=predictions_labeled, average="macro")
        self.log(self.log_prefix + "Recall_micro", recall_micro, sync_dist=True)
        self.log(self.log_prefix + "Recall_macro", recall_macro, sync_dist=True)
        return recall_weighted
