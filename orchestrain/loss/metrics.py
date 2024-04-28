from __future__ import annotations

from typing import Any
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
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
        self.device_fn = device_fn
        self.log = log_fn
        self.log_image = log_image_fn
        self.log_prefix_fn = log_prefix_fn

    @property
    def log_prefix(self):
        return self.log_prefix_fn()

    @log_prefix.setter
    def log_prefix(self, value):
        self.log_prefix = value

    def on_epoch_end(self):
        pass


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

        return torch.tensor(accuracy)


class ClassWiseMetricBase(BaseMetric):
    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = range(n_classes)
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This metric class is created to be a backbone for class wise metric calculators. You cannot call it.")

    def _get_confusion_matrix(self, predictions: torch.tensor, labels: torch.tensor) -> torch.tensor:
        preds = torch.softmax(predictions, dim=1).argmax(dim=1).squeeze().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        if labels.size == 1:
            labels = np.array([labels])
            preds = np.array([preds])
        matrix = confusion_matrix(labels, preds, labels=self.labels)
        return matrix


class ClassAccuracy(ClassWiseMetricBase):
    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(n_classes, *args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor) -> torch.tensor:
        self.confusion_matrix += self._get_confusion_matrix(predictions=predictions, labels=labels)
        for idx, row in enumerate(self.confusion_matrix):
            self.log(self.log_prefix + f"Accuracy_of_{idx}", row[idx] / row.sum(), sync_dist=True, on_epoch=True, on_step=False)
        return torch.tensor(-1.0)

    def on_epoch_end(self):
        self.confusion_matrix = np.zeros_like(self.confusion_matrix)


class ClassF1(ClassWiseMetricBase):
    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(n_classes, *args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor) -> torch.tensor:
        self.confusion_matrix += self._get_confusion_matrix(predictions=predictions, labels=labels)
        tp = self.confusion_matrix.diagonal()
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        f1 = 2 * tp / (2 * tp + fp + fn + np.finfo(float).eps)
        for idx, f1_score_i in enumerate(f1):
            self.log(self.log_prefix + f"F1_of_{idx}", f1_score_i, sync_dist=True, on_epoch=True, on_step=False)
        return torch.tensor(-1.0)

    def on_epoch_end(self):
        self.confusion_matrix = np.zeros_like(self.confusion_matrix)


class ClassRecall(ClassWiseMetricBase):
    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(n_classes=n_classes, *args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor) -> torch.tensor:
        self.confusion_matrix += self._get_confusion_matrix(predictions=predictions, labels=labels)
        tp = self.confusion_matrix.diagonal()
        fn = self.confusion_matrix.sum(axis=1) - tp
        recall = tp / (tp + fn + np.finfo(float).eps)
        for idx, recall_score_i in enumerate(recall):
            self.log(self.log_prefix + f"Recall_of_{idx}", recall_score_i, sync_dist=True, on_epoch=True, on_step=False)
        return torch.tensor(-1.0)

    def on_epoch_end(self, matrix):
        self.confusion_matrix = np.zeros_like(self.confusion_matrix)


class ClassPrecision(ClassWiseMetricBase):
    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(n_classes=n_classes, *args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor) -> torch.tensor:
        self.confusion_matrix += self._get_confusion_matrix(predictions=predictions, labels=labels)
        tp = self.confusion_matrix.diagonal()
        fp = self.confusion_matrix.sum(axis=0) - tp
        precision = tp / (tp + fp + np.finfo(float).eps)
        for idx, precision_score_i in enumerate(precision):
            self.log(self.log_prefix + f"Precision_of_{idx}", precision_score_i, sync_dist=True, on_epoch=True, on_step=False)
        return torch.tensor(-1.0)

    def on_epoch_end(self):
        self.confusion_matrix = np.zeros_like(self.confusion_matrix)


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

    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = range(n_classes)
        self.confusion_matrix = np.zeros((n_classes, n_classes))

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
        torch.Tensor
            Weighted F1 score.

        Notes
        -----
        The `predictions` tensor should be an array of probabilities.
        The `labels` tensor should contain the ground truth labels.

        """
        self.confusion_matrix += self._get_confusion_matrix(predictions=predictions, labels=labels)

        labels_np = labels.cpu().numpy()
        predictions_labeled = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        if labels_np.shape != predictions_labeled.shape:
            raise RuntimeError("Dimensions of prediction and labels are not the same.")
        f1_weighted = f1_score(y_true=labels_np, y_pred=predictions_labeled, average="weighted", zero_division=0)
        f1_micro = f1_score(y_true=labels_np, y_pred=predictions_labeled, average="micro", zero_division=0)

        tp = self.confusion_matrix.diagonal()
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        f1 = 2 * tp / (2 * tp + fp + fn + np.finfo(float).eps)
        self.log(self.log_prefix + "F1_macro", f1.mean(), sync_dist=True)

        self.log(self.log_prefix + "F1_micro", f1_micro, sync_dist=True)
        return torch.tensor(f1_weighted)

    def on_epoch_end(self):
        self.confusion_matrix = np.zeros_like(self.confusion_matrix)

    def _get_confusion_matrix(self, predictions: torch.tensor, labels: torch.tensor) -> torch.tensor:
        preds = torch.softmax(predictions, dim=1).argmax(dim=1).squeeze().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        if labels.size == 1:
            labels = np.array([labels])
            preds = np.array([preds])
        matrix = confusion_matrix(labels, preds, labels=self.labels)
        return matrix


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

    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = range(n_classes)
        self.confusion_matrix = np.zeros((n_classes, n_classes))

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
        torch.Tensor
            Weighted precision score.

        Notes
        -----
        The `predictions` tensor should be an array of probabilities.
        The `labels` tensor should contain the ground truth labels.

        """
        self.confusion_matrix += self._get_confusion_matrix(predictions=predictions, labels=labels)

        labels_np = labels.cpu().numpy()
        predictions_labeled = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        if labels_np.shape != predictions_labeled.shape:
            raise RuntimeError("Dimensions of prediction and labels are not the same.")
        precision_weighted = precision_score(y_true=labels_np, y_pred=predictions_labeled, average="weighted", zero_division=0)
        precision_micro = precision_score(y_true=labels_np, y_pred=predictions_labeled, average="micro", zero_division=0)
        self.log(self.log_prefix + "Precision_micro", precision_micro, sync_dist=True)

        tp = self.confusion_matrix.diagonal()
        fp = self.confusion_matrix.sum(axis=0) - tp
        precision = tp / (tp + fp + np.finfo(float).eps)
        self.log(self.log_prefix + "Precision_macro", precision.mean(), sync_dist=True)

        return torch.tensor(precision_weighted)

    def on_epoch_end(self):
        self.confusion_matrix = np.zeros_like(self.confusion_matrix)

    def _get_confusion_matrix(self, predictions: torch.tensor, labels: torch.tensor) -> torch.tensor:
        preds = torch.softmax(predictions, dim=1).argmax(dim=1).squeeze().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        if labels.size == 1:
            labels = np.array([labels])
            preds = np.array([preds])
        matrix = confusion_matrix(labels, preds, labels=self.labels)
        return matrix


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

    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = range(n_classes)
        self.confusion_matrix = np.zeros((n_classes, n_classes))

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
        torch.Tensor
            Weighted recall score.

        Notes
        -----
        The `predictions` tensor should be an array of probabilities.
        The `labels` tensor should contain the ground truth labels.

        """
        self.confusion_matrix += self._get_confusion_matrix(predictions=predictions, labels=labels)

        labels_np = labels.cpu().numpy()
        predictions_labeled = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        if labels_np.shape != predictions_labeled.shape:
            raise RuntimeError("Dimensions of prediction and labels are not the same.")
        recall_weighted = recall_score(y_true=labels_np, y_pred=predictions_labeled, average="weighted", zero_division=0)
        recall_micro = recall_score(y_true=labels_np, y_pred=predictions_labeled, average="micro", zero_division=0)
        self.log(self.log_prefix + "Recall_micro", recall_micro, sync_dist=True)

        tp = self.confusion_matrix.diagonal()
        fn = self.confusion_matrix.sum(axis=1) - tp
        recall = tp / (tp + fn + np.finfo(float).eps)
        self.log(self.log_prefix + "Recall_macro", recall.mean(), sync_dist=True)

        return torch.tensor(recall_weighted)

    def on_epoch_end(self):
        self.confusion_matrix = np.zeros_like(self.confusion_matrix)

    def _get_confusion_matrix(self, predictions: torch.tensor, labels: torch.tensor) -> torch.tensor:
        preds = torch.softmax(predictions, dim=1).argmax(dim=1).squeeze().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        if labels.size == 1:
            labels = np.array([labels])
            preds = np.array([preds])
        matrix = confusion_matrix(labels, preds, labels=self.labels)
        return matrix


class PixelAccuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, predictions: torch.tensor, labels: torch.tensor):
        """Calculates the pixel accuracy metric for image segmentation tasks.
        This metric compares the predicted segmentation labels with the ground truth labels and measures the percentage
        of correctly classified pixels out of all labeled pixels in the input images.

        Parameters
        ----------
        predictions: torch.Tensor
            Tensor containing the predicted segmentation labels.
        labels: torch.Tensor
            Tensor containing the ground truth segmentation labels.

        Returns
        -------
        torch.Tensor
            The pixel accuracy as a scalar tensor.

        Notes
        -----
        - The input tensor `prediction` should have 4 dimensions (N,C,H,W)
        - The input tensor `labels` should have 3 dimension (N,H,W)
        - The predicted labels are obtained by taking the argmax along the second dimension of `predictions`.
        - Both `predictions` and `labels` should have integer values representing class labels.
        - Unlabeled pixels (where the value in `labels` is greater than the maximum class label) are ignored in the calculation.

        """
        predict = torch.argmax(predictions, 1)
        pixel_labeled = torch.sum(labels >= 0)
        pixel_correct = torch.sum((predict == labels) * (labels >= 0))
        pixel_accuracy = pixel_correct / (pixel_labeled + torch.finfo(torch.float32).eps)

        return pixel_accuracy


class IoU(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = -1
        self.K = 11
        self.smooth = 1e-7

    def __call__(self, predictions: torch.tensor, target: torch.tensor):
        """
        Calculates the IoU score given the predicted and ground truth labels.

        Parameters
        ----------
        predictions: torch.Tensor
            Tensor containing predicted labels.
        labels: torch.Tensor
            Tensor containing ground truth labels.

        Returns
        -------
        torch.Tensor
            mean Intersection over Union (mIoU) score.

        Notes
        -----
        - The input tensor `prediction` should have 4 dimensions (N,C,H,W)
        - The input tensor `labels` should have 3 dimension (N,H,W)
        - The predicted labels are expected to be in the form of logits or probabilities, where the channel dimension
        represents the number of classes.
        - The ground truth labels should be integers indicating the class indices.
        - The IoU score is computed as the average IoU over all classes.
        - This implementation handles cases where the number of unique labels in the ground truth is less than the
        number of classes in the predictions. Labels beyond the unique labels are ignored in the calculations.
        - The IoU score is calculated using the intersection and union of predicted and ground truth labels.

        """
        target = target
        if predictions.shape[1] == 1:
            output = torch.where(torch.sigmoid(predictions) < 0.5, 0, 1).squeeze()
        else:
            output = torch.argmax(torch.softmax(predictions, 1), 1)
        assert output.dim() in [1, 2, 3]
        assert output.shape == target.shape
        output = output.view(-1)
        target = target.view(-1)
        output[target == self.ignore_index] = self.ignore_index
        intersection = output[output == target]
        area_intersection = torch.histc(intersection.float(), bins=self.K, min=0, max=self.K - 1)
        area_output = torch.histc(output.float(), bins=self.K, min=0, max=self.K - 1)
        area_target = torch.histc(target.float(), bins=self.K, min=0, max=self.K - 1)
        area_union = area_output + area_target - area_intersection
        return (area_intersection.sum() + self.smooth) / (area_union.sum() + self.smooth)
