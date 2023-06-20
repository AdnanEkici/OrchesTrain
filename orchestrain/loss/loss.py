from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torch_snippets  # noqa
from scipy.ndimage import distance_transform_edt

import orchestrain.loss.loss_utils as loss_utils
from orchestrain import utils


class BaseLoss(torch.nn.Module):
    """The BaseLoss class is designed to be inherited from and extended to create custom loss functions tailored to specific needs.
    By subclassing BaseLoss, you can define your own loss functions with access to important attributes such as the current epoch,
    the state of the model, and the device on which the model is running. This allows you to customize the behavior during different
    training or evaluation phases.

    Parameters
    ----------
    current_epoch_fn: Callable
        A function that returns the current epoch.
    state_fn: Callable
        A function that returns the current state of the model.
    device_fn: Callable
        A function that returns the device on which the model is running.
    log_fn: Callable
        A function used for logging messages.
    log_image_fn: Callable
        A function used for logging images.
    log_prefix_fn: Callable
        A function that returns a prefix for log messages.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Attributes
    ----------
    current_epoch: int
        The current epoch of the training process.
    state: object
        The current state of the model.
    device: torch.device
        The device on which the model is running.
    log: Callable
        A function used for logging messages.
    log_image: Callable
        A function used for logging images.
    log_prefix: str
        A prefix for log messages.

    """

    def __init__(
        self,
        current_epoch_fn: Callable,
        state_fn: Callable,
        device_fn: Callable,
        log_fn: Callable,
        log_image_fn: Callable,
        log_prefix_fn: Callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.current_epoch_fn = current_epoch_fn
        self.state_fn = state_fn
        self.device_fn = device_fn
        self.log_fn = log_fn
        self.log_image_fn = log_image_fn
        self.log_prefix_fn = log_prefix_fn

    @property
    def current_epoch(self):
        # return self.__model_base.current_epoch
        return self.current_epoch_fn()

    @property
    def state(self):
        return self.state_fn()

    @property
    def device(self):
        return self.device_fn()

    @property
    def log(self):
        return self.log_fn

    @property
    def log_image(self):
        return self.log_image_fn

    @property
    def log_prefix(self):
        return self.log_prefix_fn()

    def on_train_epoch_start(self):
        """Hook method called at the start of each training epoch."""
        pass

    def on_validation_epoch_start(self):
        """Hook method called at the start of each validation epoch."""
        pass

    def on_test_start(self):
        """Hook method called at the start of the testing phase."""
        pass

    def on_validation_epoch_end(self):
        """Hook method called at the end of each validation epoch."""
        pass


class CrossEntropyLoss(BaseLoss):
    """Cross entropy loss for classification tasks.
    This class extends the BaseLoss class and implements the forward method for computing
    the cross entropy loss between model predictions and target labels.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Attributes
    ----------
    crossEntropyLoss: torch.nn.CrossEntropyLoss
        Instance of the CrossEntropyLoss class.

    Inherits
    --------
    BaseLoss
        Base class for custom loss functions.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        """Computes the cross entropy loss between prediction and target."""
        loss = self.crossEntropyLoss(prediction, target)
        # accuracy = torch.sum(torch.argmax(inputs, dim=1) == targets) / inputs.shape[0]
        return loss


class DiceLoss(BaseLoss):
    """Dice loss for segmentation tasks.
    This class extends the BaseLoss class and implements the forward method for computing
    the Dice loss between predicted and target segmentation masks.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Inherits
    --------
    BaseLoss
        Base class for custom loss functions.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, prediction, target, smooth=1, *args, **kwarg):
        """Computes the Dice loss between predicted and target segmentation masks.

        Parameters
        ----------
        prediction: torch.Tensor
            The predicted segmentation mask tensor.
        target: torch.Tensor
            The target segmentation mask tensor.
        smooth: float, optional
            Smoothing factor to avoid division by zero. Defaults to 1.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        torch.Tensor
            The computed Dice loss.

        Note
        ----
        The inputs `prediction` and `target` should have the same shape.

        """
        prediction_tmp = torch_snippets.F.sigmoid(prediction).view(-1)
        target_tmp = target.view(-1)

        intersection = (prediction_tmp * target_tmp).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (prediction_tmp.sum() + target_tmp.sum() + smooth)
        return dice_loss


class BCEWithLogitsLoss(BaseLoss):
    """Binary cross entropy loss with logits for binary classification tasks.
    This class extends the BaseLoss class and implements the forward method for computing
    the binary cross entropy loss between predicted logits and target labels.

    Parameters
    ----------
    pos_weight: float or None, optional
        Weight of positive class in the loss calculation. If None, pos_weight_equation is used. Defaults to None.
    pos_weight_equation: str or None, optional
        Equation to calculate the weight of the positive class dynamically based on the current epoch. If None, pos_weight is used. Defaults to None.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Attributes
    ----------
    tmp_pos_equ: str or None
        Temporary storage for the pos_weight_equation during training.
    pos_weight_equation: str
        Equation to calculate the weight of the positive class dynamically.
    pos_weight: torch.Tensor
        Weight of the positive class for the loss calculation.
    bce: torch.nn.BCEWithLogitsLoss
        Instance of the BCEWithLogitsLoss class.

    Inherits
    --------
    BaseLoss
        Base class for custom loss functions.
    """

    def __init__(self, pos_weight: float | None = None, pos_weight_equation: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmp_pos_equ = None

        if pos_weight is not None and pos_weight_equation is not None:
            raise Exception("Use one of the pos_weight or pos_weight_equation parameters. Not both!")

        if pos_weight is not None:
            self.pos_weight_equation = str(pos_weight)
        else:
            self.pos_weight_equation = str(pos_weight_equation)

    def on_train_epoch_start(self):
        if self.tmp_pos_equ is not None:
            self.pos_weight_equation = self.tmp_pos_equ
        self.calculate_posweights()

    def on_validation_epoch_start(self):
        self.tmp_pos_equ = self.pos_weight_equation
        self.pos_weight_equation = str(1)
        self.calculate_posweights()

    def calculate_posweights(self):
        """Calculates the weight of the positive class based on the equation and sets it for loss calculation."""
        self.pos_weight = loss_utils.calculate_equation(self.pos_weight_equation, current_epoch=self.current_epoch)
        self.log(self.log_prefix + "pos_weight", self.pos_weight, sync_dist=True)
        self.pos_weight = torch.tensor(self.pos_weight)
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(self.device))

    def forward(self, prediction, target):
        """Computes the binary cross entropy loss with logits."""
        return self.bce(prediction, target)


# TODO: We want to split current losses into metrics and loss.  Metrics cannot be used for backpropagation during training,
# but users can utilize them to calculate metrics for the training dataset.
class BoxValidationLoss(BaseLoss):
    """Box validation loss for evaluating object detection models.

    Parameters
    ----------
    prepare_images: bool, optional
        Flag indicating whether to prepare and visualize images during training. Defaults to True.
    iou_threshold: float, optional
        IoU threshold used for determining positive matches between predicted and target bounding boxes. Defaults to 0.1.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Attributes
    ----------
    prepare_images: bool
        Flag indicating whether to prepare and visualize images during training.
    iou_threshold: float
        IoU threshold used for determining positive matches between predicted and target bounding boxes.
    log_index: int
        Number to count logged images.

    Inherits
    --------
    BaseLoss
        Base class for custom loss functions.

    """

    import sklearn.metrics as metrics

    def __init__(self, prepare_images: bool = True, iou_threshold: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Consumes too much memory needs to be implemented with different approach
        self.prepare_images = prepare_images
        self.iou_threshold = iou_threshold
        self.log_index = 0

    def torch_to_numpy(img: torch.Tensor):
        """Converts a torch.Tensor image to a NumPy array.

        Parameters
        ----------
        img: torch.Tensor
            Input image tensor.

        Returns
        -------
        np.ndarray
            NumPy array representation of the image.

        """
        img_out = img.cpu().detach().numpy().transpose(1, 2, 0)
        img_out[img_out < 0] = 0
        mul = 255 / max(img_out.max(), 1e-7)
        img_out *= mul
        img_out = img_out.astype(np.uint8)
        return img_out

    def visualize_bboxes(self, img, predicted_bboxes, target_bboxes, iou_threshold=0.4):
        """Visualizes predicted and target bounding boxes on an image.

        Parameters
        ----------
        img
            Input image.
        predicted_bboxes
            Predicted bounding boxes.
        target_bboxes
            Target bounding boxes.
        iou_threshold:float, optional
            IoU threshold used to determine the color of the predicted bounding boxes. Defaults to 0.4.

        """
        img_input_colored = cv2.cvtColor(BoxValidationLoss.torch_to_numpy(img[0]), cv2.COLOR_GRAY2RGB)

        # Draw target bboxes
        for target_bbox in target_bboxes:
            img_input_colored = cv2.rectangle(
                img_input_colored, (int(target_bbox[0]), int(target_bbox[1])), (int(target_bbox[2]), int(target_bbox[3])), (0, 0, 255), 2
            )

        # Draw predicted bboxes
        for predicted_bbox in predicted_bboxes:
            max_iou = 0
            for target_bbox in target_bboxes:
                iou = utils.calculate_iou(predicted_bbox, target_bbox)
                if iou > max_iou:
                    max_iou = iou

            color = (0, 255, 0) if max_iou > iou_threshold else (255, 0, 0)
            img_input_colored = cv2.rectangle(
                img_input_colored, (int(predicted_bbox[0]), int(predicted_bbox[1])), (int(predicted_bbox[2]), int(predicted_bbox[3])), color, 2
            )

        img_h, img_w, _ = img_input_colored.shape
        img_input_resized = cv2.resize(img_input_colored, (img_w // 2, img_h // 2))

        image_name = f"BBoxValidationImage{self.log_index}.jpg"
        self.log_image(image_name=image_name, img=img_input_resized)
        self.log_index = self.log_index + 1

    def forward(self, prediction, target, bboxes, inputs, *args, **kwargs):
        """
        Computes the forward pass of the BoxValidationLoss.

        Parameters
        ----------
        prediction
            Predicted output of the model.
        target
            Target output.
        bboxes
            Bounding boxes.
        inputs
            Input data.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Computed loss value.

        """
        target_bboxes = bboxes.cpu().numpy()
        prediction = torch.sigmoid(prediction)

        predicted_bboxes = utils.mask_2_bbox([BoxValidationLoss.torch_to_numpy(mask) for mask in prediction])
        precision, recall = zip(
            *[
                utils.calculate_precision_recall(target_bbox, predicted_bbox, self.iou_threshold)
                for target_bbox, predicted_bbox in zip(target_bboxes, predicted_bboxes)
            ]
        )
        precision, recall = np.mean(precision), np.mean(recall)

        self.log(self.log_prefix + "precision", precision, sync_dist=True, on_epoch=True)
        self.log(self.log_prefix + "recall", recall, sync_dist=True, on_epoch=True)

        if self.prepare_images:
            self.visualize_bboxes(img=inputs[0], predicted_bboxes=predicted_bboxes[0], target_bboxes=bboxes[0], iou_threshold=self.iou_threshold)

        return 1 - (3 * precision + recall) / 4


class GaussianNLLLoss(BaseLoss):
    """Gaussian Negative Log-Likelihood Loss.

    Parameters
    ----------
    val_multiplication: float
        Value multiplication factor.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.

    Inherits
    --------
    BaseLoss
        Base class for custom loss functions.

    """

    def __init__(self, val_multiplication=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.var = None
        self.val_mul = val_multiplication
        self.m_loss = torch.nn.GaussianNLLLoss()

    def forward(self, prediction, target):
        """
        Computes the forward pass of the GaussianNLLLoss.

        Parameters
        ----------
        prediction
            Predicted output of the model.
        target
            Target output.

        Returns
        -------
        Computed loss value.

        """
        if self.var is None or not self.var.shape == target.shape:
            self.var = torch.tensor(torch.ones(target.shape, requires_grad=True) * self.val_mul).to(self.device)

        return self.m_loss(prediction, target, self.var)


class BoundaryLoss(BaseLoss):
    """Boundary loss class for segmentation tasks.

    Parameters
    ----------
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Inherits
    --------
    BaseLoss
        Base class for custom loss functions.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_dist_map(self, mask):
        """Creates a distance map from the mask.

        Parameters
        ----------
        mask
            Binary mask.

        Returns
        -------
        dist_map: torch.Tensor
            Distance map tensor.

        """
        mask = np.asarray(mask, dtype=bool)
        neg_mask = ~mask

        posdis = (distance_transform_edt(mask) - 1) * mask
        negdis = distance_transform_edt(neg_mask) * neg_mask
        dist_map = negdis - posdis

        return torch.tensor(dist_map, device=self.device)

    def forward(self, prediction, target):
        """Computes the forward pass of the BoundaryLoss.

        Parameters
        ----------
        prediction
            Predicted output of the model.
        target
            Target output.

        Returns
        -------
        Computed loss value.

        """
        target = target.cpu().numpy()
        dist_maps = self.create_dist_map(target)

        loss = dist_maps * prediction  # einsum("bkwh,bkwh->bkwh", inputs, targets)
        return loss.mean()
