from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import stack
from torch import Tensor


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device=torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate a coordinate grid for an image.
    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Parameters
    ----------
    height: int
        the image height (rows).
    width: int
        the image width (cols).
    normalized_coordinates: bool
        whether to normalize
        coordinates in the range :math:`[-1,1]` in order to be consistent with the
        PyTorch function :py:func:`torch.nn.functional.grid_sample`.
    device:
        the device on which the grid will be generated.
    dtype:
        the data type of the generated grid.

    Returns
    -------
    torch.Tensor
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example
    -------
    >>> create_meshgrid(2, 2)
    tensor([[[[-1., -1.],
                [ 1., -1.]],
    <BLANKLINE>
                [[-1.,  1.],
                [ 1.,  1.]]]])
    >>> create_meshgrid(2, 2, normalized_coordinates=False)
    tensor([[[[0., 0.],
                [1., 0.]],
    <BLANKLINE>
                [[0., 1.],
                [1., 1.]]]])

    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid: Tensor = stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    r"""Normalize both derivative and smoothing kernel."""
    if len(input.size()) < 2:
        raise TypeError(f"input should be at least 2D tensor. Got {input.size()}")
    norm: torch.Tensor = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = "reflect",
    normalized: bool = False,
    padding: str = "same",
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.
    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Parameters
    ----------
    input: torch.Tensor
        the input tensor with shape of :math:`(B, C, H, W)`.
    kernel: torch.Tensor
    the kernel to be convolved with the input tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
    border_type: str
        The padding mode to be applied before convolving. The expected modes are: ``'constant'``, ``'reflect'``,
    ``'replicate'`` or ``'circular'``.
    normalized: bool
        If True, kernel will be L1 normalized.
    padding: str
        This defines the type of padding. 2 modes available ``'same'`` or ``'valid'``.

    Returns
    -------
    torch.Tensor
        the convolved tensor of same size and numbers of channels as the input with shape :math:`(B, C, H, W)`.

    Example
    -------
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])

    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

    if border_type not in ["constant", "reflect", "replicate", "circular"]:
        raise ValueError(
            f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}"
        )

    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got {type(padding)}")

    if padding not in ["valid", "same"]:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == "same":
        padding_shape = _compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == "same":
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """Computes Huasdorff Distance loss for binary segmentation

    Parameters
    ----------
    seg_soft:
        softmax results,  shape=(b,2,x,y,z)
    gt:
        ground truth, shape=(b,x,y,z)
    seg_dtm:
        segmentation distance transform map; shape=(b,2,x,y,z)
    gt_dtm:
        ground truth distance transform map; shape=(b,2,x,y,z)

    Returns
    -------
    HDLoss:

    """

    delta_s = (seg_soft - gt.float()) ** 2
    s_dtm = seg_dtm**2
    g_dtm = gt_dtm**2
    dtm = s_dtm + g_dtm
    if len(delta_s.shape) == 5:  # B,C,H,W,D
        multipled = torch.einsum("bcxyz, bcxyz->bcxyz", delta_s, dtm)
    elif len(delta_s.shape) == 4:  # B,C,H,W
        multipled = torch.einsum("bcxy, bcxy->bcxy", delta_s, dtm)
    else:
        raise RuntimeError(f"Got Error dim in HD Loss {delta_s.shape}")
    # multipled = multipled.mean()

    return multipled


def distance_transform(image: torch.Tensor, kernel_size: int = 3, h: float = 0.35) -> torch.Tensor:
    r"""Approximates the Manhattan distance transform of images using cascaded convolution operations.
    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the image image.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension of the images.

    Parameters
    ----------
    image: Tensor
        Image with shape :math:`(B,C,H,W)`.
    kernel_size: int
        size of the convolution kernel.
    h: float
        value that influence the approximation of the min function.

    Returns
    -------
    torch.Tensor
        tensor with shape :math:`(B,C,H,W)`.

    Example
    -------
    >>> tensor = torch.zeros(1, 1, 5, 5)
    >>> tensor[:,:, 1, 2] = 1
    >>> dt = kornia.contrib.distance_transform(tensor)

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image type is not a torch.Tensor. Got {type(image)}")

    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # n_iters is set such that the DT will be able to propagate from any corner of the image to its far,
    # diagonally opposite corner
    n_iters: int = math.ceil(max(image.shape[2], image.shape[3]) / math.floor(kernel_size / 2))
    grid = create_meshgrid(kernel_size, kernel_size, normalized_coordinates=False, device=image.device, dtype=image.dtype)

    grid -= math.floor(kernel_size / 2)
    kernel = torch.hypot(grid[0, :, :, 0], grid[0, :, :, 1])
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    out = torch.zeros_like(image)

    # It is possible to avoid cloning the image if boundary = image, but this would require modifying the image tensor.
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        cdt = filter2d(boundary, kernel, border_type="replicate")
        cdt = -h * torch.log(cdt)

        # We are calculating log(0) above.
        # cdt = torch.nan_to_num(cdt, posinf=0.0)

        mask = torch.where(cdt > 0, 1.0, 0.0)
        if mask.sum() == 0:
            break

        offset: int = i * kernel_size // 2
        out += (offset + cdt) * mask
        boundary = torch.where(mask == 1, signal_ones, boundary)

    return out


def compute_dtm_gpu(img_gt, out_shape, kernel_size=5):
    """Computes the distance transform map of foreground in binary mask"""
    if len(out_shape) == 5:  # B,C,H,W,D
        fg_dtm = torch.cat([distance_transform(1 - img_gt[b].float(), kernel_size=kernel_size).unsqueeze(0) for b in range(out_shape[0])], axis=0)
    else:
        fg_dtm = distance_transform(1 - img_gt.float(), kernel_size=kernel_size)

    fg_dtm[~torch.isfinite(fg_dtm)] = kernel_size
    return fg_dtm


def calculate_equation(equation: str, **kwargs) -> float:
    """Exec the euqation string and return the result

    Parameters
    ----------
    equation : str
        string of equation

    Returns
    -------
    float
        the result of the equation

    Raises
    ------
    Exception
        if result is None

    """
    equation_str = f"result = {equation}"
    loc = {}
    loc.update(kwargs)
    exec(equation_str, globals(), loc)
    result = loc["result"]

    if result is None:
        raise Exception(f"equation: {equation} is not valid!")

    return result
