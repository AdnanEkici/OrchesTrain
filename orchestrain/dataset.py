from __future__ import annotations

import glob
import os
import random
from collections import OrderedDict

import albumentations as A
import cv2
import imagesize
import numpy as np
import SimpleITK as sitk
import torch
import torchvision
from PIL import Image
from tabulate import tabulate
from torchvision import transforms
from torchvision.io import read_video

import orchestrain.augments as augments
import orchestrain.utils as utils


tfms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomErasing(p=0.5),
        # transforms.RandomCrop(size=(64, 64))
        # transforms.RandomAffine(degrees=(-5, 5), translate=(0.5, 0.5), scale=(1, 2))
        # transforms.RandomRotation(degrees=(-30, 30)),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet
        # transforms.Normalize((0.5,), (0.5,)),  # imagenet
    ]
)


class Dataset(torch.utils.data.Dataset):
    to_tensor = torchvision.transforms.ToTensor()

    def __init__(self, *args, **kwargs):
        self.composed_transforms = None
        self.kwargs = kwargs

    def setup_augmentations(self, aug_config):
        transforms = []
        for class_name, params in aug_config.items():
            params = {} if params is None else params

            if "A" == class_name[0] and hasattr(A, class_name[2:]):
                transforms.append(self.prepare_albumentation_augmentor(class_name[2:], params, self.n_images, self.n_masks))
            elif hasattr(augments, class_name):
                transforms.append(getattr(augments, class_name)(**params))
            else:
                raise Exception(f":( Sorry there is no releated augmentation function called '{class_name}'...")

        composed_transforms = augments.Compose(transforms)

        self.composed_transforms = composed_transforms

    def prepare_albumentation_augmentor(self, class_name, params, n_images, n_masks):
        augmentation_func = getattr(A, class_name)(**params)

        additional_targets = {}
        if n_images > 1 or n_masks > 1:
            for idx in range(n_images - 1):
                additional_targets[f"image{idx+1}"] = "image"
            for idx in range(n_masks - 1):
                additional_targets[f"mask{idx+1}"] = "mask"

        transfom = A.Compose(
            [augmentation_func], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]), additional_targets=additional_targets
        )

        return transfom

    def apply_augmentations(self, *args, **data):
        # Lazy load transforms
        if self.composed_transforms is None:
            self.n_images = len(data["images"])
            self.n_masks = len(data["masks"]) if "masks" in data else 0
            if "augment" in self.kwargs:
                self.setup_augmentations(self.kwargs["augment"])

        # To Do: if self.composed_transforms is still None that means there is no any augmentations warn the user with logger

        return self.composed_transforms(*args, **data) if self.composed_transforms is not None else data

    def check_dataset_validity(self):
        """
        Checks dataset validity with certain assertions that are specific to dataset format.
        You must override this method if you want to check your dataset validity.

        Raise:
            - NotImplementedError: This method must be overridden in subclasses.
        """
        raise NotImplementedError("This method must be overriden according to your dataset.")

    @staticmethod
    def images_to_tensors(imgs):
        tensor = []
        for img in imgs:
            if torch.is_tensor(img):
                tensor.append(img)
            else:
                tensor.append(Dataset.to_tensor(img))
        return tensor


class TorchVisionDataset(Dataset):
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.asarray(img, dtype=np.uint8)  # .transpose(-1, 0, 1)

        transformed = self.apply_augmentations(images=[img])
        images = self.images_to_tensors(transformed["images"])[0]

        return {"inputs": images, "labels": label}


class ClassificationDataset(Dataset):
    """Custom dataset class for classification tasks.

    Parameters
    ----------
    root_path: str
        Root directory path where the dataset is located.
    classes: list
        List of class names.

    Raises
    ------
    AssertionError
        If the `root_path` is not found or any class directory is not found.

    Attributes
    ----------
    root_path: str
        Root directory path where the dataset is located.
    classes: list
        List of class names.
    image_infos: list
        List of image information, where each element is a pair [image_path, label].
    class_map: dict
        Mapping of class indices to class names.
    image_sz: tuple or list
        Size of the input images (width, height).

    """

    def __init__(
        self, root_path: str, classes: list[str], image_sz: tuple | list, extension: str | list | tuple = ["JPG", "PNG", "JPEG"], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.classes = classes
        self.extension = extension

        self.check_dataset_validity()

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, class_dir, image_path), idx]
            for idx, class_dir in enumerate(self.classes)
            for image_path in os.listdir(os.path.join(root_path, class_dir))
        ]
        self.class_map = {idx: class_name for idx, class_name in enumerate(classes)}
        print(f"Found {len(classes)} classes and {len(self)} images in {root_path}")
        self.__print_dataset_info()

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, index):
        img_path, label = self.image_infos[index]
        input_image = cv2.imread(img_path)

        # Check if image has 3 channels
        if input_image.shape[2] != 3:
            # Convert grayscale image to 3 channels
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        input_image = cv2.resize(input_image, self.image_sz)

        transformed = self.apply_augmentations(images=[input_image])
        images = self.images_to_tensors(transformed["images"])[0]

        return {"inputs": images, "labels": label}

    def check_dataset_validity(self):
        """
        Checks the validity of given classification data.

        Raises
        ------
        FileNotFoundError
            Raised when any class directory cannot be found.
        TypeError
            Raised when a file does not contain any of the permitted extensions.
        NotADirectoryError
            Raised when there is a file present in the directory where the class folders are located.
        """
        # For each class there must be a folder named as the class name in the root path
        class_dirs = os.listdir(self.root_path)
        for class_name in self.classes:
            if class_name not in class_dirs:
                raise FileNotFoundError(f"Class directory not found: {class_name}")

        for class_name in self.classes:
            if not os.path.isdir(os.path.join(self.root_path, class_name)):
                raise NotADirectoryError(f"'{class_name}' must be a directory named as one of the class names.")
            for file in os.listdir(os.path.join(self.root_path, class_name)):
                if isinstance(self.extension, str):
                    if not (file.endswith(self.extension)) or not (file.endswith(self.extension.lower())):
                        raise TypeError(f"Class directory {class_name} contains an unsupported image format: {file}.")
                else:
                    if not np.any([(file.endswith(ext) or file.endswith(ext.lower())) for ext in self.extension]):
                        raise TypeError(f"Class directory {class_name} contains an unsupported image format: {file}.")

    def __print_dataset_info(self):
        headers = [["id", "Class", "N_samples"]]
        data = [[idx, class_name, len(os.listdir(os.path.join(self.root_path, class_name)))] for idx, class_name in self.class_map.items()]
        table_data = headers + data
        table = tabulate(table_data, headers="firstrow", tablefmt="pipe")
        print(table)


class CityScapesDataset(Dataset):
    mask_postfix, img_postfix = "_gtFine_labelIds.png", "_leftImg8bit.png"

    def __init__(self, root_path: str, image_sz: tuple | list, split: str = "train", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise RuntimeError(f"Dataset root path not found: {root_path}")

        if split not in ["train", "val"]:
            raise RuntimeError(f"Split: '{split}' is not found. Expected 'train' or 'val'.")

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."

        self.image_sz = image_sz
        self.root_path = root_path
        self.mask_dir = os.path.join(root_path, "gtFine", split)
        self.image_dir = os.path.join(root_path, "leftImg8bit", split)

        self.check_dataset_validity()
        self.images = [
            [mask_path.replace(self.mask_postfix, self.img_postfix).replace(self.mask_dir, self.image_dir), mask_path]
            for mask_path in glob.glob(self.mask_dir + "/**/*" + self.mask_postfix)
            if os.path.exists(mask_path.replace(self.mask_postfix, self.img_postfix).replace(self.mask_dir, self.image_dir))
        ]

        self.id_to_trainid_lookup_table = np.array([item["trainId"] for item in CityScapesDataset.get_label_maps()], dtype=int)
        print(f"Found {len(self.images)} images and {len(self.images)} masks in {self.image_dir}")
        print(f"Dataset size: {len(self)}")

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_label_maps():
        return [
            {"name": "unlabeled", "id": 0, "trainId": -1, "category": 0, "color": (0, 0, 0)},
            {"name": "ego vehicle", "id": 1, "trainId": -1, "category": 0, "color": (0, 0, 0)},
            {"name": "rectification border", "id": 2, "trainId": -1, "category": 0, "color": (0, 0, 0)},
            {"name": "out of roi", "id": 3, "trainId": -1, "category": 0, "color": (0, 0, 0)},
            {"name": "static", "id": 4, "trainId": -1, "category": 0, "color": (0, 0, 0)},
            {"name": "dynamic", "id": 5, "trainId": -1, "category": 0, "color": (111, 74, 0)},
            {"name": "ground", "id": 6, "trainId": -1, "category": 0, "color": (81, 0, 81)},
            {"name": "road", "id": 7, "trainId": 0, "category": 0, "color": (128, 64, 128)},
            {"name": "sidewalk", "id": 8, "trainId": 1, "category": 0, "color": (244, 35, 232)},
            {"name": "parking", "id": 9, "trainId": -1, "category": 0, "color": (250, 170, 160)},
            {"name": "rail track", "id": 10, "trainId": -1, "category": 0, "color": (230, 150, 140)},
            {"name": "building", "id": 11, "trainId": 2, "category": 0, "color": (70, 70, 70)},
            {"name": "wall", "id": 12, "trainId": 3, "category": 0, "color": (102, 102, 156)},
            {"name": "fence", "id": 13, "trainId": 4, "category": 0, "color": (190, 153, 153)},
            {"name": "guard rail", "id": 14, "trainId": -1, "category": 0, "color": (180, 165, 180)},
            {"name": "bridge", "id": 15, "trainId": -1, "category": 0, "color": (150, 100, 100)},
            {"name": "tunnel", "id": 16, "trainId": -1, "category": 0, "color": (150, 120, 90)},
            {"name": "pole", "id": 17, "trainId": 5, "category": 0, "color": (153, 153, 153)},
            {"name": "polegroup", "id": 18, "trainId": -1, "category": 0, "color": (153, 153, 153)},
            {"name": "traffic light", "id": 19, "trainId": 6, "category": 0, "color": (250, 170, 30)},
            {"name": "traffic sign", "id": 20, "trainId": 7, "category": 0, "color": (200, 220, 0)},
            {"name": "vegetation", "id": 21, "trainId": 8, "category": 0, "color": (107, 142, 35)},
            {"name": "terrain", "id": 22, "trainId": 9, "category": 0, "color": (152, 251, 152)},
            {"name": "sky", "id": 23, "trainId": 10, "category": 0, "color": (70, 130, 180)},
            {"name": "person", "id": 24, "trainId": 11, "category": 0, "color": (255, 20, 60)},
            {"name": "rider", "id": 25, "trainId": 12, "category": 0, "color": (255, 0, 0)},
            {"name": "car", "id": 26, "trainId": 13, "category": 0, "color": (0, 0, 142)},
            {"name": "truck", "id": 27, "trainId": 14, "category": 0, "color": (0, 0, 70)},
            {"name": "bus", "id": 28, "trainId": 15, "category": 0, "color": (0, 60, 100)},
            {"name": "caravan", "id": 29, "trainId": -1, "category": 0, "color": (0, 0, 90)},
            {"name": "trailer", "id": 30, "trainId": -1, "category": 0, "color": (0, 0, 110)},
            {"name": "train", "id": 31, "trainId": 16, "category": 0, "color": (0, 80, 100)},
            {"name": "motorcycle", "id": 32, "trainId": 17, "category": 0, "color": (0, 0, 230)},
            {"name": "bicycle", "id": 33, "trainId": 18, "category": 0, "color": (119, 11, 32)},
            {"name": "license plate", "id": 34, "trainId": -1, "category": 0, "color": (0, 0, 142)},
        ]

    @staticmethod
    def postprocess(image: np.ndarray):
        trainId_to_color = {item["trainId"]: item["color"] for item in reversed(CityScapesDataset.get_label_maps())}
        trainid_to_color_lookup_table = np.array([item for item in reversed(trainId_to_color.values())], dtype=np.uint8)

        colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        colored_mask = trainid_to_color_lookup_table[image]

        return colored_mask

    @staticmethod
    def preprocess(image: np.ndarray, image_sz=(1024, 512)):  # Image expected as RGB
        img = cv2.resize(image, image_sz, interpolation=cv2.INTER_LINEAR)
        image = CityScapesDataset.images_to_tensors([img])[0]

        return image

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_sz, interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_sz, interpolation=cv2.INTER_NEAREST).astype(dtype=int)

        transformed = self.apply_augmentations(images=[img], masks=[mask])
        img, mask = transformed["images"][0], transformed["masks"][0]
        mask = self.id_to_trainid_lookup_table[mask]

        img = Dataset.to_tensor(img)
        mask = Dataset.to_tensor(mask).squeeze().long()

        return {"inputs": img, "mask": mask}

    def check_dataset_validity(self):
        """
        Checks the validity of given CityScapes data.

        Raises:
            - FileNotFoundError: Raised when a folder is empty or a file is not found.
            - ValueError: Raised when the size of an image mismatches the size of its mask.
            - NotADirectoryError: Raised when there is a file present in the directory where the city folders are located.
        """
        print(f"Validity check for {self.root_path}")

        # Every folder must contain at least one pair of sample.
        if len(os.listdir(self.mask_dir)) == 0:
            raise FileNotFoundError("Mask directory is empty.")
        elif len(os.listdir(self.image_dir)) == 0:
            raise FileNotFoundError("Image directory is empty.")

        else:

            def check_dirs(dir_path):
                for directory in os.listdir(dir_path):
                    city_path = os.path.join(dir_path, directory)
                    if not os.path.isdir(city_path):
                        raise NotADirectoryError(f"{city_path} is not a directory.")
                    if len(os.listdir(city_path)) == 0:
                        raise FileNotFoundError(f"Directory named '{directory}'  is empty.")

            check_dirs(self.mask_dir)
            check_dirs(self.image_dir)

        # For each mask, there must be an image corresponds to that mask.
        for mask_path in glob.glob(self.mask_dir + "/**/*" + self.mask_postfix):
            img_path = mask_path.replace(self.mask_postfix, self.img_postfix).replace(self.mask_dir, self.image_dir)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"There must be an image with the postfix '{self.img_postfix}' for each mask.")
            else:
                # if the corresponding mask exists their sizes should be equal.
                mask_size = imagesize.get(mask_path)
                img_size = imagesize.get(img_path)
                if img_size != mask_size:
                    raise ValueError(f"Size of the mask {mask_size} is not equal to image size of {img_size}.")

        # For each image, there must be a mask corresponds to that image.
        for img_path in glob.glob(self.image_dir + "/**/*" + self.img_postfix):
            mask_path = mask_path.replace(self.img_postfix, self.mask_postfix).replace(self.image_dir, self.mask_dir)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"There must be a mask with the postfix '{self.mask_postfix}' for each image.")


class PROMISE12(Dataset):
    """Custom dataset class for PROMISE2012 image segmentation challenge data.

    Parameters
    ----------
    root_path: str
        Root directory path where the dataset is located.
    image_sz: list | tuple
        Size of 3D images.

    Raises
    ------
    AssertionError
        If the `root_path` is not found or any class directory is not found.
    FileNotFoundError
        If there is a missing image with raw extension.

    Attributes
    ----------
    root_path: str
        Root directory path where the dataset is located.
    images: list
        List of meta image file paths for raw images.
    seg_masks: list
        List of meta image file paths for segmentation maks.
    image_sz: tuple or list
        Size of the input images (depth, width, height).

    """

    def __init__(self, root_path: str, image_sz: tuple | list = (128, 128, 64), *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")
        self.root_path = root_path
        self.image_sz = image_sz

        self.images = [os.path.basename(path) for path in sorted(glob.glob(root_path + os.path.sep + "Case??.mhd"))]
        self.seg_masks = [os.path.basename(path) for path in sorted(glob.glob(root_path + os.path.sep + "Case*segmentation.mhd"))]

        self.check_dataset_validity()

        print(f"Dataset size: {len(self)}")
        print(f"Image size: ({image_sz[0]}x{image_sz[1]}x{image_sz[2]})")

    def __len__(self):
        return len(self.images)

    @staticmethod
    def resize3D(image: sitk.Image, size: tuple | list):
        """Resizes the given image into the given size.

        Parameters
        ----------
        image : sitk.Image
            Image to be resized.
        size : tuple | list
            New size of the given image.

        Returns
        -------
        sitk.Image
            Resized image.
        """
        resized_image = sitk.Image(size, image.GetPixelIDValue())
        resized_image.SetOrigin(image.GetOrigin())
        resized_image.SetDirection(image.GetDirection())
        resized_image.SetSpacing([sz * spc / nsz for nsz, sz, spc in zip(size, image.GetSize(), image.GetSpacing())])

        return sitk.Resample(image, resized_image)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_path, self.images[idx])
        mask_path = os.path.join(self.root_path, self.seg_masks[idx])
        img = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        resized_image = PROMISE12.resize3D(img, self.image_sz)
        resized_mask = PROMISE12.resize3D(mask, self.image_sz)

        transformed = self.apply_augmentations(
            images=[sitk.GetArrayFromImage(resized_image)], masks=[sitk.GetArrayFromImage(resized_mask).astype(np.int64)]
        )

        image = self.images_to_tensors(transformed["images"])[0]
        mask = (self.images_to_tensors(transformed["masks"])[0]).squeeze().long()
        image = image.permute(1, 0, 2).contiguous().unsqueeze(0).float()
        mask = mask.permute(1, 0, 2).contiguous()
        return {"inputs": image, "mask": mask}

    def check_dataset_validity(self):
        """Checks the validity of Promise2012 data in the given path.

        Raises
        ------
        AssertionError
            Raised when the image size is not 3 dimensional.
        AssertionError
            Raised when the number of masks do not match the number of images.
        FileNotFoundError
            Raised if the raw image, corresponds to an mhd file, does not exist.

        """
        if len(self.image_sz) != 3:
            raise AssertionError("The image size must be specified in a 3-dimensional layout: (D, H, W)")

        # Number of masks must be equal to the number of images.
        if len(self.seg_masks) != len(self.images):
            raise AssertionError("Number of masks must be equal to number of images.")

        # there must be a file with raw extension for each mhd file
        for img_name, mask_name in zip(self.images, self.seg_masks):
            img_path = os.path.join(self.root_path, img_name.replace(".mhd", ".raw"))
            mask_path = os.path.join(self.root_path, mask_name.replace(".mhd", ".raw"))

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"There is no such file named {img_name.replace('.mhd', '.raw')} in {self.root_path}")

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"There is no such file named {mask_name.replace('.mhd', '.raw')} in {self.root_path}")


class Ade20kDataset(Dataset):
    """ADE20K dataset for segmentation tasks.

    Parameters
    ----------
    root_path: str
        Root directory path where the dataset is located.
    image_sz: list
        Size of the input images (width, height).
    split: str
        Name of dataset (train or val). The names of the train and validation folders in the dataset must be given.

    Raises
    ------
    RuntimeError
        If the `root_path` is not found.

    Attributes
    ----------
    root_path: str
        Root directory path where the dataset is located.
    images: list
        List of images and masks.
    label_map: JPEG, PNG
        LabelID map for segmentation
    image_sz: tuple or list
        Size of the input images (width, height).

    """

    mask_postfix, img_postfix = ".png", ".jpg"

    def __init__(self, root_path: str, image_sz: tuple | list, split: str = "train", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise RuntimeError(f"Dataset root path not found: {root_path}")

        if split not in ["training", "validation"]:
            raise RuntimeError(f"Split: '{split}' is not found. Expected 'train' or 'val'.")

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."

        self.image_sz = image_sz
        self.root_path = root_path
        self.mask_dir = os.path.join(root_path, "annotations", split)
        self.image_dir = os.path.join(root_path, "images", split)

        self.check_dataset_validity()
        self.images = [
            [mask_path.replace(self.mask_postfix, self.img_postfix).replace(self.mask_dir, self.image_dir), mask_path]
            for mask_path in glob.glob(self.mask_dir + "/*" + self.mask_postfix)
            if os.path.exists(mask_path.replace(self.mask_postfix, self.img_postfix).replace(self.mask_dir, self.image_dir))
        ]

        self.id_to_trainid_lookup_table = np.array([item["trainId"] for item in Ade20kDataset.get_label_maps()], dtype=int)
        print(f"Found {len(self.images)} images and {len(self.images)} masks in {self.image_dir}")
        print(f"Dataset size: {len(self)}")

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_label_maps():
        return [
            {"name": "unlabeled", "id": 0, "trainId": 0, "category": 0, "color": (0, 0, 0)},
            {"name": "wall", "id": 1, "trainId": 1, "category": 0, "color": (120, 120, 120)},
            {"name": "building", "id": 2, "trainId": 2, "category": 0, "color": (180, 120, 120)},
            {"name": "sky", "id": 3, "trainId": 3, "category": 0, "color": (6, 230, 230)},
            {"name": "floor", "id": 4, "trainId": 4, "category": 0, "color": (80, 50, 50)},
            {"name": "tree", "id": 5, "trainId": 5, "category": 0, "color": (4, 200, 3)},
            {"name": "ceiling", "id": 6, "trainId": 6, "category": 0, "color": (120, 120, 80)},
            {"name": "road", "id": 7, "trainId": 7, "category": 0, "color": (140, 140, 140)},
            {"name": "bed", "id": 8, "trainId": 8, "category": 0, "color": (204, 5, 255)},
            {"name": "window", "id": 9, "trainId": 9, "category": 0, "color": (230, 230, 230)},
            {"name": "grass", "id": 10, "trainId": 10, "category": 0, "color": (4, 250, 7)},
            {"name": "cabinet", "id": 11, "trainId": 11, "category": 0, "color": (224, 5, 255)},
            {"name": "sidewalk", "id": 12, "trainId": 12, "category": 0, "color": (235, 255, 7)},
            {"name": "person", "id": 13, "trainId": 13, "category": 0, "color": (150, 5, 61)},
            {"name": "ground", "id": 14, "trainId": 14, "category": 0, "color": (120, 120, 70)},
            {"name": "door", "id": 15, "trainId": 15, "category": 0, "color": (8, 255, 51)},
            {"name": "table", "id": 16, "trainId": 16, "category": 0, "color": (255, 6, 82)},
            {"name": "mountain", "id": 17, "trainId": 17, "category": 0, "color": (143, 255, 140)},
            {"name": "plant", "id": 18, "trainId": 18, "category": 0, "color": (204, 255, 4)},
            {"name": "curtain", "id": 19, "trainId": 19, "category": 0, "color": (255, 51, 7)},
            {"name": "chair", "id": 20, "trainId": 20, "category": 0, "color": (204, 70, 3)},
            {"name": "car", "id": 21, "trainId": 21, "category": 0, "color": (0, 102, 200)},
            {"name": "water", "id": 22, "trainId": 22, "category": 0, "color": (61, 230, 250)},
            {"name": "picture", "id": 23, "trainId": 23, "category": 0, "color": (255, 6, 51)},
            {"name": "sofa", "id": 24, "trainId": 24, "category": 0, "color": (11, 102, 255)},
            {"name": "shelf", "id": 25, "trainId": 25, "category": 0, "color": (255, 7, 71)},
            {"name": "house", "id": 26, "trainId": 26, "category": 0, "color": (255, 9, 224)},
            {"name": "sea", "id": 27, "trainId": 27, "category": 0, "color": (9, 7, 230)},
            {"name": "mirror", "id": 28, "trainId": 28, "category": 0, "color": (220, 220, 220)},
            {"name": "carpet", "id": 29, "trainId": 29, "category": 0, "color": (255, 9, 92)},
            {"name": "field", "id": 30, "trainId": 30, "category": 0, "color": (112, 9, 255)},
            {"name": "armchair", "id": 31, "trainId": 31, "category": 0, "color": (8, 255, 214)},
            {"name": "seat", "id": 32, "trainId": 32, "category": 0, "color": (7, 255, 224)},
            {"name": "fence", "id": 33, "trainId": 33, "category": 0, "color": (255, 184, 6)},
            {"name": "desk", "id": 34, "trainId": 34, "category": 0, "color": (10, 255, 71)},
            {"name": "rock", "id": 35, "trainId": 35, "category": 0, "color": (255, 41, 10)},
            {"name": "wardrobe", "id": 36, "trainId": 36, "category": 0, "color": (7, 255, 255)},
            {"name": "lamp", "id": 37, "trainId": 37, "category": 0, "color": (224, 255, 8)},
            {"name": "bath", "id": 38, "trainId": 38, "category": 0, "color": (102, 8, 255)},
            {"name": "rail", "id": 39, "trainId": 39, "category": 0, "color": (255, 61, 6)},
            {"name": "cushion", "id": 40, "trainId": 40, "category": 0, "color": (255, 194, 7)},
            {"name": "base", "id": 41, "trainId": 41, "category": 0, "color": (255, 122, 8)},
            {"name": "box", "id": 42, "trainId": 42, "category": 0, "color": (0, 255, 20)},
            {"name": "column", "id": 43, "trainId": 43, "category": 0, "color": (255, 8, 41)},
            {"name": "sign", "id": 44, "trainId": 44, "category": 0, "color": (255, 5, 153)},
            {"name": "drawers", "id": 45, "trainId": 45, "category": 0, "color": (6, 51, 255)},
            {"name": "counter", "id": 46, "trainId": 46, "category": 0, "color": (235, 12, 255)},
            {"name": "sand", "id": 47, "trainId": 47, "category": 0, "color": (160, 150, 20)},
            {"name": "sink", "id": 48, "trainId": 48, "category": 0, "color": (0, 163, 255)},
            {"name": "skyscraper", "id": 49, "trainId": 49, "category": 0, "color": (140, 140, 140)},
            {"name": "fireplace", "id": 50, "trainId": 50, "category": 0, "color": (250, 10, 15)},
            {"name": "refrigerator", "id": 51, "trainId": 51, "category": 0, "color": (20, 255, 0)},
            {"name": "grandstand", "id": 52, "trainId": 52, "category": 0, "color": (31, 255, 0)},
            {"name": "path", "id": 53, "trainId": 53, "category": 0, "color": (255, 31, 0)},
            {"name": "stairs", "id": 54, "trainId": 54, "category": 0, "color": (255, 224, 0)},
            {"name": "runway", "id": 55, "trainId": 55, "category": 0, "color": (153, 255, 0)},
            {"name": "showcase", "id": 56, "trainId": 56, "category": 0, "color": (0, 0, 255)},
            {"name": "table", "id": 57, "trainId": 57, "category": 0, "color": (255, 71, 0)},
            {"name": "pillow", "id": 58, "trainId": 58, "category": 0, "color": (0, 235, 255)},
            {"name": "screen door", "id": 59, "trainId": 59, "category": 0, "color": (0, 173, 255)},
            {"name": "staircase", "id": 60, "trainId": 60, "category": 0, "color": (31, 0, 255)},
            {"name": "river", "id": 61, "trainId": 61, "category": 0, "color": (11, 200, 200)},
            {"name": "bridge", "id": 62, "trainId": 62, "category": 0, "color": (255, 82, 0)},
            {"name": "bookcase", "id": 63, "trainId": 63, "category": 0, "color": (0, 255, 245)},
            {"name": "blind", "id": 64, "trainId": 64, "category": 0, "color": (0, 61, 255)},
            {"name": "coffee", "id": 65, "trainId": 65, "category": 0, "color": (0, 255, 112)},
            {"name": "toilet", "id": 66, "trainId": 66, "category": 0, "color": (0, 255, 133)},
            {"name": "flower", "id": 67, "trainId": 67, "category": 0, "color": (255, 0, 0)},
            {"name": "book", "id": 68, "trainId": 68, "category": 0, "color": (255, 163, 0)},
            {"name": "hill", "id": 69, "trainId": 69, "category": 0, "color": (255, 102, 0)},
            {"name": "bench", "id": 70, "trainId": 70, "category": 0, "color": (194, 255, 0)},
            {"name": "countertop", "id": 71, "trainId": 71, "category": 0, "color": (0, 143, 255)},
            {"name": "stove", "id": 72, "trainId": 72, "category": 0, "color": (51, 255, 0)},
            {"name": "palm", "id": 73, "trainId": 73, "category": 0, "color": (0, 82, 255)},
            {"name": "kitchen island", "id": 74, "trainId": 74, "category": 0, "color": (0, 255, 41)},
            {"name": "computer", "id": 75, "trainId": 75, "category": 0, "color": (0, 255, 173)},
            {"name": "swivel chair", "id": 76, "trainId": 76, "category": 0, "color": (10, 0, 255)},
            {"name": "boat", "id": 77, "trainId": 77, "category": 0, "color": (173, 255, 0)},
            {"name": "bar", "id": 78, "trainId": 78, "category": 0, "color": (0, 255, 153)},
            {"name": "arcade machine", "id": 79, "trainId": 79, "category": 0, "color": (255, 92, 0)},
            {"name": "shack", "id": 80, "trainId": 80, "category": 0, "color": (255, 0, 255)},
            {"name": "bus", "id": 81, "trainId": 81, "category": 0, "color": (255, 0, 245)},
            {"name": "towel", "id": 82, "trainId": 82, "category": 0, "color": (255, 0, 102)},
            {"name": "light", "id": 83, "trainId": 83, "category": 0, "color": (255, 173, 0)},
            {"name": "truck", "id": 84, "trainId": 84, "category": 0, "color": (255, 0, 20)},
            {"name": "tower", "id": 85, "trainId": 85, "category": 0, "color": (255, 184, 184)},
            {"name": "chandelier", "id": 86, "trainId": 86, "category": 0, "color": (0, 31, 255)},
            {"name": "awning", "id": 87, "trainId": 87, "category": 0, "color": (0, 255, 61)},
            {"name": "streetlight", "id": 88, "trainId": 88, "category": 0, "color": (0, 71, 255)},
            {"name": "booth/cubicle/stall", "id": 89, "trainId": 89, "category": 0, "color": (255, 0, 204)},
            {"name": "television", "id": 90, "trainId": 90, "category": 0, "color": (0, 255, 194)},
            {"name": "plane", "id": 91, "trainId": 91, "category": 0, "color": (0, 255, 82)},
            {"name": "dirt", "id": 92, "trainId": 92, "category": 0, "color": (0, 10, 255)},
            {"name": "clothes", "id": 93, "trainId": 93, "category": 0, "color": (0, 112, 255)},
            {"name": "pole", "id": 94, "trainId": 94, "category": 0, "color": (51, 0, 255)},
            {"name": "land/ground", "id": 95, "trainId": 95, "category": 0, "color": (0, 194, 255)},
            {"name": "balustrade", "id": 96, "trainId": 96, "category": 0, "color": (0, 122, 255)},
            {"name": "escalator", "id": 97, "trainId": 97, "category": 0, "color": (0, 255, 163)},
            {"name": "ottoman", "id": 98, "trainId": 98, "category": 0, "color": (255, 153, 0)},
            {"name": "bottle", "id": 99, "trainId": 99, "category": 0, "color": (0, 255, 10)},
            {"name": "buffet/counter", "id": 100, "trainId": 100, "category": 0, "color": (255, 112, 0)},
            {"name": "poster/placard/notice/bill/card", "id": 101, "trainId": 101, "category": 0, "color": (143, 255, 0)},
            {"name": "stage", "id": 102, "trainId": 102, "category": 0, "color": (82, 0, 255)},
            {"name": "van", "id": 103, "trainId": 103, "category": 0, "color": (163, 255, 0)},
            {"name": "ship", "id": 104, "trainId": 104, "category": 0, "color": (255, 235, 0)},
            {"name": "fountain", "id": 105, "trainId": 105, "category": 0, "color": (8, 184, 170)},
            {"name": "transporter", "id": 106, "trainId": 106, "category": 0, "color": (133, 0, 255)},
            {"name": "canopy", "id": 107, "trainId": 107, "category": 0, "color": (0, 255, 92)},
            {"name": "washing machine", "id": 108, "trainId": 108, "category": 0, "color": (184, 0, 255)},
            {"name": "toy", "id": 109, "trainId": 109, "category": 0, "color": (255, 0, 31)},
            {"name": "swimming pool", "id": 110, "trainId": 110, "category": 0, "color": (0, 184, 255)},
            {"name": "stool", "id": 111, "trainId": 111, "category": 0, "color": (0, 214, 255)},
            {"name": "barrel", "id": 112, "trainId": 112, "category": 0, "color": (255, 0, 112)},
            {"name": "basket", "id": 113, "trainId": 113, "category": 0, "color": (92, 255, 0)},
            {"name": "waterfall", "id": 114, "trainId": 114, "category": 0, "color": (0, 224, 255)},
            {"name": "tent", "id": 115, "trainId": 115, "category": 0, "color": (112, 224, 255)},
            {"name": "bag", "id": 116, "trainId": 116, "category": 0, "color": (70, 184, 160)},
            {"name": "motorbike", "id": 117, "trainId": 117, "category": 0, "color": (163, 0, 255)},
            {"name": "cradle", "id": 118, "trainId": 118, "category": 0, "color": (153, 0, 255)},
            {"name": "oven", "id": 119, "trainId": 119, "category": 0, "color": (71, 255, 0)},
            {"name": "ball", "id": 120, "trainId": 120, "category": 0, "color": (255, 0, 163)},
            {"name": "food", "id": 121, "trainId": 121, "category": 0, "color": (255, 204, 0)},
            {"name": "step", "id": 122, "trainId": 122, "category": 0, "color": (255, 0, 143)},
            {"name": "tank/storage", "id": 123, "trainId": 123, "category": 0, "color": (0, 255, 235)},
            {"name": "brand", "id": 124, "trainId": 124, "category": 0, "color": (133, 255, 0)},
            {"name": "microwave", "id": 125, "trainId": 125, "category": 0, "color": (255, 0, 235)},
            {"name": "pot", "id": 126, "trainId": 126, "category": 0, "color": (245, 0, 255)},
            {"name": "animal", "id": 127, "trainId": 127, "category": 0, "color": (255, 0, 122)},
            {"name": "bicycle", "id": 128, "trainId": 128, "category": 0, "color": (255, 245, 0)},
            {"name": "lake", "id": 129, "trainId": 129, "category": 0, "color": (10, 190, 212)},
            {"name": "dishwasher", "id": 130, "trainId": 130, "category": 0, "color": (214, 255, 0)},
            {"name": "projection", "id": 131, "trainId": 131, "category": 0, "color": (0, 204, 255)},
            {"name": "blanket", "id": 132, "trainId": 132, "category": 0, "color": (20, 0, 255)},
            {"name": "sculpture", "id": 133, "trainId": 133, "category": 0, "color": (255, 255, 0)},
            {"name": "exhaust hood", "id": 134, "trainId": 134, "category": 0, "color": (0, 153, 255)},
            {"name": "sconce", "id": 135, "trainId": 135, "category": 0, "color": (0, 41, 255)},
            {"name": "vase", "id": 136, "trainId": 136, "category": 0, "color": (0, 255, 204)},
            {"name": "traffic light", "id": 137, "trainId": 137, "category": 0, "color": (41, 0, 255)},
            {"name": "tray", "id": 138, "trainId": 138, "category": 0, "color": (41, 255, 0)},
            {"name": "trash", "id": 139, "trainId": 139, "category": 0, "color": (173, 0, 255)},
            {"name": "fan", "id": 140, "trainId": 140, "category": 0, "color": (0, 245, 255)},
            {"name": "pier", "id": 141, "trainId": 141, "category": 0, "color": (71, 0, 255)},
            {"name": "screen", "id": 142, "trainId": 142, "category": 0, "color": (122, 0, 255)},
            {"name": "plate", "id": 143, "trainId": 143, "category": 0, "color": (0, 255, 184)},
            {"name": "monitor", "id": 144, "trainId": 144, "category": 0, "color": (0, 92, 255)},
            {"name": "board/notice", "id": 145, "trainId": 145, "category": 0, "color": (184, 255, 0)},
            {"name": "shower", "id": 146, "trainId": 146, "category": 0, "color": (0, 133, 255)},
            {"name": "radiator", "id": 147, "trainId": 147, "category": 0, "color": (255, 214, 0)},
            {"name": "glass drinking", "id": 148, "trainId": 148, "category": 0, "color": (25, 194, 194)},
            {"name": "clock", "id": 149, "trainId": 149, "category": 0, "color": (102, 255, 0)},
            {"name": "flag", "id": 150, "trainId": 150, "category": 0, "color": (92, 0, 255)},
        ]

    @staticmethod
    def postprocess(image: np.ndarray):
        trainId_to_color = {item["trainId"]: item["color"] for item in reversed(Ade20kDataset.get_label_maps())}
        trainid_to_color_lookup_table = np.array([item for item in reversed(trainId_to_color.values())], dtype=np.uint8)

        colored_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        colored_mask = trainid_to_color_lookup_table[image]

        return colored_mask

    @staticmethod
    def preprocess(image: np.ndarray, image_sz=(1024, 512)):  # Image expected as RGB
        img = cv2.resize(image, image_sz, interpolation=cv2.INTER_LINEAR)
        image = Ade20kDataset.images_to_tensors([img])[0]

        return image

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_sz, interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_sz, interpolation=cv2.INTER_NEAREST).astype(dtype=int)

        transformed = self.apply_augmentations(images=[img], masks=[mask])
        img, mask = transformed["images"][0], transformed["masks"][0]
        mask = self.id_to_trainid_lookup_table[mask]

        img = Dataset.to_tensor(img)
        mask = Dataset.to_tensor(mask).squeeze().long()

        return {"inputs": img, "mask": mask}

    def check_dataset_validity(self):
        """
        Checks the validity of given CityScapes data.

        Raises:
            - FileNotFoundError: Raised when a folder is empty or a file is not found.
            - ValueError: Raised when the size of an image mismatches the size of its mask.
            - NotADirectoryError: Raised when there is a file present in the directory where the city folders are located.
        """
        print(f"Validity check for {self.root_path}")

        if len(os.listdir(self.mask_dir)) == 0:
            raise FileNotFoundError("Mask directory is empty.")
        elif len(os.listdir(self.image_dir)) == 0:
            raise FileNotFoundError("Image directory is empty.")

        # For each mask, there must be an image corresponds to that mask.
        for mask_path in glob.glob(self.mask_dir + "/**/*" + self.mask_postfix):
            img_path = mask_path.replace(self.mask_postfix, self.img_postfix).replace(self.mask_dir, self.image_dir)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"There must be an image with the postfix '{self.img_postfix}' for each mask.")
            else:
                # if the corresponding mask exists their sizes should be equal.
                mask_size = imagesize.get(mask_path)
                img_size = imagesize.get(img_path)
                if img_size != mask_size:
                    raise ValueError(f"Size of the mask {mask_size} is not equal to image size of {img_size}.")

        # For each image, there must be a mask corresponds to that image.
        for img_path in glob.glob(self.image_dir + "/**/*" + self.img_postfix):
            mask_path = img_path.replace(self.img_postfix, self.mask_postfix).replace(self.image_dir, self.mask_dir)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"There must be a mask with the postfix '{self.mask_postfix}' for each image.")


class UCF101Dataset(Dataset):
    """UCF101 is an action recognition data set of realistic action videos.

    Parameters
    ----------
    root_path: str
        Root directory path where the dataset is located.
    labels_file: str
        Path to the file containing video labels.
    class_file: str
        Path to the file containing class information.
    image_sz: tuple or list
        Size of input images (width, height).
    frames_per_clip: int, optional
        Number of frames per video clip, defaults to 16.

    Raises
    ------
    AssertionError
        If the `root_path` or `labels_file` is not found.

    Attributes
    ----------
    root_path: str
        Root directory path where the dataset is located.
    labels: list
        List of tuples containing video paths and their corresponding labels.
    frames_per_clip: int
        Number of frames per video clip.
    class_file: str
        Path to the file containing class information.
    image_sz: tuple or list
        Size of the input images (width, height).
    """

    def __init__(
        self,
        root_path: str,
        labels_file: str,
        class_file: str,
        image_sz: tuple | list,
        frames_per_clip=16,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")
        if not os.path.exists(labels_file):
            raise AssertionError(f"Dataset labels file path not found: {labels_file}")

        self.root_path = root_path
        self.labels_file = labels_file
        self.frames_per_clip = frames_per_clip
        self.class_file = class_file
        self.labels = []
        self.class_dict = {}

        with open(class_file) as c_file:
            for line in c_file:
                key, value = line.strip().split(" ")
                self.class_dict[value] = int(key) - 1

        self.check_dataset_validity()

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz
        if "trainlist" in self.labels_file:
            with open(self.labels_file) as file:
                for line in file.readlines():
                    train_video_path, train_video_class_id = line.strip().split(" ")
                    self.labels.append([os.path.join(root_path, train_video_path), int(train_video_class_id) - 1])

        else:
            with open(self.labels_file) as file:
                for line in file:
                    test_video_class, _ = line.strip().split("/")
                    test_video_class_id = self.class_dict[test_video_class]
                    self.labels.append([os.path.join(root_path, line[:-1]), test_video_class_id])

        self.unique_labels_set = {lab[1] for lab in self.labels}

        print(f"Found {len(list(self.unique_labels_set))} classes and {len(self.labels)} videos in {root_path}")

    def __len__(self):
        return len(self.labels)

    def sample_frames(self, frames):
        if len(frames) > self.frames_per_clip:
            start_idx = random.randint(0, len(frames) - self.frames_per_clip)
            frames = frames[start_idx : start_idx + self.frames_per_clip]  # noqa
        return frames

    def videos_to_tensors(self, frames):
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop((self.image_sz[0], self.image_sz[1])),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ]
        )

        pil_frames = [Image.fromarray(frame.numpy()) for frame in frames]
        videos = torch.stack([transform(frame) for frame in pil_frames]).permute(1, 0, 2, 3)
        return videos

    def __getitem__(self, index):
        video_path, label = self.labels[index]
        frames, _, _ = read_video(video_path, pts_unit="sec")
        frames = self.sample_frames(frames)
        frames = self.videos_to_tensors(frames)

        return {"inputs": frames, "labels": label}

    def check_dataset_validity(self):
        """Checks the validity of given classification data.

        Raises
        ------
        FileNotFoundError
            Raised when any class directory cannot be found.
        Exception
            Raised when video reading error.
        """
        print(f"Validity check for {self.root_path}")

        class_dirs = [f for f in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, f))]
        for class_name in class_dirs:
            if class_name not in self.class_dict.keys():
                raise FileNotFoundError(f"Class directory not found: {class_name}")

        for video_path, label in self.labels:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Error occurred while reading {video_path}")


class ISIC2016Dataset(Dataset):
    """ISIC Challenge Dataset for Task 1 of year 2016.

    Parameters
    ----------
    root_path: str
        The folder path includes two folders named 'ISBI2016_ISIC_Part1_{split}_Data' and
    'ISBI2016_ISIC_Part1_{split}_GroundTruth'
    image_sz: tuple | list
        Image size will be used to resize images.
    split: str
        Must be 'train' or 'test' to indicate which split of data to use.

    Raises
    ------
    RuntimeError
        Raised when the given root_path is not foud or the number of images is not equal to
    number of masks.
    FileNotFoundError
    -----------------
        Raised when image_dir or the mask_dir is not in the given root path, or when there is an image
    whose mask is missing.

    """

    def __init__(self, root_path: str, image_sz: tuple | list, split: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists(root_path):
            raise RuntimeError(f"Dataset root path not found: {root_path}")
        self.root_path = root_path
        self.image_sz = image_sz
        self.image_dir = "ISBI2016_ISIC_Part1_Training_Data"
        self.mask_dir = "ISBI2016_ISIC_Part1_Training_GroundTruth"
        if split == "test":
            self.image_dir = "ISBI2016_ISIC_Part1_Test_Data"
            self.mask_dir = "ISBI2016_ISIC_Part1_Test_GroundTruth"

        self.check_dataset_validity()

        self.images = sorted(os.listdir(os.path.join(root_path, self.image_dir)), key=utils.natural_keys)
        self.masks = sorted(os.listdir(os.path.join(root_path, self.mask_dir)), key=utils.natural_keys)

        print(f"Found {len(self.images)} images and {len(self.masks)} masks in {root_path}.")
        print(f"Dataset size: {len(self)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_path, self.image_dir, self.images[idx])
        mask_path = os.path.join(self.root_path, self.mask_dir, self.masks[idx])

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_sz)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_sz)

        transformed = self.apply_augmentations(images=[img], masks=[mask])
        img, mask = transformed["images"][0], transformed["masks"][0]

        img = Dataset.to_tensor(img)
        mask = Dataset.to_tensor(mask).squeeze().long()

        return {"inputs": img, "mask": mask}

    def check_dataset_validity(self):
        if not os.path.exists(os.path.join(self.root_path, self.image_dir)):
            raise FileNotFoundError(f"Given root path does not include {self.image_dir}.")
        if not os.path.exists(os.path.join(self.root_path, self.mask_dir)):
            raise FileNotFoundError(f"Given root path does not include {self.mask_dir}.")

        if len(glob.glob(os.path.join(self.root_path, self.image_dir, "*.jpg"))) != len(
            glob.glob(os.path.join(self.root_path, self.mask_dir, "*.png"))
        ):
            raise RuntimeError("Number of images must be equal to number of masks.")

        # there must be a corresponding mask in mask_dir for each image in image_dir
        for image_name in os.listdir(os.path.join(self.root_path, self.image_dir)):
            mask_name = image_name.replace(".jpg", "_Segmentation.png")
            if not os.path.exists(os.path.join(self.root_path, self.mask_dir, mask_name)):
                raise FileNotFoundError(f"There is not any mask named {mask_name} in {os.path.join(self.root_path, self.mask_dir)}")


class RescueNet(Dataset):
    """RescueNet-v2.0 dataset: ....

    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """

    # Training dataset root folders
    train_folder = "train/train-org-img/"
    train_lbl_folder = "train/train-label-img/"

    # Validation dataset root folders
    val_folder = "val/val-org-img/"
    val_lbl_folder = "val/val-label-img/"

    # Test dataset root folders
    test_folder = "test/test-org-img/"
    test_lbl_folder = "test/test-label-img/"

    # Filters to find the images
    org_img_extension = ".jpg"
    # lbl_name_filter = '.png'

    lbl_img_extension = ".png"
    lbl_name_filter = "lab"

    # The values associated with the 35 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    # The values above are remapped to the following

    new_classes = (0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0)
    # new_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict(
        [
            ("unlabeled", (0, 0, 0)),
            ("water", (61, 230, 250)),
            ("building-no-damage", (180, 120, 120)),
            ("building-medium-damage", (235, 255, 7)),
            ("building-major-damage", (255, 184, 6)),
            ("building-total-destruction", (255, 0, 0)),
            ("vehicle", (255, 0, 245)),
            ("road-clear", (140, 140, 140)),
            ("road-blocked", (160, 150, 20)),
            ("tree", (4, 250, 7)),
            ("pool", (255, 235, 0)),
        ]
    )

    def __init__(
        self, root_dir, mode="train", transform=None, label_transform=None, n_classes=8, img_sz=(720, 720), loader=utils.pil_loader, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root_dir = root_dir
        self.mode = mode
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(img_sz)])
        self.transform = transform
        if label_transform is None:
            # label_transform = transforms.Compose([transforms.Lambda(lambda x : x.astype(np.float32)), transforms.ToTensor()])
            label_transform = transforms.Compose(
                [transforms.PILToTensor(), transforms.Resize(img_sz), transforms.Lambda(lambda x: x.long().squeeze())]
            )
        self.label_transform = label_transform
        self.loader = loader
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.new_classes = (0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0) if n_classes == 8 else self.full_classes
        if self.mode.lower() == "train":
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(os.path.join(root_dir, self.train_folder), extension_filter=self.org_img_extension)

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder), name_filter=self.lbl_name_filter, extension_filter=self.lbl_img_extension
            )
        elif self.mode.lower() == "val":
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files(os.path.join(root_dir, self.val_folder), extension_filter=self.org_img_extension)

            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder), name_filter=self.lbl_name_filter, extension_filter=self.lbl_img_extension
            )
        elif self.mode.lower() == "test":
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(os.path.join(root_dir, self.test_folder), extension_filter=self.org_img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder), name_filter=self.lbl_name_filter, extension_filter=self.lbl_img_extension
            )
        elif self.mode.lower() == "vis":
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(os.path.join(root_dir, self.test_folder), extension_filter=self.org_img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder), name_filter=self.lbl_name_filter, extension_filter=self.lbl_img_extension
            )
        else:
            raise RuntimeError("Unexpected dataset mode. " "Supported modes are: train, val and test")

    def _normalize(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """

        if self.mode == "vis":
            img = Image.open(self.test_data[index]).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return self.normalize(img), os.path.basename(self.test_data[index])

        if self.mode.lower() == "train":
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == "val":
            data_path, label_path = self.val_data[index], self.val_labels[index]
        elif self.mode.lower() == "test":
            data_path, label_path = self.test_data[index], self.test_labels[index]
        else:
            raise RuntimeError("Unexpected dataset mode. " "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        # Remap class labels
        label = utils.remap(label, self.full_classes, self.new_classes)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return {"inputs": self.normalize(img), "mask": label}

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == "train":
            return len(self.train_data)
        elif self.mode.lower() == "val":
            return len(self.val_data)
        elif self.mode.lower() == "test":
            return len(self.test_data)
        elif self.mode.lower() == "vis":
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. " "Supported modes are: train, val and test")


class DOTAv2MultiClassDataset(Dataset):
    def __init__(self, root_path: str, ann_path: str, image_sz=(512, 512), transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.transform = transform
        self.ann_path = ann_path

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, "images", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "images")))
        ]
        print(f"Found {len(self.image_infos)} images in {root_path}")

        self.mask_infos = [
            [os.path.join(root_path, "masked_multiclass", image_path), idx]
            for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "masked_multiclass")))
        ]

        print(f"Found {len(self.mask_infos)} mask images in {root_path}")

        self.annotation_files = [
            [os.path.join(root_path, "annfiles", annfile_path), idx]
            for idx, annfile_path in enumerate(os.listdir(os.path.join(root_path, "annfiles")))
        ]
        print(f"Found {len(self.annotation_files)} annotation files in {root_path}")

        self.class_map = {
            "background": 0,
            "plane": 1,
            "ship": 2,
            "storage-tank": 3,
            "baseball-diamond": 4,
            "tennis-court": 5,
            "basketball-court": 6,
            "ground-track-field": 7,
            "harbor": 8,
            "bridge": 9,
            "large-vehicle": 10,
            "small-vehicle": 11,
            "helicopter": 12,
            "roundabout": 13,
            "soccer-ball-field": 14,
            "swimming-pool": 15,
            "container-crane": 16,
            "airport": 17,
            "helipad": 18,
        }
        self.unused_classes = {
            "storage-tank": 3,
            "baseball-diamond": 4,
            "tennis-court": 5,
            "basketball-court": 6,
            "ground-track-field": 7,
            "harbor": 8,
            "bridge": 9,
            "roundabout": 13,
            "soccer-ball-field": 14,
            "swimming-pool": 15,
            "container-crane": 16,
            "airport": 17,
            "helipad": 18,
        }
        for unused_class in self.unused_classes.keys():
            self.class_map.pop(unused_class)

        class_map_new = {}
        for idx, (cls, label) in enumerate(self.class_map.items()):
            class_map_new[cls] = idx

        self.train_class_map = class_map_new

    def __len__(self):
        return len(self.image_infos)

    @staticmethod
    def collate_fn(data):
        keys = data[0].keys()

        if "bboxes" in keys:  # Fix the number of bboxes of each element with padding
            max_n_bboxes = max([len(elem["bboxes"]) for elem in data])
            for elem in data:
                bboxes_padded = np.zeros((max_n_bboxes, 4), dtype=float)
                class_labels_padded = elem["class_labels"] + [-1] * (max_n_bboxes - len(elem["class_labels"]))
                bboxes = elem["bboxes"]
                if len(bboxes) > 0:
                    bboxes_padded[: len(bboxes)] = bboxes

                elem["bboxes"] = bboxes_padded
                elem["class_labels"] = class_labels_padded

        data = torch.utils.data._utils.collate.default_collate(data)
        return data

    def _process_annotation(self, annotation_content, original_size):
        annotations = []
        target = []
        lines = annotation_content

        for line in lines:
            data = line.split()
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, data[:8])  # Extract coordinates and label
            class_label = data[8]

            if class_label in self.unused_classes:
                continue

            scale_x = self.image_sz[0] / original_size[1]
            scale_y = self.image_sz[1] / original_size[0]

            x1, y1, x2, y2, x3, y3, x4, y4 = map(
                int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, x3 * scale_x, y3 * scale_y, x4 * scale_x, y4 * scale_y]
            )

            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            points = points.reshape((-1, 1, 2))

            class_id = self.train_class_map.get(class_label, len(self.train_class_map))

            annotations.append(points)
            target.append(class_id)

        return {"annotations": annotations, "target": target}

    def roll_image(self, image):
        image_array = np.asarray(image)

        # Randomly choose shift values
        pixels_roll = np.random.choice([6, 8, 10, 12, 14])
        sign = np.random.choice([-1, 1])  # -1: rolls up-left, +1: rolls down-right
        pixels_roll = sign * pixels_roll

        # Define Albumentations ShiftScaleRotate transform
        transform = A.ShiftScaleRotate(shift_limit=(pixels_roll, pixels_roll), rotate_limit=0, scale_limit=0, p=1)

        # Apply the transform to the image
        augmented = transform(image=image_array)

        # Extract the augmented image
        rolled_image = augmented["image"]

        return rolled_image

    def encode_mask(self, mask_image, num_classes):
        # Convert mask data type if necessary (assuming uint8 is compatible)
        mask_image = mask_image.astype(np.uint8)

        # Perform one-hot encoding
        mask_onehot = torch.nn.functional.one_hot(torch.from_numpy(mask_image), num_classes=num_classes)

        # Reshape back to original dimensions (optional)
        mask_onehot = mask_onehot.view(mask_image.shape[0], mask_image.shape[1], num_classes)
        print("Mask shape:", mask_onehot.shape)

        return mask_onehot

    def __getitem__(self, index):
        image_path, image_idx = self.image_infos[index]

        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract image name without extension

        annfile_path = os.path.join(self.root_path, "annfiles", f"{image_name}.txt")

        mask_path = os.path.join(self.root_path, "masked_multiclass", f"{image_name}_mask.png")

        if not os.path.exists(annfile_path):
            raise FileNotFoundError(f"Annotation file not found: {annfile_path}")

        input_image = cv2.imread(image_path)
        mask_image = cv2.imread(mask_path)
        prev_image = self.roll_image(input_image)

        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
            input_image, _, _ = cv2.split(input_image)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2YUV)
            mask_image, _, _ = cv2.split(mask_image)
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2YUV)
            prev_image, _, _ = cv2.split(prev_image)

        input_image = cv2.resize(input_image, self.image_sz)
        prev_image = cv2.resize(prev_image, self.image_sz)
        mask_image = cv2.resize(mask_image, self.image_sz).astype(np.float32)

        with open(annfile_path) as file:
            lines = file.readlines()

        processed = self._process_annotation(lines, input_image.shape[:2])

        bboxes = []

        for annots in processed["annotations"]:
            x2, y2, x4, y4 = annots[1][0][0], annots[1][0][1], annots[3][0][0], annots[3][0][1]
            bbox = [x2, y2, x4, y4]
            bboxes.append(bbox)

        class_labels = processed["target"]

        for cls, idx in self.unused_classes.items():
            mask_image[mask_image == idx] = 0

        for cls, idx in self.class_map.items():
            mask_image[mask_image == idx] = self.train_class_map[cls]

        # transformed = self.apply_augmentations(images=[input_image])
        # images = self.images_to_tensors(transformed["images"])[0]
        images = [input_image, prev_image]
        [input_image, prev_image] = self.images_to_tensors(images)
        images = [input_image, prev_image]

        # mask_image = self.images_to_tensors([mask_image])[0].long().squeeze()*255
        # mask_image = self.images_to_tensors([mask_image])[0].squeeze()
        mask_image = self.images_to_tensors([mask_image])[0].squeeze().long()

        return {"inputs": images, "mask": mask_image, "bboxes": bboxes, "class_labels": class_labels}


class VSAIDataset(Dataset):
    def __init__(self, root_path: str, ann_path: str, image_sz=(512, 512), org_sz=(1024, 1024), transform=None, *args, **kwargs):
        super().__init__()

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.transform = transform
        self.ann_path = ann_path
        self.org_sz = org_sz

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, "images", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "images")))
        ]

        self.mask_infos = [
            [os.path.join(root_path, "masked", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "masked")))
        ]

        print(f"Found {len(self.image_infos)} images in {root_path}")

        self.annotation_files = [
            [os.path.join(root_path, "annfiles", annfile_path), idx]
            for idx, annfile_path in enumerate(os.listdir(os.path.join(root_path, "annfiles")))
        ]
        print(f"Found {len(self.annotation_files)} annotation files in {root_path}")

    def __len__(self):
        return len(self.annotation_files)

    @staticmethod
    def collate_fn(data):
        keys = data[0].keys()

        if "bboxes" in keys:  # Fix the number of bboxes of each element with padding
            max_n_bboxes = max([len(elem["bboxes"]) for elem in data])
            for elem in data:
                bboxes_padded = np.zeros((max_n_bboxes, 4), dtype=float)
                class_labels_padded = elem["class_labels"] + [-1] * (max_n_bboxes - len(elem["class_labels"]))
                bboxes = elem["bboxes"]
                if len(bboxes) > 0:
                    bboxes_padded[: len(bboxes)] = bboxes

                elem["bboxes"] = bboxes_padded
                elem["class_labels"] = class_labels_padded

        data = torch.utils.data._utils.collate.default_collate(data)

        return data

    def _process_annotation(self, annotation_content, original_size):
        annotations = []
        target = []
        lines = annotation_content

        for line in lines:
            data = line.split()
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, data[:8])  # Extract coordinates and label
            label = data[8:9]
            if label == ["small-vehicle"]:
                label = 0
            else:
                label = 1

            scale_x = self.image_sz[0] / original_size[1]
            scale_y = self.image_sz[1] / original_size[0]

            x1, y1, x2, y2, x3, y3, x4, y4 = map(
                int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, x3 * scale_x, y3 * scale_y, x4 * scale_x, y4 * scale_y]
            )

            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            points = points.reshape((-1, 1, 2))

            # points = np.array([x1, y1, x2, y2, x3, y3, x4, y4], np.int32)

            annotations.append(points)
            target.append(label)

        return {"annotations": annotations, "target": target}

    def roll_image(self, image):
        image_array = np.asarray(image)

        # Randomly choose shift values
        pixels_roll = np.random.choice([6, 8, 10, 12, 14])
        sign = np.random.choice([-1, 1])  # -1: rolls up-left, +1: rolls down-right
        pixels_roll = sign * pixels_roll

        # Define Albumentations ShiftScaleRotate transform
        transform = A.ShiftScaleRotate(shift_limit=(pixels_roll, pixels_roll), rotate_limit=0, scale_limit=0, p=1)

        # Apply the transform to the image
        augmented = transform(image=image_array)

        # Extract the augmented image
        rolled_image = augmented["image"]

        return rolled_image

    def __getitem__(self, index):
        # Get image path and annotation path for the given index
        image_path, image_idx = self.image_infos[index]

        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract image name without extension
        annfile_path = os.path.join(self.root_path, "annfiles", f"{image_name}.txt")
        mask_path = os.path.join(self.root_path, "masked", f"{image_name}_mask.png")

        if not os.path.exists(annfile_path):
            raise FileNotFoundError(f"Annotation file not found: {annfile_path}")

        input_image = cv2.imread(image_path)
        mask_image = cv2.imread(mask_path)
        prev_image = self.roll_image(input_image)

        # Check if image has 3 channels
        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
            input_image, _, _ = cv2.split(input_image)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2YUV)
            mask_image, _, _ = cv2.split(mask_image)
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2YUV)
            prev_image, _, _ = cv2.split(prev_image)

        input_image = cv2.resize(input_image, self.image_sz)
        input_image = (input_image / 255.0).astype(np.float32)
        prev_image = cv2.resize(prev_image, self.image_sz)
        prev_image = (prev_image / 255.0).astype(np.float32)
        mask_image = cv2.resize(mask_image, self.image_sz)
        mask_image = (mask_image / 255.0).astype(np.float32)

        with open(annfile_path) as file:
            lines = file.readlines()

        processed = self._process_annotation(lines, self.org_sz)
        # annots = processed["annotations"]

        bboxes = []

        for annots in processed["annotations"]:
            x2, y2, x4, y4 = annots[1][0][0], annots[1][0][1], annots[3][0][0], annots[3][0][1]
            bbox = [x2, y2, x4, y4]
            bboxes.append(bbox)

        class_labels = processed["target"]

        # transformed = self.apply_augmentations(images=[input_image])
        # images = self.images_to_tensors(transformed["images"])[0]
        images = [input_image, prev_image]
        [input_image, prev_image] = self.images_to_tensors(images)
        images = [input_image, prev_image]

        mask_image = self.images_to_tensors([mask_image])[0].long()

        return {"inputs": images, "mask": mask_image, "bboxes": bboxes, "class_labels": class_labels}


class DOTAv2Dataset(Dataset):
    def __init__(self, root_path: str, ann_path: str, image_sz=(512, 512), transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.transform = transform
        self.ann_path = ann_path

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, "images", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "images")))
        ]
        print(f"Found {len(self.image_infos)} images in {root_path}")

        self.mask_infos = [
            [os.path.join(root_path, "masks", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "masks")))
        ]

        print(f"Found {len(self.mask_infos)} mask images in {root_path}")

        self.annotation_files = [
            [os.path.join(root_path, "annfiles", annfile_path), idx]
            for idx, annfile_path in enumerate(os.listdir(os.path.join(root_path, "annfiles")))
        ]
        print(f"Found {len(self.annotation_files)} annotation files in {root_path}")

        self.train_classes = {
            "background": 0,  # bunu burdan uçur
            "plane": 1,
            "ship": 1,
            "large-vehicle": 2,
            "helicopter": 2,
            "small-vehicle": 1
            # Add more classes and their corresponding labels as needed
        }
        self.unused_classes = {
            "storage-tank": 3,
            "baseball-diamond": 4,
            "tennis-court": 5,
            "basketball-court": 6,
            "ground-track-field": 7,
            "harbor": 8,
            "bridge": 9,
            "roundabout": 13,
            "soccer-ball-field": 14,
            "swimming-pool": 15,
            "container-crane": 16,
            "airport": 17,
            "helipad": 18,
        }

    def __len__(self):
        return len(self.mask_infos)

    @staticmethod
    def collate_fn(data):
        keys = data[0].keys()

        if "bboxes" in keys:  # Fix the number of bboxes of each element with padding
            max_n_bboxes = max([len(elem["bboxes"]) for elem in data])
            for elem in data:
                bboxes_padded = np.zeros((max_n_bboxes, 4), dtype=float)
                class_labels_padded = elem["class_labels"] + [-1] * (max_n_bboxes - len(elem["class_labels"]))
                bboxes = elem["bboxes"]
                if len(bboxes) > 0:
                    bboxes_padded[: len(bboxes)] = bboxes

                elem["bboxes"] = bboxes_padded
                elem["class_labels"] = class_labels_padded

        data = torch.utils.data._utils.collate.default_collate(data)

        return data

    def roll_image(self, image):
        image_array = np.asarray(image)

        # Randomly choose shift values
        pixels_roll = np.random.choice([6, 8, 10, 12, 14])
        sign = np.random.choice([-1, 1])  # -1: rolls up-left, +1: rolls down-right
        pixels_roll = sign * pixels_roll

        # Define Albumentations ShiftScaleRotate transform
        transform = A.ShiftScaleRotate(shift_limit=(pixels_roll, pixels_roll), rotate_limit=0, scale_limit=0, p=1)

        # Apply the transform to the image
        augmented = transform(image=image_array)

        # Extract the augmented image
        rolled_image = augmented["image"]

        return rolled_image

    def _process_annotation(self, annotation_content, original_size):
        annotations = []
        target = []
        lines = annotation_content

        for line in lines:
            data = line.split()
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, data[:8])  # Extract coordinates and label
            label = data[8]

            if label in self.unused_classes:
                continue
            elif label in self.train_classes:
                label = 1

            scale_x = self.image_sz[0] / original_size[1]
            scale_y = self.image_sz[1] / original_size[0]

            x1, y1, x2, y2, x3, y3, x4, y4 = map(
                int, [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, x3 * scale_x, y3 * scale_y, x4 * scale_x, y4 * scale_y]
            )

            points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            points = points.reshape((-1, 1, 2))

            # points = np.array([x1, y1, x2, y2, x3, y3, x4, y4], np.int32)

            annotations.append(points)
            target.append(label)

        return {"annotations": annotations, "target": target}

    def __getitem__(self, index):
        image_path, image_idx = self.image_infos[index]

        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract image name without extension

        annfile_path = os.path.join(self.root_path, "annfiles", f"{image_name}.txt")

        mask_path = os.path.join(self.root_path, "masks", f"{image_name}_mask.png")

        input_image = cv2.imread(image_path)
        mask_image = cv2.imread(mask_path)
        prev_image = self.roll_image(input_image)

        if not os.path.exists(annfile_path):
            raise FileNotFoundError(f"Annotation file not found: {annfile_path}")

        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
            input_image, _, _ = cv2.split(input_image)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2YUV)
            mask_image, _, _ = cv2.split(mask_image)
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2YUV)
            prev_image, _, _ = cv2.split(prev_image)

        input_image = cv2.resize(input_image, self.image_sz)
        input_image = (input_image / 255.0).astype(np.float32)
        prev_image = cv2.resize(prev_image, self.image_sz)
        prev_image = (prev_image / 255.0).astype(np.float32)
        mask_image = cv2.resize(mask_image, self.image_sz)
        mask_image = (mask_image / 255.0).astype(np.float32)

        with open(annfile_path) as file:
            lines = file.readlines()

        processed = self._process_annotation(lines, input_image.shape[:2])
        # annots = processed["annotations"]

        bboxes = []

        for annots in processed["annotations"]:
            x2, y2, x4, y4 = annots[1][0][0], annots[1][0][1], annots[3][0][0], annots[3][0][1]
            bbox = [x2, y2, x4, y4]
            bboxes.append(bbox)

        class_labels = processed["target"]

        # transformed = self.apply_augmentations(images=[input_image])
        # images = self.images_to_tensors(transformed["images"])[0]
        images = [input_image, prev_image]
        [input_image, prev_image] = self.images_to_tensors(images)
        images = [input_image, prev_image]

        mask_image = self.images_to_tensors([mask_image])[0].long()

        return {"inputs": images, "mask": mask_image, "bboxes": bboxes, "class_labels": class_labels}


class AITODDataset(Dataset):
    def __init__(self, root_path: str, image_sz=(800, 800), transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.transform = transform

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, "images", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "images")))
        ]
        print(f"Found {len(self.image_infos)} images in {root_path}")

        self.mask_infos = [
            [os.path.join(root_path, "masks", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "masks")))
        ]

        print(f"Found {len(self.mask_infos)} mask images in {root_path}")

    def __len__(self):
        return len(self.image_infos)

    @staticmethod
    def collate_fn(data):
        keys = data[0].keys()

        if "bboxes" in keys:  # Fix the number of bboxes of each element with padding
            max_n_bboxes = max([len(elem["bboxes"]) for elem in data])
            for elem in data:
                bboxes_padded = np.zeros((max_n_bboxes, 4), dtype=float)
                class_labels_padded = elem["class_labels"] + [-1] * (max_n_bboxes - len(elem["class_labels"]))
                bboxes = elem["bboxes"]
                if len(bboxes) > 0:
                    bboxes_padded[: len(bboxes)] = bboxes

                elem["bboxes"] = bboxes_padded
                elem["class_labels"] = class_labels_padded

        data = torch.utils.data._utils.collate.default_collate(data)

        return data

    def roll_image(self, image):
        image_array = np.asarray(image)

        # Randomly choose shift values
        pixels_roll = np.random.choice([6, 8, 10, 12, 14])
        sign = np.random.choice([-1, 1])  # -1: rolls up-left, +1: rolls down-right
        pixels_roll = sign * pixels_roll

        # Define Albumentations ShiftScaleRotate transform
        transform = A.ShiftScaleRotate(shift_limit=(pixels_roll, pixels_roll), rotate_limit=0, scale_limit=0, p=1)

        # Apply the transform to the image
        augmented = transform(image=image_array)

        # Extract the augmented image
        rolled_image = augmented["image"]

        return rolled_image

    def __getitem__(self, index):
        image_path, image_idx = self.image_infos[index]

        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract image name without extension

        mask_path = os.path.join(self.root_path, "masks", f"{image_name}_mask.png")

        input_image = cv2.imread(image_path)
        mask_image = cv2.imread(mask_path)
        prev_image = self.roll_image(input_image)

        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
            input_image, _, _ = cv2.split(input_image)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2YUV)
            mask_image, _, _ = cv2.split(mask_image)
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2YUV)
            prev_image, _, _ = cv2.split(prev_image)

        input_image = cv2.resize(input_image, self.image_sz)
        input_image = (input_image / 255.0).astype(np.float32)
        prev_image = cv2.resize(prev_image, self.image_sz)
        prev_image = (prev_image / 255.0).astype(np.float32)
        mask_image = cv2.resize(mask_image, self.image_sz)
        mask_image = (mask_image / 255.0).astype(np.float32)

        # transformed = self.apply_augmentations(images=[input_image])
        # images = self.images_to_tensors(transformed["images"])[0]
        images = [input_image, prev_image]
        [input_image, prev_image] = self.images_to_tensors(images)
        images = [input_image, prev_image]

        mask_image = self.images_to_tensors([mask_image])[0].long()

        return {"inputs": images, "mask": mask_image}


class VisDroneDataset(Dataset):
    def __init__(self, root_path: str, image_sz=(1024, 1024), transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.transform = transform

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, "images", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "images")))
        ]
        print(f"Found {len(self.image_infos)} images in {root_path}")

        self.mask_infos = [
            [os.path.join(root_path, "person_masks", image_path), idx]
            for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "person_masks")))
        ]

        print(f"Found {len(self.mask_infos)} mask images in {root_path}")

    def __len__(self):
        return len(self.image_infos)

    @staticmethod
    def collate_fn(data):
        keys = data[0].keys()

        if "bboxes" in keys:  # Fix the number of bboxes of each element with padding
            max_n_bboxes = max([len(elem["bboxes"]) for elem in data])
            for elem in data:
                bboxes_padded = np.zeros((max_n_bboxes, 4), dtype=float)
                class_labels_padded = elem["class_labels"] + [-1] * (max_n_bboxes - len(elem["class_labels"]))
                bboxes = elem["bboxes"]
                if len(bboxes) > 0:
                    bboxes_padded[: len(bboxes)] = bboxes

                elem["bboxes"] = bboxes_padded
                elem["class_labels"] = class_labels_padded

        data = torch.utils.data._utils.collate.default_collate(data)

        return data

    def roll_image(self, image):
        image_array = np.asarray(image)

        # Randomly choose shift values
        pixels_roll = np.random.choice([6, 8, 10, 12, 14])
        sign = np.random.choice([-1, 1])  # -1: rolls up-left, +1: rolls down-right
        pixels_roll = sign * pixels_roll

        # Define Albumentations ShiftScaleRotate transform
        transform = A.ShiftScaleRotate(shift_limit=(pixels_roll, pixels_roll), rotate_limit=0, scale_limit=0, p=1)

        # Apply the transform to the image
        augmented = transform(image=image_array)

        # Extract the augmented image
        rolled_image = augmented["image"]

        return rolled_image

    def __getitem__(self, index):
        image_path, image_idx = self.image_infos[index]

        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract image name without extension

        mask_path = os.path.join(self.root_path, "person_masks", f"{image_name}_mask.png")

        input_image = cv2.imread(image_path)
        mask_image = cv2.imread(mask_path)
        prev_image = self.roll_image(input_image)

        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
            input_image, _, _ = cv2.split(input_image)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2YUV)
            mask_image, _, _ = cv2.split(mask_image)
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2YUV)
            prev_image, _, _ = cv2.split(prev_image)

        input_image = cv2.resize(input_image, self.image_sz)
        input_image = (input_image / 255.0).astype(np.float32)
        prev_image = cv2.resize(prev_image, self.image_sz)
        prev_image = (prev_image / 255.0).astype(np.float32)
        mask_image = cv2.resize(mask_image, self.image_sz)
        mask_image = (mask_image / 255.0).astype(np.float32)

        # transformed = self.apply_augmentations(images=[input_image])
        # images = self.images_to_tensors(transformed["images"])[0]
        images = [input_image, prev_image]
        [input_image, prev_image] = self.images_to_tensors(images)
        images = [input_image, prev_image]

        mask_image = self.images_to_tensors([mask_image])[0].long()

        return {"inputs": images, "mask": mask_image}


class InriaDataset(Dataset):
    def __init__(self, root_path: str, image_sz=(1024, 1024), transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.transform = transform

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, "images", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "images")))
        ]
        print(f"Found {len(self.image_infos)} images in {root_path}")

        self.mask_infos = [
            [os.path.join(root_path, "masks", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "masks")))
        ]

        print(f"Found {len(self.mask_infos)} mask images in {root_path}")

    def __len__(self):
        return len(self.image_infos)

    @staticmethod
    def collate_fn(data):
        keys = data[0].keys()

        if "bboxes" in keys:  # Fix the number of bboxes of each element with padding
            max_n_bboxes = max([len(elem["bboxes"]) for elem in data])
            for elem in data:
                bboxes_padded = np.zeros((max_n_bboxes, 4), dtype=float)
                class_labels_padded = elem["class_labels"] + [-1] * (max_n_bboxes - len(elem["class_labels"]))
                bboxes = elem["bboxes"]
                if len(bboxes) > 0:
                    bboxes_padded[: len(bboxes)] = bboxes

                elem["bboxes"] = bboxes_padded
                elem["class_labels"] = class_labels_padded

        data = torch.utils.data._utils.collate.default_collate(data)

        return data

    def roll_image(self, image):
        image_array = np.asarray(image)

        # Randomly choose shift values
        pixels_roll = np.random.choice([6, 8, 10, 12, 14])
        sign = np.random.choice([-1, 1])  # -1: rolls up-left, +1: rolls down-right
        pixels_roll = sign * pixels_roll

        # Define Albumentations ShiftScaleRotate transform
        transform = A.ShiftScaleRotate(shift_limit=(pixels_roll, pixels_roll), rotate_limit=0, scale_limit=0, p=1)

        # Apply the transform to the image
        augmented = transform(image=image_array)

        # Extract the augmented image
        rolled_image = augmented["image"]

        return rolled_image

    def __getitem__(self, index):
        image_path, image_idx = self.image_infos[index]

        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract image name without extension

        mask_path = os.path.join(self.root_path, "masks", f"{image_name}_mask.png")

        input_image = cv2.imread(image_path)
        mask_image = cv2.imread(mask_path)
        prev_image = self.roll_image(input_image)

        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
            input_image, _, _ = cv2.split(input_image)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2YUV)
            mask_image, _, _ = cv2.split(mask_image)
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2YUV)
            prev_image, _, _ = cv2.split(prev_image)

        input_image = cv2.resize(input_image, self.image_sz)
        input_image = (input_image / 255.0).astype(np.float32)
        prev_image = cv2.resize(prev_image, self.image_sz)
        prev_image = (prev_image / 255.0).astype(np.float32)
        mask_image = cv2.resize(mask_image, self.image_sz)
        mask_image = (mask_image / 255.0).astype(np.float32)

        # transformed = self.apply_augmentations(images=[input_image])
        # images = self.images_to_tensors(transformed["images"])[0]
        images = [input_image, prev_image]
        [input_image, prev_image] = self.images_to_tensors(images)
        images = [input_image, prev_image]

        mask_image = self.images_to_tensors([mask_image])[0].long()

        return {"inputs": images, "mask": mask_image}


class AIRSDataset(Dataset):
    def __init__(self, root_path: str, image_sz=(800, 800), transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.transform = transform

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, "images", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "images")))
        ]
        print(f"Found {len(self.image_infos)} images in {root_path}")

        self.mask_infos = [
            [os.path.join(root_path, "masks", image_path), idx] for idx, image_path in enumerate(os.listdir(os.path.join(root_path, "masks")))
        ]

        print(f"Found {len(self.mask_infos)} mask images in {root_path}")

    def __len__(self):
        return len(self.image_infos)

    @staticmethod
    def collate_fn(data):
        keys = data[0].keys()

        if "bboxes" in keys:  # Fix the number of bboxes of each element with padding
            max_n_bboxes = max([len(elem["bboxes"]) for elem in data])
            for elem in data:
                bboxes_padded = np.zeros((max_n_bboxes, 4), dtype=float)
                class_labels_padded = elem["class_labels"] + [-1] * (max_n_bboxes - len(elem["class_labels"]))
                bboxes = elem["bboxes"]
                if len(bboxes) > 0:
                    bboxes_padded[: len(bboxes)] = bboxes

                elem["bboxes"] = bboxes_padded
                elem["class_labels"] = class_labels_padded

        data = torch.utils.data._utils.collate.default_collate(data)

        return data

    def roll_image(self, image):
        image_array = np.asarray(image)

        # Randomly choose shift values
        pixels_roll = np.random.choice([6, 8, 10, 12, 14])
        sign = np.random.choice([-1, 1])  # -1: rolls up-left, +1: rolls down-right
        pixels_roll = sign * pixels_roll

        # Define Albumentations ShiftScaleRotate transform
        transform = A.ShiftScaleRotate(shift_limit=(pixels_roll, pixels_roll), rotate_limit=0, scale_limit=0, p=1)

        # Apply the transform to the image
        augmented = transform(image=image_array)

        # Extract the augmented image
        rolled_image = augmented["image"]

        return rolled_image

    def __getitem__(self, index):
        image_path, image_idx = self.image_infos[index]

        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extract image name without extension

        if "quarter" in image_name:
            parts = image_name.split("_")
            quarter_index = parts.index("quarter")
            parts.insert(quarter_index, "vis")
            image_name = "_".join(parts)

        mask_path = os.path.join(self.root_path, "masks", f"{image_name}_mask.png")

        input_image = cv2.imread(image_path)
        mask_image = cv2.imread(mask_path)
        prev_image = self.roll_image(input_image)

        if input_image.shape[2] == 3:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
            input_image, _, _ = cv2.split(input_image)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2YUV)
            mask_image, _, _ = cv2.split(mask_image)
            prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2YUV)
            prev_image, _, _ = cv2.split(prev_image)

        input_image = cv2.resize(input_image, self.image_sz)
        input_image = (input_image / 255.0).astype(np.float32)
        prev_image = cv2.resize(prev_image, self.image_sz)
        prev_image = (prev_image / 255.0).astype(np.float32)
        mask_image = cv2.resize(mask_image, self.image_sz)
        mask_image = (mask_image / 255.0).astype(np.float32)

        # transformed = self.apply_augmentations(images=[input_image])
        # images = self.images_to_tensors(transformed["images"])[0]
        images = [input_image, prev_image]
        [input_image, prev_image] = self.images_to_tensors(images)
        images = [input_image, prev_image]

        mask_image = self.images_to_tensors([mask_image])[0].long()

        return {"inputs": images, "mask": mask_image}


class WeightedClassificationDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        classes: list[str],
        image_sz: tuple | list,
        extension: str | list | tuple = ["JPG", "PNG", "JPEG"],
        train=False,
        factor=0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.classes = classes
        self.extension = extension
        self.train = train
        self.factor = factor

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.image_infos = [
            [os.path.join(root_path, class_dir, image_path), idx]
            for idx, class_dir in enumerate(self.classes)
            for image_path in os.listdir(os.path.join(root_path, class_dir))
        ]
        self.class_map = {idx: class_name for idx, class_name in enumerate(classes)}
        print(f"Found {len(classes)} classes and {len(self)} images in {root_path}")

        self.__print_dataset_info()
        self.check_dataset_validity()

        if self.train:
            print("Before Process", len(self.image_infos))
            print("TRAIN STATUS:", self.train)
            unknown_indexes = [index for index, (_, label) in enumerate(self.image_infos) if label == self.classes.index("Unknown")]
            self.image_infos = self.balance_unknown_images(unknown_indexes, self.image_infos)
            print("After Process", len(self.image_infos), "\n")

    def balance_unknown_images(self, unknown_indexes, image_infos):
        balanced_image_infos = image_infos.copy()
        for _ in range(self.factor):
            balanced_image_infos.extend([image_infos[index] for index in unknown_indexes])
        return balanced_image_infos

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, index):
        img_path, label = self.image_infos[index]
        input_image = cv2.imread(img_path)

        # Check if image has 3 channels
        if input_image.shape[2] != 3:
            # Convert grayscale image to 3 channels
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
        input_image = cv2.resize(input_image, self.image_sz)
        input_image = (input_image / 255.0).astype(np.float32)

        transformed = self.apply_augmentations(images=[input_image])
        images = self.images_to_tensors(transformed["images"])[0]

        return {"inputs": images, "labels": label}

    def check_dataset_validity(self):
        """
        Checks the validity of given classification data.

        Raises
        ------
        FileNotFoundError
            Raised when any class directory cannot be found.
        TypeError
            Raised when a file does not contain any of the permitted extensions.
        NotADirectoryError
            Raised when there is a file present in the directory where the class folders are located.
        """
        # For each class there must be a folder named as the class name in the root path
        return
        class_dirs = os.listdir(self.root_path)
        for class_name in self.classes:
            if class_name not in class_dirs:
                raise FileNotFoundError(f"Class directory not found: {class_name}")

        for class_name in self.classes:
            if not os.path.isdir(os.path.join(self.root_path, class_name)):
                raise NotADirectoryError(f"'{class_name}' must be a directory named as one of the class names.")
            for file in os.listdir(os.path.join(self.root_path, class_name)):
                if isinstance(self.extension, str):
                    if not (file.endswith(self.extension)) or not (file.endswith(self.extension.lower())):
                        raise TypeError(f"Class directory {class_name} contains an unsupported image format: {file}.")
                else:
                    if not np.any([(file.endswith(ext) or file.endswith(ext.lower())) for ext in self.extension]):
                        raise TypeError(f"Class directory {class_name} contains an unsupported image format: {file}.")

    def __print_dataset_info(self):
        headers = [["id", "Class", "N_samples"]]
        data = [[idx, class_name, len(os.listdir(os.path.join(self.root_path, class_name)))] for idx, class_name in self.class_map.items()]
        table_data = headers + data
        table = tabulate(table_data, headers="firstrow", tablefmt="pipe")
        print(table)
