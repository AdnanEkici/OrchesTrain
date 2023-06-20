from __future__ import annotations

import glob
import os

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision

import orchestrain.augments as augments


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

    def __init__(self, root_path: str, classes: list[str], image_sz: tuple | list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        self.root_path = root_path
        self.classes = classes

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        class_dirs = os.listdir(root_path)

        for class_name in classes:
            if class_name not in class_dirs:
                raise AssertionError(f"Class directory not found: {class_name}")

        self.image_infos = [
            [os.path.join(root_path, class_dir, path), idx]
            for idx, class_dir in enumerate(class_dirs)
            for path in os.listdir(os.path.join(root_path, class_dir))
        ]

        self.class_map = {idx: class_name for idx, class_name in enumerate(classes)}

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


class CityScapesDataset(Dataset):
    def __init__(self, root_path: str, image_sz: tuple | list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not os.path.exists(root_path):
            raise AssertionError(f"Dataset root path not found: {root_path}")

        assert isinstance(image_sz, (tuple, list)) and len(image_sz) == 2, "Invalid image_sz format. Expected a tuple or list [width, height]."
        self.image_sz = image_sz

        self.root_path = root_path
        self.images = [
            [mask_path.replace("_gtFine_labelIds.png", "_leftImg8bit.png"), mask_path]
            for mask_path in glob.glob(self.root_path + "/**/*_labelIds.png")
            if os.path.exists(mask_path.replace("_gtFine_labelIds.png", "_leftImg8bit.png"))
        ]

        self.id_to_trainid_lookup_table = np.array([item["trainId"] for item in CityScapesDataset.get_label_maps()], dtype=np.uint8)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def get_label_maps():
        return [
            {"name": "unlabeled", "id": 0, "trainId": 19, "category": 0, "color": (0, 0, 0)},
            {"name": "ego vehicle", "id": 1, "trainId": 19, "category": 0, "color": (0, 0, 0)},
            {"name": "rectification border", "id": 2, "trainId": 19, "category": 0, "color": (0, 0, 0)},
            {"name": "out of roi", "id": 3, "trainId": 19, "category": 0, "color": (0, 0, 0)},
            {"name": "static", "id": 4, "trainId": 19, "category": 0, "color": (0, 0, 0)},
            {"name": "dynamic", "id": 5, "trainId": 19, "category": 0, "color": (111, 74, 0)},
            {"name": "ground", "id": 6, "trainId": 19, "category": 0, "color": (81, 0, 81)},
            {"name": "road", "id": 7, "trainId": 0, "category": 0, "color": (128, 64, 128)},
            {"name": "sidewalk", "id": 8, "trainId": 1, "category": 0, "color": (244, 35, 232)},
            {"name": "parking", "id": 9, "trainId": 19, "category": 0, "color": (250, 170, 160)},
            {"name": "rail track", "id": 10, "trainId": 19, "category": 0, "color": (230, 150, 140)},
            {"name": "building", "id": 11, "trainId": 2, "category": 0, "color": (70, 70, 70)},
            {"name": "wall", "id": 12, "trainId": 3, "category": 0, "color": (102, 102, 156)},
            {"name": "fence", "id": 13, "trainId": 4, "category": 0, "color": (190, 153, 153)},
            {"name": "guard rail", "id": 14, "trainId": 19, "category": 0, "color": (180, 165, 180)},
            {"name": "bridge", "id": 15, "trainId": 19, "category": 0, "color": (150, 100, 100)},
            {"name": "tunnel", "id": 16, "trainId": 19, "category": 0, "color": (150, 120, 90)},
            {"name": "pole", "id": 17, "trainId": 5, "category": 0, "color": (153, 153, 153)},
            {"name": "polegroup", "id": 18, "trainId": 19, "category": 0, "color": (153, 153, 153)},
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
            {"name": "caravan", "id": 29, "trainId": 19, "category": 0, "color": (0, 0, 90)},
            {"name": "trailer", "id": 30, "trainId": 19, "category": 0, "color": (0, 0, 110)},
            {"name": "train", "id": 31, "trainId": 16, "category": 0, "color": (0, 80, 100)},
            {"name": "motorcycle", "id": 32, "trainId": 17, "category": 0, "color": (0, 0, 230)},
            {"name": "bicycle", "id": 33, "trainId": 18, "category": 0, "color": (119, 11, 32)},
            {"name": "license plate", "id": 34, "trainId": 19, "category": 0, "color": (0, 0, 142)},
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
        mask = cv2.resize(mask, self.image_sz, interpolation=cv2.INTER_NEAREST)

        mask = self.id_to_trainid_lookup_table[mask]

        transformed = self.apply_augmentations(images=[img], masks=[mask])
        image = self.images_to_tensors(transformed["images"])[0]
        mask = (self.images_to_tensors(transformed["masks"])[0] * 255).squeeze().long()

        return {"inputs": image, "mask": mask}
