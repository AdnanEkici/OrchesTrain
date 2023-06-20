# OrchesTrain

OrchesTrain is a Pytorch-based toolbox for training and running classification, segmentation and object detection models.

# Training

To train any model in OrchesTrain you first need to create a config that defines dataset properties, model type, loss function and other training stuff. You can see examples for segmentation models in `data/semantic_segmentation` and examples for classification in `data/classification/`. Detailed explanation of configs is given in **Configs** section.

## Semantic Segmentation

To train segmentation models you only need to create/choose a config file that defines your experiment properties.

Example training command for segmentation:

```
python train.py train --data ./data/semantic_segmentation/cityscapes.yml --project cityscapes
```

## Classification

You can train torchvision models to make transfer learning or you can create a new custom model. We have provided an example of Resnet18 transfer learning training yaml configuarion and also a custom model training configuarion. We also support torchvision datasets, you can see example training configuration in `data/classification/cifar10.yaml`.

Example training command for classification:

```
python train.py train --data ./data/classification/cifar10.yml --project cifar10
```

## Configs

Designing an experiment config

* Currently for segmentation experiments all features are supported cityscapes and for classification experiments you can use ClassificationDataset and TorchVisionDataset. You can simply create a dataset for segmentation

    ```
    dataset:
        train:
            type: CityScapesDataset
            num_workers: 6
            image_sz: &image_sz [1024, 512]
            root_path: cityscapes/gtFine_trainvaltest/gtFine/train
            augment:
            A.HorizontalFlip:
                p: 0.5
            """Other augmentation methods"""
        valid:
            type: CityScapesDataset
            num_workers: 6
            image_sz: *image_sz
            root_path: cityscapes/gtFine_trainvaltest/gtFine/val
    ```

* You can specify augmentations as

    ```
    augment:
        A.Rotate:
            limit: [-10,10]
        A.VerticalFlip
    ```

We support all of the albumentation augmentations. Also you can write your own custom augmentation class.

* You can specify your model class name like:

    ```
    model:
    type: UNet
    ```

    but you need to be careful about that you have to specify path of the given class.

    ```
    model:
    type: UNet_ResPath
    import_from: train_app.models.semantic_segmentation.unet
    ```

    you can also send argumants to your model class with args:

    ```
    args:
        in_channels: 3
        out_channels: 20
    ```

* Defining loss function is can be done like:

    ```
    losses:
    - loss:
        type: CrossEntropyLoss
        weights: 1
    ```

    you can change args with respect to your loss function.

# Scripts

* [converter](/converter.py) is used to convert trained model to torch script which can be used in c++ and other applications.

# Known Issues
