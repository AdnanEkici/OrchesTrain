from __future__ import annotations

import torch.nn as nn
import torchvision

from orchestrain.models.base import ClassificationAdapter


class Resnet3D(ClassificationAdapter):
    def __init__(self, num_classes, pretrained, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.video.r3d_18(pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def __call__(self, inp):
        return self.model(inp)


class Resnet18(ClassificationAdapter):
    def __init__(self, num_classes, pretrained, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def __call__(self, inp):
        return self.model(inp)


class Resnet34(ClassificationAdapter):
    def __init__(self, num_classes, pretrained, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.resnet34(pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def __call__(self, inp):
        return self.model(inp)


class Resnet50(ClassificationAdapter):
    def __init__(self, num_classes, pretrained, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def __call__(self, inp):
        return self.model(inp)


class ConvNext_Tiny(ClassificationAdapter):
    def __init__(self, num_classes, pretrained, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.convnext_tiny(pretrained=pretrained)
        n_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(n_features, num_classes)

    def __call__(self, inp):
        return self.model(inp)


class Swin_v2_Tiny(ClassificationAdapter):
    def __init__(self, num_classes, pretrained, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.swin_v2_t(pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, num_classes)

    def __call__(self, inp):
        return self.model(inp)


class EfficientNetB4(ClassificationAdapter):
    def __init__(self, num_classes, pretrained, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.efficientnet_b4(pretrained=pretrained)
        n_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(n_features, num_classes)

    def __call__(self, inp):
        return self.model(inp)
