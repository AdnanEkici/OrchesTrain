from __future__ import annotations

import torch.nn as nn
import torchvision

from orchestrain.models.base import ClassificationAdapter


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
