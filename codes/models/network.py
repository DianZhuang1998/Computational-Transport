import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torchvision import models

import segmentation_models_pytorch as smp


class Classifier(nn.Module):
    def __init__(self, inchannels=3, classes=2):
        super().__init__()
        encoder_name = 'densenet121'
        in_channels = inchannels
        nclasses = classes

        self.model = smp.Linknet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            encoder_weights='imagenet',
        )
        self.encoder = self.model.encoder
        self.feat_dims = self.encoder.out_channels[-1]
        self.avgpool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Linear(self.feat_dims, nclasses)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        y = self.fc(self.dropout(x))

        return y