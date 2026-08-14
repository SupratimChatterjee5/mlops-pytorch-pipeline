import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(architecture: str="resnet18", num_classes: int=10, pretrained: bool=True)-> nn.Module:
    if architecture.lower()=="resnet18":
        weights=ResNet18_Weights.DEFAULT if pretrained else None
        model=resnet18(weights=weights)
        #Adapt first conv layer for 32*32 images (CIFAR-10) to avoid over-downsampling
        model.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        model.maxpool=nn.Identity()
        model.fc=nn.Linear(model.fc.in_features,num_classes)
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")