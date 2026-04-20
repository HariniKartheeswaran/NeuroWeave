import torch
import torch.nn as nn
import torchvision.models as models

def load_resnet():
    # Using a specialized CIFAR-10 pre-trained model instead of ImageNet model
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    model.eval()
    return model

# Simple CNN (lightweight)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32 -> 16x16

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def load_cnn():
    model = SimpleCNN()
    model.eval()
    return model