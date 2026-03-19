import torchvision.models as models
import torch.nn as nn


class BaseResNet18(nn.Module): # pretrained ResNet18
    def __init__(self, num_class=2, freeze_backbone=True):
        super(BaseResNet18, self).__init__()

        self.model = models.resnet18(weights='IMAGENET1K_V1')

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_class)
        
    def forward(self, x):
        return self.model(x)


class CustomCNN(nn.Module):
    def __init__(self, num_class=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),         # 224*224->112*112

            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),         # 112*112->56*56

           
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), # 4*4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_class),
        )

    def forward(self, x):
        return self.classifier(self.features(x))