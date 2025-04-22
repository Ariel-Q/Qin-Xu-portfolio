import torch.nn as nn
from torchvision.models import resnet50

class CNNMultiLabel(nn.Module):
    def __init__(self, num_labels=40):
        super().__init__()
        base = resnet50(pretrained=True)
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)
