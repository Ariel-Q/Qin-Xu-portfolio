import torch
from efficientformer_v2 import (
    EfficientFormer_depth,
    EfficientFormer_width,
    EfficientFormerV2,
    expansion_ratios_S2,
)
from torch import nn
from torchvision.models import resnet50


NEW_ATTRIBUTES = [
    # 1. basic info
    "Male",
    "Young",
    "Attractive",
    "Blurry",

    # 2. facial and skin
    "Oval_Face",
    "Chubby",
    "Double_Chin",
    "High_Cheekbones",
    "Rosy_Cheeks",
    "Pale_Skin",

    # 3. hair and head
    "Bald",
    "Receding_Hairline",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Straight_Hair",
    "Wavy_Hair",
    "Bangs",
    "Wearing_Hat",

    # 4. eyes and eyebrows
    "Arched_Eyebrows",
    "Bushy_Eyebrows",
    "Bags_Under_Eyes",
    "Narrow_Eyes",
    "Eyeglasses",

    # 5. nose and mouth
    "Big_Nose",
    "Pointy_Nose",
    "Big_Lips",
    "Mouth_Slightly_Open",
    "5_o_Clock_Shadow",
    "Mustache",
    "Goatee",
    "Sideburns",
    "No_Beard",

    # 6. accessories
    "Heavy_Makeup",
    "Wearing_Lipstick",
    "Wearing_Earrings",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Smiling",
]

class MyModel(nn.Module):
    def __init__(self, batch_size=32) -> None:
        super().__init__()

        self.batch_size = batch_size

        efficientformer = EfficientFormerV2(
            layers=EfficientFormer_depth["S2"],
            embed_dims=EfficientFormer_width["S2"],
            downsamples=[True, True, True, True, True],
            vit_num=2,
            e_ratios=expansion_ratios_S2,
            num_classes=0,
            resolution=178,
            distillation=False,
        )
        self.network = nn.Sequential(efficientformer, Head(), nn.Sigmoid())

    def forward(self, x):
        return self.network(x)


def head_block(output_features):
    num_features = EfficientFormer_width["S2"][-1]
    return nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, output_features),
    )


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        groups = [4,6,10,5,9,6]
        self.blocks = nn.ModuleList([head_block(group) for group in groups])

    def forward(self, x):
        output = [block(x) for block in self.blocks]
        output = torch.cat(output, dim=1)
        return output



