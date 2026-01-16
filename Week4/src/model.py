"""
This file contains the RepVGG model architecture.
"""
import torch
import torch.nn as nn

from src.blocks import RepBlock, ConvBN
from src.fusion import fuse_conv_bn, pad_1x1_to_3x3, identity_kernel_3x3



class Stage(nn.Module):
    """A stack of RepBlocks at one resolution."""
    def __init__(self, in_ch, out_ch, num_blocks, downsample):
        super().__init__()
        blocks = []
        stride0 = 2 if downsample else 1
        blocks.append(RepBlock(in_ch, out_ch, stride=stride0))
        for _ in range(num_blocks - 1):
            blocks.append(RepBlock(out_ch, out_ch, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class RepNet(nn.Module):
    """
    A simple RepVGG-like classifier.
      stem -> stages -> global avg pool -> fc
    """
    def __init__(self, num_classes, cfg, in_ch=3):
        super().__init__()
        # stem
        stem_ch = cfg["stem_ch"]
        self.stem = nn.Sequential(
            ConvBN(in_ch, stem_ch, k=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # stages: list of (out_ch, num_blocks, downsample)
        stages = []
        prev = stem_ch
        for out_ch, num_blocks, downsample in cfg["stages"]:
            stages.append(Stage(prev, out_ch, num_blocks, downsample))
            prev = out_ch
        self.stages = nn.Sequential(*stages)

        # At latter stages the features are semantically rich enough
        # so we can use average pooling. Applying it too early would
        # destroy needed spatial info.
        # Why not just flatten? too big of a vector, has noise. With AvgPool
        # apply strong regularization and reduce params.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

@torch.no_grad()
def switch_model_to_deploy(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, RepBlock):
            m.switch_to_deploy()