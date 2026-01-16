"""
This file contains the building blocks of the RepVGG architecture.
"""
import torch
import torch.nn as nn

from src.fusion import fuse_conv_bn, pad_1x1_to_3x3, identity_kernel_3x3



class ConvBN(nn.Module):
    """Conv2d -> BatchNorm2d. No activation."""
    def __init__(self, in_ch, out_ch, k, stride=1, padding=None, bias=False):
        super().__init__()
        if padding is None:
            # by floor division by 2 we get same in to out shape if stride = 1
            padding = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.conv(x))


class IdentityBN(nn.Module):
    """Identity path but with BN (as in RepVGG training block)."""
    def __init__(self, ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.bn(x)


class RepBlock(nn.Module):
    """
    Training-time RepVGG block:
      out = (3x3 conv+bn) + (1x1 conv+bn) + (id+bn if possible)
      out = ReLU(out)
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        self.rbr_3x3 = ConvBN(in_ch, out_ch, k=3, stride=stride, padding=1, bias=False)

        # Padding should be 0, is a 1x1 conv!
        self.rbr_1x1 = ConvBN(in_ch, out_ch, k=1, stride=stride, padding=0, bias=False)

        # Can use them only between stages, not before (when we downsample)
        self.use_identity = (in_ch == out_ch and stride == 1)
        self.rbr_id = IdentityBN(out_ch) if self.use_identity else None

        self.activation = nn.ReLU(inplace=True)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self):
        """
        Create one equivalent 3x3 kernel and bias by:
        - fusing BN into each branch (Eq. 3–4)
        - converting 1x1 to 3x3 by centering
        - converting identity to conv kernel
        - summing kernels and biases (Fig. 4)
        :contentReference[oaicite:14]{index=14}
        """
        W3, b3 = fuse_conv_bn(self.rbr_3x3.conv, self.rbr_3x3.bn)
        W1, b1 = fuse_conv_bn(self.rbr_1x1.conv, self.rbr_1x1.bn)
        W1 = pad_1x1_to_3x3(W1)

        if self.rbr_id is not None:
            Wid = identity_kernel_3x3(self.out_ch, device=W3.device, dtype=W3.dtype)
            # fuse identity BN: same as applying BN to conv output with zero bias
            bn = self.rbr_id.bn
            gamma, beta = bn.weight, bn.bias
            mean, var, eps = bn.running_mean, bn.running_var, bn.eps
            std = torch.sqrt(var + eps)
            scale = gamma / std

            Wid = Wid * scale.reshape(-1, 1, 1, 1)
            bid = (-mean) * scale + beta
        else:
            Wid = torch.zeros_like(W3)
            bid = torch.zeros_like(b3)

        W = W3 + W1 + Wid
        b = b3 + b1 + bid
        return W, b

    @torch.no_grad()
    def switch_to_deploy(self):
        """
        Replace multi-branch structure with a single Conv2d+ReLU.
        Must call in eval mode so BN uses running stats. :contentReference[oaicite:15]{index=15}
        """
        W, b = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(
            self.in_ch, self.out_ch, kernel_size=3, stride=self.stride, padding=1, bias=True
        )
        self.rbr_reparam.weight.data.copy_(W)
        self.rbr_reparam.bias.data.copy_(b)

        # remove branches
        del self.rbr_3x3, self.rbr_1x1
        if self.rbr_id is not None:
            del self.rbr_id
        self.rbr_id = None

    def forward(self, x):
        if hasattr(self, "rbr_reparam"):
            return self.activation(self.rbr_reparam(x))
        
        out = self.rbr_3x3(x) + self.rbr_1x1(x)
        if self.rbr_id is not None:
            out = out + self.rbr_id(x)
        return self.activation(out)