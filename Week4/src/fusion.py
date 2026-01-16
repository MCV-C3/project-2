"""
This file contains the functions to fuse the three branches of RepVGG into one.
"""
import torch
import torch.nn as nn



# Conv + BN can always be rewritten as a single Conv with new weights and bias.
@torch.no_grad()
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """
    Implements Eq. (3) from RepVGG: fuse BN into Conv.
    Returns (W_fused, b_fused).
    """
    W = conv.weight
    if conv.bias is None:
        b = torch.zeros(W.size(0), device=W.device, dtype=W.dtype)
    else:
        b = conv.bias

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    scale = gamma / std

    W_fused = W * scale.reshape(-1, 1, 1, 1)
    b_fused = (b - mean) * scale + beta
    return W_fused, b_fused


def pad_1x1_to_3x3(W_1x1):
    """Put a 1x1 kernel into the center of a 3x3 kernel."""
    out_ch, in_ch, _, _ = W_1x1.shape
    W_3x3 = torch.zeros((out_ch, in_ch, 3, 3), device=W_1x1.device, dtype=W_1x1.dtype)
    W_3x3[:, :, 1, 1] = W_1x1[:, :, 0, 0]
    return W_3x3


def identity_kernel_3x3(ch, device, dtype):
    """3x3 identity conv kernel: 1 on center diagonal."""
    W = torch.zeros((ch, ch, 3, 3), device=device, dtype=dtype)
    for i in range(ch):
        W[i, i, 1, 1] = 1.0
    return W