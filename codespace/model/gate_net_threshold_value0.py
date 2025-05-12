import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1, threshold=None):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        if threshold is not None:
            # 阈值判断：大于阈值的元素设置为1，其他设置为0
            # y_hard = (y_soft > threshold).float()
            y_hard = y_soft * (y_soft >= threshold).float()
        else:
            # 标准的硬Max操作：取最大值位置为1，其余为0
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)

        # Straight-through trick：保留梯度信息
        ret = y_hard - y_soft.detach() + y_soft

    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, y_soft


class GateNet(nn.Module):
    def __init__(self, input_dim, branch_num, hard=True, threshold=None):
        super(GateNet, self).__init__()
        self.hard = hard
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, branch_num)
        )
        self.threshold = threshold

    def forward(self, x):
        logits = self.MLP(x)
        weight, x_soft = DiffSoftmax(
            logits, tau=1.0, hard=self.hard, threshold=self.threshold
        )  # 硬门控
        return weight, x_soft


# output = weight[:, 0:1] * self.branch1(inputs[0]) + weight[:, 1:2] * self.branch2(inputs[1])
if __name__ == "__main__":
    block = GateNet(45 * 4, 4)
    # input = torch.rand(32, 45 * 2)

    # output = block(input)
    # print(input.size(), output.size())
    # print(output)

    fusion_hs = torch.rand(4, 32, 45)
    fusion_hs_flatten = torch.einsum("LBD->BLD", fusion_hs).flatten(1)  # 32,45*2
    fusion_hs_permuted = torch.einsum("LBD->BLD", fusion_hs)  # 32, 2,45
    weight = block(fusion_hs_flatten)
    weighted_gate_hs1 = weight[:, 0:1] * fusion_hs[0]  # 2,32,45
    weighted_gate_hs2 = weight[:, 1:2] * fusion_hs[1]  # 2,32,45
    weighted_gate_hs0 = torch.stack([weighted_gate_hs1, weighted_gate_hs2], dim=0)

    # weight = weight.unsqueeze(-1)  # 32,2,1
    # weighted_gate_hs = fusion_hs_permuted * weight  # 32, 2, 45
    # weighted_gate_hs = torch.einsum("BLD->LBD", weighted_gate_hs)  # 2, 32, 45
    # weighted_gate_hs = torch.sum(weighted_gate_hs, dim=1)  # 32, 45

    print(weighted_gate_hs0.size(), weight.shape)
