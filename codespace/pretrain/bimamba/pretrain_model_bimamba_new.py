# --------------------------------------------------------
# part of code borrowed from Quert2Label
# Written by Zhourun Wu
# --------------------------------------------------------
import sys
import os
import os.path as osp
from transformers import AutoTokenizer, AutoModel

import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import copy

from codespace.model.multihead_attention_bimamba_concat_new import (
    build_transformerEncoder,
    _get_activation_fn,
)

from mamba_ssm import Mamba


class Pre_Train_Model(nn.Module):
    def __init__(
        self, transformerEncoder, transformerDecoder, feature_len, activation, dropout
    ):
        # feature_len=modefeature_lens = [X.shape[1], Z.shape[1]]即X的列数和Z的列数
        # 其中X是PPI信息的矩阵，Z是蛋白质结构域和亚细胞位置信息的矩阵

        super().__init__()
        self.transformerEncoder = transformerEncoder
        self.transformerDecoder = transformerDecoder
        # dim_feedforward是自己设定的 512
        # dim_feedforward = transformerEncoder.dim_feedforward
        dim_feedforward = transformerEncoder.dim_feedforward

        self.input_proj_x1 = nn.Linear(feature_len[0], dim_feedforward * 2)
        self.input_proj_z1 = nn.Linear(feature_len[1], dim_feedforward * 2)

        self.input_proj_x2 = nn.Linear(dim_feedforward * 2, dim_feedforward)
        self.input_proj_z2 = nn.Linear(dim_feedforward * 2, dim_feedforward)

        self.norm_x1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_z1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_x2 = nn.LayerNorm(dim_feedforward)
        self.norm_z2 = nn.LayerNorm(dim_feedforward)

        self.dropout_x1 = nn.Dropout(dropout)
        self.dropout_z1 = nn.Dropout(dropout)
        self.dropout_x2 = nn.Dropout(dropout)
        self.dropout_z2 = nn.Dropout(dropout)

        self.activation_x1 = copy.deepcopy(activation)
        self.activation_z1 = copy.deepcopy(activation)
        self.activation_x2 = copy.deepcopy(activation)
        self.activation_z2 = copy.deepcopy(activation)

        self.W_x1 = nn.Linear(dim_feedforward, dim_feedforward * 2)
        self.W_z1 = nn.Linear(dim_feedforward, dim_feedforward * 2)
        self.W_x2 = nn.Linear(dim_feedforward * 2, feature_len[0])
        self.W_z2 = nn.Linear(dim_feedforward * 2, feature_len[1])

        self.activation_wx = copy.deepcopy(activation)
        self.activation_wz = copy.deepcopy(activation)

        self.dropout_wx = nn.Dropout(dropout)
        self.dropout_wz = nn.Dropout(dropout)

        self.norm_wx = nn.LayerNorm(dim_feedforward * 2)
        self.norm_wz = nn.LayerNorm(dim_feedforward * 2)

    def forward(self, src):

        # ----------------------------多层感知机MLP---------------------------------------

        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_x = src[0]  # 32,19385
        in_x = self.input_proj_x1(in_x)  # 线性层，输入X的列数19385，输出512*2
        in_x = self.norm_x1(in_x)  # 标准化层，512*2
        in_x = self.activation_x1(in_x)  # 激活层 gelu
        in_x = self.dropout_x1(in_x)  # dropout
        in_x = in_x  # 32,1024

        # src[1]是蛋白质结构域和亚细胞位置信息的矩阵
        # batch_size=32因此共32行，1389列
        in_z = src[1]  # 32,1389
        in_z = self.input_proj_z1(in_z)  # 线性层，输入Z的列数1389，输出512*2
        in_z = self.norm_z1(in_z)
        in_z = self.activation_z1(in_z)
        in_z = self.dropout_z1(in_z)
        in_z = in_z  # 32,1024

        in_x = in_x  # 32,1024
        in_x = self.input_proj_x2(in_x)  # 线性层，输入512*2，输出512
        in_x = self.norm_x2(in_x)  # 标准化层，512
        in_x = self.activation_x2(in_x)  # 激活层 gelu
        in_x = self.dropout_x2(in_x)  # dropout
        in_x = in_x  # 32,512

        in_z = in_z
        in_z = self.input_proj_z2(in_z)
        in_z = self.norm_z2(in_z)
        in_z = self.activation_z2(in_z)
        in_z = self.dropout_z2(in_z)
        in_z = in_z  # 32,512

        # ----------------------------多头注意力层---------------------------------------
        in_x = in_x.unsqueeze(0)  # 1,32,512
        in_z = in_z.unsqueeze(0)  # 1,32,512

        # 按第0维拼接
        in_put = torch.cat([in_x, in_z], 0)  # 2,32,512

        # 编码器解码器
        hs = self.transformerEncoder(in_put)  # B,K,d
        # 2,32,512
        rec = self.transformerDecoder(hs)  # B,K,d

        # ----------------------------多层感知机MLP---------------------------------------

        ph_x = self.W_x1(rec[0])
        ph_x = self.norm_wx(ph_x)
        ph_x = self.activation_wx(ph_x)
        ph_x = self.dropout_wx(ph_x)

        ph_z = self.W_z1(rec[1])
        ph_z = self.norm_wz(ph_z)
        ph_z = self.activation_wz(ph_z)
        ph_z = self.dropout_wz(ph_z)

        rec_x = self.W_x2(ph_x)
        rec_z = self.W_z2(ph_z)
        # import ipdb; ipdb.set_trace()

        return (rec_x, rec_z), hs


# 创建预训练模型
def build_Pre_Train_Model(args):

    model = Pre_Train_Model(
        transformerEncoder=build_transformerEncoder(args),
        transformerDecoder=build_transformerEncoder(args),
        feature_len=args.modesfeature_len,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
    )

    return model


# 测试
if __name__ == "__main__":
    # 创建模型实例
    model = Pre_Train_Model(
        transformerEncoder=None,
        transformerDecoder=None,
        feature_len=1024,
        activation=_get_activation_fn("gelu"),
        dropout=0.1,
    )
    # 打印模型结构
    print(model)
