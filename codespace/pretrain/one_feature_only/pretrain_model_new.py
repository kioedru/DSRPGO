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
from codespace.model.multihead_attention_transformer_new import (
            build_transformerEncoder,
            _get_activation_fn,
        )

# from mamba_ssm import Mamba


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

        self.input_proj_x2 = nn.Linear(dim_feedforward * 2, dim_feedforward)

        self.norm_x1 = nn.LayerNorm(dim_feedforward * 2)
        self.norm_x2 = nn.LayerNorm(dim_feedforward)

        self.dropout_x1 = nn.Dropout(dropout)
        self.dropout_x2 = nn.Dropout(dropout)

        self.activation_x1 = copy.deepcopy(activation)
        self.activation_x2 = copy.deepcopy(activation)

        self.W_x1 = nn.Linear(dim_feedforward, dim_feedforward * 2)
        self.W_x2 = nn.Linear(dim_feedforward * 2, feature_len[0])

        self.activation_wx = copy.deepcopy(activation)

        self.dropout_wx = nn.Dropout(dropout)

        self.norm_wx = nn.LayerNorm(dim_feedforward * 2)

    def forward(self, src):

        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_x = src[0]  # 32,19385
        in_x = self.input_proj_x1(in_x)  # 线性层，输入X的列数19385，输出512*2
        in_x = self.norm_x1(in_x)  # 标准化层，512*2
        in_x = self.activation_x1(in_x)  # 激活层 gelu
        in_x = self.dropout_x1(in_x)  # dropout
        in_x = in_x  # 32,1024

        in_x = in_x  # 32,1024
        in_x = self.input_proj_x2(in_x)  # 线性层，输入512*2，输出512
        in_x = self.norm_x2(in_x)  # 标准化层，512
        in_x = self.activation_x2(in_x)  # 激活层 gelu
        in_x = self.dropout_x2(in_x)  # dropout
        in_x = in_x  # 32,512

        # ----------------------------多头注意力层---------------------------------------
        in_x = in_x.unsqueeze(0)  # 1,32,512

        # in_put = torch.cat([in_x, in_z, in_s], 0)  # 3,32,512
        in_put = in_x  # 1,32,512
        # 编码器解码器
        hs = self.transformerEncoder(in_put)  # B,K,d
        # 2,32,512
        rec = self.transformerDecoder(hs)  # B,K,d

        # ----------------------------多层感知机MLP---------------------------------------

        ph_x = self.W_x1(rec[0])
        ph_x = self.norm_wx(ph_x)
        ph_x = self.activation_wx(ph_x)
        ph_x = self.dropout_wx(ph_x)

        rec_x = self.W_x2(ph_x)
        # import ipdb; ipdb.set_trace()
        rec_x = rec_x.unsqueeze(0)
        return (rec_x), hs


# 创建预训练模型
def build_Pre_Train_Model(args, featurelen=[1024]):
        

    model = Pre_Train_Model(
        transformerEncoder=build_transformerEncoder(args),
        transformerDecoder=build_transformerEncoder(args),
        feature_len=featurelen,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
    )

    return model


# if __name__ == "__main__":
#     # 创建模型实例
#     model = Pre_Train_Model(
#         transformerEncoder=None,
#         transformerDecoder=None,
#         feature_len=1024,
#         activation=_get_activation_fn("gelu"),
#         dropout=0.1,
#     )
#     # 打印模型结构
#     print(model)
