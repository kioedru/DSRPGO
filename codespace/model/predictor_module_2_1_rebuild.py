import torch
import torch.nn as nn
import math

from codespace.model.multihead_attention_transformer import _get_activation_fn,build_transformerEncoder


class FC_Decoder(nn.Module):
    def __init__(self, num_class, dim_feedforward, activation, dropout, input_num=3):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(
            dim_feedforward * input_num, dim_feedforward // input_num
        )
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // input_num, num_class)

    def forward(self, hs):  # hs[3, 32, 512]
        # 维度转换 第0维和第1维互换
        hs = hs.permute(1, 0, 2)  # [32, 3, 512]
        # 按第一维度展开
        hs = hs.flatten(1)  # [32,1536]
        # 默认(512*2,512//2)，//表示下取整
        hs = self.output_layer1(hs)
        # sigmoid
        hs = self.activation1(hs)
        hs = self.dropout1(hs)
        # (512//2,GO标签数)
        out = self.output_layer3(hs)
        # 后面还需要一个sigmoid，在测试输出后面直接加了
        return out


class Predictor(nn.Module):
    def __init__(
        self,
        args,
        seq_pre_model,
        ppi_feature_pre_model,
        num_class,
        dim_feedforward,
        activation,
        dropout,
        input_num=3,
    ):
        super().__init__()
        self.seq_pre_model = seq_pre_model
        self.ppi_feature_pre_model = ppi_feature_pre_model

        self.fusion = build_transformerEncoder(args)
        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout,
            input_num=input_num,
        )

    def forward(self, src):
        ppi_feature_src = src  # 2，32，512
        seq_src = src[2].unsqueeze(0)  # 1，32，512
        _, hs_ppi_feature = self.ppi_feature_pre_model(ppi_feature_src)
        _, hs_seq = self.seq_pre_model(seq_src)
        hs = torch.cat([hs_ppi_feature, hs_seq], dim=0)  # 4,32,512
        if self.fusion:
            hs = self.fusion(hs)

        out = self.fc_decoder(hs)
        return hs, out


def build_predictor(seq_pre_model, ppi_feature_pre_model, args):
    predictor = Predictor(
        args,
        seq_pre_model=seq_pre_model,
        ppi_feature_pre_model=ppi_feature_pre_model,
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=args.input_num,
    )

    return predictor
