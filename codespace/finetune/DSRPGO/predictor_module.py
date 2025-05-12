import torch
import torch.nn as nn
import math

from codespace.model.multihead_attention_transformer import _get_activation_fn
from codespace.model.gate_net_threshold_value0 import GateNet


class Fusion_Model39(nn.Module):
    def __init__(self, args, dim_feedforward):
        super().__init__()

        self.gate = GateNet(dim_feedforward * 3, 3, hard=False)
        self.decoder_model39 = FC_Decoder(
            args.num_class,
            dim_feedforward,
            _get_activation_fn(args.activation),
            args.dropout,
            input_num=3,
        )

    def forward(self, hs_fc, hs_model39):

        fusion_hs = hs_model39  # 3,32,512
        fusion_hs_flatten = torch.einsum("LBD->BLD", fusion_hs).flatten(1)  # 32,512*3
        fusion_hs_permuted = torch.einsum("LBD->BLD", fusion_hs)  # 32, 3, 512
        weight = self.gate(fusion_hs_flatten).unsqueeze(-1)  # 32,3,1
        weighted_gate_hs = fusion_hs_permuted * weight  # 32, 3, 512
        weighted_gate_hs = torch.einsum("BLD->LBD", weighted_gate_hs)  # 3, 32, 512

        model39_fusion = self.decoder_model39(weighted_gate_hs)  # 32,45
        return model39_fusion


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
        pre_model17,
        num_class,
        dim_feedforward,
        activation,
        dropout,
        input_num=3,
        db=False,
    ):
        super().__init__()
        self.db = db
        self.attention_layers_num = args.attention_layers
        self.seq_pre_model = seq_pre_model
        self.ppi_feature_pre_model = ppi_feature_pre_model
        self.pre_model17 = pre_model17

        if args.t_learn is True:
            self.expert_threshold = nn.Parameter(torch.tensor(args.t))
        else:
            self.expert_threshold = args.t
        self.gate = GateNet(
            dim_feedforward * 6, 6, hard=True, threshold=self.expert_threshold
        )
        self.fc_decoder = FC_Decoder(
            num_class, dim_feedforward, activation, dropout, input_num=6
        )

    def forward(self, src):
        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_x = src[0]  # 32,19385
        in_x = self.ppi_feature_pre_model.input_proj_x1(
            in_x
        )  # 线性层，输入X的列数19385，输出512*2
        in_x = self.ppi_feature_pre_model.norm_x1(in_x)  # 标准化层，512*2
        in_x = self.ppi_feature_pre_model.activation_x1(in_x)  # 激活层 gelu
        in_x = self.ppi_feature_pre_model.dropout_x1(in_x)  # dropout

        # src[1]是蛋白质结构域和亚细胞位置信息的矩阵
        # batch_size=32因此共32行，1389列
        in_z = src[1]  # 32,1389
        in_z = self.ppi_feature_pre_model.input_proj_z1(
            in_z
        )  # 线性层，输入Z的列数1389，输出512*2
        in_z = self.ppi_feature_pre_model.norm_z1(in_z)
        in_z = self.ppi_feature_pre_model.activation_z1(in_z)
        in_z = self.ppi_feature_pre_model.dropout_z1(in_z)

        in_x = self.ppi_feature_pre_model.input_proj_x2(
            in_x
        )  # 线性层，输入512*2，输出512
        in_x = self.ppi_feature_pre_model.norm_x2(in_x)  # 标准化层，512
        in_x = self.ppi_feature_pre_model.activation_x2(in_x)  # 激活层 gelu
        in_x = self.ppi_feature_pre_model.dropout_x2(in_x)  # dropout

        in_z = self.ppi_feature_pre_model.input_proj_z2(in_z)
        in_z = self.ppi_feature_pre_model.norm_z2(in_z)
        in_z = self.ppi_feature_pre_model.activation_z2(in_z)
        in_z = self.ppi_feature_pre_model.dropout_z2(in_z)

        in_x = in_x.unsqueeze(0)  # 1,32,512
        in_z = in_z.unsqueeze(0)  # 1,32,512
        ppi_feature_fc = torch.cat([in_x, in_z], dim=0)  # 2,32,512

        # src[0]是PPI信息的矩阵
        # batch_size=32因此共32行，19385列
        in_s = src[2]  # 1,32,19385
        in_s = self.seq_pre_model.input_proj_x1(
            in_s
        )  # 线性层，输入X的列数19385，输出512*2
        in_s = self.seq_pre_model.norm_x1(in_s)  # 标准化层，512*2
        in_s = self.seq_pre_model.activation_x1(in_s)  # 激活层 gelu
        in_s = self.seq_pre_model.dropout_x1(in_s)  # dropout

        in_s = self.seq_pre_model.input_proj_x2(in_s)  # 线性层，输入512*2，输出512
        in_s = self.seq_pre_model.norm_x2(in_s)  # 标准化层，512
        in_s = self.seq_pre_model.activation_x2(in_s)  # 激活层 gelu
        in_s = self.seq_pre_model.dropout_x2(in_s)  # dropout

        # ----------------------------多头注意力层---------------------------------------
        in_s = in_s.unsqueeze(0)  # 1,32,512
        seq_fc = in_s  # 1,32,512

        hs_fc = torch.cat([ppi_feature_fc, seq_fc], dim=0)  # 3,32,512

        ppi_feature_src = src  # 3,32,19385\1389\1024
        seq_src = src[2].unsqueeze(0)  # 1,32,1024

        # 获取model39的encoder结果
        # ppi_feature的预训练模型的Encoder部分
        ppi_feature_encoder_output = ppi_feature_fc  # 2,32,512
        seq_encoder_output = seq_fc  # 1,32,512
        for num in range(self.attention_layers_num):

            ppi_feature_output = (
                self.ppi_feature_pre_model.transformerEncoder.encoder.layers[num](
                    ppi_feature_encoder_output, seq_encoder_output
                )
            )
            seq_output = self.seq_pre_model.transformerEncoder.encoder.layers[num](
                seq_encoder_output, ppi_feature_encoder_output
            )
            ppi_feature_encoder_output = ppi_feature_output
            seq_encoder_output = seq_output

        hs_model39 = torch.cat(
            [ppi_feature_encoder_output, seq_encoder_output], dim=0
        )  # 3,32,512
        # fusion_model39 = self.fusion_model39(hs_fc, hs_model39)  # 32,45

        # 获取model17的encoder结果
        # ppi_feature的预训练模型的Encoder部分

        fusion_model17, fusion_model17_out = self.pre_model17(src)  # 3,32,512 # 32,45

        fusion_hs = torch.cat([fusion_model17, hs_model39], dim=0)  # 6,32,512
        fusion_hs_flatten = torch.einsum("LBD->BLD", fusion_hs).flatten(1)  # 32,512*6
        fusion_hs_permuted = torch.einsum("LBD->BLD", fusion_hs)  # 32,6,512
        weight, _ = self.gate(fusion_hs_flatten)  # 32,6,1
        weight = weight.unsqueeze(-1)
        weighted_gate_hs = fusion_hs_permuted * weight  # 32, 6, 512
        weighted_gate_hs = torch.einsum("BLD->LBD", weighted_gate_hs)  # 6,32,512
        fc_output = self.fc_decoder(weighted_gate_hs)  # 32,45
        if self.db:
            return hs_fc, fc_output, fusion_model17, hs_model39, weighted_gate_hs
        return hs_fc, fc_output


def build_predictor(seq_pre_model, ppi_feature_pre_model, pre_model17, args, db=False):
    predictor = Predictor(
        args,
        seq_pre_model=seq_pre_model,
        ppi_feature_pre_model=ppi_feature_pre_model,
        pre_model17=pre_model17,
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout,
        input_num=args.input_num,
        db=db,
    )

    return predictor
