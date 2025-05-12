import os

# 当前脚本目录：.../codespace/pretrain/one_feature_only
current_file_dir = os.path.dirname(__file__)
# 上级目录（到项目根目录 DSRPGO）
project_root = os.path.abspath(os.path.join(current_file_dir, "..",'..','..'))
import sys
sys.path.append(project_root)

import numpy as np
import pandas as pd
import argparse
import time
import random
import torch
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from loguru import logger


from codespace.model import aslloss_adaptive

from sklearn.preprocessing import minmax_scale
import csv
from codespace.model.predictor_module_2_1_rebuild import build_predictor
from codespace.pretrain.bimamba.pretrain_model_bimamba import (
    build_Pre_Train_Model as build_Pre_Train_Model_bimamba,
)
from codespace.pretrain.one_feature_only.pretrain_model import (
    build_Pre_Train_Model as build_Pre_Train_Model_transformer,
)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = "{name} {val" + self.fmt + "}"
        else:
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class multimodesDataset(torch.utils.data.Dataset):
    def __init__(self, num_modes, modes_features, labels):
        self.modes_features = modes_features
        self.labels = labels
        self.num_modes = num_modes

    def __getitem__(self, index):
        modes_features = []
        for i in range(self.num_modes):
            modes_features.append(self.modes_features[i][index])
        return modes_features, self.labels[index]

    def __len__(self):
        return self.modes_features[0].size(0)


from codespace.utils.read_finetune_data import (
    read_feature_by_index,
    read_labels,
    read_ppi_by_index,
    read_seq_embed_avgpool_prott5_1024_by_index,
)


def perf_write_to_csv(args, epoch, perf, loss, time, lr):
    if not os.path.exists(args.epoch_performance_path):
        with open(args.epoch_performance_path, "w") as f:
            csv.writer(f).writerow(
                ["epoch", "loss", "time", "lr", "m-aupr", "Fmax", "M-aupr", "F1", "acc"]
            )

    with open(args.epoch_performance_path, "a") as f:
        csv.writer(f).writerow(
            [
                epoch,
                loss,
                time,
                lr,
                perf["m-aupr"],
                perf["Fmax"],
                perf["M-aupr"],
                perf["F1"],
                perf["acc"],
            ]
        )


# 检查并创建文件夹
def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")


# prott5:[num,1024]
def get_finetune_data(args, usefor, aspect, organism_num):
    feature = read_feature_by_index(usefor, aspect, organism_num)
    ppi_matrix = read_ppi_by_index(usefor, aspect, organism_num)
    seq = read_seq_embed_avgpool_prott5_1024_by_index(usefor, aspect, organism_num)
    labels = read_labels(usefor, aspect, organism_num)
    return feature, seq, ppi_matrix, labels


def get_dataset(args, aspect, organism_num):
    train_feature, train_seq, train_ppi_matrix, train_labels = get_finetune_data(
        args, "train", aspect, organism_num
    )
    valid_feature, valid_seq, valid_ppi_matrix, valid_labels = get_finetune_data(
        args, "valid", aspect, organism_num
    )
    test_feature, test_seq, test_ppi_matrix, test_labels = get_finetune_data(
        args, "test", aspect, organism_num
    )

    combine_feature = np.concatenate((train_feature, valid_feature), axis=0)
    combine_seq = np.concatenate((train_seq, valid_seq), axis=0)
    combine_ppi_matrix = np.concatenate((train_ppi_matrix, valid_ppi_matrix), axis=0)
    combine_labels = np.concatenate((train_labels, valid_labels), axis=0)

    combine_feature = torch.from_numpy(combine_feature).float()
    combine_seq = torch.from_numpy(combine_seq).float()
    combine_ppi_matrix = torch.from_numpy(combine_ppi_matrix).float()
    combine_labels = torch.from_numpy(combine_labels).float()
    test_feature = torch.from_numpy(test_feature).float()
    test_seq = torch.from_numpy(test_seq).float()
    test_ppi_matrix = torch.from_numpy(test_ppi_matrix).float()
    test_labels = torch.from_numpy(test_labels).float()

    train_dataset = multimodesDataset(
        3, [combine_ppi_matrix, combine_feature, combine_seq], combine_labels
    )
    test_dataset = multimodesDataset(
        3, [test_ppi_matrix, test_feature, test_seq], test_labels
    )
    modefeature_lens = [
        combine_ppi_matrix.shape[1],
        combine_feature.shape[1],
        combine_seq.shape[1],
    ]
    print("combine_ppi_matrix = ", combine_ppi_matrix.shape)

    return train_dataset, test_dataset, modefeature_lens


def parser_args():
    parser = argparse.ArgumentParser(description="CFAGO main")
    parser.add_argument("--org", help="organism")
    parser.add_argument("--aspect", type=str, default='F',choices=["P", "F", "C"], help="GO aspect")
    parser.add_argument("--num_class", default=38, type=int, help="标签数")
    parser.add_argument("--pretrained_model", type=str, help="输入的预训练模型的路径")
    parser.add_argument("--finetune_model", type=str, help="输出的微调模型的路径")
    parser.add_argument("--performance_path", type=str, help="输出的指标的路径")

    parser.add_argument("--dataset_dir", help="dir of dataset")
    parser.add_argument("--output", metavar="DIR", help="path to output folder")

    parser.add_argument(
        "--optim",
        default="AdamW",
        type=str,
        choices=["AdamW", "Adam_twd"],
        help="which optim to use",
    )

    # loss
    parser.add_argument(
        "--eps", default=1e-5, type=float, help="eps for focal loss (default: 1e-5)"
    )
    parser.add_argument(
        "--dtgfl",
        action="store_true",
        default=False,
        help="disable_torch_grad_focal_loss in asl",
    )
    parser.add_argument(
        "--gamma_pos",
        default=0,
        type=float,
        metavar="gamma_pos",
        help="gamma pos for simplified asl loss",
    )
    parser.add_argument(
        "--gamma_neg",
        default=2,
        type=float,
        metavar="gamma_neg",
        help="gamma neg for simplified asl loss",
    )
    parser.add_argument(
        "--loss_dev", default=-1, type=float, help="scale factor for loss"
    )
    parser.add_argument(
        "--loss_clip", default=0.0, type=float, help="scale factor for clip"
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=32,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs",
    )

    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-2,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-2)",
        dest="weight_decay",
    )

    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--resume_omit", default=[], type=str, nargs="*")
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )

    # distribution training
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--seed", default=12301281, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )

    # data aug
    parser.add_argument(
        "--cutout", action="store_true", default=False, help="apply cutout"
    )
    parser.add_argument(
        "--n_holes", type=int, default=1, help="number of holes to cut out from image"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=-1,
        help="length of the holes. suggest to use default setting -1.",
    )
    parser.add_argument(
        "--cut_fact", type=float, default=0.5, help="mutual exclusion with length. "
    )

    parser.add_argument(
        "--norm_norm",
        action="store_true",
        default=False,
        help="using mormal scale to normalize input features",
    )

    # * Transformer
    parser.add_argument(
        "--attention_layers",
        default=6,
        type=int,
        help="Number of layers of each multi-head attention module",
    )

    parser.add_argument(
        "--dim_feedforward",
        default=512,
        type=int,
        help="Intermediate size of the feedforward layers in the multi-head attention blocks",
    )
    parser.add_argument(
        "--activation",
        default="gelu",
        type=str,
        choices=["relu", "gelu", "lrelu", "sigmoid"],
        help="Number of attention heads inside the multi-head attention module's attentions",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout applied in the multi-head attention module",
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the multi-head attention module's attentions",
    )
    parser.add_argument("--pre_norm", action="store_true")

    # * raining
    parser.add_argument("--amp", action="store_true", default=False, help="apply amp")
    parser.add_argument(
        "--early-stop", action="store_true", default=False, help="apply early stop"
    )
    parser.add_argument(
        "--kill-stop", action="store_true", default=False, help="apply early stop"
    )
    parser.add_argument(
        "--pretrain-update",
        default=2,
        type=int,
        help="参数更新方式",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument(
        "--nni",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--pre_lr",
        default=3e-3,
        type=float,
    )
    parser.add_argument(
        "--seq_pre_lr",
        default=5e-3,
        type=float,
    )
    parser.add_argument(
        "--fusion_lr",
        default=1e-4,
        type=float,
    )
    parser.add_argument(
        "--param",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--fusion",
        default='transformer',
        type=str,
    )
    parser.add_argument(
        "--seq_feature",
        default="seq1024",  # seq1024
        type=str,
    )
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args


import nni


# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_simple/finetune.py --seq_feature seq1024 --aspect P --num_class 42 --seed 12301281 &
# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_simple/finetune.py --device cuda:1 --fusion transformer --seq_feature seq1024 --aspect F --num_class 17 --seed 1329765522 &
# nohup python -u /home/Kioedru/code/SSGO/codespace/finetune/2_1_simple/finetune.py --device cuda:0 --fusion transformer --seq_feature seq1024 --aspect C --num_class 37 --seed 12301281 &
def main():
    args = get_args()
    # 指定随机种子初始化随机数生成器（保证实验的可复现性）
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    if args.param:
        params = {
            "lr": args.lr,
            "dropout": args.dropout,
            "pre_lr": args.pre_lr,
            "seq_pre_lr": args.seq_pre_lr,
            "fusion_lr": args.fusion_lr,
        }


    # 需注释的参数
    # args.seq_feature = "seq1024"
    # args.fusion = "enhanced"  # transformer bimamba mamba enhanced
    # args.aspect = "P"
    # args.num_class = int(45)
    # args.seed = int(
    #     1329765522
    # )  #  1329765522  132976111  1329765525    1329765529  1329765519

    args.input_num = 3
    args.epochs = 100
    args.pretrain_update = 2  # 0全更新，1不更新，2更新一半
    if args.pretrain_update == 0:
        args.update_epoch = args.epochs
    elif args.pretrain_update == 1:
        args.update_epoch = 0
    elif args.pretrain_update == 2:
        args.update_epoch = int(args.epochs / 2)

    args.org = "9606"

    args.model_name = f"MSLB"
    # /home/Kioedru/code/SSGO/codespace/pretrain/one_feature_only/9606/transformer_seq480_only.pkl
    args.seq_model_name = f"transformer_{args.seq_feature}_only"
    # /home/Kioedru/code/SSGO/codespace/pretrain/bimamba/9606/bimamba.pkl
    args.ppi_feature_model_name = f"bimamba"
    pretrain_path = os.path.join(project_root, 'codespace', "pretrain")
    args.pretrain_path =pretrain_path

    args.finetune_path = os.path.join(
        project_root,
        'codespace',
        "finetune",
        "MSLB",
        args.aspect,
    )
    # 预训练模型：Sequence的路径
    args.seq_pretrained_model = os.path.join(
        pretrain_path,
        "one_feature_only",
        args.org,
        f"{args.seq_model_name}.pth",
    )
    # 预训练模型：ppi+亚细胞+结构域的路径
    args.ppi_feature_pretrained_model = os.path.join(
        pretrain_path,
        args.ppi_feature_model_name,
        args.org,
        f"{args.ppi_feature_model_name}.pth",
    )
    args.finetune_model_path = os.path.join(args.finetune_path, f"epoch_model")
    check_and_create_folder(args.finetune_model_path)


    args.epoch_performance_path = os.path.join(
        args.finetune_path,
        f"epoch_performance_lr:{params['lr']},dropout:{params['dropout']},pre_lr:{params['pre_lr']},seq_pre_lr:{params['seq_pre_lr']}.csv",
    )

    args.nheads = int(8)
    args.dropout = params["dropout"]
    args.attention_layers = int(2)
    args.gamma_pos = int(0)
    args.gamma_neg = int(2)
    args.batch_size = int(32)
    # args.lr = float(1e-4)
    args.lr = params["lr"]
    args.pre_lr = params["pre_lr"]
    args.seq_pre_lr = params["seq_pre_lr"]
    args.fusion_lr = params["fusion_lr"]


    # 使用一个隐藏层
    args.h_n = 1
    return main_worker(args)


def main_worker(args):

    # 准备数据集,esm2+prott5时 seq_2=True
    train_dataset, test_dataset, args.modesfeature_len = get_dataset(
        args, args.aspect, args.org
    )
    args.encode_structure = [1024]

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # (train_sampler is None)输出为true，因此会打乱数据
        shuffle=(train_sampler is None),
        # 指定用多少个子进程来加载数据，CFAGO中默认为32
        num_workers=args.workers,
        # 是否将加载的数据保存在锁页内存中（以占用更多内存的代价，加快数据从CPU到GPU的转移速度）
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )
    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    # 定义损失函数
    loss = aslloss_adaptive.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = args.batch_size / 32

    torch.cuda.empty_cache()
    # 载入预训练模型字典
    ppi_feature_pre_model_state_dict = torch.load(
        args.ppi_feature_pretrained_model, map_location=args.device
    )
    seq_pre_model_state_dict = torch.load(
        args.seq_pretrained_model, map_location=args.device
    )
    
    args.attention_layers=6
    # 创建预训练模型
    ppi_feature_pre_model = build_Pre_Train_Model_bimamba(args).to(args.device)
    seq_pre_model = build_Pre_Train_Model_transformer(args).to(args.device)
    # 更新权重
    ppi_feature_pre_model.load_state_dict(ppi_feature_pre_model_state_dict)
    seq_pre_model.load_state_dict(seq_pre_model_state_dict)


    # 创建预测模型
    args.attention_layers=2
    predictor_model = build_predictor(seq_pre_model, ppi_feature_pre_model, args).to(args.device)


    # if args.optim == 'AdamW':
    # 参数字典列表，存储预训练模型和fc_decoder层的参数
    predictor_model_param_dicts = [
        # 预训练模型的参数使用较低的学习率1e-5（因为已经训练好了，无需大幅度调整）
        {
            "params": [
                p
                for n, p in predictor_model.ppi_feature_pre_model.named_parameters()
                if p.requires_grad
            ],
            # "lr": args.pre_lr,
        },
        {
            "params": [
                p
                for n, p in predictor_model.seq_pre_model.named_parameters()
                if p.requires_grad
            ],
            # "lr": args.seq_pre_lr,
        },
        # fc_decoder层的参数使用默认学习率
        {
            "params": [
                p
                for n, p in predictor_model.fc_decoder.named_parameters()
                if p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in predictor_model.fusion.named_parameters()
                if p.requires_grad
            ]
        },
    ]

    # 优化器，使用AdamW算法
    predictor_model_optimizer = getattr(torch.optim, "AdamW")(
        predictor_model_param_dicts,
        lr=args.lr_mult * args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
    )
    # 学习率调度器 指定优化器，step_size=50，默认gamma=0.1，每隔step_size个周期就将每个参数组的学习率*gamma
    steplr = lr_scheduler.StepLR(predictor_model_optimizer, 50)
    patience = 10
    changed_lr = False

    # 每隔2500个epoch就把学习率乘0.01
    finetune(
        args,
        train_loader,
        test_loader,
        predictor_model,
        loss,
        predictor_model_optimizer,
        steplr,
        args.epochs,
        args.device,
    )

    save_model_path=os.path.join(args.finetune_path, f"final_model.pkl")
    torch.save(predictor_model.state_dict(), save_model_path)
    logger.info(f"Model saved to {save_model_path}")

    # 加载已保存模型并验证模型参数一致性
    loaded_model = build_predictor(seq_pre_model, ppi_feature_pre_model, args)
    loaded_model.load_state_dict(torch.load(save_model_path, map_location=args.device))
    loaded_model=loaded_model.to(args.device)
    for param1, param2 in zip(predictor_model.parameters(), loaded_model.parameters()):
        if not torch.equal(param1, param2):
            logger.warning("发现不一致的参数!")
            break
    else:
        logger.info("模型参数一致.")

def finetune(
    args,
    data_loader,
    test_loader,
    model,
    loss,
    optimizer,
    steplr,
    num_epochs,
    device,
):

    net = model.to(device)
    net.train()
    print("training on", device)
    for epoch in range(num_epochs):
        if args.pretrain_update == 1:  # 不更新参数
            for p in model.seq_pre_model.parameters():
                p.requires_grad = False
            for p in model.ppi_feature_pre_model.parameters():
                p.requires_grad = False
        if args.pretrain_update == 2:  # 更新后半部分参数
            if epoch >= (args.epochs / 2):
                for p in model.seq_pre_model.parameters():
                    p.requires_grad = True
                for p in model.ppi_feature_pre_model.parameters():
                    p.requires_grad = True
            else:
                for p in model.seq_pre_model.parameters():
                    p.requires_grad = False
                for p in model.ppi_feature_pre_model.parameters():
                    p.requires_grad = False
        if args.pretrain_update == 0:  # 更新全部参数
            for p in model.seq_pre_model.parameters():
                p.requires_grad = True
            for p in model.ppi_feature_pre_model.parameters():
                p.requires_grad = True
        start = time.time()
        batch_count = 0
        train_l_sum = 0.0
        for protein_data, label in data_loader:

            protein_data[0] = protein_data[0].to(device)
            protein_data[1] = protein_data[1].to(device)
            protein_data[2] = protein_data[2].to(device)
            label = label.to(device)

            rec, output = net(protein_data)
            l = loss(output, label, rec)
            optimizer.zero_grad()
            l.backward()
            train_l_sum += l.cpu().item()
            # record loss
            optimizer.step()  # 优化方法
            batch_count += 1
        steplr.step()

        # 每轮都测试
        with torch.no_grad():
            perf = evaluate(test_loader, net, args.device)
            perf["default"] = perf["m-aupr"]
            if not args.nni:
                perf_write_to_csv(
                    args,
                    epoch,
                    perf,
                    loss=train_l_sum / batch_count,
                    time=time.time() - start,
                    lr=optimizer.param_groups[2]["lr"],
                )

            if args.nni:
                nni.report_intermediate_result(perf)

        # 保存每轮的model参数字典
        # torch.save(
        #     net.state_dict(), os.path.join(args.finetune_model_path, f"{epoch}.pkl")
        # )
    if args.nni:
        nni.report_final_result(perf)


from codespace.utils.evaluate_performance import evaluate_performance


@torch.no_grad()
def evaluate(test_loader, predictor_model, device):

    # switch to evaluate mode
    predictor_model.eval()
    all_output_sm = []
    all_label = []
    predictor_model = predictor_model.to(device)
    for proteins, label in test_loader:
        proteins[0] = proteins[0].to(device)
        proteins[1] = proteins[1].to(device)
        proteins[2] = proteins[2].to(device)
        label = label.to(device)

        # compute output
        rec, output = predictor_model(proteins)
        output_sm = torch.nn.functional.sigmoid(output)

        # collect output and label for metric calculation
        all_output_sm.append(output_sm.detach().cpu())
        all_label.append(label.detach().cpu())

    all_output_sm = torch.cat(all_output_sm, 0).numpy()
    all_label = torch.cat(all_label, 0).numpy()

    # calculate metrics
    perf = evaluate_performance(
        all_label, all_output_sm, (all_output_sm > 0.5).astype(int)
    )

    return perf


if __name__ == "__main__":
    main()
