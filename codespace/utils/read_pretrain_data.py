import sys
sys.path.append("/home/Kioedru/code/DSRPGO")
import os
import pandas as pd
import h5py
import torch

# 当前脚本目录：.../codespace/utils
current_file_dir = os.path.dirname(__file__)

# 上一级目录（到项目代码根目录 DSRPGO/codespace）
project_root = os.path.abspath(os.path.join(current_file_dir, ".."))

pretrain_data_path = os.path.join(project_root, 'data', "pretrain")

# feature特征：[19385,1389]：亚细胞位置(442)+结构域特征(947)
def read_feature(organism_num):
    file_name = f"{organism_num}_feature.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    feature = pd.read_pickle(file_path)
    return feature


# ppi特征：[19385,19385]
def read_ppi(organism_num):
    file_name = f"{organism_num}_ppi.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    ppi_data = pd.read_pickle(file_path)
    # 读取ppi稀疏矩阵，转换为稠密矩阵
    ppi_matrix = ppi_data["matrix"].toarray()
    # print(ppi_matrix.shape)
    ppi_id = ppi_data["ppi_id"]
    # print(len(ppi_id), ppi_id[0:20])
    return ppi_matrix, ppi_id

# prott5:[19385,seq_len,1024]->[19385, 1024]
def read_seq_embed_avgpool_prott5_1024(organism_num):
    file_name = f"{organism_num}_seq_embed_avgpool_prott5_1024.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq


# prott5:[19385,seq_len,1024]->[19385, 1024]
def read_seq_embed_avgpool_prott5_1024_new(organism_num):
    file_name = f"{organism_num}_seq_embed_avgpool_prott5_1024_new.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq


# 按照ppi_id的顺序排列的蛋白质氨基酸序列
def read_seq(organism_num):
    file_name = f"{organism_num}_seq.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq


# one_hot (19385,26)
def read_seq_embed_onehot(organism_num):
    file_name = f"{organism_num}_seq_embed_onehot.pkl"
    file_path = os.path.join(pretrain_data_path, file_name)
    seq = pd.read_pickle(file_path)
    return seq
