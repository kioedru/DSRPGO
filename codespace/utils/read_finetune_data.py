import sys
sys.path.append("/home/Kioedru/code/DSRPGO")

from codespace.utils.read_pretrain_data import (
    read_ppi,
    read_feature,
    read_seq_embed_avgpool_prott5_1024,
    read_seq_embed_avgpool_prott5_1024_new,
    read_seq_embed_onehot,
)
import pandas as pd
import os
import torch
from sklearn.preprocessing import minmax_scale

# 当前脚本目录：.../codespace/utils
current_file_dir = os.path.dirname(__file__)

# 上一级目录（到项目代码根目录 DSRPGO/codespace）
project_root = os.path.abspath(os.path.join(current_file_dir, ".."))

finetune_data_path = os.path.join(project_root, 'data',"finetune")

# 读取数据集
def read_df(usefor, aspect, organism_num):
    df_name = f"{usefor}_data_{aspect}.pkl"
    df_path = os.path.join(finetune_data_path, organism_num, df_name)
    df = pd.read_pickle(df_path)
    return df


# 获取ppi对应索引
def get_ppi_index(df, organism_num):
    ppi_matrix, ppi_id = read_ppi(organism_num)
    ppi_id = [x + ";" for x in ppi_id]
    index = df["Cross-reference (STRING)"].map(lambda x: ppi_id.index(x))
    return index

# 获取数据集的原生序列特征
def read_raw_seq(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    return pd.DataFrame(df['Sequence'])

# 获取数据集的ppi特征
def read_ppi_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    ppi_matrix, ppi_id = read_ppi(organism_num)
    # ppi_min = ppi_matrix.min()
    # ppi_max = ppi_matrix.max()
    # print(ppi_min,ppi_max)
    ppi_matrix = minmax_scale(ppi_matrix)
    selected_rows = ppi_matrix[index]
    return selected_rows


# 获取数据集的feature特征
def read_feature_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    feature = read_feature(organism_num)
    feature = minmax_scale(feature)
    selected_rows = feature[index]
    return selected_rows




# prott5:[num,1024]
def read_seq_embed_avgpool_prott5_1024_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    seq_prott5_emb = read_seq_embed_avgpool_prott5_1024(organism_num)
    seq_prott5_emb = minmax_scale(seq_prott5_emb)
    selected_rows_prott5 = seq_prott5_emb[index]

    return selected_rows_prott5


# prott5:[num,1024]
def read_seq_embed_avgpool_prott5_1024_and_null_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    seq_prott5_emb = read_seq_embed_avgpool_prott5_1024(organism_num)
    # Step 1: 找到全 0 行的原始索引
    original_zero_row_indices = np.where(~seq_prott5_emb.any(axis=1))[0]
    # print(original_zero_row_indices)
    seq_prott5_emb = minmax_scale(seq_prott5_emb)
    selected_rows_prott5 = seq_prott5_emb[index]

    # Step 2: 找到全 0 行是否被挑选
    selected_zero_row_indices = np.intersect1d(index, original_zero_row_indices)
    # print(selected_zero_row_indices)
    selected_rows_zero_indices = np.where(np.isin(index, selected_zero_row_indices))[0]
    # print(selected_rows_zero_indices)

    return selected_rows_prott5, selected_rows_zero_indices  # 返回embed和全0行的索引


def read_seq_embed_avgpool_prott5_1024_by_index_without_normalize(
    usefor, aspect, organism_num
):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    seq_prott5_emb = read_seq_embed_avgpool_prott5_1024(organism_num)
    # seq_prott5_emb = minmax_scale(seq_prott5_emb)
    selected_rows_prott5 = seq_prott5_emb[index]
    return selected_rows_prott5


# prott5:[num,1024]
def read_seq_embed_avgpool_prott5_1024_new_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    seq_prott5_emb = read_seq_embed_avgpool_prott5_1024_new(organism_num)
    selected_rows_prott5 = seq_prott5_emb[index]
    selected_rows = minmax_scale(selected_rows_prott5)
    return selected_rows


# onehot:[num,26]
def read_seq_embed_onehot_by_index(usefor, aspect, organism_num):
    df = read_df(usefor, aspect, organism_num)
    index = get_ppi_index(df, organism_num)
    seq_emb = read_seq_embed_onehot(organism_num)
    selected_rows = seq_emb[index]
    selected_rows = minmax_scale(selected_rows)
    return selected_rows


def read_gan_seq_embed_and_labels(aspect, organism_num, model_name="esm-480"):
    label_dim = {"P": 45, "F": 38, "C": 35}
    terms = read_terms(aspect, organism_num)
    mlb = MultiLabelBinarizer(classes=terms["terms"].tolist())

    best_epoch = pd.read_pickle(
        f"/home/Kioedru/code/SSGO/data/synthetic/{model_name}/{organism_num}/{aspect}_best_epoch.pkl"
    )
    labels = []
    seq_emb = []
    for index, row in best_epoch.iterrows():
        term = row["term"]
        best_epoch = row["best_epoch"]
        # /home/Kioedru/code/SSGO/data/synthetic/esm2-480/9606/P/GO:0000122/GO:0000122_Iteration_400_Synthetic_Training_Positive.pkl
        emb = pd.read_pickle(
            f"/home/Kioedru/code/SSGO/data/synthetic/{model_name}/{organism_num}/{aspect}/{term}/{term}_Iteration_{best_epoch}_Synthetic_Training_Positive.pkl"
        )
        seq_emb.append(emb)

        sample_num = emb.shape[0]
        encode = mlb.fit_transform([[term]])
        encode = np.array(encode)
        for i in range(sample_num):
            labels.append(encode)
    seq_emb = np.vstack(seq_emb)
    labels = np.vstack(labels)
    return seq_emb, labels


# # 未实现：获取数据集的onehot特征
# def read_seq_onehot(usefor, aspect, organism_num):
#     df = read_df(usefor, aspect, organism_num)
#     index = get_ppi_index(df, organism_num)
#     seq_emb = read_seq_one_hot_sum(organism_num)
#     seq_emb = minmax_scale(seq_emb)
#     selected_rows = seq_emb[index]

#     return selected_rows


from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


def read_terms(aspect, organism_num):
    terms_name = f"terms_{aspect}.pkl"
    file_path = os.path.join(finetune_data_path, organism_num, terms_name)
    terms = pd.read_pickle(file_path)
    return terms


# 获取数据集的labels
def read_labels(usefor, aspect, organism_num):

    df = read_df(usefor, aspect, organism_num)

    terms = read_terms(aspect, organism_num)
    mlb = MultiLabelBinarizer(classes=terms["terms"].tolist())
    if aspect == "EC":
        encode = mlb.fit_transform(df["EC number"])
    else:
        encode = mlb.fit_transform(df["annotations"])
    encode = np.array(encode)
    return encode


# 获取数据集的labels
def read_labels_F_nlp(usefor, aspect, organism_num):
    encode = read_labels(usefor, aspect, organism_num)
    encode = np.delete(encode, -8, axis=1)
    return encode


def read_terms_F_nlp(aspect, organism_num):
    F_terms = read_terms("F", "9606")
    F_terms.drop(F_terms.index[-8])
    return F_terms


# 获取微调数据集的残基embed
def read_residue(usefor, aspect, model_name, organism_num):
    residue_name = f"{usefor}_residue_{aspect}.pkl"
    file_path = os.path.join(
        finetune_data_path, organism_num, f"residue_{model_name}", residue_name
    )

    residue = pd.read_pickle(file_path)
    # print(torch.isnan(residue).any())
    # 找到最小和最大值
    min_val = torch.min(residue)
    max_val = torch.max(residue)
    # 进行归一化
    normalized_tensor = (residue - min_val) / (max_val - min_val)
    return normalized_tensor.detach()


def torch_min_max(x):
    min_val = torch.min(x)
    max_val = torch.max(x)
    # 进行归一化
    normalized_tensor = (x - min_val) / (max_val - min_val)
    return normalized_tensor.detach()


def np_min_max(x):
    min_val = np.min(x)
    max_val = np.max(x)
    # 进行归一化
    normalized_array = (x - min_val) / (max_val - min_val)
    return normalized_array


if __name__=='__main__':
    organism_num='9606'
    ppi_matrix, ppi_id = read_ppi(organism_num)
    ppi_matrix = minmax_scale(ppi_matrix)
    ppi_min = ppi_matrix.min()
    ppi_max = ppi_matrix.max()
    print(ppi_min,ppi_max)
    # print(ppi_matrix)
    is_integer = np.all(ppi_matrix == np.floor(ppi_matrix))
    print("All values are integers:", is_integer)
    