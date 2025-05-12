import os
from torch.nn import functional as F


# 检查并创建文件夹
def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "lrelu":
        return F.leaky_relu
    if activation == "gelu":
        return F.gelu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "elu":
        return F.elu
    if activation == "tanh":
        return F.tanh
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
