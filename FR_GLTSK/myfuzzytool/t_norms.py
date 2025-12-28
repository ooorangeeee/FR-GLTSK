# -*- coding: UTF-8 -*-
"""
@Author ：L.Gao
@Date ：2024/9/8 21:12
"""

from torch import log
from torch import exp
from torch import sum
import torch

def softmin(x, t=0.01):
    """
    min函数光滑化：-t*ln(求和（exp(-xi/t)）)  i 从 1 到 num_rule
    :param x: 输入数据张量 Tensor[num_sample,num_fuzzy,input_layer_size] 索引版本
    :param t: 取小函数参数
    :return: 张量 Tensor[num_sample,num_fuzzy]
    """
    return -t * log(sum(exp((-x) / t), dim=2))


def adasoftmin1(x, dim=2):
    """
    自适应min函数光滑化：-t*ln(求和（exp(-xi/t)）)  i 从 1 到 num_rule
    :param x: 输入数据张量 Tensor[num_sample,num_fuzzy,input_layer_size] 索引版本
    :param dim: 取小的维度
    :return: 张量 Tensor[num_sample,num_fuzzy]
    """
    t = torch.amin(x, dim=dim, keepdim=True) / 500

    return -t.squeeze() * log(sum(exp((-x) / t), dim=dim))


def softmin1(x, t=0.01, dim=2):
    """
    min函数光滑化：-t*ln(求和（exp(-xi/t)）)  i 从 1 到 num_rule
    :param x: 输入数据张量 Tensor[num_sample,num_fuzzy,input_layer_size] 索引版本
    :param t: 取小函数参数
    :param dim: 取小的维度
    :return: 张量 Tensor[num_sample,num_fuzzy]
    """

    return -t * log(sum(exp((-x) / t), dim=dim))