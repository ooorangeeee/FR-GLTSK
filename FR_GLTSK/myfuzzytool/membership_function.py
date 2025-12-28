# -*- coding: UTF-8 -*-
"""
@Author ：L.Gao
@Date ：2024/9/8 21:12
"""

from torch import exp


def gaussian_mf(x, *params):
    """
    高斯型隶属度函数：h * exp(-(x-m)^2 * (sigma^2))
    :param x:特征（输入数据） [num_sample, 1, input_layer_size]
    :param params:可变参数，params[0],params[1],params[2]分别表示中心m，宽度sigma，高度h
                [num_fuzzy, input_layer_size]
    :return:隶属度值，[num_sample, num_fuzzy, input_layer_size]
    """

    return params[2] * exp(-(((x - params[0]) ** 2) * params[1] ** 2))


def gauss_uncert_mean_umf(x, *params):
    """
    不确定均值高斯型上隶属度函数
    :param x:特征（输入数据） [num_sample, 1, input_layer_size]
    :param params:可变参数，params[0],params[1],params[2], params[3]
                分别表示中心下m，中心上m，宽度sigma，高度h
                [num_fuzzy, input_layer_size]
    :return:隶属度值，[num_sample, num_fuzzy, input_layer_size]
    """

    return (((x < params[0]) * gaussian_mf(x, params[0], params[2], params[3])) +
            ((x > params[1]) * gaussian_mf(x, params[1], params[2], params[3])) +
            ((x <= params[1]) * params[3] * 1 * (x >= params[0])))



def gauss_uncert_mean_lmf(x, *params):
    """
    不确定均值高斯型下隶属度函数
    :param x:特征（输入数据） [num_sample, 1, input_layer_size]
    :param params:可变参数，params[0],params[1],params[2], params[3]
                分别表示中心下m，中心上m，宽度sigma，高度h
                [num_fuzzy, input_layer_size]
    :return:隶属度值，[num_sample, num_fuzzy, input_layer_size]
    """

    return ((x > (params[0] + params[1]) / 2) * gaussian_mf(x, params[0], params[2], params[3])) + \
        ((x <= (params[0] + params[1]) / 2) * gaussian_mf(x, params[1], params[2], params[3]))


def gauss_uncert_std_umf(x, *params):
    """
    不确定方差高斯型上隶属度函数
    :param x:特征（输入数据） [num_sample, 1, input_layer_size]
    :param params:可变参数，params[0],params[1],params[2], params[3]
                分别表示中心m，宽度上sigma，宽度下sigma，高度h
                [num_fuzzy, input_layer_size]
    :return:隶属度值，[num_sample, num_fuzzy, input_layer_size]
    """

    return gaussian_mf(x, [params[0], params[2], params[3]])


def gauss_uncert_std_lmf(x, params):
    """
    不确定方差高斯型下隶属度函数
    :param x:特征（输入数据） [num_sample, 1, input_layer_size]
    :param params:可变参数，params[0],params[1],params[2], params[3]
                分别表示中心m，宽度上sigma，宽度下sigma，高度h
                [num_fuzzy, input_layer_size]
    :return:隶属度值，[num_sample, num_fuzzy, input_layer_size]
    """

    return gaussian_mf(x, [params[0], params[1], params[3]])
