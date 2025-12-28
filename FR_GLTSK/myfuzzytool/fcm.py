# -*- coding: UTF-8 -*-
"""
@Author ：L.Gao
@Date ：2024/10/11 10:30 
"""

import torch
from scipy.spatial.distance import cdist
from .typereduced import KM_algorithm_sig
from .utils import normalize_columns, normalize_power_columns


def _distance(data, centers, metric):
    """
    计算样本点与中心之间的距离
    :param data:样本点数据 tensor (n,s)
    :param centers:中心 tensor (c,s)
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :return:距离 tensor (n,c)
    """
    return torch.from_numpy(cdist(data, centers, metric=metric))


def _fp_coeff(u):
    """
    模糊划分系数
    :param u:模划分矩阵 tensor (n, c)
    :return:模糊划分矩阵 float
    """

    n = u.shape[0]

    return torch.trace(u.T @ u) / float(n)


def FCM_compute(data, u_old, m, metric):
    """
    计算隶属度矩阵和聚类中心
    :param data:样本点数据 tensor (n,s)
    :param u_old:待更新的隶属度矩阵 tensor (n, c)
    :param m:超参数 int
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :return:cntr:聚类中心  u:隶属度矩阵 jm:记录目标函数历史值 d:距离
    """

    # 归一化， 计算可能为0的元素
    u_old = normalize_columns(u_old)  # 保证每行和为1
    u_old = torch.fmax(u_old, torch.tensor(torch.finfo(u_old.dtype).eps))

    um = u_old ** m  # (n, c)

    # 更新聚类中心
    cntr = (um.T @ data) / torch.atleast_2d(um.sum(dim=0)).T  # (c, s)

    # 计算距离
    d = _distance(data, cntr, metric=metric)  # (n, c)
    d = torch.fmax(d, torch.tensor(torch.finfo(torch.float64).eps))


    jm = (um * d ** 2).sum()  # float

    # 更新隶属度矩阵
    u = normalize_power_columns(d, -2. / (m-1))  # (n, c)

    return cntr, u, jm, d


def FCM(data, c, max_iter=5000, m=2, error=1e-6, metric='euclidean', init=None, seed=None):
    """
    FCM聚类算法
    :param data: 输入数据 tensor (n, s)  n为样本个数，s为特征个数
    :param c:聚类中心个数 int
    :param max_iter:最大迭代次数 int
    :param m:模糊因子 int
    :param error:迭代停止误差 float
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :param init:初始隶属度矩阵 tensor (n, c)
    :param seed:随机种子 float
    :return:
        cntr:聚类中心 tensor (c, s)
        u:隶属度矩阵 tensor (n, c)
        u0:初始隶属度矩阵 tensor (n, c)
        d:距离 tensor (n, c)
        jm:记录目标函数历史值 tensor (p, )
        p:迭代次数 int
        fpc:模糊划分系数 float
        labels:聚类结果 tensor (, n)
    """

    data = data.double()  # 转换成float64

    # 初始化隶属度矩阵
    if init is None:
        if seed is not None:
            torch.manual_seed(seed=seed)  # 设置随机种子
        n = data.shape[0]
        u0 = torch.rand(n, c, dtype=torch.float64)
        u0 = normalize_columns(u0)  # 保证每行和为1，每个样本点与各个聚类中心的和为1
        init = u0.clone()
    u0 = init
    u = torch.fmax(u0, torch.tensor(torch.finfo(torch.float64).eps))  # 确保数组 u0 中的每个元素都不小于float64可以达到的最小精度，防止分母为0

    # 初始循环参数
    jm = torch.zeros(0)
    p = 0

    # 开始迭代
    while p < max_iter-1:
        u2 = u.clone()
        [cntr, u, Jjm, d] = FCM_compute(data, u2, m, metric)
        jm = torch.hstack((jm, Jjm))
        p += 1

        # 停止循环
        if torch.linalg.norm(u - u2) < error:  # 默认计算L2范数
            break

    # 最后计算
    error = torch.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    # 找到隶属度矩阵中每行的最大值，即该样本最大可能所属类别
    labels = torch.argmax(u, dim=1)

    return cntr, u, u0, d, jm, p, fpc, labels


def FCM_predcompute(data, cntr, u_old, m, metric):
    """
    更新测试集的隶属矩阵
    :param data:样本点数据 tensor (n,s)
    :param cntr:已训练的聚类中心 tensor (c, s)
    :param u_old:待更新的隶属度矩阵 tensor (n, c)
    :param m:超参数 int
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :return:u:隶属度矩阵 jm:记录目标函数历史值 d:距离
    """
    # 归一化， 计算可能为0的元素
    u_old = normalize_columns(u_old)  # 保证每行和为1
    u_old = torch.fmax(u_old, torch.tensor(torch.finfo(u_old.dtype).eps))

    um = u_old ** m  # (n, c)

    # 计算距离
    d = _distance(data, cntr, metric=metric)  # (n, c)
    d = torch.fmax(d, torch.tensor(torch.finfo(torch.float64).eps))

    jm = (um * d ** 2).sum()  # float

    # 更新隶属度矩阵
    u = normalize_power_columns(d, -2. / (m - 1))  # (n, c)

    return u, jm, d


def FCM_predict(data, cntr_trained, max_iter, m=2, error=1e-6, metric='euclidean', init=None, seed=None):
    """
    FCM对测试集进行预测 (虽然无监督学习没有标签一般不进行预测 但这里还是分为训练集和测试集 训练集迭代出的聚类中心对测试集进行预测)
    :param data: 输入数据 tensor (n, s)  n为样本个数，s为特征个数
    :param cntr_trained:已训练的聚类中心 tensor (c, s)
    :param max_iter:最大迭代次数 int
    :param m:模糊因子 int
    :param error:迭代停止误差 float
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :param init:初始隶属度矩阵 tensor (n, c)
    :param seed:随机种子 float
    :return:
        u:隶属度矩阵 tensor (n, c)
        u0:初始隶属度矩阵 tensor (n, c)
        d:距离 tensor (n, c)
        jm:记录目标函数历史值 tensor (p,)
        p:迭代次数 int
        fpc:模糊划分系数 float
        labels:聚类结果 tensor (, n)
    """

    data = data.double()  # 转换成float64
    c = cntr_trained.shape[0]

    # 初始化隶属度矩阵
    if init is None:
        if seed is not None:
            torch.manual_seed(seed=seed)  # 设置随机种子
        n = data.shape[0]
        u0 = torch.rand(n, c, dtype=torch.float64)
        u0 = normalize_columns(u0)  # 保证每行和为1，每个样本点与各个聚类中心的和为1
        init = u0.clone()
    u0 = init
    u = torch.fmax(u0, torch.tensor(torch.finfo(torch.float64).eps))  # 确保数组 u0 中的每个元素都不小于float64可以达到的最高精度，防止分母为0

    # 初始循环参数
    jm = torch.zeros(0)
    p = 0

    # 开始迭代
    while p < max_iter - 1:
        u2 = u.clone()
        [u, Jjm, d] = FCM_predcompute(data, cntr_trained, u2, m, metric)
        jm = torch.hstack((jm, Jjm))
        p += 1

        # 停止循环
        if torch.linalg.norm(u - u2) < error:  # 默认计算L2范数
            break

    # 最后计算
    error = torch.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    # 找到隶属度矩阵中每行的最大值，即该样本最大可能所属类别
    labels = torch.argmax(u, dim=1)

    return u, u0, d, jm, p, fpc, labels




def IT2FCM_compute(data, cntr_old, m1, m2, c, metric):
    """
    计算隶属度矩阵和聚类中心
    :param data:样本点数据 tensor (n,s)
    :param cntr_old:待更新的聚类中心 tensor (c, s)
    :param m1:模糊度 int
    :param m2:模糊度 (m2>m1>1) int
    :param c:聚类中心个数 int
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :return:cntr:聚类中心  u:隶属度矩阵
    """

    # 计算距离
    d = _distance(data, cntr_old, metric=metric)  # (n, c)
    d = torch.fmax(d, torch.tensor(torch.finfo(torch.float64).eps))

    # # 更新上、下隶属度矩阵
    # u_low = (normalize_power_columns(d, -2./(m1-1)) * (1/normalize_columns(d) >= 1/c)) + \
    #         (normalize_power_columns(d, -2./(m2-1)) * (1/normalize_columns(d) < 1/c))  # (n, c)
    # u_upper = (normalize_power_columns(d, -2./(m1-1)) * (1/normalize_columns(d) < 1/c)) + \
    #           (normalize_power_columns(d, -2./(m2-1)) * (1/normalize_columns(d) >= 1/c))  # (n, c)
    # u_low = torch.fmax(u_low, torch.tensor(torch.finfo(u_low.dtype).eps))
    # u_upper = torch.fmax(u_upper, torch.tensor(torch.finfo(u_upper.dtype).eps))
    #
    # # 校正上、下隶属度矩阵区间
    # condition = u_low > u_upper
    # temp = u_low[condition].clone()
    # u_low[condition] = u_upper[condition]
    # u_upper[condition] = temp

    # 更新上、下隶属度矩阵
    m1_part = normalize_power_columns(d, - 2. / (m1-1))
    m2_part = normalize_power_columns(d, - 2. / (m2-1))
    u_upper = torch.max(m1_part, m2_part)  # (n, c)
    u_low = torch.min(m1_part, m2_part)  # (n, c)

    u_low = torch.fmax(u_low, torch.tensor(torch.finfo(u_low.dtype).eps))
    u_upper = torch.fmax(u_upper, torch.tensor(torch.finfo(u_upper.dtype).eps))


    # a=u_low > u_upper
    # if a.any():
    #     print("asd")

    # 更新隶属度矩阵
    u = (u_low + u_upper) / 2  # (n, c)

    # 归一化隶属度矩阵
    u = normalize_columns(u)  # (n, c)

    # 更新聚类中心
    cntr_l, cntr_r = KM_algorithm_sig(u_low.T, u_upper.T, data.T)  # (c, s)
    cntr = (cntr_l + cntr_r) / 2  # (c, s)

    # a=cntr_l > cntr_r
    # if a.any():
    #     print("asd")

    # # 目标函数
    # jm1 = (u**m1 * d**2).sum()  # float
    # jm2 = (u**m2 * d**2).sum()  # float


    return cntr, cntr_l, cntr_r, u


def IT2FCM(data, c, max_iter, m1, m2, error, metric='euclidean', init=None, seed=None):
    """
    IT2FCM聚类算法 用于初始化模糊规则库(仅为不确定中心隶属度函数，规则前件部分)
    :param data: 输入数据 tensor (n, s)  n为样本个数，s为特征个数
    :param c:聚类中心个数, 即：每个特征模糊子集个数 int
    :param max_iter:最大迭代次数 int
    :param m1:模糊度 int (m2>m1>1)
    :param m2:模糊度 int
    :param error:迭代停止误差 float
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :param init:初始聚类中心, 即：初始隶属度函数中心 tensor (c, s)
    :param seed:随机种子 float
    :return:
        cntr1:隶属度函数下中心 tensor (c, s)   s代表神经网络输入的特征个数 c代表每个神经网络输入的特征对应的模糊子集个数 即：每列为网络每个输入特征的下中心
        cntr2:隶属度函数上中心 tensor (c, s)   每列为网络每个输入特征的上中心
        spread:隶属度函数宽度 tensor (c, s)    每列为网络每个输入特征的宽度

        以下用于聚类的结果，暂未返回：
        cntr:聚类中心 tensor (c, s)
        p:迭代次数 int
        labels:聚类结果 tensor (, n)
        jm1:记录上目标函数历史值 tensor (p, )
        jm2:记录下目标函数历史值 tensor (p, )
    """

    data = data.double()  # 转换成float64
    s = data.shape[1]

    # 初始化聚类中心
    if init is None:
        if seed is not None:
            torch.manual_seed(seed=seed)  # 设置随机种子
        cntr0 = torch.rand(c, s, dtype=torch.float64)
        init = cntr0.clone()
    cntr0 = init

    # 初始循环参数
    jm1 = torch.zeros(0)
    jm2 = torch.zeros(0)
    p = 0

    # 开始迭代
    while p < max_iter-1:
        cntr, cntr_l, cntr_r, u = IT2FCM_compute(data, cntr0, m1, m2, c, metric)
        # jm1 = torch.hstack((jm1, Jjm1))
        # jm2 = torch.hstack((jm2, Jjm2))
        p += 1

        # 停止循环
        if torch.linalg.norm(cntr - cntr0) < error:
            break
        cntr0 = cntr.clone()

    # 找到隶属度矩阵中每行的最大值，即该样本最大可能所属类别
    labels = torch.argmax(u, dim=1)

    # 方法一 cite：Type 2 Fuzzy Neural Structure for Identification  and Control of Time-Varying Plants
    # 初始化center1, center2
    # delta_cntrMax = torch.min(cntr-cntr_l, cntr_r-cntr)  # 计算delta_cntr最大值
    # delta_cntr = torch.rand(c, s, dtype=torch.float64) * delta_cntrMax
    # cntr1 = cntr - delta_cntr
    # cntr2 = cntr + delta_cntr
    # if (cntr1 > cntr2).any() and (cntr1 < cntr_l).any() and (cntr2 > cntr_r).any:
    #     print("ASdsa")

    # 初始化spread
    # 1、不确定均值之间的距离乘上一个系数
    # spread = torch.sqrt(cntr2**2 - cntr1**2)
    # 2、不确定均值直接相减
    # spread = cntr2 - cntr1
    # 3、样本输入于聚类中心的加权平均值
    spread = torch.sqrt(torch.einsum("nc,ncs->cs", u, ((data.unsqueeze(1) - cntr.unsqueeze(0)) ** 2)) / u.sum(dim=0).unsqueeze(1))

    # 方法二 A smoothing Group Lasso based interval type-2 fuzzy neural network for simultaneous feature selection and system identification
    delta_noise = torch.rand(c, s, dtype=torch.float64) * 0.1
    cntr1 = cntr - delta_noise
    cntr2 = cntr + delta_noise
    # spread = torch.rand(c, s, dtype=torch.float64)
    # spread = torch.sqrt(cntr2**2 - cntr1**2)

    return cntr1, cntr2, spread


def IT2FCM_predcompute(data, cntr, m1, m2, c, metric):
    """
    更新测试集的隶属矩阵
    :param data:样本点数据 tensor (n,s)
    :param cntr:已训练的聚类中心 tensor (c, s)
    :param m1:模糊度 int
    :param m2:模糊度 (m2>m1>1) int
    :param c:聚类中心个数 int
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :return: u:隶属度矩阵
    """

    # 计算距离
    d = _distance(data, cntr, metric=metric)  # (n, c)
    d = torch.fmax(d, torch.tensor(torch.finfo(torch.float64).eps))

    # 更新上、下隶属度矩阵
    u_low = (normalize_power_columns(d, -2./(m1-1)) * (1/normalize_columns(d) >= 1/c)) + \
            (normalize_power_columns(d, -2./(m2-1)) * (1/normalize_columns(d) < 1/c))  # (n, c)
    u_upper = (normalize_power_columns(d, -2./(m1-1)) * (1/normalize_columns(d) < 1/c)) + \
              (normalize_power_columns(d, -2./(m2-1)) * (1/normalize_columns(d) >= 1/c))  # (n, c)
    u_low = torch.fmax(u_low, torch.tensor(torch.finfo(u_low.dtype).eps))
    u_upper = torch.fmax(u_upper, torch.tensor(torch.finfo(u_upper.dtype).eps))

    # 校正上、下隶属度矩阵区间
    condition = u_low > u_upper
    temp = u_low[condition].clone()
    u_low[condition] = u_upper[condition]
    u_upper[condition] = temp

    # 更新隶属度矩阵
    u = (u_low + u_upper) / 2  # (n, c)

    # 归一化隶属度矩阵
    u = normalize_columns(u)  # (n, c)

    # # 目标函数
    # jm1 = (u**m1 * d**2).sum()  # float
    # jm2 = (u**m2 * d**2).sum()  # float

    return u


def IT2FCM_predict(data, cntr_trained, max_iter, m1=1.5, m2=2, error=1e-6, metric='euclidean'):
    """
    IT2FCM对测试集进行预测 (虽然无监督学习没有标签一般不进行预测 但这里还是分为训练集和测试集 训练集迭代出的聚类中心对测试集进行预测)
    :param data: 输入数据 tensor (n, s)  n为样本个数，s为特征个数
    :param cntr_trained:已训练的聚类中心 tensor (c, s)
    :param max_iter:最大迭代次数 int
    :param m1:模糊度 int (m2>m1>1)
    :param m2:模糊度 int
    :param error:迭代停止误差 float
    :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
    :return:
        cntr:聚类中心 tensor (c, s)
        p:迭代次数 int
        labels:聚类结果 tensor (, n)
    """

    data = data.double()  # 转换成float64
    c = cntr_trained.shape[0]

    # 初始化聚类中心
    cntr0 = cntr_trained

    # 初始循环参数
    jm1 = torch.zeros(0)
    jm2 = torch.zeros(0)
    p = 0

    # 开始迭代
    while p < max_iter-1:
        cntr, u = IT2FCM_predcompute(data, cntr0, m1, m2, c, metric)
        # jm1 = torch.hstack((jm1, Jjm1))
        # jm2 = torch.hstack((jm2, Jjm2))
        p += 1

        # 停止循环
        if torch.linalg.norm(cntr - cntr0) < error:
            break
        cntr0 = cntr.clone()

    # 找到隶属度矩阵中每行的最大值，即该样本最大可能所属类别
    labels = torch.argmax(u, dim=1)

    return cntr, p, labels