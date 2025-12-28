# -*- coding: UTF-8 -*-
"""
@Author ：L.Gao
@Date ：2024/9/8 21:12
"""
import torch
from torch import sum
from torch import arange, sign, where, isclose, einsum, abs


def trim(intervals):
    v = intervals[:, 3]
    i, = where(v > 0)
    if i.size == 0:
        return False
    else:
        min1 = i[0]
        max1 = i[-1] + 1

        v = intervals[:, 2]
        i, = where(v > 0)
        if i.size == 0:
            min2 = min1
            max2 = max1
        else:
            min2 = i[0]
            max2 = i[-1] + 1
        return intervals[min(min1, min2):max(max1, max2), :]


def KM_algorithm(low_f, upper_f, con_low_y, con_upper_y, order):
    """
    KM降型算法 后件为区间  向量运算同时处理矩阵的每一行执行KM
    :param low_f: 激活强度区间左端点 (num_sample, num_fuzzy)
    :param upper_f: 激活强度区间右端点 (num_sample, num_fuzzy)
    :param con_low_y:后件区间左端点 0阶TSK(ty_dim, num_fuzzy) 1阶TSK(num_sample, num_fuzzy, ty_dim)
    :param con_upper_y:后件区间右端点 0阶TSK(ty_dim, num_fuzzy) 1阶TSK(num_sample, num_fuzzy, ty_dim)
    :param order:TSK系统的阶，取值{'classi_first', 'simpl_first' , 'zero'}
    :return:降型输出y_l, y_r (num_sample, out_dim)
    """

    # 设置误差阈值
    epsilon = 1e-5

    if order == "zero":

        num_fuzzy = low_f.shape[1]
        ty_dim = con_low_y.shape[0]
        num_sample = low_f.shape[0]

        # ---------------------------------计算y_l---------------------------------

        f = (low_f + upper_f) / 2.
        f_l = f  # (num_sample, num_fuzzy)

        # 重排序
        l_ind = con_low_y.argsort(dim=1)  # 排序索引矩阵 (ty_dim, num_fuzzy)
        # 方法一 low_f = low_f.unsqueeze(1).repeat(1, ty_dim, 1)[:, torch.arange(ty_dim).repeat(num_fuzzy, 1), l_ind]
        # 方法二
        l_ind_expanded = l_ind.unsqueeze(0).expand(num_sample, -1, -1)
        low_f_expanded = low_f.unsqueeze(1).expand(-1, ty_dim, -1)
        low_f = torch.gather(low_f_expanded, 2, l_ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
        upper_f_expanded = upper_f.unsqueeze(1).expand(-1, ty_dim, -1)
        upper_f = torch.gather(upper_f_expanded, 2, l_ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
        con_low_y = torch.gather(con_low_y, 1, l_ind)  # (ty_dim, num_fuzzy)

        # 计算初始y_l
        y_l_prime_num = f_l @ con_low_y.T  # (num_sample, ty_dim)
        y_prime_den = sum(f_l, dim=1).unsqueeze(1)  # (num_sample, 1)
        y_l_prime = y_l_prime_num / y_prime_den  # (num_sample, ty_dim)

        # 设置阈值
        count = 0

        while True:
            count += 1
            # 找出满足条件的k
            con_low_y_expanded = con_low_y.unsqueeze(0).expand(num_sample, ty_dim,
                                                               num_fuzzy).contiguous()  # (num_sample, ty_dim, num_fuzzy)
            y_l_prime_expanded = y_l_prime.unsqueeze(2)  # (num_sample, ty_dim, 1)
            k_l = (torch.searchsorted(con_low_y_expanded, y_l_prime_expanded,
                                      right=True) - 1)  # (num_sample, ty_dim, 1)
            """debug k_l 是否超出范围"""

            # 重置激活强度f
            ii = arange(num_fuzzy).expand(num_sample, ty_dim, num_fuzzy).to(low_f.device)
            f_l = torch.where(ii > k_l, low_f, upper_f)  # (num_sample, ty_dim, num_fuzzy)

            # 计算y_l
            y_l_num = einsum("ijk,jk->ij", f_l, con_low_y)  # (num_sample, ty_dim)
            y_l_den = sum(f_l, dim=2)  # (num_sample, ty_dim)
            y_l = y_l_num / y_l_den  # (num_sample, ty_dim)

            mask = abs(y_l - y_l_prime) > epsilon
            if mask.any():
                y_l_prime[mask] = y_l[mask]
            else:
                break

            if count >= num_fuzzy:
                break

        # ---------------------------------计算y_r---------------------------------

        f_r = f  # (num_sample, num_fuzzy)

        # 重排序
        r_ind = con_upper_y.argsort(dim=1)  # 排序索引矩阵 (ty_dim, num_fuzzy)
        r_ind_expanded = r_ind.unsqueeze(0).expand(num_sample, -1, -1)  # (num_sample, ty_dim, num_fuzzy)
        low_f = torch.gather(low_f_expanded, 2, r_ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
        upper_f = torch.gather(upper_f_expanded, 2, r_ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
        con_upper_y = torch.gather(con_upper_y, 1, r_ind)  # (ty_dim, num_fuzzy)

        # 计算初始y_r
        y_r_prime_num = f_r @ con_upper_y.T  # (num_sample, ty_dim)
        y_r_prime = y_r_prime_num / y_prime_den  # (num_sample, ty_dim)

        # 设置阈值
        count = 0

        while True:
            count += 1
            # 找出满足条件的k
            con_upper_y_expanded = con_upper_y.unsqueeze(0).expand(num_sample, ty_dim,
                                                                   num_fuzzy).contiguous()  # (num_sample, ty_dim, num_fuzzy)
            y_r_prime_expanded = y_r_prime.unsqueeze(2)  # (num_sample, ty_dim, 1)
            k_r = (torch.searchsorted(con_upper_y_expanded, y_r_prime_expanded,
                                      right=True) - 1).contiguous()  # (num_sample, ty_dim, 1)
            """debug k_l 是否超出范围"""

            # 重置激活强度f
            ii = arange(num_fuzzy).expand(num_sample, ty_dim, num_fuzzy).to(low_f.device)
            f_r = torch.where(ii > k_r, upper_f, low_f)  # (num_sample, ty_dim, num_fuzzy)

            # 计算y_r
            y_r_num = einsum("ijk,jk->ij", f_r, con_upper_y)  # (num_sample, ty_dim)
            y_r_den = sum(f_r, dim=2)  # (num_sample, ty_dim)
            y_r = y_r_num / y_r_den  # (num_sample, ty_dim)

            mask = abs(y_r - y_r_prime) > epsilon
            if mask.any():
                y_r_prime[mask] = y_r[mask]
            else:
                break

            if count >= num_fuzzy:
                break

        return y_l, y_r

    elif order == "classi_first" or order == "simpl_first":

        num_fuzzy = low_f.shape[1]
        ty_dim = con_low_y.shape[2]
        num_sample = low_f.shape[0]

        # ---------------------------------计算y_l---------------------------------

        f = (low_f + upper_f) / 2.
        f_l = f  # (num_sample, num_fuzzy)

        # 重排序
        con_low_y = con_low_y.transpose(1, 2)  # (num_sample, ty_dim, num_fuzzy)
        l_ind = con_low_y.argsort(dim=2)  # 排序索引矩阵 (num_sample, ty_dim, num_fuzzy)
        low_f_expanded = low_f.unsqueeze(1).expand(-1, ty_dim, -1)  # (num_sample, ty_dim, num_fuzzy)
        low_f = torch.gather(low_f_expanded, 2, l_ind)  # (num_sample, ty_dim, num_fuzzy)
        upper_f_expanded = upper_f.unsqueeze(1).expand(-1, ty_dim, -1)  # (num_sample, ty_dim, num_fuzzy)
        upper_f = torch.gather(upper_f_expanded, 2, l_ind)  # (num_sample, ty_dim, num_fuzzy)
        con_low_y = torch.gather(con_low_y, 2, l_ind)  # (num_sample, ty_dim, num_fuzzy)

        # 计算初始y_l
        y_l_prime_num = torch.einsum("ijk,ik->ij", con_low_y, f_l)  # (num_sample, ty_dim)
        y_prime_den = sum(f_l, dim=1).unsqueeze(1)  # (num_sample, 1)
        # y_prime_den.data = torch.where(y_prime_den == 0, torch.full_like(y_prime_den, 1.0), y_prime_den)  # 避免分母为0
        y_l_prime = y_l_prime_num / y_prime_den  # (num_sample, ty_dim)


        # 设置阈值
        count = 0

        while True:
            count += 1
            # 找出满足条件的k
            y_l_prime_expanded = y_l_prime.unsqueeze(2)  # (num_sample, ty_dim, 1)
            k_l = (torch.searchsorted(con_low_y, y_l_prime_expanded,
                                      right=True) - 1)  # (num_sample, ty_dim, 1)
            """debug k_l 是否超出范围"""

            # 重置激活强度f
            ii = arange(num_fuzzy).expand(num_sample, ty_dim, num_fuzzy).to(low_f.device)
            f_l = torch.where(ii > k_l, low_f, upper_f)  # (num_sample, ty_dim, num_fuzzy)

            # 计算y_l
            y_l_num = einsum("ijk,ijk->ij", f_l, con_low_y)  # (num_sample, ty_dim)
            y_l_den = sum(f_l, dim=2)   # (num_sample, ty_dim)
            # if (y_l_den == 0).any():
            #     raise ValueError(f"y_l_den == 0")
            # y_l_den.data = torch.where(y_l_den == 0, torch.full_like(y_l_den, 1.0), y_l_den)  # 避免分母为0
            y_l = y_l_num / y_l_den  # (num_sample, ty_dim)

            mask = abs(y_l - y_l_prime) > epsilon
            if mask.any():
                y_l_prime[mask] = y_l[mask]
            else:
                break

            if count >= num_fuzzy:
                break

        # ---------------------------------计算y_r---------------------------------

        f_r = f  # (num_sample, num_fuzzy)

        # 设置误差阈值
        epsilon = 1e-5

        # 重排序
        con_upper_y = con_upper_y.transpose(1, 2)  # (num_sample, ty_dim, num_fuzzy)
        r_ind = con_upper_y.argsort(dim=2)  # 排序索引矩阵 (num_sample, ty_dim, num_fuzzy)
        low_f = torch.gather(low_f_expanded, 2, r_ind)  # (num_sample, ty_dim, num_fuzzy)
        upper_f = torch.gather(upper_f_expanded, 2, r_ind)  # (num_sample, ty_dim, num_fuzzy)
        con_upper_y = torch.gather(con_upper_y, 2, r_ind)  # (num_sample, ty_dim, num_fuzzy)

        # 计算初始y_r
        y_r_prime_num = torch.einsum("ijk,ik->ij", con_low_y, f_r)  # (num_sample, ty_dim)
        y_r_prime = y_r_prime_num / y_prime_den  # (num_sample, ty_dim)

        # 设置阈值
        count = 0

        while True:
            count += 1
            # 找出满足条件的k
            y_r_prime_expanded = y_r_prime.unsqueeze(2)  # (num_sample, ty_dim, 1)
            k_r = (torch.searchsorted(con_upper_y, y_r_prime_expanded,
                                      right=True) - 1)  # (num_sample, ty_dim, 1)
            """debug k_r 是否超出范围"""

            # 重置激活强度f
            ii = arange(num_fuzzy).expand(num_sample, ty_dim, num_fuzzy).to(low_f.device)
            f_r = torch.where(ii > k_r, upper_f, low_f)  # (num_sample, ty_dim, num_fuzzy)

            # 计算y_r
            y_r_num = einsum("ijk,ijk->ij", f_r, con_upper_y)  # (num_sample, ty_dim)
            y_r_den = sum(f_r, dim=2)  # (num_sample, ty_dim)
            # if (y_r_den == 0).any():
            #     raise ValueError(f"y_r_den == 0")
            # y_r_den.data = torch.where(y_r_den == 0, torch.full_like(y_r_den, 1.0), y_r_den)  # 避免分母为0
            y_r = y_r_num / y_r_den  # (num_sample, ty_dim)

            mask = abs(y_r - y_r_prime) > epsilon
            if mask.any():
                y_r_prime[mask] = y_r[mask]
            else:
                break

            if count >= num_fuzzy:
                break

        return y_l, y_r


def KM_algorithm_sig(low_f, upper_f, y):
    """
    KM降型算法 后件为0阶单点值
    :param low_f: 激活强度区间左端点 (num_sample, num_fuzzy)
    :param upper_f: 激活强度区间右端点 (num_sample, num_fuzzy)
    :param y:后件值 (ty_dim, num_fuzzy)
    :return:降型输出y_l, y_r (num_sample, out_dim)
    """

    # 设置误差阈值
    epsilon = 1e-5

    num_fuzzy = low_f.shape[1]
    ty_dim = y.shape[0]
    num_sample = low_f.shape[0]

    # ---------------------------------计算y_l---------------------------------

    f = (low_f + upper_f) / 2.
    f_l = f  # (num_sample, num_fuzzy)

    # 重排序
    ind = y.argsort(dim=1)  # 排序索引矩阵 (ty_dim, num_fuzzy)
    # 方法一 low_f = low_f.unsqueeze(1).repeat(1, ty_dim, 1)[:, torch.arange(ty_dim).repeat(num_fuzzy, 1), l_ind]
    # 方法二
    ind_expanded = ind.unsqueeze(0).expand(num_sample, -1, -1)
    low_f_expanded = low_f.unsqueeze(1).expand(-1, ty_dim, -1)
    low_f = torch.gather(low_f_expanded, 2, ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
    upper_f_expanded = upper_f.unsqueeze(1).expand(-1, ty_dim, -1)
    upper_f = torch.gather(upper_f_expanded, 2, ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
    y = torch.gather(y, 1, ind)  # (ty_dim, num_fuzzy)

    # 计算初始y_l
    y_l_prime_num = f_l @ y.T  # (num_sample, ty_dim)
    y_prime_den = sum(f_l, dim=1).unsqueeze(1)  # (num_sample, 1)
    y_l_prime = y_l_prime_num / y_prime_den  # (num_sample, ty_dim)

    # 设置阈值
    count = 0

    while True:
        count += 1
        # 找出满足条件的k
        y_expanded = y.unsqueeze(0).expand(num_sample, ty_dim, num_fuzzy).contiguous()  # (num_sample, ty_dim, num_fuzzy)
        y_l_prime_expanded = y_l_prime.unsqueeze(2)  # (num_sample, ty_dim, 1)
        k_l = (torch.searchsorted(y_expanded, y_l_prime_expanded, right=True) - 1)  # (num_sample, ty_dim, 1)
        """debug k_l 是否超出范围"""

        # 重置激活强度f
        ii = arange(num_fuzzy).expand(num_sample, ty_dim, num_fuzzy).to(low_f.device)
        f_l = torch.where(ii > k_l, low_f, upper_f)  # (num_sample, ty_dim, num_fuzzy)

        # 计算y_l
        y_l_num = einsum("ijk,jk->ij", f_l, y)  # (num_sample, ty_dim)
        y_l_den = sum(f_l, dim=2)  # (num_sample, ty_dim)
        y_l = y_l_num / y_l_den  # (num_sample, ty_dim)

        mask = abs(y_l - y_l_prime) > epsilon
        if mask.any():
            y_l_prime[mask] = y_l[mask]
        else:
            break

        if count >= num_fuzzy:
            break

    # ---------------------------------计算y_r---------------------------------

    f_r = f  # (num_sample, num_fuzzy)

    # 计算初始y_r
    y_r_prime_num = f_r @ y.T  # (num_sample, ty_dim)
    y_r_prime = y_r_prime_num / y_prime_den  # (num_sample, ty_dim)

    # 设置阈值
    count = 0

    while True:
        count += 1
        # 找出满足条件的k
        y_expanded = y.unsqueeze(0).expand(num_sample, ty_dim, num_fuzzy).contiguous()  # (num_sample, ty_dim, num_fuzzy)
        y_r_prime_expanded = y_r_prime.unsqueeze(2)  # (num_sample, ty_dim, 1)
        k_r = (torch.searchsorted(y_expanded, y_r_prime_expanded, right=True) - 1).contiguous()  # (num_sample, ty_dim, 1)
        """debug k_l 是否超出范围"""

        # 重置激活强度f
        ii = arange(num_fuzzy).expand(num_sample, ty_dim, num_fuzzy).to(low_f.device)
        f_r = torch.where(ii > k_r, upper_f, low_f)  # (num_sample, ty_dim, num_fuzzy)

        # 计算y_r
        y_r_num = einsum("ijk,jk->ij", f_r, y)  # (num_sample, ty_dim)
        y_r_den = sum(f_r, dim=2)  # (num_sample, ty_dim)
        y_r = y_r_num / y_r_den  # (num_sample, ty_dim)

        mask = abs(y_r - y_r_prime) > epsilon
        if mask.any():
            y_r_prime[mask] = y_r[mask]
        else:
            break

        if count >= num_fuzzy:
            break

    return y_l, y_r


# def KM(low_f, upper_f, con_low_y, con_upper_y, order):
#     """
#     KM降型算法
#     :param low_f: 激活强度区间左端点 (num_sample, num_fuzzy)
#     :param upper_f: 激活强度区间右端点 (num_sample, num_fuzzy)
#     :param con_low_y:后件区间左端点 0阶TSK(ty_dim, num_fuzzy) 1阶TSK(num_sample, num_fuzzy, ty_dim)
#     :param con_upper_y:后件区间右端点 0阶TSK(ty_dim, num_fuzzy) 1阶TSK(num_sample, num_fuzzy, ty_dim)
#     :param order:TSK系统的阶，取值{'first' , 'zero'}
#     :return:
#     """
#
#
#     if order == "zero":
#
#         # ---------------------------------计算y_l---------------------------------
#
#         f = (low_f + upper_f) / 2.
#         f_l = f   # (num_sample, num_fuzzy)
#
#         # 设置误差阈值
#         epsilon = 1e-5
#
#         num_fuzzy = low_f.shape[1]
#         ty_dim = con_low_y.shape[0]
#         num_sample = low_f.shape[0]
#
#         # 重排序
#         l_ind = con_low_y.argsort(dim=1)  # 排序索引矩阵 (ty_dim, num_fuzzy)
#         # 方法一 low_f = low_f.unsqueeze(1).repeat(1, ty_dim, 1)[:, torch.arange(ty_dim).repeat(num_fuzzy, 1), l_ind]
#         # 方法二
#         l_ind_expanded = l_ind.unsqueeze(0).expand(num_sample, -1, -1)
#         low_f_expanded = low_f.unsqueeze(1).expand(-1, ty_dim, -1)
#         low_f = torch.gather(low_f_expanded, 2, l_ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
#         upper_f_expanded = upper_f.unsqueeze(1).expand(-1, ty_dim, -1)
#         upper_f = torch.gather(upper_f_expanded, 2, l_ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
#         con_low_y = torch.gather(con_low_y, 1, l_ind)  # (ty_dim, num_fuzzy)
#
#         # 计算初始y_l
#         y_l_prime_num = f_l @ con_low_y.T  # (num_sample, ty_dim)
#         y_prime_den = sum(f_l, dim=1).unsqueeze(1)  # (num_sample, 1)
#         y_l_prime = y_l_prime_num / y_prime_den   # (num_sample, ty_dim)
#
#         # 设置阈值
#         count = 0
#
#         while True:
#             count += 1
#             # 找出满足条件的k
#             con_low_y_expanded = con_low_y.unsqueeze(0).expand(num_sample, ty_dim, num_fuzzy)  # (num_sample, ty_dim, num_fuzzy)
#             y_l_prime_expanded = y_l_prime.unsqueeze(2)  # (num_sample, ty_dim, 1)
#             k_l = (torch.searchsorted(con_low_y_expanded, y_l_prime_expanded, right=True)-1)  # (num_sample, ty_dim, 1)
#             """debug k_l 是否超出范围"""
#
#             # 重置激活强度f
#             ii = arange(num_fuzzy).expand(num_sample, ty_dim, num_fuzzy)
#             f_l = torch.where(ii > k_l, low_f, upper_f)  # (num_sample, ty_dim, num_fuzzy)
#
#             # 计算y_l
#             y_l_num = einsum("ijk,jk->ij", f_l, con_low_y)  # (num_sample, ty_dim)
#             y_l_den = sum(f_l, dim=2)  # (num_sample, ty_dim)
#             y_l = y_l_num / y_l_den  # (num_sample, ty_dim)
#
#             mask = abs(y_l - y_l_prime) > epsilon
#             if mask.any():
#                 y_l_prime[mask] = y_l[mask]
#             else:
#                 break
#
#             if count >= num_fuzzy:
#                 break
#
#
#     # ---------------------------------计算y_r---------------------------------
#
#         f_r = f   # (num_sample, num_fuzzy)
#
#
#         # 重排序
#         r_ind = con_upper_y.argsort(dim=1)  # 排序索引矩阵 (ty_dim, num_fuzzy)
#         r_ind_expanded = r_ind.unsqueeze(0).expand(num_sample, -1, -1)
#         low_f_expanded = low_f.unsqueeze(1).expand(-1, ty_dim, -1)
#         low_f = torch.gather(low_f_expanded, 2, r_ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
#         upper_f_expanded = upper_f.unsqueeze(1).expand(-1, ty_dim, -1)
#         upper_f = torch.gather(upper_f_expanded, 2, r_ind_expanded)  # (num_sample, ty_dim, num_fuzzy)
#         con_upper_y = torch.gather(con_upper_y, 1, r_ind)  # (ty_dim, num_fuzzy)
#
#         # 计算初始y_l
#         y_r_prime_num = f_r @ con_upper_y.T  # (num_sample, ty_dim)
#         y_r_prime = y_r_prime_num / y_prime_den   # (num_sample, ty_dim)
#
#         # 设置阈值
#         count = 0
#
#         while True:
#             # 找出满足条件的k
#             con_upper_y_expanded = con_upper_y.unsqueeze(0).expand(num_sample, ty_dim, num_fuzzy)  # (num_sample, ty_dim, num_fuzzy)
#             y_r_prime_expanded = y_r_prime.unsqueeze(2)  # (num_sample, ty_dim, 1)
#             k_r = (torch.searchsorted(con_upper_y_expanded, y_r_prime_expanded, right=True)-1)  # (num_sample, ty_dim, 1)
#             """debug k_l 是否超出范围"""
#
#             # 重置激活强度f
#             ii = arange(num_fuzzy).expand(num_sample, ty_dim, num_fuzzy)
#             f_r = torch.where(ii > k_r, low_f, upper_f)  # (num_sample, ty_dim, num_fuzzy)
#
#             # 计算y_l
#             y_r_num = einsum("ijk,jk->ij", f_r, con_upper_y)  # (num_sample, ty_dim)
#             y_r_den = sum(f_r, dim=2)  # (num_sample, ty_dim)
#             y_r = y_r_num / y_r_den  # (num_sample, ty_dim)
#
#             mask = abs(y_r - y_r_prime) > epsilon
#             if mask.any():
#                 y_r_prime[mask] = y_r[mask]
#             else:
#                 break
#
#             if count >= num_fuzzy:
#                 break
#
#         return y_l, y_r
#
#     if order == "first":
#         pass


# def KM_algorithm_fork(intervals):  # intervals = [[a1, b1, c1, d1], [a2, b2, c2, d2], ...]
#     """
#     KM algorithm
#
#     Parameters
#     ----------
#
#     intervals:
#         numpy (n, 4) array
#
#         Y = intervals[:, 0:2]
#
#         F = intervals[:, 2:4]
#
#     params:
#         List
#
#         List of parameters of algorithm, if it is needed.
#
#     Returns
#     -------
#     Tuple (l, r)
#     """
#     # left calculations
#     # intervals = trim(intervals)
#
#     if intervals is False:
#         return 0., 0.
#
#     w = (intervals[:, 2] + intervals[:, 3]) / 2.
#     w_l = w[:]
#
#     N = len(intervals)
#     intervals = intervals[intervals[:, 0].argsort()]
#     y_l_prime_num = sum(intervals[:, 0] * w_l)
#     y_prime_den = sum(w_l)
#     y_l_prime = y_l_prime_num / y_prime_den
#
#     # 设置阈值
#     count = 0
#     while True:
#         count += 1
#         k_l = 0
#         for i in range(0, N - 1):
#             if (intervals[i, 0] <= y_l_prime <= intervals[i + 1, 0]) or \
#                     isclose(intervals[i, 0], y_l_prime) or \
#                     isclose(y_l_prime, intervals[i + 1, 0]):
#                 k_l = i
#                 break
#
#         ii = arange(N)
#         w_l = (ii <= k_l) * intervals[:, 3] + (ii > k_l) * intervals[:, 2]
#         y_l_num = sum(intervals[:, 0] * w_l)
#         y_l_den = sum(w_l)
#         y_l = y_l_num / y_l_den
#         if isclose(y_l, y_l_prime, atol=1.0e-6):
#             break
#         else:
#             y_l_prime = y_l
#
#         if count >= N:
#             break
#
#     # right calculations
#     w_r = w[:]
#
#     intervals = intervals[intervals[:, 1].argsort()]
#     y_r_prime_num = sum(intervals[:, 1] * w_r)
#     y_r_prime = y_r_prime_num / y_prime_den
#
#     # 设置阈值
#     count = 0
#
#     while True:
#         count += 1
#         k_r = 0
#         for i in range(0, N - 1):
#             if (intervals[i, 1] <= y_r_prime <= intervals[i + 1, 1]) or \
#                     isclose(intervals[i, 1], y_r_prime) or \
#                     isclose(y_r_prime, intervals[i + 1, 1]):
#                 k_r = i
#                 break
#
#         ii = arange(N)
#         w_r = (ii <= k_r) * intervals[:, 2] + (ii > k_r) * intervals[:, 3]
#         y_r_num = sum(intervals[:, 1] * w_r)
#         y_r_den = sum(w_r)
#         y_r = y_r_num / y_r_den
#         if isclose(y_r, y_r_prime, atol=1.0e-6):
#             break
#         else:
#             y_r_prime = y_r
#
#         if count >= N:
#             break
#     return y_l, y_r


def EKM_algorithm(intervals, *params):
    """
    EKM algorithm

    Parameters
    ----------

    intervals:
        numpy (n, 4) array

        Y = intervals[:, 0:2]

        F = intervals[:, 2:4]

    params:
        List

        List of parameters of algorithm, if it is needed.

    Returns
    -------
    Tuple (l, r)
    """

    # Left calculations

    # intervals = trim(intervals)

    if intervals is False:
        return 0, 0

    N = len(intervals)

    k_l = round(N / 2.4)

    intervals = intervals[intervals[:, 0].argsort()]
    a_l = sum(intervals[:k_l, 0] * intervals[:k_l, 3]) + \
          sum(intervals[k_l:, 0] * intervals[k_l:, 2])
    b_l = sum(intervals[:k_l, 3]) + sum(intervals[k_l:, 2])
    y_l_prime = a_l / b_l
    while True:
        k_l_prime = 0
        for i in range(0, N - 1):
            if (intervals[i, 0] <= y_l_prime <= intervals[i + 1, 0]) or \
                    isclose(intervals[i, 0], y_l_prime, atol=1.0e-6) or \
                    isclose(y_l_prime, intervals[i + 1, 0], atol=1.0e-6):
                k_l_prime = i
                break
        if k_l_prime == k_l:
            y_l = y_l_prime
            break
        s_l = sign(torch.tensor(k_l_prime - k_l))
        imin = min(k_l, k_l_prime) + 1
        imax = max(k_l, k_l_prime)

        a_l_prime = a_l + s_l * sum(intervals[imin:imax, 0] * \
                                    (intervals[imin:imax, 3] - intervals[imin:imax, 2]))
        b_l_prime = b_l + s_l * \
                    sum(intervals[imin:imax, 3] - intervals[imin:imax, 2])
        y_l_second = a_l_prime / b_l_prime

        k_l = k_l_prime
        y_l_prime = y_l_second
        a_l = a_l_prime
        b_l = b_l_prime
    # Right calculations
    intervals = intervals[intervals[:, 1].argsort()]
    k_r = round(N / 1.7)
    a_r = sum(intervals[:k_r, 1] * intervals[:k_r, 2]) + \
          sum(intervals[k_r:, 1] * intervals[k_r:, 3])
    b_r = sum(intervals[:k_r, 2]) + sum(intervals[k_r:, 3])

    y_r_prime = a_r / b_r

    while True:
        k_r_prime = 0
        for i in range(0, N - 1):
            if (intervals[i, 1] <= y_r_prime <= intervals[i + 1, 1]) or \
                    isclose(intervals[i, 1], y_r_prime, atol=1.0e-6) or \
                    isclose(y_r_prime, intervals[i + 1, 1], atol=1.0e-6):
                k_r_prime = i
                break
        if k_r_prime == k_r:
            y_r = y_r_prime
            break

        s_r = sign(torch.tensor(k_r_prime - k_r))

        imin = min(k_r, k_r_prime) + 1
        imax = max(k_r, k_r_prime)
        a_r_prime = sum(intervals[imin:imax, 1] * (intervals[imin:imax, 3] -
                                                   intervals[imin:imax, 2]))
        b_r_prime = sum(intervals[imin:imax, 3] - intervals[imin:imax, 2])

        a_r_prime = a_r - s_r * a_r_prime
        b_r_prime = b_r - s_r * b_r_prime
        y_r_second = a_r_prime / b_r_prime
        k_r = k_r_prime
        y_r_prime = y_r_second
        a_r = a_r_prime
        b_r = b_r_prime
    return y_l, y_r


#
# def WEKM_algorithm(intervals, *params):
#     """
#     WEKM algorithm
#
#     Parameters
#     ----------
#
#     intervals:
#         numpy (n, 4) array
#
#         Y = intervals[:, 0:2]
#
#         F = intervals[:, 2:4]
#
#     params:
#         List
#
#         List of parameters of algorithm, if it is needed.
#
#     Returns
#     -------
#     Tuple (l, r)
#     """
#
#     # Left calculations
#     intervals = intervals[intervals[:, 0].argsort()]
#     intervals = trim(intervals)
#
#     if intervals is False:
#         return 0, 0
#
#     N = len(intervals)
#
#     k_l = round(N / 2.4)
#     a_l = 0
#     b_l = 0
#     for i in range(k_l):
#         a_l += params[i] * intervals[i, 0] * intervals[i, 3]
#         b_l += params[i] * intervals[i, 3]
#     for i in range(k_l, N):
#         a_l += params[i] * intervals[i, 0] * intervals[i, 2]
#         b_l += params[i] * intervals[i, 2]
#     y_l_prime = a_l / b_l
#     while True:
#         k_l_prime = 0
#         for i in range(1, N):
#             if (intervals[i - 1, 0] <= y_l_prime <= intervals[i, 0]) or \
#                     isclose(intervals[i - 1, 0], y_l_prime) or \
#                     isclose(y_l_prime, intervals[i, 0]):
#                 k_l_prime = i - 1
#                 break
#         if k_l_prime == k_l:
#             y_l = y_l_prime
#             break
#         s_l = sign(k_l_prime - k_l)
#         a_l_prime = 0
#         b_l_prime = 0
#         for i in range(min(k_l, k_l_prime) + 1, max(k_l, k_l_prime)):
#             a_l_prime += params[i] * intervals[i, 0] * (intervals[i, 3] - intervals[i, 2])
#             b_l_prime += params[i] * (intervals[i, 3] - intervals[i, 2])
#         a_l_prime = a_l + s_l * a_l_prime
#         b_l_prime = b_l + s_l * b_l_prime
#         y_l_second = a_l_prime / b_l_prime
#         k_l = k_l_prime
#         y_l_prime = y_l_second
#         a_l = a_l_prime
#         b_l = b_l_prime
#     # Right calculations
#     intervals = intervals[intervals[:, 1].argsort()]
#     k_r = round(N / 1.7)
#     a_r = 0
#     b_r = 0
#     for i in range(k_r):
#         a_r += params[i] * intervals[i, 1] * intervals[i, 2]
#         b_r += params[i] * intervals[i, 2]
#     for i in range(k_r, N):
#         a_r += params[i] * intervals[i, 1] * intervals[i, 3]
#         b_r += params[i] * intervals[i, 3]
#     y_r_prime = a_r / b_r
#     while True:
#         k_r_prime = 0
#         for i in range(1, N):
#             if (intervals[i - 1, 1] <= y_r_prime <= intervals[i, 1]) or \
#                     isclose(intervals[i - 1, 1], y_r_prime) or \
#                     isclose(y_r_prime, intervals[i, 1]):
#                 k_r_prime = i - 1
#                 break
#         if k_r_prime == k_r:
#             y_r = y_r_prime
#             break
#         s_r = sign(k_r_prime - k_r)
#         a_r_prime = 0
#         b_r_prime = 0
#         for i in range(min(k_r, k_r_prime) + 1, max(k_r, k_r_prime)):
#             a_r_prime += params[i] * intervals[i, 1] * (intervals[i, 3] - intervals[i, 2])
#             b_r_prime += params[i] * (intervals[i, 3] - intervals[i, 2])
#         a_r_prime = a_r - s_r * a_r_prime
#         b_r_prime = b_r - s_r * b_r_prime
#         y_r_second = a_r_prime / b_r_prime
#         k_r = k_r_prime
#         y_r_prime = y_r_second
#         a_r = a_r_prime
#         b_r = b_r_prime
#     return y_l, y_r
#
#
# def TWEKM_algorithm(intervals):
#     """
#     TWEKM algorithm
#
#     Parameters
#     ----------
#
#     intervals:
#         numpy (n, 4) array
#
#         Y = intervals[:, 0:2]
#
#         F = intervals[:, 2:4]
#
#     params:
#         List
#
#         List of parameters of algorithm, if it is needed.
#
#     Returns
#     -------
#     Tuple (l, r)
#     """
#     params = []
#     N = len(intervals)
#     for i in range(N):
#         if i == 0 or i == N - 1:
#             params.append(0.5)
#         else:
#             params.append(1)
#     return WEKM_algorithm(intervals, params)
#
#
# def EIASC_algorithm(intervals):
#     """
#     EIASC algorithm
#
#     Parameters
#     ----------
#
#     intervals:
#         numpy (n, 4) array
#
#         Y = intervals[:, 0:2]
#
#         F = intervals[:, 2:4]
#
#     params:
#         List
#
#         List of parameters of algorithm, if it is needed.
#
#     Returns
#     -------
#     Tuple (l, r)
#     """
#
#     # Left calculations
#
#     intervals = trim(intervals)
#
#     if intervals is False:
#         return 0, 0
#
#     N = len(intervals)
#
#     intervals = intervals[intervals[:, 0].argsort()]
#     a_l = sum(intervals[:, 0] * intervals[:, 2])
#     b_l = sum(intervals[:, 2])
#     L = 0
#     while True:
#         d = intervals[L, 3] - intervals[L, 2]
#         a_l += intervals[L, 0] * d
#         b_l += d
#         y_l = a_l / b_l
#         L += 1
#         if (y_l <= intervals[L, 0]) or isclose(y_l, intervals[L, 0]):
#             break
#             # Right calculations
#     intervals = intervals[intervals[:, 1].argsort()]
#     a_r = sum(intervals[:, 1] * intervals[:, 2])
#     b_r = sum(intervals[:, 2])
#     R = N - 1
#     while True:
#         d = intervals[R, 3] - intervals[R, 2]
#         a_r += intervals[R, 1] * d
#         b_r += d
#         y_r = a_r / b_r
#         R -= 1
#         if (y_r >= intervals[R, 1]) or isclose(y_r, intervals[R, 1]):
#             break
#     return y_l, y_r
#
#
# def WM_algorithm(intervals):
#     """
#     WM algorithm
#
#     Parameters
#     ----------
#
#     intervals:
#         numpy (n, 4) array
#
#         Y = intervals[:, 0:2]
#
#         F = intervals[:, 2:4]
#
#     params:
#         List
#
#         List of parameters of algorithm, if it is needed.
#
#     Returns
#     -------
#     Tuple (l, r)
#     """
#     intervals = intervals[intervals[:, 0].argsort()]
#     intervals = trim(intervals)
#
#     if intervals is False:
#         return 0, 0
#
#     F = intervals[:, 2:4]
#     Y = intervals[:, 0:2]
#     y_l_sup = min(sum(F[:, 0] * Y[:, 0]) / sum(F[:, 0]),
#                   sum(F[:, 1] * Y[:, 0]) / sum(F[:, 1]))
#     y_r_inf = min(sum(F[:, 1] * Y[:, 1]) / sum(F[:, 1]),
#                   sum(F[:, 0] * Y[:, 1]) / sum(F[:, 0]))
#     c = sum(F[:, 1] - F[:, 0]) / (sum(F[:, 0]) * sum(F[:, 1]))
#     y_l_inf = y_l_sup - c * (sum(F[:, 0] * (Y[:, 0] - Y[0, 0])) *
#                              sum(F[:, 1] * (Y[-1, 0] - Y[:, 0]))) / (
#                       sum(F[:, 0] * (Y[:, 0] - Y[0, 0])) + sum(F[:, 1] * (Y[-1, 0] - Y[:, 0])))
#     y_r_sup = y_r_inf + c * (sum(F[:, 1] * (Y[:, 1] - Y[0, 1])) *
#                              sum(F[:, 0] * (Y[-1, 1] - Y[:, 1]))) / (
#                       sum(F[:, 1] * (Y[:, 1] - Y[0, 1])) + sum(F[:, 0] * (Y[-1, 1] - Y[:, 1])))
#     y_l = (y_l_sup + y_l_inf) / 2
#     y_r = (y_r_sup + y_r_inf) / 2
#     return y_l, y_r
#
#
# def BMM_algorithm(intervals, params):
#     """
#     BMM algorithm
#
#     Parameters
#     ----------
#
#     intervals:
#         numpy (n, 4) array
#
#         Y = intervals[:, 0:2]
#
#         F = intervals[:, 2:4]
#
#     params:
#         List
#
#         List of parameters of algorithm, if it is needed.
#
#     Returns
#     -------
#     float
#
#     Crisp output
#     """
#     intervals = intervals[intervals[:, 0].argsort()]
#     intervals = trim(intervals)
#
#     if intervals is False:
#         return 0
#
#     F = intervals[:, 2:4]
#     Y = (intervals[:, 0] + intervals[:, 1]) / 2.
#     m = params[0]
#     n = params[1]
#     # Y = Y.reshape((Y.size,))
#     return m * sum(F[:, 0] * Y) / sum(F[:, 0]) + n * sum(F[:, 1] * Y) / sum(F[:, 1])
#
#
# def LBMM_algorithm(intervals, params):
#     """
#     LBMM algorithm (BMM extended by Li et al.)
#
#     Ref. An Overview of Alternative Type-ReductionApproaches for
#     Reducing the Computational Costof Interval Type-2 Fuzzy Logic
#     Controllers
#
#     Parameters
#     ----------
#
#     intervals:
#         numpy (n, 4) array
#
#         Y = intervals[:, 0:2]
#
#         F = intervals[:, 2:4]
#
#     params:
#         List
#
#         List of parameters of algorithm, if it is needed.
#
#     Returns
#     -------
#     float
#
#     Crisp output
#     """
#     intervals = intervals[intervals[:, 0].argsort()]
#     intervals = trim(intervals)
#
#     if intervals is False:
#         return 0
#
#     F = intervals[:, 2:4]
#     Y = intervals[:, 0:2]
#     m = params[0]
#     n = params[1]
#     return m * sum(F[:, 0] * Y[:, 0]) / sum(F[:, 0]) + n * sum(F[:, 1] * Y[:, 1]) / sum(F[:, 1])


# def NT_algorithm(intervals, params=[]):
#     """
#     NT algorithm
#
#     Parameters
#     ----------
#
#     intervals:
#         numpy (n, 4) array
#
#         Y = intervals[:, 0:2]
#
#         F = intervals[:, 2:4]
#
#     params:
#         List
#
#         List of parameters of algorithm, if it is needed.
#
#     Returns
#     -------
#     float
#
#     Crisp output
#     """
#     intervals = intervals[intervals[:, 0].argsort()]
#     intervals = trim(intervals)
#
#     if intervals is False:
#         return 0
#
#     F = intervals[:, 2:4]
#     Y = (intervals[:, 0] + intervals[:, 1]) / 2.
#     return (sum(Y * F[:, 1]) + sum(Y * F[:, 0])) / (sum(F[:, 0]) + sum(F[:, 1]))


def NT_algorithm(low_f, upper_f, con_low_y, con_upper_y, order):
    tmp1 = low_f + upper_f
    tmp2 = tmp1 / tmp1.sum(dim=1).unsqueeze(1)
    if order == "zero":
        y_l = tmp2 @ con_low_y.T
        y_r = tmp2 @ con_upper_y.T
        return y_l, y_r
    if order == "classi_first" or order == "simpl_first":
        y_l = torch.einsum("ijk,ij->ik", con_low_y, tmp2)
        # y_l = torch.bmm(tmp2.unsqueeze(1), con_low_y).squeeze()
        y_r = torch.einsum("ijk,ij->ik", con_upper_y, tmp2)
        return y_l, y_r
