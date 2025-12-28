# -*- coding: UTF-8 -*-
"""
@Author ：L.Gao
@Date ：2024/9/8 21:12
"""

import torch
import torch.nn as nn
from .membership_function import *
from .t_norms import *
from .typereduced import *
from .fcm import IT2FCM
from .utils import gate_1, piecewise_func, threshold_fun_1, threshold_fun_2, cali_inteval

# torch.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


class Input_to_MemF_layer(nn.Module):
    """
    输入层到隶属度函数层
    """

    def __init__(self, input_layer_size, num_fuzzy):
        """
        初始化Input_to_MemF_layer对象
        :param input_layer_size: 输入维度
        :param num_fuzzy: 每个输入特征的模糊子集个数
        """
        super(Input_to_MemF_layer, self).__init__()

        self.input_layer_size = input_layer_size
        self.num_fuzzy = num_fuzzy
        self.low = 0
        self.high = 1

        # 初始化隶属度函数参数
        # center1 center2 spread dim: num_fuzzy x input_layer_size
        self.center1 = nn.Parameter(
            torch.DoubleTensor(self.num_fuzzy, self.input_layer_size).uniform_(self.low, self.high))
        self.center2 = nn.Parameter(
            torch.rand(self.num_fuzzy, self.input_layer_size, dtype=torch.float64) * (
                        self.high - self.center1) + self.center1)
        self.spread = nn.Parameter(
            torch.DoubleTensor(self.num_fuzzy, self.input_layer_size).uniform_(self.low, self.high))

        # 保存初始参数
        self.init_center1 = self.center1.detach().clone()
        # detach()，浅拷贝，共享内存，不跟踪梯度，dtype一致
        # clone()，深拷贝，开辟内存，跟踪梯度，dtype一致 两者结合深拷贝，不跟踪梯度
        self.init_center2 = self.center2.detach().clone()
        self.init_spread = self.spread.detach().clone()

        if (self.center2 < self.center1).any():
            print("self.center2 < self.center1")

    def reinit_ant_IT2FCM(self, x, c, max_iter, m1, m2, error, metric, init, seed):
        """
        IT2FCM聚类算法 用于初始化模糊规则库(仅为不确定中心隶属度函数，规则前件部分)
        :param x: 输入数据 tensor (n, s)  n为样本个数，s为特征个数
        :param c:每个特征模糊子集个数 int
        :param max_iter:最大迭代次数 int
        :param m1:模糊度 int (m2>m1>1)
        :param m2:模糊度 int
        :param error:迭代停止误差 float
        :param metric:选择距离计算公式 string 取值{'braycurtis', 'canberra', 'chebyshev', 'cityblock',
            'correlation','cosine', 'dice', 'euclidean'(default), 'hamming', 'jaccard', 'jensenshannon',
            'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
            'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'}
        :param init:初始隶属度函数中心 tensor (c, s)
        :param seed:随机种子 float
        :return:空
        """
        # IT2FCM生成初始规则前件参数
        center1, center2, spread = IT2FCM(data=x, c=c, max_iter=max_iter, m1=m1, m2=m2, error=error, metric=metric,
                                          init=init, seed=seed)

        self.center1.data = center1.detach().clone()
        self.center2.data = center2.detach().clone()
        self.spread.data = spread.detach().clone()

        # 保存初始参数
        self.init_center1 = self.center1.detach().clone()
        self.init_center2 = self.center2.detach().clone()
        self.init_spread = self.spread.detach().clone()

    def reinit_spread_zero(self):
        """
        将前件宽度初始化接近于0
        :return:
        """
        self.spread.data = torch.DoubleTensor(self.num_fuzzy, self.input_layer_size).uniform_(0.00001, 0.00002)
        # self.spread.data = torch.full_like(self.spread, 0.01)

        # 保存初始参数
        self.init_spread = self.spread.detach().clone()

    def reinit_spread_one(self):
        """
        将前件宽度初始化1
        :return:
        """
        self.spread.data = torch.full_like(self.spread, 1)

        # 保存初始参数
        self.init_spread = self.spread.detach().clone()

    def forward(self, x):
        """
        向前传播，计算隶属度
        :param x: 输入数据
        :return: 上隶属度，下隶属度
        """

        # lmf umf dim: num_sample x num_fuzzy x input_layer_size
        lmf = gauss_uncert_mean_lmf(x.unsqueeze(1), self.center1, self.center2, self.spread, 1)
        umf = gauss_uncert_mean_umf(x.unsqueeze(1), self.center1, self.center2, self.spread, 1)

        if (umf < lmf).any():
            print("umf<lmf")

        if (umf < 0).any():
            print("(umf < 0).any()")

        if (lmf < 0).any():
            print("(lmf < 0).any()")

        return lmf, umf


class MemF_to_Rule_layer(nn.Module):
    """
    隶属度函数层到规则层
    """

    def __init__(self, input_layer_size, rule_layer_size, t_norm, frb):
        """
        初始化MemF_to_Rule_layer对象
        :param input_layer_size: 输入维度
        :param rule_layer_size: 规则层维度（与模糊子集个数相等）
        :param t_norm: 计算规则激活强度的算子
        :param frb:规则库类型
        """
        super(MemF_to_Rule_layer, self).__init__()

        self.t_norm = t_norm
        self.rule_layer_size = rule_layer_size
        self.input_layer_size = input_layer_size

        # 生成FRB的前件模糊集索引，CoCo-FRB, En-FRB, Cross-FRB，FuCo-FRB
        self.FRB = self._init_frb(input_layer_size, self.rule_layer_size, frb_type=frb)

    def _init_frb(self, input_layer_size, num_fuzzy, frb_type):
        """
        计算每个特征的模糊子集索引
        :param input_layer_size:  输入维数
        :param num_fuzzy: 每维下定义的模糊子集数
        :param frb_type: 规则库类型，取值{'CoCo-FRB' (default), 'En-FRB', 'Cross-FRB', 'FuCo-FRB'}
        :return: 计算激活强度要用到的各特征下的模糊集的索引
        """
        # 生成FRB的前件模糊集索引，CoCo-FRB, En-FRB, Cross-FRB，FuCo-FRB
        # fs_ind dim: num_fuzzy x input_layer_size
        if frb_type == 'CoCo-FRB':
            """
            每一行包含了每个模糊变量在样本维度中的索引
            生成CoCo-FRB的前件模糊集索引
            [0,0,0,0],[1,1,1,1],[2,2,2,2]
            """
            fs_ind = torch.tensor(range(num_fuzzy)).unsqueeze(1).repeat_interleave(input_layer_size, dim=1)
            return fs_ind.long()
        elif frb_type == 'En-FRB':
            """
            生成En-FRB的前件模糊集索引
            以[2,2,2,2]等为轴
            演变[1,2,2,2],[2,1,2,2],[2,2,1,2],[2,2,2,1],[3,2,2,2],[2,3,2,2],[2,2,3,2],[2,2,2,3]
            最终S + S * (2D)个规则
            """
            fs_ind = []
            eye = torch.eye(input_layer_size)

            # 由CoCo-FRB，向左-1，向右+1，构成En-FRB
            if input_layer_size == 1 or num_fuzzy == 1:  # 即为CoCo-FRB，此时R=S
                fs_ind.append(torch.tensor(range(num_fuzzy)).unsqueeze(1).repeat_interleave(input_layer_size, dim=1))
            elif input_layer_size == 2 or num_fuzzy == 2:  # En-FRB会有重复规则，此时R=(D+1)*S
                for i in range(num_fuzzy):
                    fs_ind_temp = torch.ones(1, input_layer_size) * i
                    fs_ind.append(fs_ind_temp)
                    if i == 0:
                        fs_ind_left = fs_ind_temp + eye * (num_fuzzy - 1)
                    else:
                        fs_ind_left = fs_ind_temp - eye
                    fs_ind.append(fs_ind_left)
            else:  # 常规情况的En-FRB，此时R=(2D+1)*S
                for i in range(num_fuzzy):
                    fs_ind_temp = torch.ones(1, input_layer_size) * i
                    fs_ind.append(fs_ind_temp)
                    if i == 0:
                        fs_ind_left = fs_ind_temp + eye * (num_fuzzy - 1)
                        fs_ind_right = fs_ind_temp + eye
                    elif i < num_fuzzy - 1:
                        fs_ind_left = fs_ind_temp - eye
                        fs_ind_right = fs_ind_temp + eye
                    else:
                        fs_ind_left = fs_ind_temp - eye
                        fs_ind_right = fs_ind_temp - eye * i
                    fs_ind.append(fs_ind_left)
                    fs_ind.append(fs_ind_right)
            return torch.cat(fs_ind).long()
        elif frb_type == 'FuCo-FRB':
            """
            生成FuCo-FRB的前件模糊集索引
            [0,0,0,0],[0,0,0,1],[0,0,0,2]
            """
            fs_ind = torch.zeros([num_fuzzy ** input_layer_size, input_layer_size])
            for i, ii in enumerate(reversed(range(input_layer_size))):  # i--正序下标；ii--倒叙下标
                fs_ind[:, ii] = torch.tensor(range(num_fuzzy)).repeat_interleave(num_fuzzy ** i).repeat(
                    num_fuzzy ** ii)
            return fs_ind.long()
        else:
            raise ValueError(
                "Invalid value for frb: '{}', expected 'CoCo-FRB', 'En-FRB', 'Cross-FRB', 'FuCo-FRB'".format(
                    self.frb))

    def forward(self, lmf, umf):
        """
        向前传播，计算激活强度
        :param lmf: 上隶属度
        :param umf: 下隶属度
        :return: 上下激活强度
        """

        # 计算激活强度区间
        input_layer_size, fs_ind = self.input_layer_size, self.FRB
        # low_f, upper_f dim: num_sample x num_fuzzy
        if self.t_norm == "product":
            low_f = lmf[:, fs_ind, range(input_layer_size)].prod(dim=2)
            upper_f = umf[:, fs_ind, range(input_layer_size)].prod(dim=2)

            # range(input_layer_size)可改成torch.arange(input_layer_size).unsqueeze(0).repeat_interleave(num_fuzzy,dim=0)
            """
            lmf整数索引之后维数不变，2维数据每行为待处理的节点
            eg：(2,3,4) [样本a： [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]],
                        样本b：  [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]]
            如果frb_type == 'CoCo-FRB' ：
            a11 a12 a13分别为第一、二、三个输入节点第一个模糊集的数据，[a11,a12,a13]此行为待处理的节点

            """

        elif self.t_norm == "softmin1":
            low_f = softmin1(lmf[:, fs_ind, range(input_layer_size)])
            upper_f = softmin1(umf[:, fs_ind, range(input_layer_size)])
            # 此softmin可能导致数量级很小的值为负 在此直接变成0
            with torch.no_grad():
                low_f.data[low_f < 0] = 0
                upper_f.data[upper_f < 0] = 0
            cali_inteval(low_f, upper_f)

        elif self.t_norm == "adasoftmin1":
            low_f = adasoftmin1(lmf[:, fs_ind, range(input_layer_size)])
            upper_f = adasoftmin1(umf[:, fs_ind, range(input_layer_size)])
            # 此softmin可能导致数量级很小的值为负 在此直接变成0
            with torch.no_grad():
                low_f.data[low_f < 0] = 0
                upper_f.data[upper_f < 0] = 0
            cali_inteval(low_f, upper_f)

        else:
            raise ValueError(f"Invalid value for tnorm: {self.t_norm}")

        if (upper_f < low_f).any():
            print("upper_f < low_f")
            # ind = upper_f < low_f
            # print(f"{low_f[ind]}")
            # print(f"{upper_f[ind]}")

        if (low_f < 0).any():
            print("(low_f < 0).any()")

        if (upper_f < 0).any():
            print("(upper_f < 0).any()")

        return low_f, upper_f


class Normalization2d(nn.Module):
    def __init__(self, dim=1):
        """
        归一化层 【0，1】
        :param dim: 归一化维度 int 1(default)
        """
        super(Normalization2d, self).__init__()
        self.dim = dim

    def forward(self, low, upper):

        dim = self.dim
        low_normalized = low / low.sum(dim=dim, keepdim=True)
        upper_normalized = upper / upper.sum(dim=dim, keepdim=True)


        return low_normalized, upper_normalized


# class MinMaxNorm2d(nn.Module):
#     def __init__(self, dim=1):
#         """
#         归一化层 【0，1】
#         :param dim: 归一化维度 int 1(default)
#         """
#         super(MinMaxNorm2d, self).__init__()
#         self.dim = dim
#
#     def forward(self, low, upper):
#
#         dim = self.dim
#
#         min_low_val = low.amin(dim=dim, keepdim=True)
#         max_low_val = low.amax(dim=dim, keepdim=True)
#         min_upper_val = upper.amin(dim=dim, keepdim=True)
#         max_upper_val = upper.amax(dim=dim, keepdim=True)
#
#         low_normalized = (low - min_low_val) / (max_low_val - min_low_val)
#         upper_normalized = (upper - min_upper_val) / (max_upper_val - min_upper_val)
#
#         condition = low_normalized > upper_normalized
#         low_normalized.data[condition] = upper_normalized.data[condition]
#
#         # 将 low 和 upper 中的 0 替换为一个非常小的数（例如 1e-3）
#         low_normalized.data = torch.where(low_normalized == 0, torch.full_like(low_normalized, 1e-3), low_normalized)
#         upper_normalized.data = torch.where(upper_normalized == 0, torch.full_like(upper_normalized, 1e-3), upper_normalized)
#
#         # 将 low 和 upper 中的 Nan 替换为 1 (可能除法中分母为0出错)
#         low_normalized.data = torch.where(torch.isnan(low_normalized), torch.full_like(low_normalized, 1), low_normalized)
#         upper_normalized.data = torch.where(torch.isnan(upper_normalized), torch.full_like(upper_normalized, 1), upper_normalized)
#
#         return low_normalized, upper_normalized

class MinMaxNorm2d(nn.Module):
    def __init__(self, dim=1, eps=1e-6):
        super(MinMaxNorm2d, self).__init__()
        self.dim = dim
        self.eps = eps  # 避免除零

    def forward(self, low, upper):
        dim = self.dim

        min_low_val = low.amin(dim=dim, keepdim=True)
        max_low_val = low.amax(dim=dim, keepdim=True)
        min_upper_val = upper.amin(dim=dim, keepdim=True)
        max_upper_val = upper.amax(dim=dim, keepdim=True)

        # 给分母加eps，避免除零
        low_normalized = (low - min_low_val) / (max_low_val - min_low_val + self.eps)
        upper_normalized = (upper - min_upper_val) / (max_upper_val - min_upper_val + self.eps)

        # 保证 low ≤ upper
        condition = low_normalized > upper_normalized
        low_normalized.data[condition] = upper_normalized.data[condition]

        # 给0加上一个小常数，避免梯度为0
        low_normalized = torch.clamp(low_normalized, min=self.eps)
        upper_normalized = torch.clamp(upper_normalized, min=self.eps)

        return low_normalized, upper_normalized



class Rule_to_TypeReduced_layer(nn.Module):
    """
    规则层到降型层，仅为后件为TSK模糊推理 0阶，1阶
    """

    def __init__(self, rule_layer_size, typereduced_layer_size, input_layer_size, order, typereduced, gl_fea_sel,
                 opt_function, tau_fea):
        """
        初始化Rule_to_TypeReduced_layer对象
        :param rule_layer_size: 规则层维度（等于模糊子集个数）
        :param typereduced_layer_size: 降型层维度（等于输出维度）
        :param input_layer_size: 输入维度
        :param order: TSK模糊系统的阶，取值{'zero','classi_first','simpl_first'}
        :param typereduced: 降型器的类型，取值{”KM“, "NT"}
        :param gl_fea_sel:是否使用Group Lasso进行特征选择 取值{False (default), True}
        :param opt_function: 1阶TSK特征选择消除前件对后件影响的函数选择 {piecewise_func(default), gate_1}
        :param tau_fea: float 特征选择单个变量阈值
        """
        super(Rule_to_TypeReduced_layer, self).__init__()

        self.rule_layer_size = rule_layer_size
        self.input_layer_size = input_layer_size
        self.typereduced_layer_size = typereduced_layer_size
        self.order = order
        self.typereduced = typereduced
        self.gl_fea_sel = gl_fea_sel
        self.opt_function = opt_function
        self.tau_fea = tau_fea

        # 初始后件参数
        if self.order == "zero":
            # con_low_param con_upper_param dim: typereduced_layer_size X num_fuzzy
            self.con_low_param = nn.Parameter(
                torch.DoubleTensor(self.typereduced_layer_size, self.rule_layer_size).uniform_(0, 1))
            self.con_upper_param = nn.Parameter(
                torch.rand(self.typereduced_layer_size, self.rule_layer_size, dtype=torch.float64) * (
                            1 - self.con_low_param) + self.con_low_param)
        elif self.order == "simpl_first":
            # con_low_param con_upper_param dim: typereduced_layer_size X num_fuzzy X input_layer_size+1
            self.con_low_param = nn.Parameter(torch.DoubleTensor(self.typereduced_layer_size, self.rule_layer_size,
                                                                 self.input_layer_size + 1).uniform_(0, 1))
            self.con_upper_param = nn.Parameter(
                torch.rand(self.typereduced_layer_size, self.rule_layer_size, self.input_layer_size + 1,
                           dtype=torch.float64) * (1 - self.con_low_param) + self.con_low_param)
        elif self.order == "classi_first":
            # con_low_param(spread) con_upper_param(center) dim: typereduced_layer_size X num_fuzzy X input_layer_size+1
            self.con_low_param = nn.Parameter(torch.DoubleTensor(self.typereduced_layer_size, self.rule_layer_size,
                                                                 self.input_layer_size + 1).uniform_(0, 1))
            self.con_upper_param = nn.Parameter(
                torch.rand(self.typereduced_layer_size, self.rule_layer_size, self.input_layer_size + 1,
                           dtype=torch.float64) * (1 - self.con_low_param) + self.con_low_param)
        else:
            raise ValueError(f"Invalid value for order: {self.order}, expected 'zero', 'simpl_first', 'classi_first'")

        # 记录初始值
        self.init_con_low_param = self.con_low_param.detach().clone()
        self.init_con_upper_param = self.con_upper_param.detach().clone()

        if (self.con_upper_param < self.con_low_param).any():
            print("self.con_upper_param < self.con_low_param")

    def elimi_effect(self, input_data, ant_norm, tau_fea, opt_function="piecewise_func"):
        """
        特征选择时，消除前件对后件的影响
        :param input_data: 输入数据
        :param ant_norm: 前件宽度范数
        :param tau_fea: float 特征选择参数阈值
        :param opt_function: 消除影响的函数选择 {piecewise_func(default), gate_1}
        :return:处理后的输入数据
        """
        if self.order != "zero":
            # threshold = threshold_fun_1(ant_norm.min(), ant_norm.max(), tau_fea)
            # threshold = tau_fea * torch.sqrt(torch.tensor(self.rule_layer_size))  # (门限值 = tau*根号下向量个数)
            threshold = threshold_fun_2(ant_norm, tau_fea)
            # 1阶TSK才需要消除前件对后件的影响
            if opt_function == "gate_1":
                input_data = input_data * gate_1(ant_norm, self.rule_layer_size)
            elif opt_function == "piecewise_func":
                input_data = input_data * piecewise_func(ant_norm, threshold)
            else:
                raise ValueError(f"Invalid value for optim_type: '{opt_function}'")

            # 将临时未被选取的特征对应的后件置0
            ind_zero = torch.nonzero(input_data[0, :] < 1e-6).squeeze(1) + 1
            self.con_low_param.data[:, :, ind_zero] = 0.0
            self.con_upper_param.data[:, :, ind_zero] = 0.0


        return input_data

    def reinit_con_zero(self):
        """
        将后件参数初始化接近于0
        :return:
        """
        # self.con_low_param.data = torch.full_like(self.con_low_param, 0.001)
        # self.con_upper_param.data = torch.full_like(self.con_upper_param, 0.001)

        self.con_low_param.data = torch.torch.full_like(self.con_low_param, 0.001).uniform_(0.00001, 0.000015)
        self.con_upper_param.data = torch.torch.full_like(self.con_upper_param, 0.001).uniform_(0.000015, 0.00002)

        # 记录初始值
        self.init_con_low_param = self.con_low_param.detach().clone()
        self.init_con_upper_param = self.con_upper_param.detach().clone()


    def forward(self, input_data, low_f, upper_f, ant_norm):
        """
        向前传播，降型计算
        :param input_data: 输入数据
        :param low_f: 下激活强度
        :param upper_f: 上激活强度
        :param ant_norm: 模型前件宽度范数
        :return: 降型输出
        """

        if self.gl_fea_sel and self.order != "zero":  # 如果对1阶TSK进行特征选择 需要对后件进行处理
            input_data = self.elimi_effect(input_data, ant_norm=ant_norm, opt_function=self.opt_function,
                                           tau_fea=self.tau_fea)

        # 计算后件输出
        if self.order == "zero":
            # con_low_y con_upper_y dim: typereduced_layer_size X num_fuzzy
            con_low_y = self.con_low_param
            con_upper_y = self.con_upper_param
        elif self.order == "simpl_first":
            # con_low_y con_upper_y dim: num_sample X num_fuzzy X typereduced_layer_size
            con_low_y = (self.con_low_param[:, :, 1:] @ input_data.T).permute(2, 1, 0) + self.con_low_param[:, :, 0].T
            con_upper_y = (self.con_upper_param[:, :, 1:] @ input_data.T).permute(2, 1, 0) + self.con_upper_param[:, :, 0].T
        elif self.order == "classi_first":
            # 如果con_low_param(宽度)为负值 修正为正值
            with torch.no_grad():
                negative_ind = self.con_low_param < 0
                self.con_low_param.data[negative_ind] = torch.abs(self.con_low_param.data[negative_ind])
            con_low_y = (self.con_upper_param[:, :, 1:] @ input_data.T - self.con_low_param[:, :, 1:] @ torch.abs(
                input_data.T)).permute(2, 1, 0) + (self.con_upper_param[:, :, 0] - self.con_low_param[:, :, 0]).T
            con_upper_y = (self.con_upper_param[:, :, 1:] @ input_data.T + self.con_low_param[:, :, 1:] @ torch.abs(
                input_data.T)).permute(2, 1, 0) + (self.con_upper_param[:, :, 0] + self.con_low_param[:, :, 0]).T

            if (self.con_low_param < 0).any():
                print("(self.con_low_param < 0).any()")

        else:
            raise ValueError(f"Invalid value for order: {self.order}, expected 'simpl_first', 'classi_first', 'zero'")

        if (con_upper_y < con_low_y).any():
            print("con_upper_y < con_low_y")

        # 降型
        # yl yr dim: num_sample X typereduced_layer_size
        if self.typereduced == "KM":
            yl, yr = KM_algorithm(low_f, upper_f, con_low_y, con_upper_y, self.order)
        elif self.typereduced == "EKM":
            num_sample = low_f.shape[0]
            yl = torch.zeros(num_sample, con_low_y.shape[0])
            yr = torch.zeros(num_sample, con_low_y.shape[0])
            if self.order == "zero":
                for i in range(num_sample):
                    intervals = torch.stack((con_low_y,  # (typereduced_layer_size, num_fuzzy)
                                             con_upper_y,  # (typereduced_layer_size, num_fuzzy)
                                             low_f[i, :].unsqueeze(0).expand(self.typereduced_layer_size, -1),
                                             # (typereduced_layer_size, num_fuzzy)
                                             upper_f[i, :].unsqueeze(0).expand(self.typereduced_layer_size, -1)),
                                            # (typereduced_layer_size, num_fuzzy)
                                            dim=2)  # (typereduced_layer_size, num_fuzzy, 4)
                    for j in range(self.typereduced_layer_size):
                        yl[i, j], yr[i, j] = EKM_algorithm(intervals[j])

            elif self.order == "classi_first" or self.order == "simpl_first":
                for i in range(num_sample):
                    intervals = torch.stack((
                        con_low_y[i].T,  # (typereduced_layer_size, num_fuzzy)
                        con_upper_y[i].T,  # (typereduced_layer_size, num_fuzzy)
                        low_f[i].unsqueeze(0).expand(self.typereduced_layer_size, -1),
                        # (typereduced_layer_size, num_fuzzy)
                        upper_f[i].unsqueeze(0).expand(self.typereduced_layer_size, -1)
                        # (typereduced_layer_size, num_fuzzy)
                    ), dim=2)  # (typereduced_layer_size, num_fuzzy, 4)
                    for j in range(self.typereduced_layer_size):
                        yl[i, j], yr[i, j] = EKM_algorithm(intervals[j])
        elif self.typereduced == "NT":
            yl, yr = NT_algorithm(low_f, upper_f, con_low_y, con_upper_y, self.order)
        else:
            raise ValueError(f"Invalid value for typereduced algorithm: {self.typereduced}")

        return yl, yr


class TypeReduced_to_Output_layer(nn.Module):
    """
    降型层到输出层
    """

    def __init__(self, q):
        """
        初始化降型层到输出层
        :param q: 控制输出系数
        """
        super(TypeReduced_to_Output_layer, self).__init__()

        self.q = q

    def forward(self, yl, yr):
        """
        向前传播，计算系统输出
        :param yl: 降型输出左端点
        :param yr: 降型输出右端点
        :return: 系统输出
        """
        # y dim: num_sample X output_layer_size
        y = self.q * yl + (1 - self.q) * yr
        return y
