# -*- coding: UTF-8 -*-
"""
@Author ：L.Gao
@Date ：2024/9/8 21:12
"""

from .layer import *
import torch
import torch.nn as nn
from .utils import threshold_fun_1, threshold_fun_2
import copy
from .train_test import *

class IT2TSK(nn.Module):
    """
    区间二型TSK模糊系统，可指定一阶或零阶 (目前只支持不确定中心高斯隶属度函数)
    """

    def __init__(self, input_layer_size, num_fuzzy, output_layer_size=1, frb='CoCo-FRB', tnorm='product',
                 order='zero', typereduced="NT", firing_normalized=False, q=0.5, gl_fea_sel=False, gl_rule_extr=False,
                 opt_function="piecewise_func", tau_fea=0.005):
        """
        初始化IT2TSK模糊系统
        :param input_layer_size: 输入维度
        :param num_fuzzy: 每个特征下的模糊集数
        :param output_layer_size: 输出维度
        :param frb: 规则库类型，取值{'CoCo-FRB' (default), 'En-FRB', 'Cross-FRB', 'FuCo-FRB'}
        :param tnorm: 计算规则激活强度的算子，取值{'product' (default),'softmin1', 'adasoftmin1'}
        :param order: 取值{zero (default), simpl_first, classi_first}，前者，零阶TSK模型；后两者，一阶TSK模型 simpl_first：后件为[wl, wr], classi_first:后件为[c-s, c+s]
        :param typereduced:降阶器, 取值{KM, NT (default)}
        :param firing_normalized: 是否添加激活强度归一化层, 取值{True, False (default)}
        :param q: 控制输出系数,取值{[0, 1], 0.5(default)}
        :param gl_fea_sel: 是否使用Group Lasso进行特征选择 取值{False (default), True}
        :param gl_rule_extr: 是否使用Group Lasso进行规则提取 取值{False (default), True}
        :param opt_function: 1阶TSK特征选择消除前件对后件影响的函数选择 {piecewise_func(default)}
        :param tau_fea: 特征选择时变量的阈值 0.005(default)
        """
        super(IT2TSK, self).__init__()

        self.input_layer_size = input_layer_size
        self.num_fuzzy = num_fuzzy
        self.rule_layer_size = num_fuzzy
        self.typereduced_layer_size = output_layer_size
        self.output_layer_size = output_layer_size
        self.order = order
        self.tnorm = tnorm
        self.typereduced = typereduced
        self.firing_normalized = firing_normalized
        self.frb = frb
        self.gl_fea_sel = gl_fea_sel
        self.gl_rule_extr = gl_rule_extr
        self.opt_function = opt_function
        self.tau_fea = tau_fea


        # 构建输出层到隶属度函数层
        self.input_to_memf_layer = Input_to_MemF_layer(self.input_layer_size, self.num_fuzzy)

        # 构建隶属度函数层到规则层
        self.memf_to_rule_layer = MemF_to_Rule_layer(self.input_layer_size, self.rule_layer_size, self.tnorm, self.frb)

        if self.firing_normalized:
            # 构建激活强度归一化层
            self.firing_normalized_layer = MinMaxNorm2d(dim=1)

        # 构建规则层到降型层
        self.rule_to_typereduced_layer = Rule_to_TypeReduced_layer(self.rule_layer_size, self.typereduced_layer_size,
                                                                   self.input_layer_size, self.order, self.typereduced,
                                                                   self.gl_fea_sel, self.opt_function, self.tau_fea)

        # 构建降型层到输出层
        self.typereduced_to_output_layer = TypeReduced_to_Output_layer(q)

    def forward(self, x):
        """
        系统向前传播
        :param x: 输入数据，[num_sample, input_layer_size]
        :return: 系统输出，[num_sample, output_layer_size]
        """

        lmf, umf = self.input_to_memf_layer(x)

        low_f, upper_f = self.memf_to_rule_layer(lmf, umf)

        # 是否对激活强度层归一化
        if self.firing_normalized:
            low_f_normalized, upper_f_normalized = self.firing_normalized_layer(low_f, upper_f)
            yl, yr = self.rule_to_typereduced_layer(x, low_f_normalized, upper_f_normalized, self.ant_norm)
        else:
            yl, yr = self.rule_to_typereduced_layer(x, low_f, upper_f, self.ant_norm)

        y = self.typereduced_to_output_layer(yl, yr)

        # if torch.isnan(y).any():
        #     print("NaN detected! Entering debug mode...")


        return y

    def trained_param(self, tra_param=('all',)):
        """
        指定哪些参数是需要训练的
        :param tra_param: tuple 被训练的参数元组，取值{'center', 'spread', 'IF','THEN','IF_THEN','all'(default)}
               center: 前件中心 spread: 前件宽度 IF：前件参数；THEN：后件参数；IF_THEN：前件后件参数；all：全部参数
        :return:
        """
        # 先将所有参数都指定为不求梯度
        for each in self.parameters():
            each.requires_grad = False

        # 再指定某些参数可求梯度
        for param in tra_param:
            if param == 'None':
                pass
            elif param == 'center':
                self.input_to_memf_layer.center1 = nn.Parameter(self.input_to_memf_layer.center1)
                self.input_to_memf_layer.center2 = nn.Parameter(self.input_to_memf_layer.center2)
            elif param == 'spread':
                self.input_to_memf_layer.spread = nn.Parameter(self.input_to_memf_layer.spread)
            elif param == 'IF':
                self.input_to_memf_layer.center1 = nn.Parameter(self.input_to_memf_layer.center1)
                self.input_to_memf_layer.center2 = nn.Parameter(self.input_to_memf_layer.center2)
                self.input_to_memf_layer.spread = nn.Parameter(self.input_to_memf_layer.spread)
            elif param == 'THEN':
                self.rule_to_typereduced_layer.con_low_param = nn.Parameter(self.rule_to_typereduced_layer.con_low_param)
                self.rule_to_typereduced_layer.con_upper_param = nn.Parameter(
                    self.rule_to_typereduced_layer.con_upper_param)
            elif param == 'IF_THEN':
                self.input_to_memf_layer.center1 = nn.Parameter(self.input_to_memf_layer.center1)
                self.input_to_memf_layer.center2 = nn.Parameter(self.input_to_memf_layer.center2)
                self.input_to_memf_layer.spread = nn.Parameter(self.input_to_memf_layer.spread)
                self.rule_to_typereduced_layer.con_low_param = nn.Parameter(self.rule_to_typereduced_layer.con_low_param)
                self.rule_to_typereduced_layer.con_upper_param = nn.Parameter(
                    self.rule_to_typereduced_layer.con_upper_param)
            elif param == 'all':  # 默认，全部参数都要训练
                for each in self.parameters():
                    each.requires_grad = True
            else:
                raise ValueError(f"Invalid value for tra_param: '{tra_param}'")

    def reinit_ant_IT2FCM(self, x, max_iter=5000, m1=1.5, m2=2, error=1e-6, metric='euclidean', init=None, seed=None):
        """
        IT2FCM聚类算法 用于初始化模糊规则库(仅为不确定中心隶属度函数，规则前件部分)
        :param x: 输入数据 tensor (n, s)  n为样本个数，s为特征个数
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

        self.input_to_memf_layer.reinit_ant_IT2FCM(x, self.num_fuzzy, max_iter=max_iter, m1=m1, m2=m2, error=error,
                                                   metric=metric, init=init, seed=seed)

    def cor_interval(self):
        """
        校正区间左右端点值大小
        :return: 返回空
        """
        # 1、出现左端点大于右端点时，low = upper
        with torch.no_grad():
            # 校正前件
            condition1 = self.input_to_memf_layer.center1 > self.input_to_memf_layer.center2
            self.input_to_memf_layer.center1[condition1] = self.input_to_memf_layer.center2[condition1]
            self.input_to_memf_layer.center1[condition1] -= 0.01
            # 校正后件
            condition2 = self.rule_to_typereduced_layer.con_low_param > self.rule_to_typereduced_layer.con_upper_param
            self.rule_to_typereduced_layer.con_low_param[condition2] = self.rule_to_typereduced_layer.con_upper_param[
                condition2]
            self.rule_to_typereduced_layer.con_low_param[condition2] -= 0.01

        # 2、出现左端点大于右端点时，upper = low  upper += 0.01
        # with torch.no_grad():
        #     # 校正前件
        #     condition1 = self.input_to_memf_layer.center1 > self.input_to_memf_layer.center2
        #     self.input_to_memf_layer.center2[condition1] = self.input_to_memf_layer.center1[condition1]
        #     # self.input_to_memf_layer.center2[condition1] += 0.01
        #     # 校正后件
        #     condition2 = self.rule_to_typereduced_layer.con_low_param > self.rule_to_typereduced_layer.con_upper_param
        #     self.rule_to_typereduced_layer.con_upper_param[condition2] = self.rule_to_typereduced_layer.con_low_param[condition2]
        #     # self.rule_to_typereduced_layer.con_upper_param[condition2] += 0.01

        # 2、出现左端点大于右端点时，两者交换
        # with torch.no_grad():
        #     # 校正前件
        #     condition1 = self.input_to_memf_layer.center1 > self.input_to_memf_layer.center2
        #     temp1 = self.input_to_memf_layer.center1[condition1].clone()
        #     self.input_to_memf_layer.center1[condition1] = self.input_to_memf_layer.center2[condition1]
        #     self.input_to_memf_layer.center2[condition1] = temp1
        #     # 校正后件
        #     condition2 = self.rule_to_typereduced_layer.con_low_param > self.rule_to_typereduced_layer.con_upper_param
        #     temp2 = self.rule_to_typereduced_layer.con_low_param[condition2].clone()
        #     self.rule_to_typereduced_layer.con_low_param[condition2] = self.rule_to_typereduced_layer.con_upper_param[condition2]
        #     self.rule_to_typereduced_layer.con_upper_param[condition2] = temp2

    @property
    def con_norm(self):
        """
        计算后件规则参数的L2范数, 返回的tensor每一个数代表一条规则的L2范数
        :return:后件参数L2范数 con_param_norm tensor [num_fuzzy, ]
        """
        if self.order == "simpl_first" or self.order == "classi_first":
            con_param_norm = torch.sqrt((self.rule_to_typereduced_layer.con_low_param ** 2).sum(dim=(0, 2)) +
                                        (self.rule_to_typereduced_layer.con_upper_param ** 2).sum(dim=(0, 2)))

        else:
            con_param_norm = torch.sqrt((self.rule_to_typereduced_layer.con_low_param ** 2).sum(dim=0) +
                                        (self.rule_to_typereduced_layer.con_upper_param ** 2).sum(dim=0))

        return con_param_norm

    @property
    def ant_norm(self):
        """
        计算前件宽度参数的L2范数
        :return:前件宽度参数L2范数 ant_spread_norm tensor [input_dim, ]
        """

        # ant_spread_norm = torch.sqrt((self.input_to_memf_layer.spread ** 2).sum(dim=0))
        ant_spread_norm = self.input_to_memf_layer.spread.norm(p=2, dim=0)

        return ant_spread_norm

    def con_change(self, num):
        """
        修改后件参数
        :num:需要置为0的规则 int
        :return: 空
        """
        if self.order == "simpl_first" or self.order == "classi_first":
            self.rule_to_typereduced_layer.con_low_param.data[:, num, :] = 0
            self.rule_to_typereduced_layer.con_upper_param.data[:, num, :] = 0
        elif self.order == "zero":
            self.rule_to_typereduced_layer.con_low_param.data[:, num] = 0
            self.rule_to_typereduced_layer.con_upper_param.data[:, num] = 0

    def model_param(self):
        """
        保存模型参数
        :return: 参数字典
        """
        param_dic = {
            'init_center1': self.input_to_memf_layer.init_center1,
            'init_center2': self.input_to_memf_layer.init_center2,
            'init_spread': self.input_to_memf_layer.spread,
            'init_con_low_param': self.rule_to_typereduced_layer.init_con_low_param,
            'init_con_upper_param': self.rule_to_typereduced_layer.init_con_upper_param,
            'model_state_dict': self.state_dict(),  # 保存模型的状态字典
        }

        return param_dic

    def model_checkpoint(self, epoch, loss, optimizer):
        """
        保存断点信息
        :return: 参数字典
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        return checkpoint

    # def gl_fea_select(self, tau_fea):
    #     """
    #     根据前件宽度范数选择特征
    #     调整系统参数
    #     :param tau_fea: 特征选择阈值的参数
    #     :return: 特征选择后的特征索引
    #     """
    #     # 计算前件宽度范数  * 计算特征选择后的特征索引  (门限值 = tau*根号下向量个数)
    #     ant_sread_norm = self.ant_norm
    #     selected_fea_ind = ant_sread_norm.gt(tau_fea * torch.sqrt(torch.tensor(self.num_fuzzy)))\
    #         .nonzero().squeeze(1).tolist()
    #
    #     if len(selected_fea_ind) == 0:
    #         raise Exception("selected_fea_ind is null, all features are selected!")
    #
    #     return selected_fea_ind
    #
    # def gl_rule_extract(self, tau_rule):
    #     """
    #     根据后件范数提取规则
    #     调整系统参数
    #     :param tau_rule: 规则提取阈值的参数
    #     :return: 规则提取后的规则索引
    #     """
    #     # 计算后件规则范数 * 计算规则提取后的规则索引   (门限值 = tau*根号下向量个数)
    #     con_param_norm = self.con_norm
    #     if self.order == "simpl_first" or self.order == "classi_first":
    #         extracted_rule_ind = con_param_norm.gt(tau_rule * torch.sqrt(torch.tensor(2 * (self.input_layer_size + 1)
    #                                                * self.output_layer_size))).nonzero().squeeze(1).tolist()
    #
    #     else:
    #         extracted_rule_ind = con_param_norm.gt(tau_rule * torch.sqrt(torch.tensor(2))) \
    #             .nonzero().squeeze(1).tolist()
    #
    #     if len(extracted_rule_ind) == 0:
    #         raise Exception("extracted_rule_ind is null, all rules are extracted!")
    #
    #     return extracted_rule_ind

    def gl_fea_select(self, tau_fea, min_num_fea=None):
        """
        根据前件宽度范数选择特征
        调整系统参数
        :param tau_fea: 特征选择阈值的参数
        :param min_num_fea: 最少被选择特征数
        :return: 特征选择后的特征索引
        """
        min_num_fea = 2 if min_num_fea is None else min_num_fea

        # 计算前件宽度范数  * 计算特征选择后的特征索引
        ant_sread_norm = self.ant_norm

        selected_fea_ind = ant_sread_norm.gt(threshold_fun_2(ant_sread_norm, tau_fea))\
            .nonzero().squeeze(1).tolist()

        # 如果选择的特征数低于min_num_fea个，强制提取前min_num_fea个特征
        if len(selected_fea_ind) < min_num_fea:
            selected_fea_ind = ant_sread_norm.sort(descending=True).indices[:min_num_fea].tolist()
            # selected_fea_ind = sorted(ant_sread_norm.topk(min_num_fea).indices.tolist())

        return selected_fea_ind

    def gl_rule_extract(self, tau_rule, min_num_rule=None):
        """
        根据后件范数提取规则
        调整系统参数
        :param tau_rule: 规则提取阈值的参数
        :param min_num_rule: 最少提取的规则数
        :return: 规则提取后的规则索引
        """
        min_num_rule = 2 if min_num_rule is None else min_num_rule
        # 计算后件规则范数 * 计算规则提取后的规则索引
        con_param_norm = self.con_norm

        if self.order == "simpl_first" or self.order == "classi_first":
            extracted_rule_ind = con_param_norm.gt(threshold_fun_2(con_param_norm, tau_rule)).nonzero().squeeze(1).tolist()

        else:
            extracted_rule_ind = con_param_norm.gt(threshold_fun_2(con_param_norm, tau_rule)).nonzero().squeeze(1).tolist()

        # 如果提取的规则数低于min_num_rule个，强制提取前min_num_rule个规则
        if len(extracted_rule_ind) < min_num_rule:
            extracted_rule_ind = con_param_norm.sort(descending=True).indices[:min_num_rule].tolist()
            # extracted_rule_ind = sorted(con_param_norm.topk(min_num_rule).indices.tolist())

        return extracted_rule_ind

    # def gl_fea_select(self, tau_fea):
    #     """
    #     根据前件宽度范数选择特征
    #     调整系统参数
    #     :param tau_fea: 特征选择阈值的参数
    #     :return: 特征选择后的特征索引
    #     """
    #
    #     # 根据范数计算阈值 * 计算被选择特征的索引
    #     threshold = threshold_fun(self.ant_norm.min(), self.ant_norm.max(), tau_fea)
    #     fea_selected_ind = self.ant_norm.gt(threshold).nonzero().squeeze(1).tolist()
    #
    #     if len(fea_selected_ind) == 0:
    #         raise Exception("fea_selected_ind is null, all features are selected!")
    #
    #     return fea_selected_ind
    #
    # def gl_rule_extract(self, tau_rule):
    #     """
    #     根据后件范数提取规则
    #     调整系统参数
    #     :param tau_rule: 规则提取阈值的参数
    #     :return: 规则提取后的规则索引
    #     """
    #
    #     # 根据范数计算阈值 * 计算被选择特征的索引
    #     threshold = threshold_fun(self.con_norm.min(), self.con_norm.max(), tau_rule)
    #     extracted_rule_ind = self.con_norm.gt(threshold).nonzero().squeeze(1).tolist()
    #
    #     if len(extracted_rule_ind) == 0:
    #         raise Exception("extracted_rule_ind is null, all rules are extracted!")
    #
    #     return extracted_rule_ind

    def prune_structure(self, fea_selected: list = None, rule_extracted: list = None):
        """
        根据指定的特征或者规则，修剪网络结构
        :param fea_selected: 特征选择后的特征索引
        :param rule_extracted: 规则提取后的规则索引
        :return:
        """

        if fea_selected:  # 根据所选择的特征调整系统参数
            self.input_to_memf_layer.center1 = nn.Parameter(self.input_to_memf_layer.center1.data[:, fea_selected])
            self.input_to_memf_layer.center2 = nn.Parameter(self.input_to_memf_layer.center2.data[:, fea_selected])
            self.input_to_memf_layer.spread = nn.Parameter(self.input_to_memf_layer.spread.data[:, fea_selected])
            # self.memf_to_rule_layer.FRB = self.memf_to_rule_layer.FRB[:, fea_selected]  # 未重新生成FRB
            if self.order != "zero":
                self.rule_to_typereduced_layer.con_low_param = nn.Parameter(
                    self.rule_to_typereduced_layer.con_low_param.data[:, :, [0] + [i + 1 for i in fea_selected]])
                self.rule_to_typereduced_layer.con_upper_param = nn.Parameter(
                    self.rule_to_typereduced_layer.con_upper_param.data[:, :, [0] + [i + 1 for i in fea_selected]])

            self.input_to_memf_layer.input_layer_size = len(fea_selected)
            self.memf_to_rule_layer.input_layer_size = self.input_to_memf_layer.input_layer_size
            self.rule_to_typereduced_layer.input_layer_size = self.input_to_memf_layer.input_layer_size
            self.input_layer_size = self.input_to_memf_layer.input_layer_size
            self.memf_to_rule_layer.FRB = self.memf_to_rule_layer._init_frb(self.input_layer_size, self.rule_layer_size, frb_type=self.frb)  # 重新生成FRB

        if rule_extracted:  # 调整系统参数
            # # 该方法其实并未剪枝第一层 再剪枝后使用IT2FNN 模糊子集个数会出错
            # self.memf_to_rule_layer.FRB = self.memf_to_rule_layer.FRB[rule_extracted, :]
            # if self.order == "zero":
            #     self.rule_to_typereduced_layer.con_low_param = nn.Parameter(
            #         self.rule_to_typereduced_layer.con_low_param.data[:, rule_extracted])
            #     self.rule_to_typereduced_layer.con_upper_param = nn.Parameter(
            #         self.rule_to_typereduced_layer.con_upper_param.data[:, rule_extracted])
            # else:
            #     self.rule_to_typereduced_layer.con_low_param = nn.Parameter(
            #         self.rule_to_typereduced_layer.con_low_param.data[:, rule_extracted, :])
            #     self.rule_to_typereduced_layer.con_upper_param = nn.Parameter(
            #         self.rule_to_typereduced_layer.con_upper_param.data[:, rule_extracted, :])


            # 此方法剪枝第一层，但需要重生成FRB
            self.input_to_memf_layer.num_fuzzy = len(rule_extracted)
            self.rule_layer_size = self.input_to_memf_layer.num_fuzzy
            self.rule_to_typereduced_layer.rule_layer_size = self.input_to_memf_layer.num_fuzzy
            self.memf_to_rule_layer.rule_layer_size = self.input_to_memf_layer.num_fuzzy
            self.num_fuzzy = self.input_to_memf_layer.num_fuzzy

            self.input_to_memf_layer.center1 = nn.Parameter(self.input_to_memf_layer.center1.data[rule_extracted, :])
            self.input_to_memf_layer.center2 = nn.Parameter(self.input_to_memf_layer.center2.data[rule_extracted, :])
            self.input_to_memf_layer.spread = nn.Parameter(self.input_to_memf_layer.spread.data[rule_extracted, :])
            self.memf_to_rule_layer.FRB = self.memf_to_rule_layer._init_frb(self.input_layer_size, self.rule_layer_size, frb_type=self.frb)
            if self.order == "zero":
                self.rule_to_typereduced_layer.con_low_param = nn.Parameter(
                    self.rule_to_typereduced_layer.con_low_param.data[:, rule_extracted])
                self.rule_to_typereduced_layer.con_upper_param = nn.Parameter(
                    self.rule_to_typereduced_layer.con_upper_param.data[:, rule_extracted])
            else:
                self.rule_to_typereduced_layer.con_low_param = nn.Parameter(
                    self.rule_to_typereduced_layer.con_low_param.data[:, rule_extracted, :])
                self.rule_to_typereduced_layer.con_upper_param = nn.Parameter(
                    self.rule_to_typereduced_layer.con_upper_param.data[:, rule_extracted, :])

    def search_tau_fs_re(self, tau_fs_list, tau_re_list, tra_sam, tra_tar, task="classification", pre_sca=None):
        test_dict = {}
        for each_tau_fs in tau_fs_list:
            selected_fea_ind = self.gl_fea_select(each_tau_fs, min_num_fea=2)
            for each_tau_re in tau_re_list:
                extracted_rule_ind = self.gl_rule_extract(each_tau_re, min_num_rule=2)

                self_copy = copy.deepcopy(self)
                self_copy.prune_structure(selected_fea_ind, extracted_rule_ind)

                if task == "classification":
                    _, _, metric = test(self_copy, tra_sam[:, selected_fea_ind], tra_tar, task=task)
                if task == "regression":
                    _, _, test_rmse = test(self_copy, tra_sam[:, selected_fea_ind], tra_tar, task="regression",
                                           pre_sca=pre_sca)
                    metric = -test_rmse

                test_dict[(each_tau_fs, each_tau_re)] = metric
        tau_fs, tau_re = max(test_dict, key=test_dict.get)

        return tau_fs, tau_re
