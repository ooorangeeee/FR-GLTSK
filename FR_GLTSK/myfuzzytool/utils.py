# -*- coding: UTF-8 -*-
"""
@Author ：L.Gao
@Date ：2024/9/8 21:12
"""

import torch
from numpy import sqrt
from torch.optim.optimizer import Optimizer
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score, accuracy_score


def myRMSEloss(y, output):
    """
    系统输出y与实际输出各元素对应相减，然后平方，求和，除以样本数，开根号
    :param y: 系统输出 [num_sample, output_layer_size]
    :param output: 实际输出[num_sample, output_layer_size]
    :return: loss损失 标量
    """
    sample_num = y.shape[0]
    loss = torch.sqrt((1. / sample_num) * torch.sum((y - output) ** 2))
    return loss


def myMSEloss(y, output):
    """
    系统输出y与实际输出各元素对应相减，然后平方，求和，乘 0.5（一般是除以样本数，但是这样使得loss变得很小，在FNN效果不好）
    :param y: 系统输出 [num_sample, output_layer_size]
    :param output: 实际输出[num_sample, output_layer_size]
    :return: loss损失 标量
    """
    sample_num = y.shape[0]
    loss = 0.5 * torch.sum((y - output) ** 2)
    return loss


def myMSEloss_GL(y, output, model, belta, coef_ant_lambda, coef_con_lambda):
    """
    基于Group Lasso的损失函数
    系统输出y与实际输出各元素对应相减，然后平方，求和，乘 0.5 + 正则项
    :param y: 系统输出 [num_sample, output_layer_size]
    :param output: 实际输出[num_sample, output_layer_size]
    :param model:模型
    :param belta:光滑逼近Group Lasso 0点处函数的系数
    :param coef_ant_lambda:前件正则化系数
    :param coef_con_lambda:后件正则化系数
    :return: loss损失 标量
    """

    loss_mymse = torch.sum((y - output) ** 2) / (2*y.size(0))
    loss_ant, loss_con = 0, 0

    if model.gl_rule_extr:
        loss_con = coef_con_lambda * (torch.where(model.con_norm > torch.tensor(belta),
                                                  model.con_norm,
                                                  model.con_norm ** 2 / (2 * belta) + belta / 2)
                                      .sum())

    if model.gl_fea_sel:
        loss_ant = coef_ant_lambda * (torch.where(model.ant_norm > torch.tensor(belta),
                                                  model.ant_norm,
                                                  (model.ant_norm ** 2) / (2 * belta) + belta / 2)
                                      .sum())

    loss = loss_mymse + loss_con + loss_ant

    # loss = 0.5 * torch.sum((y - output) ** 2) + coef_lambda * torch.sqrt(model.rule_to_typereduced_layer.con_low_param ** 2 +
    # model.rule_to_typereduced_layer.con_upper_param ** 2).sum()

    return loss


# def myMSEloss_GL(y, output, model, coef_ant_lambda=0.5, coef_con_lambda=0):
#     """
#     基于Group Lasso的损失函数
#     系统输出y与实际输出各元素对应相减，然后平方，求和，乘 0.5 + 正则项
#     :param y: 系统输出 [num_sample, output_layer_size]
#     :param output: 实际输出[num_sample, output_layer_size]
#     :param model:模型
#     :param coef_ant_lambda:前件正则化系数
#     :param coef_con_lambda:后件正则化系数
#     :return: loss损失 标量
#     """
#     sample_num = y.shape[0]
#     if model.order == "simpl_first" or model.order == "classi_first":
#         loss = 0.5 * torch.sum((y - output) ** 2) + \
#                coef_con_lambda * (torch.sqrt((model.rule_to_typereduced_layer.con_low_param ** 2).
#                                              sum(dim=(0, 2)) +
#                                              (model.rule_to_typereduced_layer.con_upper_param ** 2).
#                                              sum(dim=(0, 2)))).sum() + \
#                coef_ant_lambda * model.input_to_memf_layer.spread.norm(p=2, dim=0).sum()
#
#     elif model.order == "zero":
#         loss = 0.5 * torch.sum((y - output) ** 2) + \
#                coef_con_lambda * (torch.sqrt((model.rule_to_typereduced_layer.con_low_param ** 2).
#                                              sum(dim=0) +
#                                              (model.rule_to_typereduced_layer.con_upper_param ** 2).
#                                              sum(dim=0))).sum() + \
#                coef_ant_lambda * model.input_to_memf_layer.spread.norm(p=2, dim=0).sum()
#
#     # loss = 0.5 * torch.sum((y - output) ** 2) + coef_lambda * torch.sqrt(model.rule_to_typereduced_layer.con_low_param ** 2 +
#     # model.rule_to_typereduced_layer.con_upper_param ** 2).sum()
#
#     return loss

def myCrossEntropyLoss_GL(y, output, model, belta, coef_ant_lambda, coef_con_lambda):
    """
    基于Group Lasso的损失函数
    在CrossEntropyLoss(softmax+log+nll_loss)后面添加正则项
    :param y: 系统输出 [num_sample, output_layer_size]
    :param output: 实际输出[num_sample, ]
    :param model:模型
    :param belta:光滑逼近Group Lasso 0点处函数的系数
    :param coef_ant_lambda:前件正则化系数
    :param coef_con_lambda:后件正则化系数
    :return: loss损失 标量
    """
    criterion = CrossEntropyLoss(reduction='mean')  # 实例化 CrossEntropyLoss()是一个类
    loss_cross = criterion(y, output)
    loss_ant, loss_con = 0, 0

    if model.gl_rule_extr:
        loss_con = coef_con_lambda * (torch.where(model.con_norm > torch.tensor(belta),
                                                  model.con_norm,
                                                  model.con_norm ** 2 / (2 * belta) + belta / 2)
                                      .sum())

    if model.gl_fea_sel:
        loss_ant = coef_ant_lambda * (torch.where(model.ant_norm > torch.tensor(belta),
                                                  model.ant_norm,
                                                  (model.ant_norm ** 2) / (2 * belta) + belta / 2)
                                      .sum())

    loss = loss_cross + loss_con + loss_ant

    return loss

# def myCrossEntropyLoss_GL(y, output, model, coef_ant_lambda=0.5, coef_con_lambda=0):
#     """
#     基于Group Lasso的损失函数
#     在CrossEntropyLoss(softmax+log+nll_loss)后面添加正则项
#     :param y: 系统输出 [num_sample, output_layer_size]
#     :param output: 实际输出[num_sample, ]
#     :param model:模型
#     :param coef_ant_lambda:前件正则化系数
#     :param coef_con_lambda:后件正则化系数
#     :return: loss损失 标量
#     """
#     loss_1 = CrossEntropyLoss()  # 实例化 CrossEntropyLoss()是一个类
#     if model.order == "simpl_first" or model.order == "classi_first":
#         loss = loss_1(y, output) + \
#                coef_con_lambda * (torch.sqrt((model.rule_to_typereduced_layer.con_low_param ** 2).
#                                              sum(dim=(0, 2)) +
#                                              (model.rule_to_typereduced_layer.con_upper_param ** 2).
#                                              sum(dim=(0, 2)))).sum() + \
#                coef_ant_lambda * model.input_to_memf_layer.spread.norm(p=2, dim=0).sum()
#     elif model.order == "zero":
#         loss = loss_1(y, output) + \
#                coef_con_lambda * (torch.sqrt((model.rule_to_typereduced_layer.con_low_param ** 2).
#                                              sum(dim=0) +
#                                              (model.rule_to_typereduced_layer.con_upper_param ** 2).
#                                              sum(dim=0))).sum() + \
#                coef_ant_lambda * model.input_to_memf_layer.spread.norm(p=2, dim=0).sum()
#
#     return loss


def eval_classification(model_output, target_output):
    """
    单标签多分类问题指标计算
    :param model_output:模型预测结果 tensor (num_sample, )
    :param target_output: 模型真实标签 tensor (num_sample, )
    :return:混淆矩阵，分类报告
    """

    # 数据转换成numpy才能调用sklearn
    model_output = model_output.detach().numpy()
    target_output = target_output.detach().numpy()

    conf_matrix = confusion_matrix(target_output, model_output)
    report = classification_report(target_output, model_output)
    acc = accuracy_score(target_output, model_output)

    return conf_matrix, report, acc


# def eval_regression(model_output, target_output):
#     """
#     回归问题指标计算
#     :param model_output:模型预测结果 tensor (num_sample, )
#     :param target_output:模型真实输出 tensor (num_sample, )
#     :return:分类报告
#     """
#     # 数据转换成numpy才能调用sklearn
#     model_output = model_output.detach().numpy()
#     target_output = target_output.detach().numpy()
#
#     mse = mean_squared_error(target_output, model_output)
#     mae = mean_absolute_error(target_output, model_output)
#     rmse = sqrt(mse)
#     r2 = r2_score(target_output, model_output)
#
#     report = f" mse: {mse:.4f}\n mae: {mae:.4f}\n rmse: {rmse:.4f}\n r2: {r2:.4f}"
#     return report, rmse


# def modif_sigmoid(ant_norm, t=10, threshold=0.01):
#     """
#     特征选择时，使用改进的sigmoid函数消除后件影响
#     :param ant_norm: 规则前件宽度范数
#     :param t: int 常数>10，控制斜率，越大逼近效果越好
#     :param threshold: float 门限值，大于等于门限值，函数趋于1，小于门限值，函数趋于0
#     :return: 函数 tensor [input_dim, ]
#     """
#     return 1/(1 + torch.exp((-t)*(ant_norm - threshold)))

def eval_regression(model_output, target_output):
    """
    回归问题指标计算
    :param model_output:模型预测结果 tensor (num_sample, 1)
    :param target_output:模型真实输出 tensor (num_sample, 1)
    :return:分类报告
    """

    mse = torch.mean((model_output - target_output) ** 2)
    mae = torch.mean((model_output - target_output).abs())
    rmse = sqrt(mse)
    r2 = 1 - ((torch.sum((model_output - target_output) ** 2)) / (torch.sum((model_output - model_output.mean())**2)))

    report = f" mse: {mse:.4f}\n mae: {mae:.4f}\n rmse: {rmse:.4f}\n r2: {r2:.4f}"
    return report, rmse


def gate_1(ant_norm, n):
    """
    门函数
    $1 - e^{-(1/N)x^2}$
    :param ant_norm: 规则前件宽度范数
    :param n: 规则数
    :return: 函数 tensor [input_dim, ]
    """
    return 1-torch.exp((-1/n) * ant_norm**2)



def piecewise_func(ant_norm, threshold):
    """
    特征选择时，使用分段函数消除后件影响
    :param ant_norm: 规则前件宽度范数
    :param threshold: 门限值，大于等于门限值，函数趋于1，小于门限值，函数趋于0
    :return: 函数 tensor [input_dim, ]
    """
    return ant_norm >= threshold




class MyGD(Optimizer):
    """
    梯度下降优化器
    """

    def __init__(self, params, lr):
        """
        初始化参数
        :param params: 需要求梯度的参数
        :param lr: 学习率
        """
        defaults = dict(lr=lr)
        super(MyGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        更新梯度
        :param closure:闭包
        :return: 空
        """
        with torch.no_grad():
            # 更新参数时禁用梯度计算
            for group in self.param_groups:
                # param_groups: List[Dict[str, Any]] = [] Any一般也为列表 存放需要求导的参数
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    lr = group["lr"]
                    param -= lr * param.grad


def normalize_columns(columns):
    """
    归一化矩阵的行
    :param columns: 输入张量 tensor (m,n)
    :return: 返回归一化后张量 tensor (m,n)
    """
    normalized_columns = columns / torch.sum(columns, dim=1, keepdim=True)

    return normalized_columns


def normalize_power_columns(x, exponent):
    """
    归一化矩阵的行 (x**exponent) 更安全的形式
    :param x: 输入张量 tensor (m,n)
    :param exponent: 指数 float
    :return: 返回归一化后张量 tensor (m,n)
    """
    assert torch.all(x >= 0.0)

    x = x.to(torch.float64)

    # values in range [0, 1]
    x = x / torch.amax(x, dim=1, keepdim=True)

    # values in range [eps, 1]
    x = torch.fmax(x, torch.tensor(torch.finfo(x.dtype).eps))

    if exponent < 0:
        # values in range [1, 1/eps]
        x /= torch.amin(x, dim=1, keepdim=True)

        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        x = x ** exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        x = x ** exponent

    result = normalize_columns(x)

    return result


def threshold_fun_1(minimum, maximum, zeta):
    """
    计算阈值，介于最大值与最小值之间
    :param minimum: 最小值
    :param maximum: 最大值
    :param zeta: 参数
    :return:
    """
    return maximum - zeta * (maximum - minimum)


def threshold_fun_2(norm, tau):
    """
    计算阈值，Q3 - tau * IQR   数据维度很高时不太好用（Q3 和 Q1都落在 0处）
    在高维数据中Q3取1  在地位数据中Q3取0.75
    :param norm: 范数
    :param tau: 参数
    :return:
    """
    q1 = torch.quantile(norm, 0.25)
    q3 = torch.quantile(norm, 1)

    iqr = (q3 - q1).abs()

    return q3 - tau * iqr


def cali_inteval(low, upper):
    """
    校正区间
    :param low:
    :param upper:
    :return:
    """
    condition = low > upper
    low.data[condition] = upper.data[condition]
