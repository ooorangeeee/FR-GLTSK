# -*- coding: UTF-8 -*-
"""
@Author ：L.Gao
@Date ：2024/9/8 21:12
"""

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from .utils import myRMSEloss, myMSEloss, MyGD, myMSEloss_GL, myCrossEntropyLoss_GL, eval_classification, eval_regression
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter


def train_full_batch(model, model_input, target_output, learning_rate, epoch, optim_type='SGD',
                     lr_sched=False, gpu=True, task='regression', visual_path=None,
                     belta=0.05, coef_ant_lambda=0, coef_con_lambda=0):
    """
    训练模型
    :param model: 实例化的模型
    :param model_input: 模型输入
    :param target_output: 模型理想输出
    :param learning_rate: 学习率
    :param epoch: 整个训练集迭代次数
    :param optim_type: 取值{'SGD' (default), 'Adam'}，优化器的选择
    :param lr_sched: 是否使用学习率更新策略， 取值{Ture, False(default)}
    :param gpu: 是否使用gpu，取值{Ture, False(default)}
    :param task: 分类任务还是回归任务，取值{classification,regression(default)}
    :param visual_path: 使用tensorboard可视化路径 None(default)
    :param belta: :param belta:光滑逼近Group Lasso 0点处函数的系数 0.05(default)
    :param coef_ant_lambda: 使用GL进行特征选择时，惩罚项系数  0(default)
    :param coef_con_lambda: 使用GL进行规则提取时，惩罚项系数  0(default)
    :return: 训练好的模型
    """

    # # 训练模式,打开 dropout 层(其实没有设置dropout)
    # model.train()

    if gpu:  # 如果使用GPU加速
        device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型和数据推上GPU
        model.to(device_gpu)
        model_input, target_output = model_input.to(device_gpu), target_output.to(device_gpu)

    # 定义损失函数和优化器
    if task == "regression":
        criterion = myMSEloss_GL
        # criterion = MSELoss()
    elif task == "classification":
        criterion = myCrossEntropyLoss_GL  # target必须从0开始编号
        # criterion = myMSEloss_GL  # hardmax
    else:
        raise ValueError(f"Invalid value for task: {task}, expected 'classification', 'regression'")

    if optim_type == 'MyGD':
        optimizer = MyGD(model.parameters(), lr=learning_rate)
    elif optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid value for optim_type: '{optim_type}'")

    if lr_sched:
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 判断是否需要可视化
    # if visual_path is not None:
    #     writer = SummaryWriter(visual_path)
    #     writer.add_graph(model, model_input)

    # torch.autograd.set_detect_anomaly(True)

    # 迭代更新
    loss_his = torch.zeros(epoch)  # 准备记录训练过程中的loss
    con_norm_his = torch.zeros((epoch+1, model.num_fuzzy))  # 准备记录训练过程中的后件范数(包括初始范数)
    ant_norm_his = torch.zeros((epoch+1, model.input_layer_size))  # 准备记录训练过程中的前件宽度范数(包括初始范数)
    con_norm_his[0], ant_norm_his[0] = model.con_norm, model.ant_norm
    model_output = None
    loss = None
    for i in range(epoch):
        # 正向传播
        model_output = model(model_input)

        # 计算损失
        if task == "regression":
            loss = criterion(model_output, target_output, model, belta=belta, coef_ant_lambda=coef_ant_lambda, coef_con_lambda=coef_con_lambda)
            loss_his[i] = loss
            # 可视化
            # if visual_path is not None:
            #     writer.add_scalar("Loss", loss.item(), i)

            # 输出
            if (i + 1) % 5 == 0:
                print('第{}次迭代，训练集loss：{:.4f}'.format(i + 1, loss.data))
            # print('第{}次迭代，训练集loss：{:.4f}'.format(i, loss.data))

        elif task == "classification":
            loss = criterion(model_output, target_output, model, belta=belta, coef_ant_lambda=coef_ant_lambda, coef_con_lambda=coef_con_lambda)
            loss_his[i] = loss
            acc = (target_output == model_output.argmax(dim=1)).sum().float() / target_output.shape[0]  # softmax target_output要从0开始编号

            # 可视化
            # if visual_path is not None:
            #     writer.add_scalar("Loss", loss.item(), i)
            #     writer.add_scalar("Accuracy", acc.item(), i)

            # 输出
            if (i + 1) % 5 == 0:
                # print('第{}次迭代，训练集loss：{:.4f}'.format(i + 1, loss.data))
                print('第{}次迭代，训练集loss：{:.4f}， accuracy：{:.4f}'.format(i + 1, loss.data, acc))
            # print('第{}次迭代，训练集loss：{:.4f}， accuracy：{:.4f}'.format(i, loss.data, acc))

        # 梯度清0
        optimizer.zero_grad()

        # 反向传播，计算梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        if lr_sched:
            # 更新学习率
            lr_scheduler.step()

        # 校正区间
        model.cor_interval()

        # 记录后件规则范数
        con_norm_his[i+1] = model.con_norm
        ant_norm_his[i+1] = model.ant_norm

        # 可视化
        # if visual_path is not None:
        #     writer.add_scalars("con_norm", {f"r{j}": con_norm_his[i][j].item() for j in range(model.num_fuzzy)}, i)
        # 可视化，每个epoch，记录梯度，权值
        # if visual_path is not None:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad and param.grad is not None:
        #             writer.add_histogram(name + '_grad', param.grad, i)
        #             writer.add_histogram(name + '_data', param, i)


    if gpu:  # 模型在gpu上训练后，拉回cpu上
        device_cpu = torch.device("cpu")
        model.to(device_cpu)
        model_output = model_output.to(device_cpu)

    # 关闭tensorboard
    # if visual_path is not None:
    #     writer.close()

    return loss_his, con_norm_his, ant_norm_his, model_output


def train(model, model_input, target_output, batch_size, learning_rate, epoch, optim_type='SGD', gpu=False, task="regression",
          belta=0.05, coef_ant_lambda=0, coef_con_lambda=0):
    """
    训练模型
    :param model: 实例化的模型
    :param model_input: 模型输入
    :param target_output: 模型理想输出
    :param batch_size: 每个batch的尺寸
    :param learning_rate: 学习率
    :param epoch: 1个epoch是多个iteration，1个iteration是1个batch_size的迭代
    :param optim_type: 取值{"MyGD", 'SGD' (default), 'Adam'}，优化器的选择
    :param gpu: 是否使用gpu，取值{Ture, False(default)}
    :param task: 分类任务还是回归任务，取值{classification,regression(default)}
    :param belta: :param belta:光滑逼近Group Lasso 0点处函数的系数 0.05(default)
    :param coef_ant_lambda: 使用GL进行特征选择时，惩罚项系数  0(default)
    :param coef_con_lambda: 使用GL进行规则提取时，惩罚项系数  0(default)
    :return: 训练好的模型
    """

    # # 训练模式,打开 dropout 层(其实没有设置dropout)
    # model.train()

    if gpu:  # 如果使用GPU加速
        device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型和数据推上GPU
        model.to(device_gpu)
        model_input, target_output = model_input.to(device_gpu), target_output.to(device_gpu)

    # 数据按照batch_size存放
    data_loader = DataLoader(
        dataset=TensorDataset(model_input, target_output),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )

    # 定义损失函数和优化器
    if task == "regression":
        criterion = myMSEloss_GL
        # criterion = MSELoss()
    elif task == "classification":
        criterion = myCrossEntropyLoss_GL  # target必须从0开始编号
        # criterion = myMSEloss_GL  # hardmax
    else:
        raise ValueError(f"Invalid value for task: {task}, expected 'classification', 'regression'")

    if optim_type == 'MyGD':
        optimizer = MyGD(model.parameters(), lr=learning_rate)
    elif optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid value for optim_type: '{optim_type}'")

    # 迭代更新
    loss_epoch = []  # 记录loss，一次epoch一个loss
    con_norm_his = torch.zeros((epoch + 1, model.num_fuzzy))  # 准备记录训练过程中的后件范数(包括初始范数)
    ant_norm_his = torch.zeros((epoch + 1, model.input_layer_size))  # 准备记录训练过程中的前件宽度范数(包括初始范数)
    con_norm_his[0], ant_norm_his[0] = model.con_norm, model.ant_norm
    for iteration in range(epoch):
        loss_iter = []  # 记录loss，一次iteration一个loss
        for model_input_batch, target_output_batch in data_loader:

            # 正向传播
            model_output_batch = model.forward(model_input_batch)

            # 计算损失
            loss = criterion(model_output_batch, target_output_batch, model, belta=belta, coef_ant_lambda=coef_ant_lambda, coef_con_lambda=coef_con_lambda)
            loss_iter.append(loss.data)

            # 梯度清0
            optimizer.zero_grad()

            # 反向传播，计算梯度
            loss.backward()

            # 更新参数
            optimizer.step()

            # 校正区间
            model.cor_interval()

        # 记录后件规则范数
        con_norm_his[iteration+1] = model.con_norm
        ant_norm_his[iteration+1] = model.ant_norm

        # 输出
        loss_epoch.append(torch.tensor(loss_iter).sum())
        print('第{}次epoch，训练集损失值为：{:.4f}'.format(iteration, loss_epoch[-1]))

    if gpu:  # 模型在gpu上训练后，拉回cpu上
        device_cpu = torch.device("cpu")
        model.to(device_cpu)

    return torch.tensor(loss_epoch), con_norm_his, ant_norm_his


def test(model, model_input, target_output, task='regression', pre_sca=None):
    """
    测试模型性能
    :param model_input: 模型输入
    :param model: 训练好的模型
    :param target_output: 模型输入的理想输出
    :param task: 分类任务还是回归任务，取值{classification,regression(default)}
    :param pre_sca: preprocessing_scaler，对理想输出的标准化器
            仅当 task='regression' 时指定这个参数
    :return: 模型性能指标：损失值（误差），分类精度
    """

    # # 评估模式,关闭 dropout 层(其实没有设置dropout)
    # model.eval()

    # 计算模型输出，在测试集上
    model_output = model(model_input)

    if task == 'classification':  # 分类
        # criterion = CrossEntropyLoss()
        # # criterion = myMSEloss  # hardmax
        #
        # loss = criterion(model_output, target_output)
        # accuracy = accuracy_score(target_output, model_output_argmax)  # softmax target_output要从0开始编号
        # accuracy = accuracy_score(target_output.argmax(dim=1), model_output.argmax(dim=1))  # hardmax

        model_output_argmax = model_output.argmax(dim=1)  # 模型输出是logits(原始输出，没有经过转换)，选取最大下标为预测结果
        conf_matrix, report, acc = eval_classification(model_output_argmax, target_output)
        # conf_matrix, report, acc = eval_classification(model_output_argmax, target_output.argmax(dim=1)) # hardmax
        print(f"测试集confusion matrix：\n{conf_matrix}")
        print(f"测试集classification report：\n{report}")


        return conf_matrix, report, acc

    elif task == 'regression':  # 回归
        # criterion = myMSEloss
        # # criterion = MSELoss()
        #
        # loss = criterion(model_output, target_output)
        # rmse = loss.sqrt()
        report, rmse = eval_regression(
                            torch.DoubleTensor(pre_sca.inverse_transform(model_output.detach().numpy())),
                            torch.DoubleTensor(target_output.detach().numpy()))

        print(f"测试集regression report: \n{report}")

        return model_output, report, rmse

    raise ValueError(f"Invalid value for task: {task}, expected 'classification', 'regression'")


def train_test(model, tra_input, tra_target, test_input, test_target, batch_size, learning_rate, epoch, optim_type='SGD', gpu=False, task="regression"):
    """
        训练模型
        :param model: 实例化的模型
        :param tra_input: 训练集输入
        :param tra_target: 训练集目标标签
        :param test_input: 测试集输入
        :param test_target: 测试集目标标签
        :param batch_size: 每个batch的尺寸
        :param learning_rate: 学习率
        :param epoch: 1个epoch是多个iteration，1个iteration是1个batch_size的迭代
        :param optim_type: 取值{"MyGD", 'SGD' (default), 'Adam'}，优化器的选择
        :param gpu: 是否使用gpu，取值{Ture, False(default)}
        :param task: 分类任务还是回归任务，取值{classification,regression(default)}
        :return: 训练好的模型
        """

    # # 训练模式,打开 dropout 层(其实没有设置dropout)
    # model.train()

    if gpu:  # 如果使用GPU加速
        device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型和数据推上GPU
        model.to(device_gpu)
        tra_input, tra_target = tra_input.to(device_gpu), tra_target.to(device_gpu)
        test_input, test_target = test_input.to(device_gpu), test_target.to(device_gpu)

    # 数据按照batch_size存放
    tra_dataloader = DataLoader(
        dataset=TensorDataset(tra_input, tra_target),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )
    test_dataloader = DataLoader(
        dataset=TensorDataset(test_input, test_target),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
    )


    # 定义损失函数和优化器
    if task == "regression":
        # criterion = myMSEloss_GL
        criterion = myMSEloss
    elif task == "classification":
        criterion = CrossEntropyLoss()
        # criterion = myCrossEntropyLoss_GL
        # criterion = myMSEloss_GL  # hardmax
    else:
        raise ValueError(f"Invalid value for task: {task}, expected 'classification', 'regression'")

    if optim_type == 'MyGD':
        optimizer = MyGD(model.parameters(), lr=learning_rate)
    elif optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid value for optim_type: '{optim_type}'")

    # 迭代更新
    loss_epoch = []  # 记录loss，一次epoch一个loss
    for iteration in range(epoch):
        print(f"------第{epoch}轮epoch------")
        loss_iter = []  # 记录loss，一次iteration一个loss
        for tra_input_batch, tra_target_batch in tra_dataloader:
            # 正向传播
            tra_output_batch = model.forward(tra_input_batch)

            # 计算损失
            loss = criterion(tra_output_batch, tra_target_batch)
            loss_iter.append(loss.data)

            # 梯度清0
            optimizer.zero_grad()

            # 反向传播，计算梯度
            loss.backward()

            # 更新参数
            optimizer.step()

            # 校正区间
            model.cor_interval()

        # 输出
        loss_epoch.append(torch.tensor(loss_iter).sum())
        print('第{}次epoch，训练集损失值为：{:.4f}'.format(epoch, loss_epoch[-1]))

        # 评估模式,关闭 dropout 层(其实没有设置dropout)
        # model.eval()
        with torch.no_grad():
            acc_iter = []
            loss_iter = []
            for test_input_batch, test_target_batch in test_dataloader:
                # 计算模型输出，在测试集上
                test_output_batch = model(test_input_batch)

                if task == 'classification':  # 如果是分类，则计算精度
                    acc_batch = (test_output_batch.argmax(1) == test_target_batch).sum()  # acc分子
                    acc_iter.append(acc_batch)

                elif task == 'regression':  # 如果是回归，仅计算损失
                    ex_loss = myRMSEloss(test_output_batch, test_target_batch)
                    loss_iter.append(ex_loss)

            if task == 'classification':  # 如果是分类，则计算精度
                acc = torch.tensor(acc_iter).sum() / test_target.shape[0] * 100
                print(f"第{epoch}次epoch，测试集准确率为：{acc:.2f}%")

            elif task == 'regression':  # 如果是回归，仅计算损失
                ex_loss = torch.tensor(loss_iter).sum()
                print(f'第{epoch}次epoch，测试集损失值为：{ex_loss:.4f}')

        torch.save(model.model_param(), fr"../result/_param{epoch}.pth")

    if gpu:  # 模型在gpu上训练后，拉回cpu上
        device_cpu = torch.device("cpu")
        model.to(device_cpu)
