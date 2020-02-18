"""
Created by 陈辰柄 
datetime:2020/2/18 23:40
Describe: 使用小型数据集训练
"""

import numpy as np


def mean_squared_error(y, t):
    """
    均方差损失函数
    :param y: 神经网络输出
    :param t: 监督数据
    :return:
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """
    交叉嫡损失函数
    :param y: 神经网络输出
    :param t: 监督数据
    :return:
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7))/batch_size
