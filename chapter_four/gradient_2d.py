"""
Created by 陈辰柄 
datetime:2020/2/20 11:37
Describe: 导数、偏导数、梯度
"""

import numpy as np


def numerical_diff(f, x):
    """数值微分"""
    h = 1e-4
    return (f(x + h) - (f(x - h))) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return np.sum(x ** 2)


def numerical_gradient(f, x: np.array):
    """求梯度"""
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    梯度下降
    :param f:需要进行优化的函数
    :param init_x: 初始值
    :param lr: 学习率
    :param step_num: 梯度法的重复次数
    :return:
    """
    x = init_x

    # 沿梯度方向向最小值靠近step_num次
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


if __name__ == '__main__':
    print(numerical_gradient(function_2, np.array([-3.0, 4.0])))

    print(gradient_descent(function_2, np.array([-3.0, 4.0]), lr=10.0))
    print(gradient_descent(function_2, np.array([-3.0, 4.0])))
