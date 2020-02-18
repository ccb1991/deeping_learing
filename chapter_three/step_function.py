"""
Created by 陈辰柄 
datetime:2020/2/14 13:42
Describe: 阶跃函数
"""

import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    """阶跃函数"""
    # dtype参数可以作为将np数组中的数据转换为相应的类型
    return np.array(x > 0, dtype=np.int)


def plt_show(x, y):
    """绘图函数"""
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def sigmoid(x):
    """sigmoid函数"""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """relu函数"""
    # 0 与x中的元素比较取大值，返回np数组
    return np.maximum(0, x)


def identity_function(x):
    """恒等函数"""
    return x


def softmax(a):
    """softmax函数"""
    max_c=np.max(a)
    exp_a = np.exp(a-max_c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def init_network():
    """初始化神经网络"""
    network = {}
    network["w1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["w2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["w3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    """向前传播"""
    a1 = sigmoid(np.dot(x, network['w1']) + network['b1'])
    a2 = sigmoid(np.dot(a1, network['w2']) + network['b2'])
    a3 = np.dot(a2, network['w3']) + network['b3']
    return identity_function(a3)


if __name__ == '__main__':
    # # 绘制阶跃函数
    # x = np.arange(-5.0, 5.0, 0.1)
    # y = step_function(x)
    # plt_show(x, y)
    #
    # # 绘制sigmoid函数
    # y = sigmoid(x)
    # plt_show(x, y)

    # 三层神经网络
    x = np.array([1.0, 0.5])
    netword = init_network()
    y = forward(netword, x)
    print(y)

    #softmax函数
    a=np.array([0.3,2.9,4.0])
    print(softmax(a))
