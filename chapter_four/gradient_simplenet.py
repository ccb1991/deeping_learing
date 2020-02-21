"""
Created by 陈辰柄 
datetime:2020/2/22 3:00
Describe: 神经网络的梯度
"""

import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 用高斯分布进行初始化权重参数

    def predict(self, x):
        """神经网络输出，即预测值"""
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)
