"""
Created by 陈辰柄 
datetime:2020/2/23 14:18
Describe: 激活函数层实现
"""
import numpy as np
from common.functions import softmax, cross_entropy_error


class Relu:
    """ReLu激活函数"""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0  # 会将数据索引为true的字段变为0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    """Sigmoid激活函数层"""

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class Affine:
    """仿射层"""

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = self.dw = self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    """Softmax损失函数与交叉熵"""

    def __init__(self):
        self.loss = None
        self.y = self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        return cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx


if __name__ == '__main__':
    x = np.array([2, -1, -3])
    relu = Relu()
    y = relu.forward(x)
