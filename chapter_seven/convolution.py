"""
Created by 陈辰柄 
datetime:2020/3/9 0:20
Describe: 卷积层实现
"""
import pickle
from collections import OrderedDict

from common.functions import softmax
from common.layers import Relu, Affine, SoftmaxWithLoss
from common.util import im2col, col2im
import numpy as np


class Convolution:
    """卷积层"""

    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape  # FN、C、FH、FW 分别是FilterNumber（滤波器数量）
        # 、Channel、Filter Height、Filter Width 的缩写。
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        self.col = im2col(x, FH, FW, self.stride, self.pad)
        self.col_W = self.W.reshape(FN, -1).T  # 滤波器的展开 .T方法为矩阵的转置操作
        out = np.dot(self.col, self.col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.x = x
        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    """池化层"""

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        """
        初始化方法
        :param pool_h: 池高
        :param pool_w: 池宽
        :param stride: 步幅
        :param pad: 填充
        """
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        # 计算输出矩阵的大小
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        self.arg_max = np.argmax(col, axis=1)
        # 求最大值
        out = np.max(col, axis=1)

        # 转换
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = x
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


class SimpleConvNet:
    """CNN"""

    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num': 30, "filter_size": 5,
                                                          'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        """

        :param input_dim:输入数据的维度：（通道，高，长）
        :param conv_param:卷积层的超参数（字典）。字典的关键字如下：
                            filter_num―滤波器的数量
                            filter_size―滤波器的大小
                            stride―步幅
                            pad―填充
        :param hidden_size:隐藏层（全连接）的神经元数量
        :param output_size:输出层（全连接）的神经元数量
        :param weight_init_std:初始化时权重的标准差
        """
        filter_num = conv_param['filter_num']  # 滤波器数量
        filter_size = conv_param['filter_size']  # 滤波器大小
        filter_pad = conv_param['pad']  # 滤波器填充
        filter_stride = conv_param['stride']  # 滤波器步幅
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / \
                           filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) *
                               (conv_output_size / 2))

        # 权重参数初始化
        self.params = {'W1': weight_init_std * np.random.randn(filter_num, input_dim[0],
                                                               filter_size, filter_size),  # 卷积层权重
                       'b1': np.zeros(filter_num),  # 卷积层偏置
                       'W2': weight_init_std * np.random.randn(pool_output_size,
                                                               hidden_size),
                       'b2': np.zeros(hidden_size),
                       'W3': weight_init_std * np.random.randn(hidden_size, output_size),
                       'b3': np.zeros(output_size)}

        # 生成必要的层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """预测"""
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """损失函数"""
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        """方向传播发求梯度"""
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {'W1': self.layers['Conv1'].dW, 'b1': self.layers['Conv1'].db,
                 'W2': self.layers['Affine1'].dW, 'b2': self.layers['Affine1'].db,
                 'W3': self.layers['Affine2'].dW, 'b3': self.layers['Affine2'].db}
        return grads

    def save_params(self, file_name="params.pkl"):
        """保存权重参数"""
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """读取权重参数"""
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]

    def accuracy(self, x, t, batch_size=100):
        """计算精度"""
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]
