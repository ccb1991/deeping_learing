"""
Created by 陈辰柄 
datetime:2020/3/1 23:24
Describe:
"""
import numpy as np


class Momentum:
    """采用动量的方法更高效的求权重"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        """更新参数"""
        if self.v is None: # 如果没有初始速度，则初始化速度
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            # 当某一方向梯度变化非常大时，self.v[key]对应的值变化越大，params更新的速度就会越快。
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """学习率衰减方法更新权重参数"""
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None # 变量保存所有梯度值的平台和

    def update(self,params,grads):
        """通过学习率衰减更新参数"""
        if self.h is None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)

        for key in params.keys():
            self.h[key]+=grads[key]*grads[key]
            # 通过乘以1/根号h就可以调整学习的尺度，意味着，参数的元素中变动较大的元素的学习率将变小
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)
