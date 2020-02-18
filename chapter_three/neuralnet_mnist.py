"""
Created by 陈辰柄 
datetime:2020/2/17 0:09
Describe:
"""
import pickle
import numpy as np
from step_function import sigmoid, softmax

from mnist import load_mnist


def get_data():
    """从数据集中获取数据"""
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """初始化神经网络"""
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    return softmax(a3)


if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    batch_size = 100  # 批数量

    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = predict(network, x_batch)  # 输入图像将图像从0~9分类
        # 从输出结果中获取np数组中最大的值,axis=1表示从行方向找最大值
        p = np.argmax(y_batch, axis=1)
        # 如果预测的最高概率与测试数据集的标签相同则
        # p==t[i:i+batch_size]运算会得到包含true false的np数组，false为0，true为1
        accuracy_cnt += np.sum(p == t[i:i + batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
