"""
Created by 陈辰柄 
datetime:2020/2/28 0:38
Describe:
"""

"""
Created by 陈辰柄 
datetime:2020/2/22 3:52
Describe:使用mnist数据集训练
"""

import numpy as np
from dataset.mnist import load_mnist
from chapter_five.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                  one_hot_label=True)
train_loss_list = []
train_acc_list = []
test_acc_list = []
train_size = t_train.shape[0]

# 超参数
iters_num = 10000
batch_size = 100
learning_rate = 0.1

# epoch是一个单位，一个epoch表示学习中所有训练数据均被使用过一次时的更新次数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 每个epoch计算识别精度
    if i % iter_per_epoch == 0:
        # 用整个数据集预测精度
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
