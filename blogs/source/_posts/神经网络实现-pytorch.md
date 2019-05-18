---
title: 神经网络实现-pytorch
categories:
  - 编程
tags:
  - 机器学习
  - 深度学习
abbrlink: 5a77dbca
date: 2019-05-18 15:01:30
---

参考：

[神经网络实现-numpy](https://www.zhujian.tech/posts/ba2ca878.html#more)

[使用softmax回归进行mnist分类](https://www.zhujian.tech/posts/dd673751.html#more)

[PyTorch 进阶之路（四）：在 GPU 上训练深度神经网络](https://www.jiqizhixin.com/articles/2019-04-09-9)

使用`pytorch`实现`3`层神经网络模型`ThreeNet`

网络参数如下：

```
# 批量大小
batch_size = 256
# 输入维数
D = 784
# 隐藏层大小
H1 = 200
H2 = 60
# 输出类别
K = 10

# 学习率
learning_rate = 1e-3

# 迭代次数
epoches = 500
```

完整代码如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-5-18 下午3:03
# @Author  : zj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import copy
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# 批量大小
batch_size = 256
# 输入维数
D = 784
# 隐藏层大小
H1 = 200
H2 = 60
# 输出类别
K = 10

# 学习率
learning_rate = 1e-3

# 迭代次数
epoches = 500


def load_data(batch_size=128, shuffle=False):
    data_dir = '../data/'

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_data_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def compute_accuracy(module, dataLoader, device=torch.device('cpu')):
    """
    计算精度
    :param module: 计算模型
    :param dataLoader: 数据加载器

    """
    accuracy = 0
    for i, items in enumerate(dataLoader, 0):
        data, labels = items
        data = data.reshape((data.size()[0], -1))
        data, labels = data.to(device=device), labels.to(device=device)

        scores = module.forward(data)
        predictions = torch.argmax(scores, dim=1)
        res = (predictions == labels.squeeze())
        accuracy += 1.0 * torch.sum(res).item() / scores.size()[0]
    return accuracy / dataLoader.__len__()


def draw(loss_list, title='损失图', ylabel='损失值', xlabel='迭代/20次'):
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.plot(loss_list)
    plt.show()


class NNModule(nn.Module):

    def __init__(self):
        super(NNModule, self).__init__()
        self.fc1 = nn.Linear(D, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, K)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x))
        return x

    def __copy__(self, device):
        module = NNModule().to(device=device)
        module.fc1.weight = copy.deepcopy(self.fc1.weight)
        module.fc1.bias = copy.deepcopy(self.fc1.bias)

        module.fc2.weight = copy.deepcopy(self.fc2.weight)
        module.fc2.bias = copy.deepcopy(self.fc2.bias)

        module.fc3.weight = copy.deepcopy(self.fc3.weight)
        module.fc3.bias = copy.deepcopy(self.fc3.bias)

        return module


def compute_gradient_descent(batch_size=8, epoches=2000):
    train_loader, test_loader = load_data(batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # softmax模型
    module = NNModule().to(device=device)
    # 损失函数
    criterion = nn.NLLLoss().to(device=device)
    # 优化器
    optimizer = optim.SGD(module.parameters(), lr=learning_rate)

    loss_list = []
    accuracy_list = []
    bestA = 0
    bestModule = None

    batch_len = train_loader.__len__()
    for i in range(epoches):
        start = time.time()
        for j, items in enumerate(train_loader, 0):
            data, labels = items
            data = data.reshape((data.size()[0], -1))
            data, labels = data.to(device=device), labels.to(device=device)

            scores = module.forward(data)
            loss = criterion(scores, labels.squeeze())
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            if j == (batch_len - 1):
                loss_list.append(loss.item())
        end = time.time()
        print('epoch： %d time: %.2f s' % (i + 1, end - start))
        if i % 20 == 19:  # 每个20次进行一次检测
            start = time.time()
            accuracy = compute_accuracy(module, train_loader, device)
            accuracy_list.append(accuracy)
            if accuracy >= bestA:
                bestA = accuracy
                bestModule = module.__copy__(device)
            end = time.time()
            print('epoch: %d time: %.2f s accuracy: %.3f %%' % (i + 1, end - start, accuracy * 100))

    draw(loss_list, title='mnist', xlabel='迭代/次')
    draw(accuracy_list, title='训练精度', ylabel='检测精度', xlabel='迭代/20次')

    test_accuracy = compute_accuracy(bestModule, test_loader, device)

    print('best train accuracy is %.3f %%' % (bestA * 100))
    print('test accuracy is %.3f %%' % (test_accuracy * 100))


if __name__ == '__main__':
    start = time.time()
    compute_gradient_descent(batch_size=batch_size, epoches=epoches)
    end = time.time()
    print('all train and test need time: %.2f minutes' % ((end - start) / 60.0))
```

训练`500`次精度如下：

```
best train accuracy is 99.997 %
test accuracy is 97.959 %
all train and test need time: 71.90 minutes
```

![](/imgs/神经网络实现-pytorch/mnist_loss.png)

![](/imgs/神经网络实现-pytorch/mnist_accuracy.png)