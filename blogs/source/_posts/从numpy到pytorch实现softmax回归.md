---
title: 从numpy到pytorch实现softmax回归
categories:
  - 编程
tags:
  - 机器学习
  - 深度学习
abbrlink: 1c195604
date: 2019-04-28 11:13:16
---

使用`pytorch`实现`softmax`回归，首先使用基本数学运算函数实现，然后逐步使用各种封装函数和优化包进行替换

超参数如下：

* batch_size = 8
* lambda = 2e-4
* alpha = 2e-4

使用数据库

* [Iris Species](https://www.kaggle.com/uciml/iris)

## `numpy`实现

参考[softmax回归](https://www.zhujian.tech/posts/2626bec3.html#more)

## `pytorch`实现 - 基本数学运算函数

先利用`numpy`获取`iris`数据，再转换`为torch.Tensor`结构

```
    x_train, x_test, y_train, y_test, y_train_indicator = load_data()
    
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    y_train_indicator = torch.FloatTensor(y_train_indicator)
```

初始化权重，生成标准正态分布随机数组

```
def init_weights(inputs, outputs, requires_grad=False):
    """
    初始化权重
    使用torch.randn生成标准正态分布
    """
    w = 0.01 * torch.randn(inputs, outputs, requires_grad=requires_grad, dtype=torch.float)
    b = 0.01 * torch.randn(1, requires_grad=requires_grad, dtype=torch.float)
    return w, b
```

执行线性运算和`softmax`运算

```
def linear(x, w, b):
    """
    线性操作
    :param x: 大小为(m,n)
    :param w: 大小为(k,n)
    :return: 大小为(m,k)
    """
    return x.mm(w) + b


def softmax(x):
    """
    softmax归一化计算
    :param x: 大小为(m, k)
    :return: 大小为(m, k)
    """
    x -= torch.unsqueeze(torch.max(x, 1)[0], 1)
    exps = torch.exp(x)
    return exps / torch.unsqueeze(torch.sum(exps, dim=1), 1)
```

计算预测结果

```
def compute_scores(W, b, X):
    """
    计算精度
    :param X: 大小为(m,n)
    :param W: 大小为(k,n)
    :param b: 1
    :return: (m,k)
    """
    return softmax(linear(X, W, b))
```

计算损失值和梯度值

```
def compute_loss(scores, indicator, W, b, la=2e-4):
    """
    计算损失值
    :param scores: 大小为(m, n)
    :param indicator: 大小为(m, n)
    :param W: (n, k)
    :return: (m,1)
    """
    loss = -1 / scores.size()[0] * torch.sum(torch.log(scores) * indicator)
    reg = la / 2 * (torch.sum(W ** 2) + b ** 2)

    return (loss + reg).item()

def compute_gradient(indicator, scores, x, W, la=2e-4):
    """
    计算梯度
    :param indicator: 大小为(m,k)
    :param scores: 大小为(m,k)
    :param x: 大小为(m,n)
    :param W: (n, k)
    :return: (n,k)
    """
    dloss = -1 / scores.size()[0] * x.t().mm(torch.sub(indicator, scores))
    dreg = la * W
    return dloss + dreg
```

最后计算精度

```
def compute_accuracy(scores, Y):
    """
    计算精度
    :param scores: (m,k)
    :param Y: (m,1)
    """
    predictions = torch.argmax(scores, dim=1)
    res = (predictions == Y.squeeze())
    return 1.0 * torch.sum(res).item() / scores.size()[0]
```

完整代码如下

```
# -*- coding: utf-8 -*-

# @Time    : 19-4-27 下午3:05
# @Author  : zj

import torch
import numpy as np
from sklearn import utils
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = '../data/iris-species/Iris.csv'


def load_data(shuffle=True, tsize=0.8):
    """
    加载iris数据
    """
    data = pd.read_csv(data_path, header=0, delimiter=',')

    if shuffle:
        data = utils.shuffle(data)

    # 示性函数
    pd_indicator = pd.get_dummies(data['Species'])
    indicator = np.array(
        [pd_indicator['Iris-setosa'], pd_indicator['Iris-versicolor'], pd_indicator['Iris-virginica']]).T

    species_dict = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['Species'] = data['Species'].map(species_dict)

    data_x = np.array(
        [data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm']]).T
    data_y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    y_train = np.atleast_2d(y_train).T
    y_test = np.atleast_2d(y_test).T

    y_train_indicator = np.atleast_2d(indicator[:y_train.shape[0]])

    return torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float(), torch.from_numpy(
        y_train), torch.from_numpy(y_test), torch.from_numpy(y_train_indicator).float()


def linear(x, w):
    """
    线性操作
    :param x: 大小为(m,n+1)
    :param w: 大小为(n+1,k)
    :return: 大小为(m,k)
    """
    return x.mm(w)


def softmax(x):
    """
    softmax归一化计算
    :param x: 大小为(m, k)
    :return: 大小为(m, k)
    """
    x -= torch.unsqueeze(torch.max(x, 1)[0], 1)
    exps = torch.exp(x)
    return exps / torch.unsqueeze(torch.sum(exps, dim=1), 1)


def compute_scores(X, W):
    """
    计算精度
    :param X: 大小为(m,n)
    :param W: 大小为(k,n)
    :return: (m,k)
    """
    return softmax(linear(X, W))


def compute_loss(scores, indicator, W, la=2e-4):
    """
    计算损失值
    :param scores: 大小为(m, k)
    :param indicator: 大小为(m, k)
    :param W: (n+1, k)
    :return: (1)
    """
    loss = -1 / scores.size()[0] * torch.sum(torch.log(scores) * indicator)
    reg = la / 2 * torch.sum(W ** 2)

    return (loss + reg).item()


def compute_gradient(indicator, scores, x, W, la=2e-4):
    """
    计算梯度
    :param indicator: 大小为(m,k)
    :param scores: 大小为(m,k)
    :param x: 大小为(m,n+1)
    :param W: (n+1, k)
    :return: (n+1,k)
    """
    dloss = -1 / scores.size()[0] * x.t().mm(torch.sub(indicator, scores))
    dreg = la * W
    return dloss + dreg


def compute_accuracy(scores, Y):
    """
    计算精度
    :param scores: (m,k)
    :param Y: (m,1)
    """
    predictions = torch.argmax(scores, dim=1)
    res = (predictions == Y.squeeze())
    return 1.0 * torch.sum(res).item() / scores.size()[0]


def draw(res_list, title=None, xlabel=None):
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.plot(res_list)
    plt.show()


def compute_gradient_descent(batch_size=8, epoches=2000, alpha=2e-4):
    x_train, x_test, y_train, y_test, y_train_indicator = load_data()

    m, n = x_train.size()[:2]
    k = 3
    # print(m, n, k)

    W = 0.01 * torch.randn(n + 1, k, requires_grad=False, dtype=torch.float)
    # print(w)
    # 插入一列
    x_train = torch.from_numpy(np.insert(x_train.numpy(), 0, np.ones(m), axis=1))
    x_test = torch.from_numpy(np.insert(x_test.numpy(), 0, np.ones(x_test.size()[0]), axis=1))

    loss_list = []
    accuracy_list = []
    bestW = None
    bestA = 0
    range_list = list(range(0, m - batch_size, batch_size))
    for i in range(epoches):
        for j in range_list:
            data = x_train[j:j + batch_size]
            labels = y_train_indicator[j:j + batch_size]

            scores = compute_scores(data, W)
            tempW = W - alpha * compute_gradient(labels, scores, data, W)
            W = tempW

            if j == range_list[-1]:
                loss = compute_loss(scores, labels, W)
                loss_list.append(loss)

                accuracy = compute_accuracy(compute_scores(x_train, W), y_train)
                accuracy_list.append(accuracy)
                if accuracy >= bestA:
                    bestA = accuracy
                    bestW = W.clone()
                break

    draw(loss_list, title='损失值')
    draw(accuracy_list, title='训练精度')

    print(bestA)
    print(compute_accuracy(compute_scores(x_test, bestW), y_test))


if __name__ == '__main__':
    compute_gradient_descent(batch_size=8, epoches=100000)
```

测试结果：

```
# 测试集精度
0.975
# 验证集精度
1.0
```

![](/imgs/从numpy到pytorch实现softmax回归/pytorch_basic_softmax_loss.png)

![](/imgs/从numpy到pytorch实现softmax回归/pytorch_basic_softmax_accuracy.png)

## `pytorch`实现 - 使用`nn`包优化`softmax`回归模型和损失函数

`pytorch`在包`nn`中提供了大量算法和损失函数实现，并且能够自动计算梯度

使用线性模型和`softmax`回归模型

```
# softmax回归模型和权重
linearModel = nn.Linear(n, k)
softmaxModel = nn.LogSoftmax()
w = linearModel.weight
b = linearModel.bias

scores = softmaxModel.forward(linearModel.forward(data))
```

使用交叉熵损失类计算损失和计算梯度

```
# 损失函数
criterion = nn.CrossEntropyLoss()

loss = criterion(scores, labels.squeeze())
# 反向传播
loss.backward()
# 梯度更新
with torch.no_grad():
    w -= w.grad * alpha
    w.grad.zero_()
    b -= b.grad * alpha
    b.grad.zero_()
```

完整代码如下：

```
import torch
import torch.nn as nn
import numpy as np
from sklearn import utils
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

data_path = '../data/iris-species/Iris.csv'


def load_data(shuffle=True, tsize=0.8):
    """
    加载iris数据
    """
    data = pd.read_csv(data_path, header=0, delimiter=',')

    if shuffle:
        data = utils.shuffle(data)

    # 示性函数
    pd_indicator = pd.get_dummies(data['Species'])
    indicator = np.array(
        [pd_indicator['Iris-setosa'], pd_indicator['Iris-versicolor'], pd_indicator['Iris-virginica']]).T

    species_dict = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['Species'] = data['Species'].map(species_dict)

    data_x = np.array(
        [data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm']]).T
    data_y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    y_train = np.atleast_2d(y_train).T
    y_test = np.atleast_2d(y_test).T

    y_train_indicator = np.atleast_2d(indicator[:y_train.shape[0]])

    return torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float(), torch.from_numpy(
        y_train), torch.from_numpy(y_test), torch.from_numpy(y_train_indicator).float()


def compute_accuracy(scores, Y):
    """
    计算精度
    :param scores: (m,k)
    :param Y: (m,1)
    """
    predictions = torch.argmax(scores, dim=1)
    res = (predictions == Y.squeeze())
    return 1.0 * torch.sum(res).item() / scores.size()[0]


def draw(res_list, title=None, xlabel=None):
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.plot(res_list)
    plt.show()


def compute_gradient_descent(batch_size=8, epoches=2000, alpha=2e-4):
    x_train, x_test, y_train, y_test, y_train_indicator = load_data()

    m, n = x_train.size()[:2]
    k = 3
    # print(m, n, k)

    # softmax回归模型和权重
    linearModel = nn.Linear(n, k)
    softmaxModel = nn.LogSoftmax()
    w = linearModel.weight
    b = linearModel.bias
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    loss_list = []
    accuracy_list = []
    bestW = None
    bestB = None
    bestA = 0
    range_list = list(range(0, m - batch_size, batch_size))
    for i in range(epoches):
        for j in range_list:
            data = x_train[j:j + batch_size]
            labels = y_train[j:j + batch_size]

            scores = softmaxModel.forward(linearModel.forward(data))
            loss = criterion(scores, labels.squeeze())
            # 反向传播
            loss.backward()
            # 梯度更新
            with torch.no_grad():
                w -= w.grad * alpha
                w.grad.zero_()
                b -= b.grad * alpha
                b.grad.zero_()

            if j == range_list[-1]:
                loss_list.append(loss.item())

                accuracy = compute_accuracy(softmaxModel(linearModel(x_train)), y_train)
                accuracy_list.append(accuracy)
                if accuracy >= bestA:
                    bestA = accuracy
                    bestW = w.clone()
                    bestB = b.clone()
                break

    draw(loss_list, title='损失值')
    draw(accuracy_list, title='训练精度')

    linearModel.weight = nn.Parameter(bestW)
    linearModel.bias = nn.Parameter(bestB)
    print(bestA)
    print(compute_accuracy(softmaxModel.forward(linearModel.forward(x_test)), y_test))


if __name__ == '__main__':
    compute_gradient_descent(batch_size=8, epoches=50000)
```

测试结果：

```
# 测试集精度
0.9833333333333333
# 验证集精度
1.0
```

![](/imgs/从numpy到pytorch实现softmax回归/pytorch_advanced_softmax_loss.png)

![](/imgs/从numpy到pytorch实现softmax回归/pytorch_advanced_softmax_accuracy.png)

## `pytorch`实现 - 使用优化器和自定义`softmax`实现类

自定义类，实现`softmax`运算以及参数设置

```
class SoftmaxModule(nn.Module):

    def __init__(self, inputs, outputs):
        super(SoftmaxModule, self).__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        x = self.linear.forward(input)
        x = self.softmax.forward(x)
        return x

    def getParameter(self):
        return self.linear.weight, self.linear.bias

    def setParameter(self, weight, bias):
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)
```

`pytorch`提供了优化器包`torch.optim`来辅助进行梯度更新

```
# 优化器
optimizer = optim.SGD(module.parameters(), lr=alpha)

optimizer.zero_grad()
# 反向传播
loss.backward()
# 梯度更新
optimizer.step()
```

更新代码如下：

```
class SoftmaxModule(nn.Module):

    def __init__(self, inputs, outputs):
        super(SoftmaxModule, self).__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        x = self.linear.forward(input)
        x = self.softmax.forward(x)
        return x

    def getParameter(self):
        return self.linear.weight, self.linear.bias

    def setParameter(self, weight, bias):
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)


def compute_gradient_descent(batch_size=8, epoches=2000, alpha=2e-4):
    x_train, x_test, y_train, y_test, y_train_indicator = load_data()

    m, n = x_train.size()[:2]
    k = 3
    # print(m, n, k)

    # softmax模型
    module = SoftmaxModule(n, k)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(module.parameters(), lr=alpha)

    loss_list = []
    accuracy_list = []
    bestW = None
    bestB = None
    bestA = 0
    range_list = list(range(0, m - batch_size, batch_size))
    for i in range(epoches):
        for j in range_list:
            data = x_train[j:j + batch_size]
            labels = y_train[j:j + batch_size]

            scores = module.forward(data)
            loss = criterion(scores, labels.squeeze())
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            if j == range_list[-1]:
                loss_list.append(loss.item())

                accuracy = compute_accuracy(module.forward(x_train), y_train)
                accuracy_list.append(accuracy)
                if accuracy >= bestA:
                    bestA = accuracy
                    bestW, bestB = module.getParameter()
                break

    draw(loss_list, title='损失值')
    draw(accuracy_list, title='训练精度')

    module.setParameter(bestW, bestB)
    print(bestA)
    print(compute_accuracy(module.forward(x_test), y_test))
```

测试结果：

```
# 测试集精度
0.975
# 验证集精度
1.0
```

![](/imgs/从numpy到pytorch实现softmax回归/pytorch_advanced_softmax_loss_v2.png)

![](/imgs/从numpy到pytorch实现softmax回归/pytorch_advanced_softmax_accuracy_v2.png)

## `pytorch`实现 - 使用`TensorDataset`和`DataLoader`简化批量数据操作

`pytorch.util.data`包提供了类`TensorDataset`和`DataLoader`，用于批量加载数据

`TensorDataset`是一个数据集包装类；`DataLoader`是一个数据加载类，能够实现批量采样、数据打乱

```
# 包装数据集和标记
dataset = TensorDataset(x_train, y_train)
# 加载包装类，设置批量和打乱数据
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 获取批量数据个数
batch_len = dataloader.__len__()
# 依次获取批量数据
for j, items in enumerate(dataloader, 0):
    data, labels = items
```

完整代码如下：

```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn import utils
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

data_path = '../data/iris-species/Iris.csv'


def load_data(shuffle=True, tsize=0.8):
    """
    加载iris数据
    """
    data = pd.read_csv(data_path, header=0, delimiter=',')

    if shuffle:
        data = utils.shuffle(data)

    species_dict = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    data['Species'] = data['Species'].map(species_dict)

    data_x = np.array(
        [data['SepalLengthCm'], data['SepalWidthCm'], data['PetalLengthCm'], data['PetalWidthCm']]).T
    data_y = data['Species']

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=False)

    y_train = np.atleast_2d(y_train).T
    y_test = np.atleast_2d(y_test).T

    return torch.from_numpy(x_train).float(), torch.from_numpy(x_test).float(), torch.from_numpy(
        y_train), torch.from_numpy(y_test)


def compute_accuracy(scores, Y):
    """
    计算精度
    :param scores: (m,k)
    :param Y: (m,1)
    """
    predictions = torch.argmax(scores, dim=1)
    res = (predictions == Y.squeeze())
    return 1.0 * torch.sum(res).item() / scores.size()[0]


def draw(res_list, title=None, xlabel=None):
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.plot(res_list)
    plt.show()


class SoftmaxModule(nn.Module):

    def __init__(self, inputs, outputs):
        super(SoftmaxModule, self).__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        x = self.linear.forward(input)
        x = self.softmax.forward(x)
        return x

    def getParameter(self):
        return self.linear.weight, self.linear.bias

    def setParameter(self, weight, bias):
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)


def compute_gradient_descent(batch_size=8, epoches=2000, alpha=2e-4):
    x_train, x_test, y_train, y_test = load_data()

    m, n = x_train.size()[:2]
    k = 3
    # print(m, n, k)

    # softmax模型
    module = SoftmaxModule(n, k)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(module.parameters(), lr=alpha)

    loss_list = []
    accuracy_list = []
    bestW = None
    bestB = None
    bestA = 0

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    batch_len = dataloader.__len__()
    for i in range(epoches):
        for j, items in enumerate(dataloader, 0):
            data, labels = items

            scores = module.forward(data)
            loss = criterion(scores, labels.squeeze())
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度更新
            optimizer.step()

            if j == (batch_len - 1):
                loss_list.append(loss.item())
                accuracy = compute_accuracy(module.forward(x_train), y_train)
                accuracy_list.append(accuracy)
                if accuracy >= bestA:
                    bestA = accuracy
                    bestW, bestB = module.getParameter()

    draw(loss_list, title='损失值')
    draw(accuracy_list, title='训练精度')

    module.setParameter(bestW, bestB)
    print(bestA)
    print(compute_accuracy(module.forward(x_test), y_test))


if __name__ == '__main__':
    compute_gradient_descent(batch_size=8, epoches=50000)
```

测试结果：

```
# 测试集精度
0.975
# 验证集精度
1.0
```

![](/imgs/从numpy到pytorch实现softmax回归/pytorch_advanced_softmax_loss_v3.png)

![](/imgs/从numpy到pytorch实现softmax回归/pytorch_advanced_softmax_accuracy_v3.png)