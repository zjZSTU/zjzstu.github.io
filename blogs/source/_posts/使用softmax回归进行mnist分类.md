---
title: 使用softmax回归进行mnist分类
categories:
  - [算法]
  - [编程]
tags:
  - 机器学习
  - 深度学习
  - python
  - pytorch
abbrlink: dd673751
date: 2019-04-28 19:56:03
---

参考：[PyTorch进阶之路（三）：使用logistic回归实现图像分类](https://www.jiqizhixin.com/articles/2019-03-15-17)

`MNIST`是手写数字数据库，共有`60000`张训练图像和`10000`张测试图像，分别表示数字`0-9`

利用`softmax`回归模型进行`mnist`分类

## `numpy`实现

首先需要进行数据库下载和解压，参考[Python MNIST解压](https://blog.csdn.net/u012005313/article/details/84453316)

加载数据集并将训练数据打乱

```
def load_data(shuffle=True):
    """
    加载mnist数据
    """
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    train_file_list = []
    for i in cate_list:
        data_dir = os.path.join(train_dir, str(i))
        file_list = os.listdir(data_dir)
        for filename in file_list:
            file_path = os.path.join(data_dir, filename)
            train_file_list.append(file_path)

        data_dir = os.path.join(test_dir, str(i))
        file_list = os.listdir(data_dir)
        for filename in file_list:
            file_path = os.path.join(data_dir, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape[:2]
                x_test.append(img.reshape(h * w))
                y_test.append(i)

    train_file_list = np.array(train_file_list)
    if shuffle:
        np.random.shuffle(train_file_list)

    for file_path in train_file_list:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
            x_train.append(img.reshape(h * w))
            y_train.append(int(os.path.split(file_path)[0].split('/')[-1]))

    df = pd.DataFrame(y_train)
    df.columns = ['label']
    y_train_indicator = pd.get_dummies(df.label)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), y_train_indicator.values
```

对数据集进行预处理，压缩到`[-1,1]`

```
# 计算均值，进行图像预处理
mu = np.mean(x_train, axis=0)
x_train = (x_train - mu) / 255
x_test = (x_test - mu) / 255
```

`softmax`回归`numpy`实现参考[softmax回归](https://www.zhujian.tech/posts/2626bec3.html#more)

完整代码如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-4-29 上午10:00
# @Author  : zj

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

data_path = '../data/mnist/'

cate_list = list(range(10))


def load_data(shuffle=True):
    """
    加载mnist数据
    """
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    train_file_list = []
    for i in cate_list:
        data_dir = os.path.join(train_dir, str(i))
        file_list = os.listdir(data_dir)
        for filename in file_list:
            file_path = os.path.join(data_dir, filename)
            train_file_list.append(file_path)

        data_dir = os.path.join(test_dir, str(i))
        file_list = os.listdir(data_dir)
        for filename in file_list:
            file_path = os.path.join(data_dir, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape[:2]
                x_test.append(img.reshape(h * w))
                y_test.append(i)

    train_file_list = np.array(train_file_list)
    if shuffle:
        np.random.shuffle(train_file_list)

    for file_path in train_file_list:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
            x_train.append(img.reshape(h * w))
            y_train.append(int(os.path.split(file_path)[0].split('/')[-1]))

    df = pd.DataFrame(y_train)
    df.columns = ['label']
    y_train_indicator = pd.get_dummies(df.label)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), y_train_indicator.values


def linear(x, w):
    """
    线性操作
    :param x: 大小为(m,n+1)
    :param w: 大小为(n+1,k)
    :return: 大小为(m,k)
    """
    return x.dot(w)


def softmax(x):
    """
    softmax归一化计算
    :param x: 大小为(m, k)
    :return: 大小为(m, k)
    """
    x -= np.atleast_2d(np.max(x, axis=1)).T
    exps = np.exp(x)
    return exps / np.atleast_2d(np.sum(exps, axis=1)).T


def compute_scores(X, W):
    """
    计算精度
    :param X: 大小为(m,n+1)
    :param W: 大小为(n+1,k)
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
    cost = -1 / scores.shape[0] * np.sum(np.log(scores) * indicator)
    reg = la / 2 * np.sum(W ** 2)
    return cost + reg


def compute_gradient(scores, indicator, x, W, la=2e-4):
    """
    计算梯度
    :param scores: 大小为(m,k)
    :param indicator: 大小为(m,k)
    :param x: 大小为(m,n+1)
    :param W: (n+1, k)
    :return: (n+1,k)
    """
    return -1 / scores.shape[0] * x.T.dot((indicator - scores)) + la * W


def compute_accuracy(scores, Y):
    """
    计算精度
    :param scores: (m,k)
    :param Y: (m,1)
    """
    res = np.dstack((np.argmax(scores, axis=1), Y.squeeze())).squeeze()

    return len(list(filter(lambda x: x[0] == x[1], res[:]))) / len(res)


def draw(res_list, title=None, xlabel=None):
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.plot(res_list)
    plt.show()


def compute_gradient_descent(batch_size=8, epoches=2000, alpha=2e-4):
    x_train, x_test, y_train, y_test, y_train_indicator = load_data(shuffle=True)

    m, n = x_train.shape[:2]
    k = y_train_indicator.shape[1]

    # 计算均值，进行图像预处理
    mu = np.mean(x_train, axis=0)
    x_train = (x_train - mu) / 255
    x_test = (x_test - mu) / 255

    # 初始化权重(n+1,k)
    W = 0.01 * np.random.normal(loc=0.0, scale=1.0, size=(n + 1, k))
    x_train = np.insert(x_train, 0, np.ones(m), axis=1)
    x_test = np.insert(x_test, 0, np.ones(x_test.shape[0]), axis=1)

    loss_list = []
    accuracy_list = []
    bestW = None
    bestA = 0
    range_list = np.arange(0, m - batch_size, step=batch_size)
    for i in range(epoches):
        for j in range_list:
            data = x_train[j:j + batch_size]
            labels = y_train_indicator[j:j + batch_size]

            # 计算分类概率
            scores = np.atleast_2d(compute_scores(data, W))
            # 更新梯度
            tempW = W - alpha * compute_gradient(scores, labels, data, W)
            W = tempW

            if j == range_list[-1]:
                loss = compute_loss(scores, labels, W)
                print(loss)
                loss_list.append(loss)

                accuracy = compute_accuracy(compute_scores(x_train, W), y_train)
                print('epoch: %d accuracy is %.2f %%' % (i + 1, accuracy * 100))
                accuracy_list.append(accuracy)
                if accuracy >= bestA:
                    bestA = accuracy
                    bestW = W.copy()
                break

    draw(loss_list, title='损失值')
    draw(accuracy_list, title='训练精度')

    test_accuracy = compute_accuracy(compute_scores(x_test, bestW), y_test)
    print('best train accuracy is %.2f %%' % (bestA * 100))
    print('test accuracy is %.2f %%' % (test_accuracy * 100))


if __name__ == '__main__':
    compute_gradient_descent(batch_size=128, epoches=10000)
```

批量大小为`128`，学习率为`2e-4`，共训练`1`万次，结果如下：

```
best train accuracy is 92.33 %
test accuracy is 92.15 %
```

![](/imgs/使用softmax回归进行mnist分类/numpy_softmax_mnist_loss.png)

![](/imgs/使用softmax回归进行mnist分类/numpy_softmax_mnist_accuracy.png)

## `pytorch`实现 - `cpu`训练

`pytorch`提供了类[torchvision.datasets.ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html?highlight=imagefolder#torchvision.datasets.ImageFolder)用于自动加载排列如下的数据集：

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

`pytorch`也提供了类[torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)用于数据格式转换以及数据处理

默认`ImageFolder`读取得到的是彩色`PIL Image`格式图像，需要转换成灰度`Tensor`格式

```
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
```

**使用`ToTensor`进行`PIL Image`或`numpy.ndarray`格式图像转换，从通道（`H, W, C`）、取值（`0,255`）转换为通道（`C, H, W`）、取值（`0,1`）**

最后将数据集载入`pytorch`提供的`DataLoader`，用于批量处理和数据打乱

```
def load_data(batch_size=128, shuffle=True):
    """
    加载iris数据
    """
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_data_set = ImageFolder(train_dir, transform=transform)
    test_data_set = ImageFolder(test_dir, transform=transform)

    return DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle), DataLoader(test_data_set,
                                                                                          batch_size=batch_size,
                                                                                          shuffle=shuffle)
```

完整代码如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-4-28 下午7:55
# @Author  : zj

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import time
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

data_path = '../data/mnist/'

cate_list = list(range(10))


def load_data(batch_size=128, shuffle=True):
    """
    加载iris数据
    """
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_data_set = ImageFolder(train_dir, transform=transform)
    test_data_set = ImageFolder(test_dir, transform=transform)

    return DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle), DataLoader(test_data_set,
                                                                                          batch_size=batch_size,
                                                                                          shuffle=shuffle)


def compute_accuracy(module, dataLoader):
    """
    计算精度
    :param module: 计算模型
    :param dataLoader: 数据加载器
    """
    accuracy = 0
    for i, items in enumerate(dataLoader, 0):
        data, labels = items
        data = data.reshape((data.size()[0], -1))
        scores = module.forward(data)

        predictions = torch.argmax(scores, dim=1)
        res = (predictions == labels.squeeze())
        accuracy += 1.0 * torch.sum(res).item() / scores.size()[0]
    return accuracy / dataLoader.__len__()


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
    train_loader, test_loader = load_data(batch_size=batch_size, shuffle=True)

    n = 784
    k = 10

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

    batch_len = train_loader.__len__()
    for i in range(epoches):
        start = time.time()
        for j, items in enumerate(train_loader, 0):
            data, labels = items
            data = data.reshape((data.size()[0], -1))

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
            accuracy = compute_accuracy(module, train_loader)
            accuracy_list.append(accuracy)
            if accuracy >= bestA:
                bestA = accuracy
                bestW, bestB = module.getParameter()
            end = time.time()
            print('epoch: %d time: %.2f s accuracy: %.3f %%' % (i + 1, end - start, accuracy * 100))

    draw(loss_list, title='损失值')
    draw(accuracy_list, title='训练精度/20次')

    module.setParameter(bestW, bestB)
    test_accuracy = compute_accuracy(module, test_loader)

    print('best train accuracy is %.3f %%' % (bestA * 100))
    print('test accuracy is %.3f %%' % (test_accuracy * 100))


if __name__ == '__main__':
    start = time.time()
    compute_gradient_descent(batch_size=128, epoches=200)
    end = time.time()
    print('all train and test need time: %.2f minutes' % ((end - start) / 60.0))
```

使用`8`核`CPU`（`Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz`）训练`200`次，需要`54.97`分钟，训练和测试精度如下：

```
best train accuracy is 87.474 %
test accuracy is 88.301 %
all train and test need time: 54.97 minutes
```

## `pytorch`实现 - `gpu`训练

`pytorch`提供了`gpu`相关代码，用于加速训练过程

首先判断当前设备是否保存gpu

```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

然后使用`to`函数将模型、损失函数、数据和标签转入`gpu`进行训练

```
# softmax模型
module = SoftmaxModule(n, k).to(device=device)
# 损失函数
criterion = nn.CrossEntropyLoss().to(device=device)
...
data, labels = data.to(device=device), labels.to(device=device)
```

完整代码如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-4-29 下午3:48
# @Author  : zj

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import time
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

data_path = '../data/mnist/'

cate_list = list(range(10))


def load_data(batch_size=128, shuffle=True):
    """
    加载iris数据
    """
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_data_set = ImageFolder(train_dir, transform=transform)
    test_data_set = ImageFolder(test_dir, transform=transform)

    return DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle), DataLoader(test_data_set,
                                                                                          batch_size=batch_size,
                                                                                          shuffle=shuffle)


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
    train_loader, test_loader = load_data(batch_size=batch_size, shuffle=True)

    n = 784
    k = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # softmax模型
    module = SoftmaxModule(n, k).to(device=device)
    # 损失函数
    criterion = nn.CrossEntropyLoss().to(device=device)
    # 优化器
    optimizer = optim.SGD(module.parameters(), lr=alpha)

    loss_list = []
    accuracy_list = []
    bestW = None
    bestB = None
    bestA = 0

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
                bestW, bestB = module.getParameter()
            end = time.time()
            print('epoch: %d time: %.2f s accuracy: %.3f %%' % (i + 1, end - start, accuracy * 100))

    draw(loss_list, title='损失值')
    draw(accuracy_list, title='训练精度/20次')

    module.setParameter(bestW, bestB)
    test_accuracy = compute_accuracy(module, test_loader, device)

    print('best train accuracy is %.3f %%' % (bestA * 100))
    print('test accuracy is %.3f %%' % (test_accuracy * 100))


if __name__ == '__main__':
    start = time.time()
    compute_gradient_descent(batch_size=128, epoches=200)
    end = time.time()
    print('all train and test need time: %.2f minutes' % ((end - start) / 60.0))
```

使用`GeForce GTX 1080`训练`200`次，需要`29.88`分钟，训练和测试精度如下：

```
best train accuracy is 87.442 %
test accuracy is 88.439 %
all train and test need time: 29.88 minutes
```

## 使用`torchvision`内置`mnist`

`torchvision`模块中包含了许多常用数据集的加载类，其中就包括了[MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist)

类`torchvision.datasets.MNIST`能够实现`MNIST`数据集的下载，保存和处理

实现如下：

```
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

def load_data(batch_size=128, shuffle=False):
    data_dir = '../data/mnist/'

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_data_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader
```