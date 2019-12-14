---
title: Fashion-MNIST数据集解析
categories: 
- [数据集]
- [编程]
- [代码库]
tags: 
- fashion-mnist
- pytorch
- python
- torchvision
- matplotlib
abbrlink: 631c599a
date: 2019-12-10 19:08:55
---

之前识别测试最常用的是手写数字数据集[MNIST](http://yann.lecun.com/exdb/mnist/)，今天遇到一个新的基准数据集 - [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

![](/imgs/..//imgs/fashion-mnist/fashion-mnist-sprite.png)

## 简介

`Fashion-MNIST`是关于衣物饰品的数据集，其格式类似于`MNIST`，包含了`10`类标签，每张图片为`28x28`大小灰度图像，共`60000`张训练集和`10000`张测试集

| Label | Description |
| --- | --- |
| 0 | T-shirt/top(T恤) |
| 1 | Trouser(裤子) |
| 2 | Pullover(套衫) |
| 3 | Dress(连衣裙) |
| 4 | Coat(外套) |
| 5 | Sandal(凉鞋) |
| 6 | Shirt(衬衫) |
| 7 | Sneaker(运动鞋) |
| 8 | Bag(包) |
| 9 | Ankle boot(短靴) |

## fashion-mnist vs. mnist

`Fashion-MNIST`希望替代`MNIST`成为新的基准数据集，给出了以下`3`个理由：

1. `MNIST`没有挑战性。使用最新的卷积神经网络算法能够达到`99.7%`，使用传统的机器学习算法也能够达到`97%`
2. `MNIST`没有新奇感。不解释
3. `MNIST`无法代表现代`CV`任务。某位大牛说的

总的来说，在使用相同样本格式的情况下，`Fashion-MNIST`的难度大于`MNIST`，参考[benchmarks](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#)

与此同时，`Fashion-MNIST`还提供了更丰富的语言接口

## 下载

`Fashion-MNIST`的存储格式和`MNIST`一样，具体格式参考[Python MNIST解压](https://blog.csdn.net/u012005313/article/details/84453316)

下载地址如下：

1. [train-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz)
2. [train-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz)
3. [t10k-images-idx3-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz)
4. [t10k-labels-idx1-ubyte.gz](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)

或者下载仓库

```
$ git clone git@github.com:zalandoresearch/fashion-mnist.git
```

数据集保存在`/data/fashion`路径下

## 解析

`Fashion-MNIST`提供了多种语言的数据加载接口，包括`Python/C/C++/Java`等，同时基于多种机器学习库，比如`PyTorch/Torch`

### Numpy

下载整个仓库

```
$ git clone git@github.com:zalandoresearch/fashion-mnist.git
```

使用`utils/mnist_reader`脚本加载图片

```
utils$ python
Python 3.7.4 (default, Aug 13 2019, 20:35:49) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mnist_reader
>>> X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
>>> X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
>>> 
>>> 
>>> X_train.shape
(60000, 784)
>>> y_train.shape
(60000,)
>>> X_test.shape
(10000, 784)
>>> y_test.shape
(10000,)
```

### PyTorch

```
import torch
import torchvision
import torchvision.transforms as transforms

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)
```

### C++

参考：[wichtounet/mnist](https://github.com/wichtounet/mnist)

## 测试

分别使用`MNIST`和`Fashion-MNIST`，使用`AlexNet`卷积神经网络模型，利用`PyTorch`实现，训练参数如下：

* 批量大小：`256`
* 学习率：`1e-3`
* 动量大小：`0.9`
* 迭代次数：`30`

实现代码如下：

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   mnist.py
@time:   2019-12-10
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

root = './data'
bsize = 256
shuffle = True
num_work = 8

learning_rate = 1e-3
moment = 0.9
epoches = 30

classes = range(10)


def load_mnist(transform, root, bsize, shuffle, num_work):
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=shuffle, num_workers=num_work)

    test_dataset = datasets.MNIST(root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=shuffle, num_workers=num_work)

    return train_loader, test_loader


def load_fashion_mnist(transform, root, bsize, shuffle, num_work):
    train_dataset = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=bsize, shuffle=shuffle, num_workers=num_work)

    test_dataset = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=bsize, shuffle=shuffle, num_workers=num_work)

    return train_loader, test_loader


def compute_accuracy(loader, net, device):
    total_accu = 0.0
    num = 0

    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = net.forward(inputs)
        predicted = torch.argmax(outputs, dim=1)
        total_accu += torch.mean((predicted == labels).float()).item()
        num += 1
    return total_accu / num


class AlexNet(nn.Module):

    def __init__(self, num_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_loader, test_loader = load_mnist(transform, root, bsize, shuffle, num_work)
    # train_loader, test_loader = load_fashion_mnist(transform, root, bsize, shuffle, num_work)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = AlexNet(num_channels=1, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

    train_accu_list = list()
    test_accu_list = list()
    loss_list = list()

    for epoch in range(epoches):
        num = 0
        total_loss = 0.0
        start = time.time()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimer.step()

            total_loss += loss.item()
            num += 1
        end = time.time()

        avg_loss = total_loss / num
        print('[%d] loss: %.5f, time: %.3f' % (epoch + 1, avg_loss, end - start))
        loss_list.append(avg_loss)

        train_accu = compute_accuracy(train_loader, net, device)
        test_accu = compute_accuracy(test_loader, net, device)
        print('train: %.3f, test: %.3f' % (train_accu, test_accu))
        train_accu_list.append(train_accu)
        test_accu_list.append(test_accu)

    print(loss_list)
    print(train_accu_list)
    print(test_accu_list)
```

训练完成后通过`matplotlib`绘制

```
import matplotlib.pyplot as plt
import numpy as np

def draw(mnist_list,fashion_list, xlabel,ylabel,title):
    fig=plt.figure()
    x = list(range(30))
    plt.plot(x, mnist_list, label='mnist')
    plt.plot(x, fashion_list, label='fashion-mnist')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

draw(mnist_loss_list, fashion_loss_list, 'epoch','loss_value','loss')
draw(mnist_train_accu_list, fashion_train_accu_list, 'epoch','accurancy','train accurancy')
draw(mnist_test_accu_list, fashion_test_accu_list, 'epoch','accurancy','test accurancy')
```

![](/imgs/..//imgs/fashion-mnist/mnist_fashion_loss.png)

![](/imgs/..//imgs/fashion-mnist/mnist_fashion_train.png)

![](/imgs/..//imgs/fashion-mnist/mnist_fashion_test.png)

从训练结果可以发现，相比较于`MNIST`，数据集`Fashion-MNIST`更具挑战性