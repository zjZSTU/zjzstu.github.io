---
title: 随机失活-pytorch
abbrlink: 2bee4fce
date: 2019-06-08 10:49:36
categories:
  - [最优化]
  - [编程]
tags:
  - 随机失活
  - python
  - pytorch
---

`pytorch`提供了多种失活函数实现

1. [torch.nn.Dropout](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout)
2. [torch.nn.Dropout2d](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout2d)
3. [torch.nn.Dropout3d](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout3d)
4. [torch.nn.AlphaDropout](https://pytorch.org/docs/stable/nn.html#torch.nn.AlphaDropout)

下面首先介绍`Dropout`和`Dropout2d`的使用，然后通过`LeNet-5`模型进行`cifar-10`的训练

## Dropout

对每个神经元进行随机失活

>CLASS torch.nn.Dropout(p=0.5, inplace=False)

默认失活概率为$p=0.5$

输入数组可以是任意大小，输出数组大小和输出数组一致

```
>>> dropout = nn.Dropout()
>>> inputs = torch.randn(2,4)
>>> dropout(inputs)
tensor([[ 3.5830,  5.0388, -0.0000,  0.0000],
        [ 2.4098, -2.1856, -0.7015,  2.0616]])
>>> dropout(inputs)
tensor([[ 3.5830,  5.0388, -0.0000,  0.0000],
        [ 0.0000, -2.1856, -0.0000,  0.0000]])
>>> dropout(inputs)
tensor([[0.0000, 0.0000, -0.0000, 1.7565],
        [0.0000, -0.0000, -0.0000, 2.0616]])
```

**注意：参数$p$表示失活概率，$p=1$表示全部置为$0$，$p=0$表示不执行失活操作**

```
>>> dropout = nn.Dropout(p=0)
>>> inputs = torch.randn(2,4)
>>> dropout(inputs)
tensor([[ 1.2098,  0.3409,  1.4093,  0.6397],
        [ 1.2380, -0.8287,  0.6893,  0.9666]])
>>> dropout(inputs)
tensor([[ 1.2098,  0.3409,  1.4093,  0.6397],
        [ 1.2380, -0.8287,  0.6893,  0.9666]])
>>> dropout = nn.Dropout(p=1)
>>> dropout(inputs)
tensor([[0., 0., 0., 0.],
        [0., -0., 0., 0.]])
>>> dropout(inputs)
tensor([[0., 0., 0., 0.],
        [0., -0., 0., 0.]])
```

## Dropout2d

对每个通道（一个通道表示一个激活图）进行随机失活

>CLASS torch.nn.Dropout2d(p=0.5, inplace=False)

默认失活概率为$p=0.5$

输入数组大小至少为`2`维，默认为$[N, C, H, W]$，输出数组大小和输出数组一致

>RuntimeError: Feature dropout requires at least 2 dimensions in the input

```
>>> dropout = nn.Dropout2d()
>>> inputs = torch.randn(2,3,2,2)
>>> dropout(inputs)
tensor([[[[ 2.0601,  0.0035],
          [-0.7429,  1.2160]],

         [[-0.0000,  0.0000],
          [-0.0000,  0.0000]],

         [[-1.3138, -1.9364],
          [-1.1147,  0.6847]]],


        [[[ 0.0000, -0.0000],
          [-0.0000, -0.0000]],

         [[-0.0000, -0.0000],
          [-0.0000, -0.0000]],

         [[-0.0000,  0.0000],
          [-0.0000,  0.0000]]]])
```

**注意：参数$p$表示失活概率，$p=1$表示全部置为$0$，$p=0$表示不执行失活操作**

## 训练/测试阶段实现

`Pytorch`实现采用**反向失活**方式，在训练阶段，除了进行随机失活操作外，还将结果乘以缩放因子$\frac {1}{1-p}$，这样在测试阶段直接计算全部神经元即可

所以需要区分训练阶段和测试阶段，有两种方式

1. 设置标志位
2. 添加测试函数

### 设置标志位

参考：

[Model.train() and model.eval() vs model and model.eval()](https://discuss.pytorch.org/t/model-train-and-model-eval-vs-model-and-model-eval/5744)

[torch.nn.Module.eval](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval)

[torch.nn.Module.train](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train)

`Pytorch`采用设置标志位的方式判断训练和测试阶段

```
def train(self, mode=True):
        self.training = mode
        for module in self.children():
                module.train(mode)
        return self
def eval(self):
        return self.train(False)
```

```
net.train()  # 训练模式
net.eval()  # 测试模式
```

### 添加测试函数

另一种方式是重写测试函数，将训练和测试实现分开即可

```
def forward(self, inputs): # 训练实现
        a1 = F.relu(self.conv1(inputs))
        a1 = self.dropout2d(a1)
        z2 = self.pool(a1)

        a3 = F.relu(self.conv2(z2))
        a3 = self.dropout2d(a3)
        z4 = self.pool(a3)

        a5 = F.relu(self.conv3(z4))
        a5 = self.dropout2d(a5)

        x = a5.view(-1, self.num_flat_features(a5))

        a6 = F.relu(self.fc1(x))
        a6 = self.dropout(a6)
        return self.fc2(a6)

def predict(self, inputs): # 测试实现
        a1 = F.relu(self.conv1(inputs))
        z2 = self.pool(a1)

        a3 = F.relu(self.conv2(z2))
        z4 = self.pool(a3)

        a5 = F.relu(self.conv3(z4))

        x = a5.view(-1, self.num_flat_features(a5))

        a6 = F.relu(self.fc1(x))
        return self.fc2(a6)
```

## LeNet-5测试


```
class LeNet5(nn.Module):

    def __init__(self, in_channels, p=0.0):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, bias=True)

        self.pool = nn.MaxPool2d((2, 2), stride=2)

        self.fc1 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc2 = nn.Linear(84, 10, bias=True)

        self.p = p
        self.dropout2d = nn.Dropout2d(p=p)
        self.dropout = nn.Dropout(p=p)

    def forward(self, inputs):
        a1 = F.relu(self.conv1(inputs))
        a1 = self.dropout2d(a1)
        z2 = self.pool(a1)

        a3 = F.relu(self.conv2(z2))
        a3 = self.dropout2d(a3)
        z4 = self.pool(a3)

        a5 = F.relu(self.conv3(z4))
        a5 = self.dropout2d(a5)

        x = a5.view(-1, self.num_flat_features(a5))

        a6 = F.relu(self.fc1(x))
        a6 = self.dropout(a6)
        return self.fc2(a6)

    def predict(self, inputs):
        a1 = F.relu(self.conv1(inputs))
        z2 = self.pool(a1)

        a3 = F.relu(self.conv2(z2))
        z4 = self.pool(a3)

        a5 = F.relu(self.conv3(z4))

        x = a5.view(-1, self.num_flat_features(a5))

        a6 = F.relu(self.fc1(x))
        return self.fc2(a6)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

共测试`4`个网络

* 网络$A$：标准神经网络
* 网络$B$：对全连接层进行失活操作
* 网络$C$：对卷积层进行失活操作
* 网络$D$：对所有隐藏层进行失活操作

参考细节如下：

* 批量大小`batch_size=256`
* 迭代次数`epochs=1000`
* 学习率`lr=1e-2`
* 失活率`p=0.5`
* 动量因子`momentum=0.9`
* 每隔`150`轮迭代衰减一半学习率

每隔`20`轮进行一次精度检测，实现如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-6-7 下午3:09
# @Author  : zj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
import time

# 批量大小
batch_size = 256
# 迭代次数
epochs = 1000

# 学习率
lr = 1e-2
# 失活率
p_h = 0.5


def load_cifar_10_data(batch_size=128, shuffle=False):
    data_dir = '/home/lab305/Documents/data/cifar_10/'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_data_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_data_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


class LeNet5(nn.Module):
...
...

def compute_accuracy(loader, net, device):
    total = 0
    correct = 0
    for item in loader:
        data, labels = item
        data = data.to(device)
        labels = labels.to(device)

        scores = net.predict(data)
        predicted = torch.argmax(scores, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return correct / total


if __name__ == '__main__':
    train_loader, test_loader = load_cifar_10_data(batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = LeNet5(3, p=p_h).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    stepLR = lr_scheduler.StepLR(optimer, step_size=150, gamma=0.5)

    best_train_accuracy = 0.99
    best_test_accuracy = 0

    loss_list = []
    train_list = []
    for i in range(epochs):
        num = 0
        total_loss = 0
        start = time.time()
        net.train()  # 训练模式
        for j, item in enumerate(train_loader, 0):
            data, labels = item
            data = data.to(device)
            labels = labels.to(device)

            scores = net.forward(data)
            loss = criterion.forward(scores, labels)

            optimer.zero_grad()
            loss.backward()
            optimer.step()

            total_loss += loss.item()
            num += 1
        end = time.time()
        stepLR.step()

        avg_loss = total_loss / num
        loss_list.append(float('%.4f' % avg_loss))
        print('epoch: %d time: %.2f loss: %.4f' % (i + 1, end - start, avg_loss))

        if i % 20 == 19:
            # 计算训练数据集检测精度
            net.eval()  # 测试模式
            train_accuracy = compute_accuracy(train_loader, net, device)
            train_list.append(float('%.4f' % train_accuracy))
            if best_train_accuracy < train_accuracy:
                best_train_accuracy = train_accuracy

                test_accuracy = compute_accuracy(test_loader, net, device)
                if best_test_accuracy < test_accuracy:
                    best_test_accuracy = test_accuracy

            print('best train accuracy: %.2f %%   best test accuracy: %.2f %%' % (
                best_train_accuracy * 100, best_test_accuracy * 100))
            print(loss_list)
            print(train_list)
```

`1000`轮迭代后的测试精度如下：

|   | 最好训练集精度 | 最好测试集精度 |
|:-:|:--------------:|:--------------:|
| A |      100%      |     60.45 %    |
| B |     99.84%     |     61.47%     |
| C |     57.04%     |        /       |
| D |     50.93%     |        /       |

其损失值和训练集精度值变化如下：

![](/imgs/随机失活-pytorch/lenet_5_loss.png)

![](/imgs/随机失活-pytorch/lenet_5_accuracy.png)

## 小结

从训练结果看出

1. 失活网络需要更多的时间训练才能收敛
2. 失活操作能够提高泛化能力
3. 对卷积层进行失活操作会导致损失值过早收敛