---
title: 线性SVM分类器-PyTorch实现
categories:
  - - 算法
    - 机器学习
  - - 算法
    - 分类器
  - - 编程
    - 编程语言
  - - 编程
    - 代码库
  - - 算法
    - 损失函数
tags:
  - 支持向量机
  - python
  - pytorch
  - torchvision
  - 折页损失
abbrlink: 4d25cbab
date: 2020-03-01 14:31:21
---

之前使用`Numpy`实现了线性`SVM`分类器 - [线性SVM分类器](https://zhujian.tech/posts/ebe205e.html)。这一次使用`PyTorch`实现

## 简介

线性`SVM`（`support vector machine`，支持向量机）分类器定义为特征空间上间隔最大的线性分类器模型，其学习策略是使得分类间隔最大化

其训练结果是使得正确类别的成绩至少比错误类别成绩高一个间隔$\triangle $

训练过程如下：

* 首先对输入数据进行线性映射，得到分类成绩；
* 然后，使用折页损失（`hinge loss`）函数计算损失值
* 最后根据损失值进行梯度求导，反向传播

## Hinge Loss

完整的损失值包括折页损失+正则化项

$$
L = \frac {1}{N} \sum_{i} L_{i} + \lambda R(W)
$$

折页损失（`hinge loss`）计算表达式如下：

$$
L_{i} = \sum_{j\neq y_{i}} \max(0, s_{j} - s_{y_{i}} + \triangle )
$$

其中$i$表示批量数据中第$i$个样本，$y_{i}$表示第$i$个样本的正确类别，$j$表示不正确类别

正则化项使用`L2`范数:

$$
R(W) = \sum_{k} \sum_{l} W_{k,l}^{2}
$$

## 前向计算

输入参数：

$$
X \in R^{N\times D}
$$

$$
y \in R^{N}
$$

$$
W \in R^{D\times C}
$$

$$
b \in R^{1\times C}
$$

$$
delta \in R^{1}
$$

前向计算如下：

$$
scores = X\cdot W + b \in R^{N\times C}
$$

$$
corrects = scores[y] \in R^{N\times 1}
$$

$$
margins = \frac {1}{N}\sum_{i=1}^{N}\max (0, scores_{i} - corrects_{i} + delta)
$$

$$
loss = margins + \lambda R(W)
$$

```
def hinge_loss(outputs, labels):
    """
    折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss
```

## MNIST训练

使用线性`SVM`训练`MNIST`数据集，训练参数如下：

1. 学习率：`1e-3`
2. 动量因子：`0.9`
3. 批量大小：`128`
4. 最大间隔：`1.0`

完整代码如下：

```
# -*- coding: utf-8 -*-

"""
@date: 2020/3/1 下午2:38
@file: svm.py
@author: zj
@description: 
"""

import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_loaders = {}
    data_sizes = {}
    for name in ['train', 'val']:
        data_set = MNIST('./data', download=True, transform=transform)
        # 测试
        # img, target = data_set.__getitem__(0)
        # print(img.shape)
        # print(target)
        # exit(0)

        data_loader = DataLoader(data_set, shuffle=True, batch_size=128, num_workers=8)
        data_loaders[name] = data_loader
        data_sizes[name] = len(data_set)
    return data_loaders, data_sizes


def hinge_loss(outputs, labels):
    """
    折页损失计算
    :param outputs: 大小为(N, num_classes)
    :param labels: 大小为(N)
    :return: 损失值
    """
    num_labels = len(labels)
    corrects = outputs[range(num_labels), labels].unsqueeze(0).T

    # 最大间隔
    margin = 1.0
    margins = outputs - corrects + margin
    loss = torch.sum(torch.max(margins, 1)[0]) / len(labels)

    # # 正则化强度
    # reg = 1e-3
    # loss += reg * torch.sum(weight ** 2)

    return loss


def train_model(data_loaders, model, criterion, optimizer, lr_scheduler, num_epochs=25, device=None):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                # print(inputs.shape)
                # print(labels.shape)
                inputs = inputs.reshape(-1, 28 * 28)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                lr_scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_loaders, data_sizes = load_data()
    # print(data_loaders)
    # print(data_sizes)

    model = nn.Linear(28 * 28, 10).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = hinge_loss
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_schduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(data_loaders, model, criterion, optimizer, lr_schduler, num_epochs=25, device=device)
```

`25`轮迭代训练结果如下：

```
Epoch 0/24
----------
train Loss: 1.0502 Acc: 0.7911
val Loss: 1.0268 Acc: 0.8372

Epoch 1/24
----------
train Loss: 1.0214 Acc: 0.8622
val Loss: 1.0238 Acc: 0.8432

Epoch 2/24
----------
train Loss: 1.0180 Acc: 0.8713
val Loss: 1.0150 Acc: 0.8852
...
...
Epoch 24/24
----------
train Loss: 1.0075 Acc: 0.9049
val Loss: 1.0075 Acc: 0.9047

Training complete in 2m 52s
Best val Acc: 0.910633
```