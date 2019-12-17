---
title: '[PyTorch]Tensorboard使用实践'
categories:
  - [编程, 编程语言]
  - [编程, 代码库]
  - - 工具
tags:
  - python
  - pytorch
  - torchvision
  - tensorboard
abbrlink: f793688d
date: 2019-12-11 19:29:58
---

学习了`PyTorch`环境下的`Tensorboard`使用 - [[PyTorch]Tensorboard可视化实现](https://zhujian.tech/posts/eb6f2b71.html#more)。`PyTorch`也提供了`Tensorboard`学习教程 - [Visualizing Models, Data, and Training with TensorBoard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html#visualizing-models-data-and-training-with-tensorboard)

下面结合一个完整的训练过程，通过`Tensorboard`实现可视化

## 示例

利用`LeNet-5`模型训练并测试`Fashion-MNIST`，训练参数如下：

* 批量大小：`256`
* 学习率：`1e-3`
* 动量：`0.9`
* 迭代次数：`50`

操作流程如下：

1. 加载训练集，新建模型，损失器和优化器，转换数据和模型到`GPU`
2. 迭代数据集训练网络，每轮完成训练后计算损失值，训练集精度和测试集精度
3. 绘制损失图和精度图

完整代码如下：

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   tensorboard-fashion-mnist.py
@time:   2019-12-11
"""

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.utils

learning_rate = 1e-3
moment = 0.9
epoches = 50
bsize = 256

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def load_data(bsize):
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=bsize,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


def draw(values, xlabel, ylabel, title, label):
    fig = plt.figure()
    plt.plot(list(range(len(values))), values, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.show()


def train(trainloader, testloader, net, criterion, optimizer, device):
    train_accu_list = list()
    test_accu_list = list()
    loss_list = list()

    for epoch in range(epoches):  # loop over the dataset multiple times
        num = 0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num += 1
        # 每轮迭代完成后，记录损失值，计算训练集和测试集的检测精度
        avg_loss = running_loss / num
        print('[%d] loss: %.4f' % (epoch + 1, avg_loss))
        loss_list.append(avg_loss)

        train_accu = compute_accuracy(trainloader, net, device)
        test_accu = compute_accuracy(testloader, net, device)
        print('train: %.4f, test: %.4f' % (train_accu, test_accu))
        train_accu_list.append(train_accu)
        test_accu_list.append(test_accu)

    print('Finished Training')
    return train_accu_list, test_accu_list, loss_list


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=moment)

    trainloader, testloader = load_data(bsize)

    train_accu_list, test_accu_list, loss_list = train(trainloader, testloader, net, criterion, optimizer, device)

    draw(train_accu_list, 'epoch', 'accuracy', 'train accuracy', 'fashion-mnist')
    draw(test_accu_list, 'epoch', 'accuracy', 'test accuracy', 'fashion-mnist')
    draw(loss_list, 'epoch', 'loss_value', 'loss', 'fashion-mnist')
```

![](/imgs/tensorboard-work/normal-training-process-result.png)

## Tensorboard实践

实现流程如下：

1. 启动`Tensorboard`
2. 写入样本图像
3. 写入模型
4. 高维特征投影
5. 追踪训练过程

### 启动Tensorboard

```
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```

打开新的命令行窗口，在同一路径下输入命令：

```
$ tensorboard --logdir=runs --host=192.168.0.112 --port=7878
```

打开浏览器，输入`192.168.0.112:7878`，即可打开`Tensorboard`

### 写入样本图像

修改数据加载函数，分离转换器，以便能偶加载未标准化的数据集

```
def load_data(bsize, tf=None):
    # datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
                                                 download=True,
                                                 train=True,
                                                 transform=tf)
    testset = torchvision.datasets.FashionMNIST('./data',
                                                download=True,
                                                train=False,
                                                transform=tf)

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize,
                                              shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(testset, batch_size=bsize,
                                             shuffle=False, num_workers=4)
    return trainloader, testloader
```

加载数据集，写入图像。`torchvision`提供了函数[make_grid](https://pytorch.org/docs/stable/torchvision/utils.html#torchvision.utils.make_grid)将`Tensor`数组转换成单个图像（`[64, 1, 28, 28] -> [3, 242, 242]`）

```
transform = transforms.Compose(
        [transforms.ToTensor()])

trainloader, testloader = load_data(64, tf=transform)
print(trainloader)

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.__next__()
print(images.size())

# create grid of images
img_grid = torchvision.utils.make_grid(images)
print(img_grid.size())

# write to tensorboard
writer.add_image('fashion_mnist_images', img_grid)
writer.close()
```

打开`Tensorboard IMAGES`页面，选择`fashion_mnist_images`标签的图像

![](/imgs/tensorboard-work/fashion-mnist-images.png)

### 写入模型

```
net = Net()

writer.add_graph(net, images)
writer.close()
```

打开`Tensorboard GRAPHS`页面，在右侧类别`Runs`中选择当前写入的文件`fashion-mnist-lenet5`

![](/imgs/tensorboard-work/graph-lenet-5.png)

### 高维特征投影

```
# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.__next__()
print(images.size())

# select random images and their target indices
# images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
print(features.size())
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images)
writer.close()
```

随机提取批量大小数据集，转换成向量数组，输入`add_embedding`函数中

打开`Tensorboard GRAPHS`页面，在右侧类别`Runs`中选择当前写入的文件`fashion-mnist-lenet5`，可在右下角选择不同的投影规则（默认`PCA`）

![](/imgs/tensorboard-work/projector-pca.png)

### 追踪训练过程

每轮迭代完成后，计算其损失值，训练集和测试集精度值，输入到`add_scalar(s)`函数中

```
for epoch in range(epoches):  # loop over the dataset multiple times
    num = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num += 1
    # 每轮迭代完成后，记录损失值，计算训练集和测试集的检测精度
    avg_loss = running_loss/num
    print('[%d] loss: %.4f' % (epoch+1, avg_loss))

    train_accu = compute_accuracy(trainloader, net, device)
    test_accu = compute_accuracy(testloader, net, device)
    print('train: %.4f, test: %.4f' % (train_accu, test_accu))
    
    # 添加损失值
    writer.add_scalar("training loss", avg_loss, epoch)

    # 添加训练集和测试集精度
    writer.add_scalars("training accurancy", {'loss': avg_loss,
                          'train_accu': train_accu,
                          'test_accu': test_accu}, epoch)

print('Finished Training')
```

打开`Tensorboard SCALARS`页面，在右下角选择类别

![](/imgs/tensorboard-work/train-test-loss.png)