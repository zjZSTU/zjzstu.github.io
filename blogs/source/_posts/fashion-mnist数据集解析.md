---
title: Fashion-MNIST数据集解析
categories: 数据集
tags: fashion-mnist
abbrlink: 631c599a
date: 2019-12-10 19:08:55
---

之前识别测试最常用的是手写数字数据集[MNIST](http://yann.lecun.com/exdb/mnist/)，今天遇到一个新的基准数据集 - [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

![](./imgs/../../imgs/fashion-mnist/fashion-mnist-sprite.png)

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