---
title: NIN-numpy
categories:
  - [深度学习]
  - [编程]
tags:
  - NIN
  - python
abbrlink: 55877cae
date: 2019-06-20 13:59:12
---

`numpy`实现[NIN](https://www.zhujian.tech/posts/359ae103.html#more)模型，利用`cifar-10`、`cifar-100`和`mnist`数据集进行`MLPConv`和`GAP`的测试

完整实现：[zjZSTU/PyNet](https://github.com/zjZSTU/PyNet)

## MLPConv实现

`MLPConv`对局部连接执行微神经网络操作，在`NIN`模型中，每个`MLPConv`包含一个`3`层`MLP`

首先需要对输入数据体提取局部连接，可使用一个常规卷积操作实现

比如输入数据体大小为$128\times 3\times 32\times 32$

第一层执行$5\times 5$卷积核大小，步长为$1$，零填充为$2$，滤波器个数为192的卷积操作

输出数据体大小为$128\times 192\times 32\times 32$，特征图上的一点即为输入数据体的单个局部连接结果

第二和第三层执行$1\times 1$卷积核大小，步长为$1$，零填充为$0$的卷积操作

输出数据体空间尺寸为$32\times 32$，不改变输入数据体大小，仅执行深度方向的降维重组

## GAP实现

输入张量大小为$N\times C\times H\times W$，对每个特征图进行均值运算，输出大小为$N\times C$，前向运算如下

```
>>> a = np.arange(36).reshape(2,2,3,3)
>>> a
array([[[[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8]],

        [[ 9, 10, 11],
         [12, 13, 14],
         [15, 16, 17]]],


       [[[18, 19, 20],
         [21, 22, 23],
         [24, 25, 26]],

        [[27, 28, 29],
         [30, 31, 32],
         [33, 34, 35]]]])
>>> b = a.reshape(2,2,-1) # 2-D特征图转换成1-D向量
>>> b
array([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8],
        [ 9, 10, 11, 12, 13, 14, 15, 16, 17]],

       [[18, 19, 20, 21, 22, 23, 24, 25, 26],
        [27, 28, 29, 30, 31, 32, 33, 34, 35]]])
>>> c = np.mean(b, axis=2) # 计算每个特征图均值
>>> c
array([[ 4., 13.],
       [22., 31.]])

```

反向操作中，输入梯度大小为$N\times C$，单个梯度图所有像素点有相同梯度，输出数据体大小为$N\times C\times H\times W$，反向计算如下：

```
>>> a = np.arange(4).reshape(2,2)
>>> a
array([[0, 1],
       [2, 3]])
>>> b = a.reshape(4, -1)
>>> b
array([[0],
       [1],
       [2],
       [3]])
>>> c = np.repeat(b, 9, axis=1)
>>> c
array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3, 3, 3, 3]])
>>> d = c.reshape(2,2,3,3)
>>> d
array([[[[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]],


       [[[2, 2, 2],
         [2, 2, 2],
         [2, 2, 2]],

        [[3, 3, 3],
         [3, 3, 3],
         [3, 3, 3]]]])
```

所以`GAP`类完整实现及测试如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-6-21 上午11:18
# @Author  : zj


import numpy as np
from nn.Layer import *

__all__ = ['GAP']


class GAP(Layer):
    """
    global average pooling layer
    全局平均池化层
    """

    def __init__(self):
        super(GAP, self).__init__()
        self.input_shape = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # input.shape == [N, C, H, W]
        assert len(inputs.shape) == 4
        N, C, H, W = inputs.shape[:4]

        z = np.mean(inputs.reshape(N, C, -1), axis=2)
        self.input_shape = inputs.shape

        return z

    def backward(self, grad_out):
        N, C, H, W = self.input_shape[:4]
        dz = grad_out.reshape(N * C, -1)
        da = np.repeat(dz, H * W, axis=1)

        return da.reshape(N, C, H, W)


if __name__ == '__main__':
    gap = GAP()

    inputs = np.arange(36).reshape(2, 2, 3, 3)
    res = gap(inputs)
    print(res)

    grad_out = np.arange(4).reshape(2, 2)
    da = gap.backward(grad_out)
    print(da)
```

输出如下：

```
[[ 4. 13.]
 [22. 31.]]
[[[[0 0 0]
   [0 0 0]
   [0 0 0]]
  [[1 1 1]
   [1 1 1]
   [1 1 1]]]
 [[[2 2 2]
   [2 2 2]
   [2 2 2]]
  [[3 3 3]
   [3 3 3]
   [3 3 3]]]]
```

## NIN定义

假定输入数据体空间尺寸为$32\times 32$，输出类别为$10$

共有`3`层`MLPConv`和`1`层`GAP`，其中每个`MLPConv`为`3`层`MLP`，完整实现如下：

```
class NIN(Net):
    """
    NIN网络
    """

    def __init__(self, in_channels=1, out_channels=10, momentum=0, nesterov=False, p_h=1.0):
        super(NIN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 5, 5, 192, stride=1, padding=2, momentum=momentum, nesterov=nesterov)
        self.conv2 = nn.Conv2d(96, 5, 5, 192, stride=1, padding=2, momentum=momentum, nesterov=nesterov)
        self.conv3 = nn.Conv2d(192, 3, 3, 192, stride=1, padding=1, momentum=momentum, nesterov=nesterov)

        self.mlp1 = nn.Conv2d(192, 1, 1, 160, stride=1, padding=0, momentum=momentum, nesterov=nesterov)
        self.mlp2 = nn.Conv2d(160, 1, 1, 96, stride=1, padding=0, momentum=momentum, nesterov=nesterov)

        self.mlp2_1 = nn.Conv2d(192, 1, 1, 192, stride=1, padding=0, momentum=momentum, nesterov=nesterov)
        self.mlp2_2 = nn.Conv2d(192, 1, 1, 192, stride=1, padding=0, momentum=momentum, nesterov=nesterov)

        self.mlp3_1 = nn.Conv2d(192, 1, 1, 192, stride=1, padding=0, momentum=momentum, nesterov=nesterov)
        self.mlp3_2 = nn.Conv2d(192, 1, 1, out_channels, stride=1, padding=0, momentum=momentum, nesterov=nesterov)

        self.maxPool1 = nn.MaxPool(2, 2, 96, stride=2)
        self.maxPool2 = nn.MaxPool(2, 2, 192, stride=2)

        self.gap = nn.GAP()

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()
        self.relu9 = nn.ReLU()

        self.dropout = nn.Dropout2d()

        self.p_h = p_h
        self.U1 = None
        self.U2 = None

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        # inputs.shape = [N, C, H, W]
        assert len(inputs.shape) == 4
        x = self.relu1(self.conv1(inputs))
        x = self.relu2(self.mlp1(x))
        x = self.relu3(self.mlp2(x))
        x = self.maxPool1(x)
        self.U1 = self.dropout(x.shape, self.p_h)
        x *= self.U1

        x = self.relu4(self.conv2(x))
        x = self.relu5(self.mlp2_1(x))
        x = self.relu6(self.mlp2_2(x))
        x = self.maxPool2(x)
        self.U2 = self.dropout(x.shape, self.p_h)
        x *= self.U2

        x = self.relu7(self.conv3(x))
        x = self.relu8(self.mlp3_1(x))
        x = self.relu9(self.mlp3_2(x))

        x = self.gap(x)
        return x

    def backward(self, grad_out):
        # grad_out.shape = [N, C]
        assert len(grad_out.shape) == 2
        da11 = self.gap.backward(grad_out)

        dz11 = self.relu9.backward(da11)
        da10 = self.mlp3_2.backward(dz11)
        dz10 = self.relu8.backward(da10)
        da9 = self.mlp3_1.backward(dz10)
        dz9 = self.relu7.backward(da9)
        da8 = self.conv3.backward(dz9)

        da8 *= self.U2
        da7 = self.maxPool2.backward(da8)
        dz7 = self.relu6.backward(da7)
        da6 = self.mlp2_2.backward(dz7)
        dz6 = self.relu5.backward(da6)
        da5 = self.mlp2_1.backward(dz6)
        dz5 = self.relu4.backward(da5)
        da4 = self.conv2.backward(dz5)

        da4 *= self.U1
        da3 = self.maxPool1.backward(da4)
        dz3 = self.relu3.backward(da3)
        da2 = self.mlp2.backward(dz3)
        dz2 = self.relu2.backward(da2)
        da1 = self.mlp1.backward(dz2)
        dz1 = self.relu1.backward(da1)
        da0 = self.conv1.backward(dz1)

    def update(self, lr=1e-3, reg=1e-3):
        self.mlp3_2.update(learning_rate=lr, regularization_rate=reg)
        self.mlp3_1.update(learning_rate=lr, regularization_rate=reg)
        self.conv3.update(learning_rate=lr, regularization_rate=reg)

        self.mlp2_2.update(learning_rate=lr, regularization_rate=reg)
        self.mlp2_1.update(learning_rate=lr, regularization_rate=reg)
        self.conv2.update(learning_rate=lr, regularization_rate=reg)

        self.mlp2.update(learning_rate=lr, regularization_rate=reg)
        self.mlp1.update(learning_rate=lr, regularization_rate=reg)
        self.conv1.update(learning_rate=lr, regularization_rate=reg)

    def predict(self, inputs):
        # inputs.shape = [N, C, H, W]
        assert len(inputs.shape) == 4
        x = self.relu1(self.conv1(inputs))
        x = self.relu2(self.mlp1(x))
        x = self.relu3(self.mlp2(x))
        x = self.maxPool1(x)

        x = self.relu4(self.conv2(x))
        x = self.relu5(self.mlp2_1(x))
        x = self.relu6(self.mlp2_2(x))
        x = self.maxPool2(x)

        x = self.relu7(self.conv3(x))
        x = self.relu8(self.mlp3_1(x))
        x = self.relu9(self.mlp3_2(x))

        x = self.gap(x)
        return x

    def get_params(self):
        out = dict()
        out['conv1'] = self.conv1.get_params()
        out['conv2'] = self.conv2.get_params()
        out['conv3'] = self.conv3.get_params()

        out['mlp1'] = self.mlp1.get_params()
        out['mlp2'] = self.mlp2.get_params()
        out['mlp2_1'] = self.mlp2_1.get_params()
        out['mlp2_2'] = self.mlp2_2.get_params()
        out['mlp3_1'] = self.mlp3_1.get_params()
        out['mlp3_2'] = self.mlp3_2.get_params()

        out['p_h'] = self.p_h

        return out

    def set_params(self, params):
        self.conv1.set_params(params['conv1'])
        self.conv2.set_params(params['conv2'])
        self.conv3.set_params(params['conv3'])

        self.mlp1.set_params(params['mlp1'])
        self.mlp2.set_params(params['mlp2'])
        self.mlp2_1.set_params(params['mlp2_1'])
        self.mlp2_2.set_params(params['mlp2_1'])
        self.mlp3_1.set_params(params['mlp3_1'])
        self.mlp3_2.set_params(params['mlp3_1'])

        self.p_h = params.get('p_h', 1.0)
```

## 测试

测试代码如下：

```
def nin_train():
    x_train, x_test, y_train, y_test = vision.data.load_cifar10(data_path, shuffle=True)

    # 标准化
    x_train = x_train / 255.0 - 0.5
    x_test = x_test / 255.0 - 0.5

    net = models.nin(in_channels=3, p_h=p_h)
    criterion = nn.CrossEntropyLoss()

    accuracy = vision.Accuracy()

    loss_list = []
    train_list = []
    test_list = []
    best_train_accuracy = 0.995
    best_test_accuracy = 0.995

    range_list = np.arange(0, x_train.shape[0] - batch_size, step=batch_size)
    for i in range(epochs):
        total_loss = 0
        num = 0
        start = time.time()
        for j in range_list:
            data = x_train[j:j + batch_size]
            labels = y_train[j:j + batch_size]

            scores = net(data)
            loss = criterion(scores, labels)
            total_loss += loss
            num += 1

            grad_out = criterion.backward()
            net.backward(grad_out)
            net.update(lr=learning_rate, reg=reg)
        end = time.time()
        print('one epoch need time: %.3f' % (end - start))
        print('epoch: %d loss: %f' % (i + 1, total_loss / num))
        loss_list.append(total_loss / num)

        if (i % 20) == 19:
            # # 每隔20次降低学习率
            # learning_rate *= 0.5

            train_accuracy = accuracy.compute_v2(x_train, y_train, net, batch_size=batch_size)
            test_accuracy = accuracy.compute_v2(x_test, y_test, net, batch_size=batch_size)
            train_list.append(train_accuracy)
            test_list.append(test_accuracy)

            print(loss_list)
            print(train_list)
            print(test_list)
            if train_accuracy > best_train_accuracy and test_accuracy > best_test_accuracy:
                path = 'nin-epochs-%d.pkl' % (i + 1)
                utils.save_params(net.get_params(), path=path)
                break

    draw = vision.Draw()
    draw(loss_list, xlabel='迭代/20次')
    draw.multi_plot((train_list, test_list), ('训练集', '测试集'), title='精度图', xlabel='迭代/20次', ylabel='精度值')
```

*`numpy`实现的速度太慢，就看看如何实现的吧，不建议运行*
