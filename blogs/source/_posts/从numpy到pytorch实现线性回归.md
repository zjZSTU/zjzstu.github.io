---
title: 从numpy到pytorch实现线性回归
categories:
  - 编程
tags:
  - 机器学习
  - 深度学习
abbrlink: ca2079f0
date: 2019-04-16 20:13:01
---

参考：

[Linear Regression and Gradient Descent from scratch in PyTorch](https://medium.com/jovian-io/linear-regression-with-pytorch-3dde91d60b50)

[PyTorch进阶之路（二）：如何实现线性回归](https://www.jiqizhixin.com/articles/2019-03-15-5)

[线性回归](https://www.zhujian.tech/posts/ec419bd2.html#more)

[特征缩放](https://www.zhujian.tech/posts/dea583b1.html#more)

首先利用`numpy`实现梯度下降解决多变量线性回归问题，然后逐步将操作转换成`pytorch`

实现步骤如下：

1. 加载训练数据
2. 初始化权重
3. 计算预测结果
4. 计算损失函数
5. 梯度更新
6. 重复`3-5`步，直到完成迭代次数
7. 绘制损失图

多变量线性回归测试数据参考[ex1data2.txt](https://github.com/peedeep/Coursera/blob/master/ex1/ex1data2.txt)

## `numpy`实现随机梯度下降

参考：[梯度下降](https://www.zhujian.tech/posts/3c50d4b7.html#more)

**随机梯度下降**实现如下

```
# -*- coding: utf-8 -*-

# @Author  : zj

"""
梯度下降法计算线性回归问题
"""

import matplotlib.pyplot as plt
import numpy as np


def load_ex1_multi_data():
    """
    加载多变量数据
    """
    path = '../data/coursera2.txt'
    datas = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datas.append(line.strip().split(','))
    data_arr = np.array(datas)
    data_arr = data_arr.astype(np.float)

    X = data_arr[:, :2]
    Y = data_arr[:, 2]
    return X, Y


def draw_loss(loss_list):
    """
    绘制损失函数值
    """
    fig = plt.figure()
    plt.plot(loss_list)

    plt.show()


def init_weight(size):
    """
    初始化权重，使用均值为0,方差为1的标准正态分布
    """
    return np.random.normal(loc=0.0, scale=1.0, size=size)


def compute_loss(w, x, y):
    """
    计算损失值
    """
    n = y.shape[0]
    return (x.dot(w) - y).T.dot(x.dot(w) - y) / n


def using_stochastic_gradient_descent():
    """
    随机梯度下降
    """
    x, y = load_ex1_multi_data()
    extend_x = np.insert(x, 0, values=np.ones(x.shape[0]), axis=1)
    w = init_weight(extend_x.shape[1])
    # print(w)
    print(w.shape)

    # 打乱数据
    np.random.shuffle(extend_x)
    print(extend_x.shape)
    print(y.shape)

    n = y.shape[0]
    epoches = 10
    alpha = 1e-8
    loss_list = []
    for i in range(epoches):
        for j in range(n):
            temp = w - alpha * (extend_x[j].dot(w) - y[j]) * extend_x[j].T / 2
            w = temp
            loss_list.append(compute_loss(w, extend_x, y))
    draw_loss(loss_list)


if __name__ == '__main__':
    using_stochastic_gradient_descent()
```

![](/imgs/从numpy到pytorch实现线性回归/numpy_sgd.png)

## `pytorch`实现批量梯度下降

`pytorch`使用`tensor`作为数据保存结构，使用函数`from_numpy`可以将`numpy array`数组转换成`tensor`类型

```
torch.from_numpy(X), torch.from_numpy(Y)
```

使用`torch.randn`可以生成符合标准正态分布的随机数组，用于生成权重和偏置值

```
torch.randn(h, 1, requires_grad=True, dtype=torch.double), torch.randn(1, requires_grad=True,                                                                                  dtype=torch.double)
```

`pytorch`内置了`autograd`包，计算预测结果和损失函数后，调用函数`backward()`就能够自动计算出梯度

首先需要开启权重和偏置值的梯度开关，然后在调用函数后进行梯度更新

```
with torch.no_grad():
      w -= w.grad * lr
      b -= b.grad * lr
      w.grad.zero_()
      b.grad.zero_()
```

使用`torch.no_grad`能够保证梯度更新过程中不再计算梯度值，计算完成后需要将梯度归零，避免下次叠加

使用`pytorch`实现**批量梯度下降**计算多变量线性回归问题

```
# -*- coding: utf-8 -*-

# @Author  : zj

"""
梯度下降法计算线性回归问题
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_ex1_multi_data():
    """
    加载多变量数据
    """
    path = '../data/coursera2.txt'
    datas = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datas.append(line.strip().split(','))
    data_arr = np.array(datas)
    data_arr = data_arr.astype(np.float)

    X = data_arr[:, :2]
    Y = data_arr[:, 2]

    return torch.from_numpy(X), torch.from_numpy(Y)


def init_weight(h):
    """
    初始化权重，使用均值为0,方差为1的标准正态分布
    """
    return torch.randn(h, 1, requires_grad=True, dtype=torch.double), torch.randn(1, requires_grad=True,
                                                                                  dtype=torch.double)


def predict_result(w, b, x):
    """
    预测结果
    """
    return x.mm(w) + b


def compute_loss(w, b, x, y):
    """
    计算损失值 MSE
    """
    diff = y - predict_result(w, b, x)
    return torch.sum(diff * diff) / diff.numel()


def draw_loss(loss_list):
    """
    绘制损失函数值
    """
    fig = plt.figure()
    plt.plot(loss_list)

    plt.show()


def using_batch_gradient_descent():
    """
    批量梯度下降
    """
    x, y = load_ex1_multi_data()
    w, b = init_weight(x.shape[1])

    epoches = 20
    lr = 1e-7
    loss_list = []
    for i in range(epoches):
        # 计算损失值
        loss = compute_loss(w, b, x, y)
        # 保存损失值
        loss_list.append(loss)
        # 反向更新
        loss.backward()
        # 梯度更新
        with torch.no_grad():
            w -= w.grad * lr
            b -= b.grad * lr
            w.grad.zero_()
            b.grad.zero_()
    draw_loss(loss_list)


if __name__ == '__main__':
    using_batch_gradient_descent()
```

![](/imgs/从numpy到pytorch实现线性回归/pytorch_batch.png)

## `pytorch`实现随机梯度下降

`pytorch`提供了许多类和函数用于计算，下面实现**随机梯度下降**解决多变量线性回归

首先在`numpy`数组转换成`pytorch tensor`类型前先打乱数据

```
# 打乱数据
indexs = np.arange(X.shape[0])
np.random.shuffle(indexs)
X = X[indexs]
Y = Y[indexs]
```

`pytorch.nn`包提供了类`Linear`用于线性计算

```
# 定义线性模型
model = nn.Linear(x.size()[1], 1)
# 获取初始权重和偏置值
w = model.weight
b = model.bias
# 计算预测结果，计算损失值
diff = y - model(x)
```

`pytorch.nn.function`包提供了函数`mse_loss`用于计算均方误差

也可以使用包装类`nn.MSELoss`

```
# 损失函数
loss_fn = F.mse_loss
# 计算损失值
loss = loss_fn(model(x), y)
# 或者
# 损失函数
criterion = nn.MSELoss()
# 计算损失值
loss = criterion(model(x), y)
```

`pytorch.optim`提供了类`SGD`用于计算随机梯度下降

```
# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=2e-7, momentum=0.9)
# 清空梯度
optimizer.zero_grad()
# 计算梯度
loss.backward()
# 更新
optimizer.step()
```

实现如下：

```
# -*- coding: utf-8 -*-

# @Author  : zj

"""
梯度下降法计算线性回归问题
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def load_ex1_multi_data():
    """
    加载多变量数据
    """
    path = '../data/coursera2.txt'
    datas = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datas.append(line.strip().split(','))
    data_arr = np.array(datas)
    data_arr = data_arr.astype(np.float)

    X = data_arr[:, :2]
    Y = data_arr[:, 2]
    
    # 打乱数据
    indexs = np.arange(X.shape[0])
    np.random.shuffle(indexs)
    X = X[indexs]
    Y = Y[indexs]

    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def draw_loss(loss_list):
    """
    绘制损失函数值
    """
    fig = plt.figure()
    plt.plot(loss_list)

    plt.show()


def using_stochastic_gradient_descent():
    """
    随机梯度下降
    """
    x, y = load_ex1_multi_data()

    # 定义线性模型
    model = nn.Linear(x.size()[1], 1)
    # 获取初始权重和偏置值
    w = model.weight
    b = model.bias

    # 损失函数
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=1e-10, momentum=0.9)

    epoches = 10
    loss_list = []
    for i in range(epoches):
        for j, item in enumerate(x, 0):
            # 计算损失值
            loss = criterion(model(item), y[j])
            # 清空梯度
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新
            optimizer.step()
            # 保存损失值
            loss_list.append(loss)
    draw_loss(loss_list)


if __name__ == '__main__':
    using_stochastic_gradient_descent()
```

![](/imgs/从numpy到pytorch实现线性回归/pytorch_stochastic.png)

## `pytorch`实现小批量梯度下降

实际训练过程中最常使用的梯度下降方法是小批量梯度下降，

`pytorch`提供了类`torch.utils.data.TensorDataset`以及`torch.utils.data.DataLoader`来实现数据的加载、打乱和批量化

```
batch_size = 8
data_ts = TensorDataset(x, y)
data_loader = DataLoader(data_ts, batch_size=batch_size, shuffle=True)
for j, item in enumerate(data_loader, 0):
    inputs, targets = item
    # 计算损失值
    loss = criterion(model(inputs), targets)
```

实现如下：

```
# -*- coding: utf-8 -*-

# @Author  : zj

"""
梯度下降法计算线性回归问题
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def load_ex1_multi_data():
    """
    加载多变量数据
    """
    path = '../data/coursera2.txt'
    datas = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            datas.append(line.strip().split(','))
    data_arr = np.array(datas)
    data_arr = data_arr.astype(np.float)

    X = data_arr[:, :2]
    Y = data_arr[:, 2]

    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def draw_loss(loss_list):
    """
    绘制损失函数值
    """
    fig = plt.figure()
    plt.plot(loss_list)

    plt.show()


def using_small_batch_gradient_descent():
    """
    小批量梯度下降
    """
    x, y = load_ex1_multi_data()

    batch_size = 8
    data_ts = TensorDataset(x, y)
    data_loader = DataLoader(data_ts, batch_size=batch_size, shuffle=True)

    # 定义线性模型
    model = nn.Linear(x.size()[1], 1)
    # 获取初始权重和偏置值
    w = model.weight
    b = model.bias

    # 损失函数
    criterion = nn.MSELoss()
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=1e-10, momentum=0.9)

    epoches = 200
    loss_list = []
    for i in range(epoches):
        for j, item in enumerate(data_loader, 0):
            # print(item)
            inputs, targets = item
            # 计算损失值
            loss = criterion(model(inputs), targets)
            # 清空梯度
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新
            optimizer.step()
            # 保存损失值
            loss_list.append(loss)
    draw_loss(loss_list)


if __name__ == '__main__':
    using_small_batch_gradient_descent()
```

![](/imgs/从numpy到pytorch实现线性回归/pytorch_small_batch.png)

## 小结

`pytorch`使用到的类库如下所示

![](/imgs/从numpy到pytorch实现线性回归/torch-class.png)

