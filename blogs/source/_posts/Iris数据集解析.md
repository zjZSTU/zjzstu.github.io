---
title: Iris数据集解析
categories: 
- [数据, 数据集]
- [编程, 代码库]
- [编程, 编程语言]
tags: 
- iris
- python
- matplotlib
- numpy
- pandas
- sklearn
abbrlink: ffa9d775
date: 2019-12-14 21:08:21
---

参考：[Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris)

`Iris`数据集包含`3`个类别`4`个属性，共`150`个实例

## 解析

`4`个属性分别表示（单位`cm`）：

1. 萼片(`sepal`)长度
2. 萼片宽度
3. 花瓣(`petal`)长度
4. 花瓣宽度

每个类别有`50`个实例，整个数据集共`150`个

## 下载

下载地址：[machine-learning-databases/iris](http://archive.ics.uci.edu/ml/machine-learning-databases/iris/)

文件`iris.data`中共`150`行，每行`5`个值，前`4`个表示属性，最后一个表示类别，每列用逗号隔开

*`Kaggle`也提供了`Iris`数据集，其数据集格式略有差别，具体参考[鸢尾数据集](https://zhujian.tech/posts/2626bec3.html)*

## 解析

通过`pandas`库解析`csv`文件，通过`sklearn`库分离训练集和测试集，通过`matplotlib`绘制图像

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   iris.py
@time:   2019-12-14
"""

import pandas as pd
import sklearn.utils as utils
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def iris_str_to_int(x):
    if 'Iris-setosa'.__eq__(x):
        return 0
    elif 'Iris-versicolor'.__eq__(x):
        return 1
    else:
        return 2


def load_iris_data(data_path, shuffle=True, tsize=0.8):
    data_list = pd.read_csv(data_path, header=None, sep=',')

    data_array = data_list.values
    height, width = data_array.shape[:2]
    data_x = data_array[:, :(width - 1)].astype(np.float)
    data_y = data_array[:, (width - 1)]

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=shuffle)
    y_train = np.array(list(map(lambda x: iris_str_to_int(x), y_train)))
    y_test = np.array(list(map(lambda x: iris_str_to_int(x), y_test)))

    return x_train, x_test, y_train, y_test


def draw_iris(x_data, y_data, title, xlabel, ylabel):
    fig = plt.figure()

    x = x_data[y_data == 0]
    plt.scatter(x[:, 0], x[:, 1], c='r', marker='<', label='Iris-setosa')

    x = x_data[y_data == 1]
    plt.scatter(x[:, 0], x[:, 1], c='g', marker='8', label='Iris-versicolor')

    x = x_data[y_data == 2]
    plt.scatter(x[:, 0], x[:, 1], c='y', marker='*', label='Iris-virginica')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    iris_path = '/home/zj/data/iris-species/iris.data'
    x_train, x_test, y_train, y_test = load_iris_data(iris_path, shuffle=True, tsize=0.8)

    x_train = x_train.astype(np.double)
    x_test = x_test.astype(np.double)
    # 计算训练集每个属性的均值和方差
    mu = np.mean(x_train, axis=0)
    var = np.var(x_train, axis=0)
    eps = 1e-8
    # 将数据变换为均值为0，方差为1的标准正态分布
    x_train = (x_train - mu) / np.sqrt(var + eps)
    x_test = (x_test - mu) / np.sqrt(var + eps)

    draw_iris(x_train[:, :2], y_train, 'sepal height/width', 'height', 'width')
    draw_iris(x_train[:, 2:], y_train, 'petal height/width', 'height', 'width')
```

![](/imgs/iris/iris-sepal.png)

![](/imgs/iris/iris-petal.png)

## sklearn

`sklearn`库提供了`iris`数据集的封装

```
# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.model_selection import train_test_split

def load_data():
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
```

输出如下：

```
(75, 4)
(75, 4)
(75,)
(75,)
```