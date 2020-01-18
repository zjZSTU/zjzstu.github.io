---
title: '[数据集]German Credit Data'
categories:
  - [数据, 数据集]
  - [编程, 编程语言]
  - [编程, 代码库]
tags:
  - german credit data
  - python
  - numpy
  - sklearn
  - pandas
abbrlink: 833d7df4
date: 2019-12-13 20:32:06
---

参考：[Statlog (German Credit Data) Data Set](http://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data))

德国信用卡数据（`German Credit Data`）提供了一个二分类数据集，下载地址 - [statlog/german](http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/)

## 解析

* 文件`german.data`提供了`20`个属性（`13`个类别属性+`7`个数字属性）共`1000`个实例，最后一列表示类别（分别为`1`和`2`）
* 文件`german.data-numeric`提供了`24`个数字属性，共`1000`个实例，最后一列表示类别（分别为`1`和`2`）

最开始数据集仅提供了文件`german.data`，后续提供了文件`german.data-numeric`，添加了属性并全部转换成数值

## python读取

利用`pandas`库读取`csv`文件，利用`sklearn`库分离训练集和数据集，对数据进行标准化操作

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   german.py
@time:   2019-12-13
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_german_data(data_path, shuffle=True, tsize=0.8):
    data_list = pd.read_csv(data_path, header=None, sep='\s+')

    data_array = data_list.values
    height, width = data_array.shape[:2]
    data_x = data_array[:, :(width - 1)]
    data_y = data_array[:, (width - 1)]

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, train_size=tsize, test_size=(1 - tsize),
                                                        shuffle=shuffle)

    y_train = np.array(list(map(lambda x: 1 if x == 2 else 0, y_train)))
    y_test = np.array(list(map(lambda x: 1 if x == 2 else 0, y_test)))

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    data_path = '/home/zj/data/german/german.data-numeric'
    x_train, x_test, y_train, y_test = load_german_data(data_path)

    x_train = x_train.astype(np.double)
    x_test = x_test.astype(np.double)
    # 计算训练集每个属性的均值和方差
    mu = np.mean(x_train, axis=0)
    var = np.var(x_train, axis=0)
    eps = 1e-8
    # 将数据变换为均值为0，方差为1的标准正态分布
    x_train = (x_train - mu) / np.sqrt(var + eps)
    x_test = (x_test - mu) / np.sqrt(var + eps)

    print(x_test)
```