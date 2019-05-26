---
title: im2col解析1
categories:
  - 编程
tags:
  - 深度学习
abbrlink: cc37c46b
date: 2019-05-24 09:51:33
---

在`cs231n`课程[Convolutional Neural Networks: Architectures, Convolution / Pooling Layers ](http://cs231n.github.io/convolutional-networks/#fc)中提到使用矩阵乘法方式完成卷积层及池化层操作，同时在[Assignment #2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets](http://cs231n.github.io/assignments2019/assignment2/)中给出了一个卷积层转全连接层的实现 - `im2col.py`

`im2col`表示将滤波器局部连接矩阵向量化为列向量（`column vector`），在行方向进行堆叠，最终得到`2-D`矩阵

`im2col.py`使用 ***花式下标求解*** 的方式，让我觉得应该写篇文章好好学习一下

本文介绍一些`numpy`实现，下一篇介绍`im2col`实现，第三篇实现`im2row`，第四篇介绍另一种实现图像和行向量互换的方式，最后实现池化层图像和行向量的互换`pool2row`

1. 数组扩展
2. 数组变形
3. 数组填充
4. 维数转换
5. 矩阵提取
6. 数据叠加

## 数组扩展

`numpy`提供了[numpy.repeat](https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html)以及[numpy.tile](https://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html)实现基于数组原先数据的扩展

`numpy.repeat`会重复数组中的每个元素

>numpy.repeat(a, repeats, axis=None)

* `a`表示输入数组
* `repeats`表示重复次数
* `axis`表示沿着哪个轴进行重复，`axis=0`表示沿着列，`axis=1`表示沿着行（注意：`axis`不能超出`a`的维度）。默认会将输入数组拉平（`flattened`），再沿着行进行重复

```
# 对1-D数组进行扩展
>>> x = np.arange(3)
>>> x
array([0, 1, 2])
>>> np.repeat(x, 4)
array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
>>> np.repeat(x, 4, axis=0) # 当x仅有1维时，axis失效
array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
```

```
# 对2-D数组进行扩展
>>> x = np.arange(6).reshape(2,3)
>>> x
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.repeat(x, 2) # 默认情况下会先将输入数组拉平，再进行扩展
array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
>>> np.repeat(x, 2, axis=0) # axis=0表示沿着列进行扩展
array([[0, 1, 2],
       [0, 1, 2],
       [3, 4, 5],
       [3, 4, 5]])
>>> np.repeat(x, 2, axis=1) # axis=1表示沿着行进行扩展
array([[0, 0, 1, 1, 2, 2],
       [3, 3, 4, 4, 5, 5]])
>>> np.repeat(x, (2,3), axis=0) # 还可以指定扩展个数，本例表示沿着列进行扩展，其中输入数组第一行扩展2次，第二行扩展3次
array([[0, 1, 2],
       [0, 1, 2],
       [3, 4, 5],
       [3, 4, 5],
       [3, 4, 5]])
```

**`np.repeat`对逐元素进行重复，而`np.tile`对逐数组进行重复**

>numpy.tile(A, reps)

* `A`表示输入数组
* `reps`表示重复次数

有两种情况

1. 当`A.ndim < reps.ndim`时，输入数组会扩展到reps相同维数，比如`A.shape=(3,)`，那么扩展到`2-D`就是`(3,) -> (1,3)`，扩展到`3-D`就是`(3,) -> (1,1,3)`
2. 当`A.ndim > reps.ndim`时，`reps`会先扩展到和A相同维数，扩展维数用`1`填充，比如`A.shape = (2,3,4,5)，reps.shape = (2,3)`，那么`reps`首先扩展到`(1,1,2,3)`，再对`A`进行扩展，结果为`(2,3,8,15)`

```
>>> a = np.arange(3)
>>> a
array([0, 1, 2])
>>> a.shape
(3,)
>>> x
array([[0, 1, 2],
       [3, 4, 5]])
```

```
# A.ndim = reps.ndim，不改变数组维数，沿着维数方向进行扩展
>>> np.tile(a, 3) 
array([0, 1, 2, 0, 1, 2, 0, 1, 2])
>>> np.tile(a, (2,3))
array([[0, 1, 2, 0, 1, 2, 0, 1, 2],
       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
>>> np.tile(x, 2)
array([[0, 1, 2, 0, 1, 2],
       [3, 4, 5, 3, 4, 5]])
>>> np.tile(x, (2,2))
array([[0, 1, 2, 0, 1, 2],
       [3, 4, 5, 3, 4, 5],
       [0, 1, 2, 0, 1, 2],
       [3, 4, 5, 3, 4, 5]])
```

```
# A.ndim < reps.ndim，扩展A维数，再进行扩展
>>> np.tile(a, (3,1)) # 扩展到2-D，再沿着列方向进行扩展
array([[0, 1, 2],
       [0, 1, 2],
       [0, 1, 2]])
>>> np.tile(a, (1,3)) # 扩展到2-D，再沿着行方向进行扩展
array([[0, 1, 2, 0, 1, 2, 0, 1, 2]])
```

```
# A.ndim > reps.ndim, 先扩展reps维度，再扩展A数组
>>> A=np.ones((2,3,4,5))
>>> reps=(2,3)
>>> np.tile(A, reps).shape
(2, 3, 8, 15)
```

## 数组变形

`numpy`提供了方法[numpy.reshape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)用于数组变形

`numpy.reshape`在不改变数组数据的情况下变换数组大小

>numpy.reshape(a, newshape, order='C')

* `a`表示输入数组
* `newshape`表示新的数组大小，其中一个维度值可以为`-1`，这样该维度大小将通过数组长度和剩余维数判断得出。**注意：新的数组大小应该和输入数组大小一致**
* `order`表示索引读取元素顺序，默认读取方式和`C`语言一致（大多数情况下就是这种方式）：最后一个轴索引（`the last axis index`）变化最快，第一个轴索引（`the first axis index`）变化最慢。比如`2-D`数组，先变化第二个轴，也就是按行读取，再变化第一个轴，也就是按列读取

```
>>> x = np.arange(24)
>>> x
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23])

>>> y = x.reshape(2,3,4)
>>> y
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
# 最后一个轴先进行读取
>>> y[0,0,:]
# 然后是中间轴
array([0, 1, 2, 3])
>>> y[0,1,:]
array([4, 5, 6, 7])
>>> y[0,2,:]
array([ 8,  9, 10, 11])
# 最后是第一个轴
>>> y[1,:,:]
array([[12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23]])
```

## 数组填充

在进行卷积神经网络计算时，需要零填充操作，`numpy`提供了[numpy.pad](https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html#numpy-pad)操作

>numpy.pad(array, pad_width, mode, **kwargs)

* `array`表示输入数组
* `pad_width`表示每轴填充的宽度，其格式为`((before_1, after_1), … (before_N, after_N))`。如果所有轴都使用相同填充宽度，可以简化为`(before, after)`，或者`(pad,)`
* `mode`表示填充模式，最常用的是`constant` - 填充一个常数，默认为`0`
* `constant_values`是可选参数，表示常数值

```
>>> a = np.arange(1, 5).reshape(2,2)
>>> a
array([[1, 2],
       [3, 4]])
```

```
# 以下三种方法等价
>>> np.pad(a, (2), mode='constant')
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 2, 0, 0],
       [0, 0, 3, 4, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0]])
>>> np.pad(a, (2,2), mode='constant')
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 2, 0, 0],
       [0, 0, 3, 4, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0]])
>>> np.pad(a, ((2,2),(2,2)), mode='constant')
array([[0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 1, 2, 0, 0],
       [0, 0, 3, 4, 0, 0],
       [0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0]])
```

```
#可以指定不同的轴填充个数
>>> a
array([[[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]],

       [[13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]]])
>>> a.shape
(2, 3, 4)
# 仅填充最后一轴
>>> np.pad(a, ((0,),(0,),(1,)), mode='constant')
array([[[ 0,  1,  2,  3,  4,  0],
        [ 0,  5,  6,  7,  8,  0],
        [ 0,  9, 10, 11, 12,  0]],

       [[ 0, 13, 14, 15, 16,  0],
        [ 0, 17, 18, 19, 20,  0],
        [ 0, 21, 22, 23, 24,  0]]])
>>> np.pad(a, ((0,),(0,),(1,)), mode='constant').shape
(2, 3, 6)
```

## 维数转换

`numpy`提供了[numpy.transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)用于维数转换

>numpy.transpose(a, axes=None)

* `a`表示输入数组
* `axes`表示待转换的轴，是`int`类型的元组（`tuple`）

```
# 2维数组转换相当于矩阵转置操作
>>> x = np.arange(12).reshape(3,4)
>>> x
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> x.shape
(3, 4)
>>> np.transpose(x, (1,0))
array([[ 0,  4,  8],
       [ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11]])
>>> np.transpose(x, (1,0)).shape
(4, 3)
```

```
>>> x = np.arange(24).reshape(2,3,4)
>>> x
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
>>> x.shape
(2, 3, 4)
>>> np.transpose(x, (1,2,0))
array([[[ 0, 12],
        [ 1, 13],
        [ 2, 14],
        [ 3, 15]],

       [[ 4, 16],
        [ 5, 17],
        [ 6, 18],
        [ 7, 19]],

       [[ 8, 20],
        [ 9, 21],
        [10, 22],
        [11, 23]]])
>>> np.transpose(x, (1,2,0)).shape
(3, 4, 2)
>>> np.transpose(x, (1,0,2))
array([[[ 0,  1,  2,  3],
        [12, 13, 14, 15]],

       [[ 4,  5,  6,  7],
        [16, 17, 18, 19]],

       [[ 8,  9, 10, 11],
        [20, 21, 22, 23]]])
>>> np.transpose(x, (1,0,2)).shape
(3, 2, 4)
```

## 矩阵提取

当在多维数组中取一个`2`维矩阵时，通常会使用切片方式

```
>>> a
array([[[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]],

       [[13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]]])
>>> a[0, 0:2, 0:2]
array([[1, 2],
       [5, 6]])
>>> a[0, slice(0, 2, 1), slice(0, 2, 1)]
array([[1, 2],
       [5, 6]])
```

还可以先获取矩阵对应数据下标，再进行计算

```
>>> x
[[0, 0], [1, 1]]
>>> y
[[0, 1], [0, 1]]
>>> a[0, x, y]
array([[1, 2],
       [5, 6]])
```

```
# 结合矩阵扩展和维数转换方法
>>> z
array([0, 1])
>>> x = np.tile(z, (2,1))
>>> x
array([[0, 1],
       [0, 1]])
>>> x = np.transpose(x, (1,0))
>>> x
array([[0, 0],
       [1, 1]])
>>> y = np.tile(z, (2,1))
>>> y
array([[0, 1],
       [0, 1]])
>>> a[0, x, y]
array([[1, 2],
       [5, 6]])
```

## 数据叠加

参考：[np.add.at indexing with array](https://stackoverflow.com/questions/45473896/np-add-at-indexing-with-array)

`numpy`提供函数[numpy.add.at](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html)用于批量添加数据

>ufunc.at(a, indices, b=None)

* `a`表示目标数组
* `indices`表示指定下标，可以是数组或者元组结构
* `b`是添加的数据，可以是标量或者数组，必须能够在`a`上可广播（`broadcastable`）

`np.add.at`函数功能等同于`a[indices]=b`，区别在于`np.add.at`执行无缓冲即时操作（`unbuffered in place operation`），也就是说同一个下标可以累加多次

```
>>> x = np.arange(5)
>>> x
array([0, 1, 2, 3, 4])
>>> np.add.at(x, [0, 1, 2, 3, 3], 2) # 累加下标3共2次
>>> x
array([2, 3, 4, 7, 4])
```

```
>>> x = np.zeros((2,3))
>>> x
array([[0., 0., 0.],
       [0., 0., 0.]])

>>> np.add.at(x, ([0,1], [0,2]), 3) # 指定行列坐标
>>> x
array([[3., 0., 0.],
       [0., 0., 3.]])
>>> np.add.at(x, ([1], [0,2]), 1)
>>> x
array([[3., 0., 0.],
       [1., 0., 4.]])
>>> np.add.at(x, ([1], [0,2]), [5,6])
>>> x
array([[ 3.,  0.,  0.],
       [ 6.,  0., 10.]])
```