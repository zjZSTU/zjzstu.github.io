---
title: im2col解析3
categories:
  - [深度学习]
  - [编程, 编程语言]
  - [编程, 代码库]
tags:
  - 卷积神经网络
  - python
  - numpy
abbrlink: b77e018f
date: 2019-05-25 14:05:25
---

前面实现了图像转列向量，在之前推导过程中使用的是行向量，所以修改`im2col.py`，实现`im2row`的功能

卷积核大小为$2\times 2$，步长为`1`，零填充为`0`

* field_height = 2
* field_width = 2
* stride = 1
* padding = 0

`2`维图像大小为$3\times 3$，3维图像大小为$2\times 3\times 3$，4维图像大小为$2\times 2\times 3\times 3$

所以输出数据体的空间尺寸为$2\times 2$，深度为`2`，数量为`2`

* out_height = 2
* out_width = 2
* depth = 2
* N = 2

## 图像转行向量

### 2维图像

```
>>> a = np.arange(9).reshape(3,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
```

#### 行坐标矩阵

对于行坐标矩阵而言，每一行表示一个局部连接矩阵

局部连接矩阵中同一行的行坐标相等，相邻行的行坐标加`1`

```
# i1 = np.repeat(np.arange(field_height), field_width)
>>> i1 = np.repeat(np.arange(2), 2)
>>> i1
array([0, 0, 1, 1])
```

行坐标矩阵的列数表示局部连接的个数

图像中同一行局部连接的行坐标相等，相邻行之间的局部连接行坐标相差`stride`

```
# i0 = stride * np.repeat(np.arange(out_height), out_width)
>>> i0 = 1 * np.repeat(np.arange(2), 2)
>>> i0
array([0, 0, 1, 1])
```

计算行坐标矩阵

```
# i = i0.reshape(-1, 1) + i1.reshape(1, -1)
>>> i = i0.reshape(-1, 1) + i1.reshape(1, -1)
>>> i
array([[0, 0, 1, 1],
       [0, 0, 1, 1],
       [1, 1, 2, 2],
       [1, 1, 2, 2]])
```

#### 列坐标矩阵

对于列坐标矩阵而言，每一行表示一个局部连接矩阵

局部连接矩阵中同一列的列坐标相等，相邻列的列坐标加`1`

```
# j1 = np.tile(np.arange(field_width), field_height)
>>> j1 = np.tile(np.arange(2), 2)
>>> j1
array([0, 1, 0, 1])
```

列坐标矩阵的列数表示局部连接的个数

图像中同一行的相邻局部连接相差`stride`距离，同一列的局部连接距离该行最左端的距离相等

```
# j0 = stride * np.tile(np.arange(out_width), out_height)
>>> j0 = 1 * np.tile(np.arange(2), 2)
>>> j0
array([0, 1, 0, 1])
```

计算列坐标矩阵

```
# j = j0.reshape(-1, 1) + j1.reshape(1, -1)
>>> j = j0.reshape(-1, 1) + j1.reshape(1, -1)
>>> j
array([[0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2]])
```

#### 行向量

```
>>> a[i,j]
array([[0, 1, 3, 4],
       [1, 2, 4, 5],
       [3, 4, 6, 7],
       [4, 5, 7, 8]])
```

### 3维图像

```
>>> a = np.arange(18).reshape(2,3,3)
>>> a
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]]])
```

#### 行坐标矩阵

多通道图像仅改变单个局部连接矩阵大小，不改变数量

并且单个局部连接在每个通道的行坐标相同

```
# i1 = np.repeat(np.arange(field_height), field_width)
# i1 = np.tile(i1, C)
>>> i1 = np.repeat(np.arange(2), 2)
>>> i1
array([0, 0, 1, 1])
>>> i1 = np.tile(i1, 2)
>>> i1
array([0, 0, 1, 1, 0, 0, 1, 1])
```

计算行坐标矩阵

```
>>> i0 = 1 * np.repeat(np.arange(2), 2)
>>> i0
array([0, 0, 1, 1])
>>> i = i0.reshape(-1, 1) + i1.reshape(1, -1)
>>> i
array([[0, 0, 1, 1, 0, 0, 1, 1],
       [0, 0, 1, 1, 0, 0, 1, 1],
       [1, 1, 2, 2, 1, 1, 2, 2],
       [1, 1, 2, 2, 1, 1, 2, 2]])
```

#### 列坐标矩阵

多通道图像仅改变单个局部连接矩阵大小，不改变数量

并且单个局部连接在每个通道的列坐标相同

```
# j1 = np.tile(np.arange(field_width), field_height * C)
>>> j1 = np.tile(np.arange(2), 2*2)
>>> j1
array([0, 1, 0, 1, 0, 1, 0, 1])
```

计算列坐标矩阵

```
>>> j0 = 1 * np.tile(np.arange(2), 2)
>>> j = j0.reshape(-1, 1) + j1.reshape(1, -1)
>>> j
array([[0, 1, 0, 1, 0, 1, 0, 1],
       [1, 2, 1, 2, 1, 2, 1, 2],
       [0, 1, 0, 1, 0, 1, 0, 1],
       [1, 2, 1, 2, 1, 2, 1, 2]])
```

#### 通道向量

需要指定哪个通道图像进行数据提取

```
# k = np.repeat(np.arange(C), field_width * field_height).reshape(-1, 1)
>>> k = np.repeat(np.arange(2), 2*2).reshape(1,-1)
>>> k
array([[0, 0, 0, 0, 1, 1, 1, 1]])
```

#### 行向量

```
>>> a[k,i,j]
array([[ 0,  1,  3,  4,  9, 10, 12, 13],
       [ 1,  2,  4,  5, 10, 11, 13, 14],
       [ 3,  4,  6,  7, 12, 13, 15, 16],
       [ 4,  5,  7,  8, 13, 14, 16, 17]])
```

### 4维图像

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
```

对于批量图像进行行向量转换

```
>>> rows = a[:,k,i,j]
>>> rows
array([[[ 0,  1,  3,  4,  9, 10, 12, 13],
        [ 1,  2,  4,  5, 10, 11, 13, 14],
        [ 3,  4,  6,  7, 12, 13, 15, 16],
        [ 4,  5,  7,  8, 13, 14, 16, 17]],

       [[18, 19, 21, 22, 27, 28, 30, 31],
        [19, 20, 22, 23, 28, 29, 31, 32],
        [21, 22, 24, 25, 30, 31, 33, 34],
        [22, 23, 25, 26, 31, 32, 34, 35]]])
```

还需要进一步变形

```
>>> rows.shape
(2, 4, 8)
>>> rows.reshape(-1, 8)
array([[ 0,  1,  3,  4,  9, 10, 12, 13],
       [ 1,  2,  4,  5, 10, 11, 13, 14],
       [ 3,  4,  6,  7, 12, 13, 15, 16],
       [ 4,  5,  7,  8, 13, 14, 16, 17],
       [18, 19, 21, 22, 27, 28, 30, 31],
       [19, 20, 22, 23, 28, 29, 31, 32],
       [21, 22, 24, 25, 30, 31, 33, 34],
       [22, 23, 25, 26, 31, 32, 34, 35]])
```

**最终实现结果的采样方式：逐图像按照从左到右、从上到下的顺序采集局部连接矩阵**

如果要实现逐坐标的采集局部连接矩阵，需要先进行维数转换，再完成变形

```
>>> rows = np.transpose(rows, (1,0,2))
>>> rows.shape
(4, 2, 8)
>>> rows.reshape(-1, 8)
array([[ 0,  1,  3,  4,  9, 10, 12, 13],
       [18, 19, 21, 22, 27, 28, 30, 31],
       [ 1,  2,  4,  5, 10, 11, 13, 14],
       [19, 20, 22, 23, 28, 29, 31, 32],
       [ 3,  4,  6,  7, 12, 13, 15, 16],
       [21, 22, 24, 25, 30, 31, 33, 34],
       [ 4,  5,  7,  8, 13, 14, 16, 17],
       [22, 23, 25, 26, 31, 32, 34, 35]])
```

## 行向量转图像

### 2维图像

已知图像大小，行/列坐标矩阵和行向量矩阵，用`numpy.add.at`就能完成映射

```
>>> b = np.zeros(a.shape)
>>> np.add.at(b, (i,j), rows)
>>> b
array([[ 0.,  2.,  2.],
       [ 6., 16., 10.],
       [ 6., 14.,  8.]])
```

计算叠加倍数，得到原始图像

```
>>> c = np.ones(a.shape)
>>> rows_c = c[i,j]
>>> d = np.zeros(c.shape)
>>> np.add.at(d, (i,j), rows_c)
>>> d
array([[1., 2., 1.],
       [2., 4., 2.],
       [1., 2., 1.]])
```

```
>>> b/d
array([[0., 1., 2.],
       [3., 4., 5.],
       [6., 7., 8.]])
```

#### 3维图像

已知图像大小，行/列坐标矩阵、深度向量和行向量矩阵，用`numpy.add.at`就能完成映射

```
>>> b = np.zeros(a.shape)
>>> np.add.at(b, (k,i,j), rows)
>>> b
array([[[ 0.,  2.,  2.],
        [ 6., 16., 10.],
        [ 6., 14.,  8.]],

       [[ 9., 20., 11.],
        [24., 52., 28.],
        [15., 32., 17.]]])
```

计算叠加倍数，得到原始图像

```
>>> c = np.ones(a.shape)
>>> rows_c = c[k,i,j]
>>> d = np.zeros(a.shape)
>>> np.add.at(d, (k,i,j), rows_c)
>>> d
array([[[1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.]],

       [[1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.]]])
```

```
>>> b/d
array([[[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.]],

       [[ 9., 10., 11.],
        [12., 13., 14.],
        [15., 16., 17.]]])
```

#### 4维图像

已知图像大小，行/列坐标矩阵、深度向量和行向量矩阵，用`numpy.add.at`就能完成映射

```
>>> rows
array([[ 0,  1,  3,  4,  9, 10, 12, 13],
       [ 1,  2,  4,  5, 10, 11, 13, 14],
       [ 3,  4,  6,  7, 12, 13, 15, 16],
       [ 4,  5,  7,  8, 13, 14, 16, 17],
       [18, 19, 21, 22, 27, 28, 30, 31],
       [19, 20, 22, 23, 28, 29, 31, 32],
       [21, 22, 24, 25, 30, 31, 33, 34],
       [22, 23, 25, 26, 31, 32, 34, 35]])
>>> rows_reshaped = rows.reshape(2, 4, 8)
>>> rows_reshaped
array([[[ 0,  1,  3,  4,  9, 10, 12, 13],
        [ 1,  2,  4,  5, 10, 11, 13, 14],
        [ 3,  4,  6,  7, 12, 13, 15, 16],
        [ 4,  5,  7,  8, 13, 14, 16, 17]],

       [[18, 19, 21, 22, 27, 28, 30, 31],
        [19, 20, 22, 23, 28, 29, 31, 32],
        [21, 22, 24, 25, 30, 31, 33, 34],
        [22, 23, 25, 26, 31, 32, 34, 35]]])
>>> b = np.zeros(a.shape)
>>> np.add.at(b, (slice(None),k,i,j), rows_reshaped)
>>> b
array([[[[  0.,   2.,   2.],
         [  6.,  16.,  10.],
         [  6.,  14.,   8.]],

        [[  9.,  20.,  11.],
         [ 24.,  52.,  28.],
         [ 15.,  32.,  17.]]],


       [[[ 18.,  38.,  20.],
         [ 42.,  88.,  46.],
         [ 24.,  50.,  26.]],

        [[ 27.,  56.,  29.],
         [ 60., 124.,  64.],
         [ 33.,  68.,  35.]]]])
```

计算叠加倍数，得到原始图像

```
>>> c = np.ones(a.shape)
>>> cols_c = c[:,k,i,j]
>>> d = np.zeros(a.shape)
>>> np.add.at(d, (slice(None),k,i,j), cols_c)
>>> d
array([[[[1., 2., 1.],
         [2., 4., 2.],
         [1., 2., 1.]],

        [[1., 2., 1.],
         [2., 4., 2.],
         [1., 2., 1.]]],


       [[[1., 2., 1.],
         [2., 4., 2.],
         [1., 2., 1.]],

        [[1., 2., 1.],
         [2., 4., 2.],
         [1., 2., 1.]]]])
```

```
>>> b/d
array([[[[ 0.,  1.,  2.],
         [ 3.,  4.,  5.],
         [ 6.,  7.,  8.]],

        [[ 9., 10., 11.],
         [12., 13., 14.],
         [15., 16., 17.]]],


       [[[18., 19., 20.],
         [21., 22., 23.],
         [24., 25., 26.]],

        [[27., 28., 29.],
         [30., 31., 32.],
         [33., 34., 35.]]]])
```

## im2row

仿照`im2col.py`，`im2row`实现如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-5-25 下午4:17
# @Author  : zj

from builtins import range
import numpy as np


def get_im2row_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    # 行坐标
    i0 = stride * np.repeat(np.arange(out_height), out_width)
    i1 = np.repeat(np.arange(field_height), field_width)
    i1 = np.tile(i1, C)

    # 列坐标
    j0 = stride * np.tile(np.arange(out_width), out_height)
    j1 = np.tile(np.arange(field_width), field_height * C)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(1, -1)

    return (k, i, j)


def im2row_indices(x, field_height, field_width, padding=1, stride=1):
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2row_indices(x.shape, field_height, field_width, padding, stride)

    rows = x_padded[:, k, i, j]
    C = x.shape[1]
    # 逐图像采集
    rows = rows.reshape(-1, field_height * field_width * C)
    return rows


def row2im_indices(rows, x_shape, field_height=3, field_width=3, padding=1, stride=1, isstinct=False):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=rows.dtype)
    k, i, j = get_im2row_indices(x_shape, field_height, field_width, padding,
                                 stride)
    rows_reshaped = rows.reshape(N, -1, C * field_height * field_width)
    np.add.at(x_padded, (slice(None), k, i, j), rows_reshaped)

    if isstinct:
        # 计算叠加倍数，恢复原图
        x_ones = np.ones(x_padded.shape)
        rows_ones = x_ones[:, k, i, j]
        x_zeros = np.zeros(x_padded.shape)
        np.add.at(x_zeros, (slice(None), k, i, j), rows_ones)
        x_padded = x_padded / x_zeros

    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]


if __name__ == '__main__':
    pass
```

修改如下：

1. 实现图像转行向量
2. 实现逐图像的行向量转换（*`im2cols.py`实现的是逐坐标的列向量转换*）
3. 添加行向量转换回原图功能（*符号位`isstinct`*）

## 卷积层和全连接层相互转换

批量图像数据大小为$2\times 3\times 4\times 4$，卷积核大小为$3\times 3$，步长为$1$，零填充为0

单个局部连接矩阵大小为$3\times 3\times 3=27$，共有8个

行向量矩阵大小为$8\times 27$

```
x = np.arange(96).reshape(2, 3, 4, 4)
# print(x)
rows = im2row_indices(x, 3, 3, padding=0, stride=1)
print(rows)
print(rows.shape)
# output = row2im_indices(rows, x.shape, field_height=3, field_width=3, padding=0, stride=1)
# print(output)
output = row2im_indices(rows, x.shape, field_height=3, field_width=3, padding=0, stride=1, isstinct=True)
print(output)
```

```
[[ 0  1  2  4  5  6  8  9 10 16 17 18 20 21 22 24 25 26 32 33 34 36 37 38
  40 41 42]
 [ 1  2  3  5  6  7  9 10 11 17 18 19 21 22 23 25 26 27 33 34 35 37 38 39
  41 42 43]
 [ 4  5  6  8  9 10 12 13 14 20 21 22 24 25 26 28 29 30 36 37 38 40 41 42
  44 45 46]
 [ 5  6  7  9 10 11 13 14 15 21 22 23 25 26 27 29 30 31 37 38 39 41 42 43
  45 46 47]
 [48 49 50 52 53 54 56 57 58 64 65 66 68 69 70 72 73 74 80 81 82 84 85 86
  88 89 90]
 [49 50 51 53 54 55 57 58 59 65 66 67 69 70 71 73 74 75 81 82 83 85 86 87
  89 90 91]
 [52 53 54 56 57 58 60 61 62 68 69 70 72 73 74 76 77 78 84 85 86 88 89 90
  92 93 94]
 [53 54 55 57 58 59 61 62 63 69 70 71 73 74 75 77 78 79 85 86 87 89 90 91
  93 94 95]]
(8, 27)
[[[[ 0.  1.  2.  3.]
   [ 4.  5.  6.  7.]
   [ 8.  9. 10. 11.]
   [12. 13. 14. 15.]]
  [[16. 17. 18. 19.]
   [20. 21. 22. 23.]
   [24. 25. 26. 27.]
   [28. 29. 30. 31.]]
  [[32. 33. 34. 35.]
   [36. 37. 38. 39.]
   [40. 41. 42. 43.]
   [44. 45. 46. 47.]]]
 [[[48. 49. 50. 51.]
   [52. 53. 54. 55.]
   [56. 57. 58. 59.]
   [60. 61. 62. 63.]]
  [[64. 65. 66. 67.]
   [68. 69. 70. 71.]
   [72. 73. 74. 75.]
   [76. 77. 78. 79.]]
  [[80. 81. 82. 83.]
   [84. 85. 86. 87.]
   [88. 89. 90. 91.]
   [92. 93. 94. 95.]]]]
```