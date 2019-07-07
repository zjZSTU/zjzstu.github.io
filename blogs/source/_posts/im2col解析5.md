---
title: im2col解析5
categories:
  - [算法]
  - [编程]
tags:
  - 深度学习
  - python
abbrlink: 5e1da4ba
date: 2019-05-26 14:05:27
---

前面实现了卷积层和全连接层的相互转换，下面实现池化层和全连接层的相互转换

## 原理解析

**池化层操作和卷积层操作的不同之处在于池化层操作没有零填充，不包含深度，它仅对输入数据体的每一个激活图进行`2`维操作**

比如输入数据体大小为$128\times 6\times 28\times 28$，池化层过滤器空间尺寸为$2\times 2$，步长为$2$

$$
(28 - 2)/2 + 1=14
$$

那么输出数据体大小为$128\times 6\times 14\times 14$

## 代码实现

```
def get_pool2row_indices(x_shape, field_height, field_width, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H - field_height) % stride == 0
    assert (W - field_height) % stride == 0
    out_height = int((H - field_height) / stride + 1)
    out_width = int((W - field_width) / stride + 1)

    # 行坐标
    i0 = stride * np.repeat(np.arange(out_height), out_width)
    i0 = np.tile(i0, C)
    i1 = np.repeat(np.arange(field_height), field_width)

    # 列坐标
    j0 = stride * np.tile(np.arange(out_width), out_height * C)
    j1 = np.tile(np.arange(field_width), field_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), out_height * out_width).reshape(-1, 1)

    return (k, i, j)


def pool2row_indices(x, field_height, field_width, stride=1):
    k, i, j = get_pool2row_indices(x.shape, field_height, field_width, stride)

    rows = x[:, k, i, j]
    C = x.shape[1]
    # 逐图像采集
    rows = rows.reshape(-1, field_height * field_width)
    return rows


def row2pool_indices(rows, x_shape, field_height=2, field_width=2, stride=2, isstinct=False):
    N, C, H, W = x_shape
    x = np.zeros(x_shape, dtype=rows.dtype)
    k, i, j = get_pool2row_indices(x_shape, field_height, field_width, stride)
    rows_reshaped = rows.reshape(N, -1, field_height * field_width)
    np.add.at(x, (slice(None), k, i, j), rows_reshaped)

    if isstinct and (stride < field_height or stride < field_width):
        x_ones = np.ones(x.shape)
        rows_ones = x_ones[:, k, i, j]
        x_zeros = np.zeros(x.shape)
        np.add.at(x_zeros, (slice(None), k, i, j), rows_ones)
        return x / x_zeros

    return x
```

修改如下：

1. 行向量大小为`field_width * field_height`，每个激活图的行/列坐标矩阵，所以在坐标矩阵列方向重复
2. 当步长等于滤波器长宽时，行向量转图像得到的就是原来的数据体，不需要进一步转换