---
title: im2col解析2
categories:
  - [算法, 深度学习, 卷积神经网络]
  - [编程, 编程语言]
  - [编程, 代码库]
tags:
  - python
  - numpy
abbrlink: 597060a3
date: 2019-05-25 12:39:38
---

参考：

[在 Caffe 中如何计算卷积？](https://www.zhihu.com/question/28385679)

[卷积神经网络推导-批量图片矩阵计算](https://www.zhujian.tech/posts/ab1e719c.html#more)

`im2col`表示`image to column`，将图像转换成列向量

**卷积操作步骤**：首先将卷积核映射到`x_padded`左上角，然后沿着行方向操作，每次滑动`stride`距离；到达最右端后，将卷积核往列方向滑动`stride`距离，再实现从左到右的滑动

## 图像转列向量

在以下操作中，假设感受野大小为`field_height = field_width = 2`，零填充`padding = 0`，步长`stride = 2`

### 2维图像

**以(3,3)大小矩阵为例**

```
>>> a = np.arange(9).reshape(3,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
```

那么得到的局部连接数为

$$
(3 - 2 + 2*0)/1 + 1 = 2\\
num = 2*2 = 4
$$

所以共有4个局部连接，分别是

$$
\begin{bmatrix}
0 & 1\\ 
3 & 4
\end{bmatrix}\ \ 
\begin{bmatrix}
1 & 2\\ 
4 & 5
\end{bmatrix}\ \  
\begin{bmatrix}
3 & 4\\ 
6 & 7
\end{bmatrix}
\begin{bmatrix}
4 & 5\\ 
7 & 8
\end{bmatrix}
$$

其坐标分别为

$$
\begin{bmatrix}
(0,0) & (0,1)\\ 
(1,0) & (1,1)
\end{bmatrix}\ \ 
\begin{bmatrix}
(0,1) & (0,2)\\
(1,1) & (1,2)
\end{bmatrix}\ \  
\begin{bmatrix}
(1,0) & (1,1)\\
(2,0) & (2,1)
\end{bmatrix}
\begin{bmatrix}
(1,1) & (1,2)\\
(2,1) & (2,2)
\end{bmatrix}
$$

将其列向量化，可得

$$
matrix=
\begin{bmatrix}
0 & 1 & 3 & 4\\ 
1 & 2 & 4 & 5\\ 
3 & 4 & 6 & 7\\ 
4 & 5 & 7 & 8
\end{bmatrix}
$$

$$
indexs=
\begin{bmatrix}
(0,0) & (0,1) & (1,0) & (1,1)\\ 
(0,1) & (0,2) & (1,1) & (1,2)\\ 
(1,0) & (1,1) & (2,0) & (2,1)\\ 
(1,1) & (1,2) & (2,1) & (2,2)
\end{bmatrix}
$$

进行行列坐标分离

$$
rows_{index}=
\begin{bmatrix}
0 & 0 & 1 & 1\\ 
0 & 0 & 1 & 1\\ 
1 & 1 & 2 & 2\\ 
1 & 1 & 2 & 2
\end{bmatrix}
$$

$$
columns_{index}=
\begin{bmatrix}
0 & 1 & 0 & 1\\ 
1 & 2 & 1 & 2\\ 
0 & 1 & 0 & 1\\ 
1 & 2 & 1 & 2
\end{bmatrix}
$$

对卷积核而言，每行的行坐标一致，共有`field_width`个，每个卷积核有`field_height`行

```
# i0 = np.repeat(np.arange(field_height), field_width)
>>> i0 = np.repeat(np.arange(2), 2)
>>> i0
array([0, 0, 1, 1])
```

每行共有`out_width`个局部连接矩阵，每个矩阵相隔`stride`，共有`out_height`行

```
# i1 = stride * np.repeat(np.arange(out_height), out_width)
>>> i1 = 1 * np.repeat(np.arange(2), 2)
>>> i1
array([0, 0, 1, 1])
```

对于局部连接矩阵的行坐标为

```
# i = i0.reshape(-1, 1) + i1.reshape(1, -1)
>>> i0.reshape(-1,1)
array([[0],
       [0],
       [1],
       [1]])
>>> i1.reshape(1,-1)
array([[0, 0, 1, 1]])
>>> i = i0.reshape(-1,1)+i1.reshape(1,-1)
>>> i
array([[0, 0, 1, 1],
       [0, 0, 1, 1],
       [1, 1, 2, 2],
       [1, 1, 2, 2]])
```

同样的，对于卷积核的列来说，其相邻列相差`1`，长`field_width`，共有`field_height`行

```
# j0 = np.tile(np.arange(field_width), field_height * C)
>>> j0 = np.tile(np.arange(2),2)
>>> j0
array([0, 1, 0, 1])
```

每行有`out_width`个局部连接矩阵，矩阵之间相差`stride`步长，同一列矩阵相对于该行最左侧的距离相同

```
# j1 = stride * np.tile(np.arange(out_width), out_height)
>>> j1 = 1*np.tile(np.arange(2), 2)
>>> j1
array([0, 1, 0, 1])
```

计算局部连接矩阵的行坐标

```
# j = j0.reshape(-1, 1) + j1.reshape(1, -1)
>>> j0.reshape(-1,1)
array([[0],
       [1],
       [0],
       [1]])
>>> j1.reshape(1,-1)
array([[0, 1, 0, 1]])
>>> j = j0.reshape(-1,1) + j1.reshape(1,-1)
>>> j
array([[0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2]])
```

得到列向量矩阵的行坐标和列坐标后，求取局部连接矩阵的列向量矩阵

```
>>> a[i,j]
array([[0, 1, 3, 4],
       [1, 2, 4, 5],
       [3, 4, 6, 7],
       [4, 5, 7, 8]])
```

### 3维图像

如果图像有多通道，每个通道图像的卷积操作一致，局部连接总数不变，仅扩展每个卷积矩阵的大小，所以仅需在行/列坐标矩阵的列方向扩展即可

比如有$2\times 3\times 3$大小图像，通道数为`2`

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

局部连接矩阵大小为$2\times 2\times 2$，比如

```
array([[[ 0,  1],
        [ 3,  4],

       [[ 9, 10],
        [12, 13]]])
```

对于行坐标矩阵

```
>>> i0 = np.repeat(np.arange(2), 2)
>>> i0
array([0, 0, 1, 1])
>>> i0 = np.tile(i0, 2)
>>> i0
array([0, 0, 1, 1, 0, 0, 1, 1])

>>> i1 = 1 * np.repeat(np.arange(2), 2)
>>> i1
array([0, 0, 1, 1])

>>> i = i0.reshape(-1,1) + i1.reshape(1, -1)
>>> i
array([[0, 0, 1, 1],
       [0, 0, 1, 1],
       [1, 1, 2, 2],
       [1, 1, 2, 2],
       [0, 0, 1, 1],
       [0, 0, 1, 1],
       [1, 1, 2, 2],
       [1, 1, 2, 2]])
```

对于列坐标矩阵

```
>>> j0 = np.tile(np.arange(2), 2*2)
>>> j0
array([0, 1, 0, 1, 0, 1, 0, 1])
>>> j1 = 1 * np.tile(np.arange(2), 2)
>>> j1
array([0, 1, 0, 1])
>>> j = j0.reshape(-1, 1) + j1.reshape(1, -1)
>>> j
array([[0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2]])
```

**还需要计算通道向量k，用于指定哪个通道图像**

```
>>> k = np.repeat(np.arange(2), 2*2).reshape(-1,1)
>>> k
array([[0],
       [0],
       [0],
       [0],
       [1],
       [1],
       [1],
       [1]])
```

最后求取局部连接矩阵的列向量矩阵

```
>>> a[k,i,j]
array([[ 0,  1,  3,  4],
       [ 1,  2,  4,  5],
       [ 3,  4,  6,  7],
       [ 4,  5,  7,  8],
       [ 9, 10, 12, 13],
       [10, 11, 13, 14],
       [12, 13, 15, 16],
       [13, 14, 16, 17]])
```

### 4维图像

批量处理多通道图像，比如批量图像数据大小为$2\times 2\times 3\times 3$，共`2`张图片，每张图像`2`通道，大小为$3\times 3$

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

对于行/列坐标矩阵`i,j`以及通道向量`k`与`3`维图像操作一致

```
>>> a[:,k,i,j]
array([[[ 0,  1,  3,  4],
        [ 1,  2,  4,  5],
        [ 3,  4,  6,  7],
        [ 4,  5,  7,  8],
        [ 9, 10, 12, 13],
        [10, 11, 13, 14],
        [12, 13, 15, 16],
        [13, 14, 16, 17]],

       [[18, 19, 21, 22],
        [19, 20, 22, 23],
        [21, 22, 24, 25],
        [22, 23, 25, 26],
        [27, 28, 30, 31],
        [28, 29, 31, 32],
        [30, 31, 33, 34],
        [31, 32, 34, 35]]])
>>> a[:,k,i,j].shape
(2, 8, 4)
```

得到的是一个`3`维数据体，第一维表示图像数，第二维表示单个矩阵向量，第三维表示每个图片的局部矩阵数

先进行维数转换，再变形为2维矩阵

```
>>> c = np.transpose(b, (1,2,0))
>>> c.shape
(8, 4, 2)
>>> c.reshape(8, -1)
array([[ 0, 18,  1, 19,  3, 21,  4, 22],
       [ 1, 19,  2, 20,  4, 22,  5, 23],
       [ 3, 21,  4, 22,  6, 24,  7, 25],
       [ 4, 22,  5, 23,  7, 25,  8, 26],
       [ 9, 27, 10, 28, 12, 30, 13, 31],
       [10, 28, 11, 29, 13, 31, 14, 32],
       [12, 30, 13, 31, 15, 33, 16, 34],
       [13, 31, 14, 32, 16, 34, 17, 35]])
```

最后得到了`2`维矩阵，每列表示一个局部连接矩阵向量，其排列方式为依次加入每个图像的相同位置局部连接矩阵，再向左向下滑动（**`im2col.py`实现方式**）

**如果想要先完成单个图像所有局部连接矩阵，再进行下一个图像的转换，可以修改维数变换如下**

```
>>> c = np.transpose(b, (1,0,2))
>>> c.shape
(8, 2, 4)
>>> c.reshape(8, -1)
array([[ 0,  1,  3,  4, 18, 19, 21, 22],
       [ 1,  2,  4,  5, 19, 20, 22, 23],
       [ 3,  4,  6,  7, 21, 22, 24, 25],
       [ 4,  5,  7,  8, 22, 23, 25, 26],
       [ 9, 10, 12, 13, 27, 28, 30, 31],
       [10, 11, 13, 14, 28, 29, 31, 32],
       [12, 13, 15, 16, 30, 31, 33, 34],
       [13, 14, 16, 17, 31, 32, 34, 35]])
```

## 列向量转图像

将图像转列向量小节中得到的列向量矩阵重新映射回图像

### 2维图像

已知图像大小为$3\times 3$，卷积核大小为$2\times 2$，步长为`2`，零填充为`0`

```
>>> a = np.arange(9).reshape(3,3)
>>> a
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
# 列向量矩阵
>>> a[i,j]
array([[0, 1, 3, 4],
       [1, 2, 4, 5],
       [3, 4, 6, 7],
       [4, 5, 7, 8]])
```

根据图像数据和参数获取行/列坐标矩阵

```
>>> i
array([[0, 0, 1, 1],
       [0, 0, 1, 1],
       [1, 1, 2, 2],
       [1, 1, 2, 2]])
>>> j
array([[0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2]])
```

获取`2`维列矩阵

```
>>> cols = a[i,j]
>>> cols
array([[0, 1, 3, 4],
       [1, 2, 4, 5],
       [3, 4, 6, 7],
       [4, 5, 7, 8]])
```

将`2`维列矩阵映射到图像

```
>>> b = np.zeros(a.shape)
>>> b
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
>>> np.add.at(b, (i,j), cols)
>>> b
array([[ 0.,  2.,  2.],
       [ 6., 16., 10.],
       [ 6., 14.,  8.]])
```

**反向映射得到的图像数据和原先图像数据不一致，因为卷积操作中许多下标的位置被多次采集**

如果想要得到原图，**需要除以叠加的倍数**

```
>>> c = np.zeros(a.shape)
>>> c
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
>>> c = np.ones(a.shape)
>>> c
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
>>> cols_c = c[i,j]
>>> cols_c
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
>>> d = np.zeros(c.shape)
>>> d
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
>>> np.add.at(d, (i,j), cols_c)
>>> d
array([[1., 2., 1.],
       [2., 4., 2.],
       [1., 2., 1.]])
>>> b/d
array([[0., 1., 2.],
       [3., 4., 5.],
       [6., 7., 8.]])
>>> b/d == a
array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]])
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

>>> i
array([[0, 0, 1, 1],
       [0, 0, 1, 1],
       [1, 1, 2, 2],
       [1, 1, 2, 2],
       [0, 0, 1, 1],
       [0, 0, 1, 1],
       [1, 1, 2, 2],
       [1, 1, 2, 2]])

>>> j
array([[0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2],
       [0, 1, 0, 1],
       [1, 2, 1, 2]])

>>> k
array([[0],
       [0],
       [0],
       [0],
       [1],
       [1],
       [1],
       [1]])

>>> cols = a[k,i,j]
>>> cols
array([[ 0,  1,  3,  4],
       [ 1,  2,  4,  5],
       [ 3,  4,  6,  7],
       [ 4,  5,  7,  8],
       [ 9, 10, 12, 13],
       [10, 11, 13, 14],
       [12, 13, 15, 16],
       [13, 14, 16, 17]])
```

反向计算图像

```
>>> np.add.at(b, (k, i, j), cols)
>>> b
array([[[ 0.,  2.,  2.],
        [ 6., 16., 10.],
        [ 6., 14.,  8.]],

       [[ 9., 20., 11.],
        [24., 52., 28.],
        [15., 32., 17.]]])
```

除以叠加倍数，转变回原图

```
>>> c = np.ones(a.shape)
>>> cols_c = c[k,i,j]
>>> cols_c
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
>>> d = np.zeros(c.shape)
>>> d
array([[[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

       [[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]])
>>> np.add.at(d, (k,i,j), cols_c)
>>> d
array([[[1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.]],

       [[1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.]]])
>>> b/d
array([[[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.]],

       [[ 9., 10., 11.],
        [12., 13., 14.],
        [15., 16., 17.]]])
>>> b/d == a
array([[[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]],

       [[ True,  True,  True],
        [ True,  True,  True],
        [ True,  True,  True]]])
```

### 4维图像

**最终要实现的是批量图片列向量矩阵的反卷积操作**

从批量图像数据中通过坐标矩阵获取的列向量矩阵是`3`维大小，还需要通过维数转换和变形

列向量转图像需要执行反向操作，首先进行数据变形，再进行维数转换，最后通过坐标矩阵叠加

在上一小节中最后得到了两种排列的列向量矩阵，一种是**先提取同一位置局部连接矩阵**，另一种是**先提取同一图片局部连接矩阵**

如果前向操作如下（第二种）

```
>>> cols = np.transpose(b, (1,2,0))
>>> cols.shape
(8, 4, 2)
>>> cols.reshape(8, -1)
array([[ 0,  1,  3,  4, 18, 19, 21, 22],
       [ 1,  2,  4,  5, 19, 20, 22, 23],
       [ 3,  4,  6,  7, 21, 22, 24, 25],
       [ 4,  5,  7,  8, 22, 23, 25, 26],
       [ 9, 10, 12, 13, 27, 28, 30, 31],
       [10, 11, 13, 14, 28, 29, 31, 32],
       [12, 13, 15, 16, 30, 31, 33, 34],
       [13, 14, 16, 17, 31, 32, 34, 35]])
```

那么反向操作为

```
>>> N=2
>>> cols_reshaped = cols.reshape(cols.shape[0], N, -1)
>>> cols_reshaped.shape
(8, 2, 4)
>>> np.transpose(cols_reshaped, (1,0,2))
array([[[ 0,  1,  3,  4],
        [ 1,  2,  4,  5],
        [ 3,  4,  6,  7],
        [ 4,  5,  7,  8],
        [ 9, 10, 12, 13],
        [10, 11, 13, 14],
        [12, 13, 15, 16],
        [13, 14, 16, 17]],

       [[18, 19, 21, 22],
        [19, 20, 22, 23],
        [21, 22, 24, 25],
        [22, 23, 25, 26],
        [27, 28, 30, 31],
        [28, 29, 31, 32],
        [30, 31, 33, 34],
        [31, 32, 34, 35]]])

```

批量图片大小为$2\times 2\times 3\times 3$，得到最终的反向结果

```
b = np.zeros(a.shape)
>>> np.add.at(b, (slice(None), k, i,j), cols_reshaped)
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

除以叠加倍数，得到最初的图像

```
>>> c = np.ones(a.shape)
>>> cols_c = c[:,k,i,j]
>>> cols_c
array([[[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]],

       [[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]]])
>>> d = np.zeros(c.shape)
>>> d
array([[[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]],


       [[[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]]]])
>>> np.add.at(d, (slice(None), k,i,j), cols_c)
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
>>> b/d == a
array([[[[ True,  True,  True],
         [ True,  True,  True],
         [ True,  True,  True]],

        [[ True,  True,  True],
         [ True,  True,  True],
         [ True,  True,  True]]],


       [[[ True,  True,  True],
         [ True,  True,  True],
         [ True,  True,  True]],

        [[ True,  True,  True],
         [ True,  True,  True],
         [ True,  True,  True]]]])
```

## im2col.py

`im2col.py`实现代码如下

```
from builtins import range
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
```

包含两部分功能：图像转列向量以及列向量转图像

函数`get_im2col_indices`的功能是计算单个图像行/列坐标矩阵以及通道向量

函数`im2col_indices`的功能是实现图像转列向量

函数`col2im_indices`的功能是实现列向量转图像

**注意，`col2im_indices`得到的图像不等于原图，是叠加后的结果**