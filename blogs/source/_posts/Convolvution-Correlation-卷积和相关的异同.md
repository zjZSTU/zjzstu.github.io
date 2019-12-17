---
title: '[Convolvution][Correlation]卷积和相关的异同'
categories:
  - [算法, 图像处理, 图像滤波]
tags:
  - 卷积
  - 相关
abbrlink: b79f94c7
date: 2019-11-22 20:12:51
---

参考：

[卷积运算和相关运算的区别与物理含义?](https://www.zhihu.com/question/32067344)

[通俗理解【卷】积+互相关与卷积](https://blog.csdn.net/Sunny_HQ/article/details/80875664)

[Convolution Vs Correlation](https://stackoverflow.com/questions/20321296/convolution-vs-correlation/37847548#37847548)

学习`OpenCV`的`2`维线性滤波器[filter2D](https://docs.opencv.org/4.1.0/d4/dbd/tutorial_filter_2d.html)，发现一句话

```
Correlation

In a very general sense, correlation is an operation between every part of an image and an operator (kernel).
```

之前有接触过`correlation`(相关)的存在，但是没有仔细理清相关和卷积的异同，以及与之衍生而来的互相关（`cross-correlation`）和滤波（`filter`）的概念

## 卷积

参考：[图像卷积](https://blog.csdn.net/u012005313/article/details/84068337)

$$
(f*g)(t) = \int_{-\infty }^{\infty } f(\tau )g(t - \tau)dt
$$

卷积核需要先翻转`180`度。从信号处理的角度来看，翻转卷积核是为了表现累加信号的过程，即某一时刻的输出不仅包括当前信号的响应，还包括之前信号的响应

对于二维图像而言，图像卷积就是卷积核按步长对图像局部像素块进行加权求和的过程

## 相关

参考：

[Cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation)

[Autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation)

在信号处理中，相关（`corrlation`）是自身序列平移后的相似性度量

$$
R(\tau) = \int_{-\infty }^{\infty }x(t )x(t - \tau)dt
$$

互相关（`cross-correlation`）是两个序列相似性的度量，是一个序列相对于另一个序列位移的函数

$$
(f*g)(t) = \int_{-\infty }^{\infty }f(\tau )g(\tau - t)dt
$$

对于`2`维图像而言，互相关其计算方式和卷积一样，利用模板滑动图像进行点积操作，但不需要翻转卷积核

## 小结

>* 卷积运算，反映了事物的相互作用，并且这种相互作用受制于同一个影响因子。
>* 相关运算，在于反应已有事物的内在关联，并不是事物之间的相互影响。

只有在内核沿`x`轴和`y`轴对称时，卷积操作和互相关操作一致，比如高斯卷积