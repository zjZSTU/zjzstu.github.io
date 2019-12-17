---
title: '颜色空间小结(RGB/YUV/HSV/Lab/...)'
categories:
  - [算法, 图像处理, 颜色空间]
tags:
  - rgb
  - hsv
  - yuv
  - lab
abbrlink: cc213461
date: 2019-11-26 20:04:45
---

经常需要在不同的颜色空间下进行图像处理

## 颜色空间

参考：

[颜色空间](https://baike.baidu.com/item/%E9%A2%9C%E8%89%B2%E7%A9%BA%E9%97%B4)

[色彩空間](https://zh.wikipedia.org/wiki/%E8%89%B2%E5%BD%A9%E7%A9%BA%E9%96%93)

>颜色空间（color space），又称为颜色模型（color model），是坐标系统和子空间的阐述，位于系统的每种颜色都有单个点表示

换句话说，所有颜色都被颜色空间进行编码，每种颜色都可以使用基于该颜色模型的一组数字来描述。常用的颜色空间有：

* `RGB`
* `YUV`
* `HSV`
* `Lab`

## RGB

参考：

[RGB](https://baike.baidu.com/item/RGB)

[RGB color model](https://en.wikipedia.org/wiki/RGB_color_model#Color_depth)

`RGB`颜色模型是一种加法颜色模型，其中红色、绿色和蓝色的光以各种方式相加以得到广泛的颜色阵列。模型的名称来自三种附加原色（红色、绿色和蓝色）的首字母。其主要作用于各种显示器（电脑/摄像机/手机等等）

用于`RGB`颜色的总位数通常称为颜色深度（`color depth`），每种颜色通常有`1、2、4、5、8`和`16`位编码

## YUV

参考：[YUV2RGB Opencv](https://blog.csdn.net/u012005313/article/details/70304922)

## HSV

参考：

[HSL and HSV](https://en.wikipedia.org/wiki/HSL_and_HSV)

[HSV （HSV颜色模型）](https://baike.baidu.com/item/HSV/547122)

[opencv HSV 颜色模型(H通道取值 && CV_BGR2HSV＿FULL)](https://blog.csdn.net/u012005313/article/details/46678883)

`HSV`颜色模型由色调（`hue`）、饱和度（`saturation`）和明度（`value`）组成。其更符合人眼对于色彩的直观感受，因为人的视觉对亮度的敏感程度远强于对颜色浓淡的敏感程度

## Lab

参考：

[Lab 颜色模型](https://baike.baidu.com/item/%E9%A2%9C%E8%89%B2%E6%A8%A1%E5%9E%8B/7558583?fromtitle=Lab&fromid=1514615)

[CIELAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space)

`Lab`颜色模型是由`CIE`（国际照明委员会）制定的一种色彩模式，它的色彩空间比`RGB`空间还要大。另外这种模式是以数字化方式来描述人的视觉感应，与设备无关，所以它弥补了`RGB`和`CMYK`必须依赖于设备色彩特性的不足