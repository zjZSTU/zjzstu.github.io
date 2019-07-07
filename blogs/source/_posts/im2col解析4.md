---
title: im2col解析4
categories:
  - [算法]
  - [编程]
tags:
  - 深度学习
  - python
abbrlink: 291a942c
date: 2019-05-25 20:31:41
---

之前实现了一个图像和行向量相互转换的函数，逐图像进行局部连接矩阵的转换

其实现原理较下标计算更易理解，**通过循环，逐个图像对局部连接矩阵进行切片操作，得到矩阵后拉平为向量，以行向量方式进行保存**

反向转换图像可以设置标志位`isstinct`，是否返回叠加图像还是原图，**其实现原理是在指定位置赋值过程中是执行累加还是执行覆盖**

## 实现代码

```
def convert_conv_to_fc(input, filter_height=3, filter_width=3, stride=1, padding=0):
    input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                          'constant', constant_values=(0, 0))
    # [N, C, H, W]
    num, depth, height, width = input_padded.shape[:4]

    res = []
    for k in range(num):
        i = 0
        while i < height:
            j = 0
            while j < width:
                arr = input_padded[k, :, i:i + filter_height, j:j + filter_width]
                res.append(arr.flatten())
                j += stride
                if (j + filter_width) > width:
                    break
            i += stride
            if (i + filter_height) > height:
                break

    return np.array(res)


def deconvert_fc_to_conv(input, output_shape, filter_height=3, filter_width=3, stride=2, padding=0, isstinct=False):
    output = np.zeros(output_shape)
    output_padded = np.pad(output, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                           'constant', constant_values=(0, 0))
    # [N, C, H, W]
    num, depth, height, width = output_padded.shape[:4]

    number = 0
    for k in range(num):
        i = 0
        while i < height:
            j = 0
            while j < width:
                if isstinct:
                    output_padded[k, :, i:i + filter_height, j:j + filter_width] = \
                        input[number].reshape(depth, filter_height, filter_width)
                else:
                    output_padded[k, :, i:i + filter_height, j:j + filter_width] += \
                        input[number].reshape(depth, filter_height, filter_width)
                j += stride
                number += 1
                if (j + filter_width) > width:
                    break
            i += stride
            if (i + filter_height) > height:
                break

    if padding == 0:
        return output_padded

    return output_padded[:, :, padding:-padding, padding:-padding]
```

## 时间测试

|                      | 大小(128x3x32x32) 卷积核(3x3) 步长(1) 零填充(0) | 大小(128x3x227x227) 卷积核(3x3) 步长(1) 零填充(0) |
|:--------------------:|:-----------------------------------------------:|:-------------------------------------------------:|
|  im2row 图像转行向量 |                      0.087                      |                       0.941                       |
|  自定义 图像转行向量 |                      0.257                      |                       3.475                       |
| im2row  行向量转图像 |                      0.238                      |                       3.519                       |
| 自定义  行向量转图像 |                      0.411                      |                       7.413                       |

经过测试发现，下标计算方式快过循环计算方式，并且图像转行向量操作比行向量转图像操作快