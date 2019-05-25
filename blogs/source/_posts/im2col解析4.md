---
title: im2col解析4
categories:
  - 编程
tags:
  - 深度学习
abbrlink: 291a942c
date: 2019-05-25 20:31:41
---

之前实现了一个图像和行向量相互转换的函数，逐图像进行局部连接矩阵的转换

其实现原理较下标计算更易理解，**就是逐个图像对局部连接矩阵进行切片操作，得到矩阵后拉平为向量，以行向量方式进行保存**

反向转换图像可以设置标志位`isstinct`，是否返回叠加图像还是原图，**其实现原理是在指定位置赋值过程中是否累加**

```
def convert_conv_to_fc(input, filter_size=3, stride=1, padding=0):
    input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                          'constant', constant_values=(0, 0))
    # 批量大小、深度、长度、宽度
    num, depth, height, width = input_padded.shape[:4]

    res = []
    for k in range(num):
        i = 0
        while i < height:
            j = 0
            while j < width:
                arr = input_padded[k, :, i:i + filter_size, j:j + filter_size]
                res.append(arr.flatten())
                j += stride
                if (j + filter_size) > width:
                    break
            i += stride
            if (i + filter_size) > height:
                break

    return np.array(res)


def deconvert_fc_to_conv(input, output, filter_height=3, filter_width=3, stride=2, padding=0, isstinct=False):
    # 批量大小、深度、长度、宽度
    num, depth, height, width = output.shape[:4]

    number = 0
    for k in range(num):
        i = 0
        while i < height:
            j = 0
            while j < width:
                if isstinct:
                    output[k, :, i:i + filter_height, j:j + filter_width] = \
                        input[number].reshape(depth, filter_height, filter_width)
                else:
                    output[k, :, i:i + filter_height, j:j + filter_width] += \
                        input[number].reshape(depth, filter_height, filter_width)
                j += stride
                number += 1
                if (j + filter_width) > width:
                    break
            i += stride
            if (i + filter_height) > height:
                break

    if padding == 0:
        return output

    return output[:, :, padding:-padding, padding:-padding]
```

测试如下：

1. 图像转行向量

    ```
    import im2row

    import numpy as np

    if __name__ == '__main__':
        x = np.arange(32).reshape(2, 1, 4, 4)
        rows = im2row_indices(x, 2, 2, padding=0, stride=2)
        print(rows)
        print(rows.shape)
        rows2 = convert_conv_to_fc(x, filter_size=2, stride=2, padding=0)
        print(rows2)
        print(rows2.shape)
        print(rows == rows2)
    ```

    ```
    [[ 0  1  4  5]
    [ 2  3  6  7]
    [ 8  9 12 13]
    [10 11 14 15]
    [16 17 20 21]
    [18 19 22 23]
    [24 25 28 29]
    [26 27 30 31]]
    (8, 4)
    [[ 0  1  4  5]
    [ 2  3  6  7]
    [ 8  9 12 13]
    [10 11 14 15]
    [16 17 20 21]
    [18 19 22 23]
    [24 25 28 29]
    [26 27 30 31]]
    (8, 4)
    [[ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]]
    ```

2. 行向量转图像
    
    ```
    if __name__ == '__main__':
        x = np.arange(32).reshape(2, 1, 4, 4)
        rows = im2row_indices(x, 2, 2, padding=0, stride=2)
        output = row2im_indices(rows, x.shape, field_height=2, field_width=2, padding=0, stride=2)
        print(output)
        print(output.shape)
        output2 = deconvert_fc_to_conv(rows, np.zeros(x.shape), filter_height=2, filter_width=2, stride=2, padding=0)
        print(output2)
        print(output2.shape)
        print(output2 == output)
    ```

    ```
    [[[[ 0  1  2  3]
    [ 4  5  6  7]
    [ 8  9 10 11]
    [12 13 14 15]]]
    [[[16 17 18 19]
    [20 21 22 23]
    [24 25 26 27]
    [28 29 30 31]]]]
    (2, 1, 4, 4)
    [[[[ 0.  1.  2.  3.]
    [ 4.  5.  6.  7.]
    [ 8.  9. 10. 11.]
    [12. 13. 14. 15.]]]
    [[[16. 17. 18. 19.]
    [20. 21. 22. 23.]
    [24. 25. 26. 27.]
    [28. 29. 30. 31.]]]]
    (2, 1, 4, 4)
    [[[[ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]]]
    [[[ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]
    [ True  True  True  True]]]]
    ```