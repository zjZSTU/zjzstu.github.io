---
title: cifar-10数据集解析
categories: 
- [数据集]
- [编程, 编程语言]
- [编程, 代码库]
tags: 
- cifar10
- python
- pickle
- numpy
- opencv
abbrlink: 43d7ec86
date: 2019-04-02 19:44:57
---

`cifar-10`数据集保存`10`类，每类`6000`张图像。其中`50000`张训练图像和`10000`张测试图像

训练图像保存在`5`个文件中，每个文件有`10000`张图像，测试图像保存在一个文件，训练和测试图像都以随机顺序保存

官网：[The CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)

`cifar-10`提供了使用不同语言生成的压缩包，包括`python/matlab/c`

## `python`解析

训练文件命名为`data_batch_1/data_batch_2/data_batch_3/data_batch_4/data_batch_5`

测试文件命名为`test_batch`

另外还有一个元数据文件`batches.meta`，保存了标签名对应的类名，比如`label_names[0] == "airplane", label_names[1] == "automobile"`

`python`版压缩包使用`pickle`模块进行保存，解析程序如下，返回一个`dict`

```
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
```

以测试集文件为例，其包含以下键值对

```
dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
```

其中`b'labels'`保存类别标签（`0-9`），`b'data'`保存对应图像数据，`b'filenames'`保存图像文件名

```
if __name__ == '__main__':
    data_dir = '/home/zj/data/cifar-10-batches-py/'
    test_data_dir = data_dir + 'test_batch'
    di = unpickle(test_data_dir)
    # 打印所有键值
    print(di.keys())

    batch_label = di.get(b'batch_label')
    filenames = di.get(b'filenames')
    labels = di.get(b'labels')
    data = di.get(b'data')

    print(batch_label)
    print(filenames[0])
    print(labels[0])
    print(data[0])

dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
b'testing batch 1 of 1'
b'domestic_cat_s_000907.png'
3
[158 159 165 ... 124 129 110]
```

## 数据格式

键`b'data'`保存了图像数据，其值类型是`numpy.ndarray`，每一行都保存了`32x32`大小图像数据

其中前`1024`个字节是红色分量值，中间`1024`个字节是绿色分量值，最后`1024`个字节是蓝色分量值

解析程序如下：

```
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convert_data_to_img(data):
    """
    转换数据为图像
    :param data: 3072维
    :return: img
    """
    num = 1024
    red = data[:num]
    green = data[num:(num * 2)]
    blue = data[(num * 2):]

    img = np.ndarray((32, 32, 3))
    for i in range(32):
        for j in range(32):
            img[i, j, 0] = blue[i * 32 + j]
            img[i, j, 1] = green[i * 32 + j]
            img[i, j, 2] = red[i * 32 + j]

    return img


if __name__ == '__main__':
    data_dir = '/home/zj/data/cifar-10-batches-py/'
    test_data_dir = data_dir + 'test_batch'
    di = unpickle(test_data_dir)

    filenames = di.get(b'filenames')
    data = di.get(b'data')

    filename = str(filenames[0], encoding='utf-8')
    img = convert_data_to_img(data[0])

    cv.imwrite(filename, img)
```

![](/imgs/cifar-10数据集解析/domestic_cat_s_000907.png)

## 解压代码

完整解压代码如下：

```
# -*- coding: utf-8 -*-

# @Time    : 19-6-5 下午8:20
# @Author  : zj

import numpy as np
import pickle
import os
import cv2

data_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

res_data_dir = '/home/zj/data/decompress_cifar_10'


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def write_img(data, labels, filenames, isTrain=True):
    if isTrain:
        data_dir = os.path.join(res_data_dir, 'train')
    else:
        data_dir = os.path.join(res_data_dir, 'test')

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    N = len(labels)
    for i in range(N):
        cate_dir = os.path.join(data_dir, str(labels[i]))
        if not os.path.exists(cate_dir):
            os.mkdir(cate_dir)
        img_path = os.path.join(cate_dir, str(filenames[i], encoding='utf-8'))

        # r = data[i][:1024].reshape(32, 32)
        # g = data[i][1024:2048].reshape(32, 32)
        # b = data[i][2048:].reshape(32, 32)
        # img = cv2.merge((b, g, r))

        img = data[i].reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, img)


if __name__ == '__main__':

    for item in data_list:
        data_dir = '/home/zj/data/cifar-10-batches-py/'
        data_dir = os.path.join(data_dir, item)
        di = unpickle(data_dir)

        batch_label = str(di.get(b'batch_label'), encoding='utf-8')
        filenames = di.get(b'filenames')
        labels = di.get(b'labels')
        data = di.get(b'data')

        if 'train' in batch_label:
            write_img(data, labels, filenames)
        else:
            write_img(data, labels, filenames, isTrain=False)
```

## 读取图像

读取解压后的图像并显示，代码如下：

```
if __name__ == '__main__':
    data_dir = '/home/zj/data/cifar-10-batches-py/test_batch'

    di = unpickle(data_dir)

    batch_label = str(di.get(b'batch_label'), encoding='utf-8')
    filenames = di.get(b'filenames')
    labels = di.get(b'labels')
    data = di.get(b'data')

    N = 10
    W = 32
    H = 32
    ex = np.zeros((H * N, W * N, 3))
    for i in range(N):
        for j in range(N):
            img = data[i * N + j].reshape(3, H, W)
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            ex[i * H:(i + 1) * H, j * W:(j + 1) * W] = img
    plt.imshow(ex.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()
```

![](/imgs/cifar-10数据集解析/mix.png)