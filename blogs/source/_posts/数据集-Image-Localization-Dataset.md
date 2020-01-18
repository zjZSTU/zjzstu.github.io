---
title: '[数据集]Image Localization Dataset'
categories:
  - - 数据
    - 数据集
  - - 编程
    - 编程语言
  - - 编程
    - 代码库
tags:
  - image localization dataset
  - python
  - numpy
  - sklearn
  - xmltodict
  - sklearn
  - glob
  - opencv
abbrlink: a2d65e1
date: 2020-01-18 19:00:29
---

图像定位数据集（`image localization dataset`）是一个简单的用于图像定位实验的数据集，参考[Image Localization Dataset](https://www.kaggle.com/mbkinaci/image-localization-dataset/data)

## 简介

* 包含`3`类：`Cucumber`（黄瓜）、`Eggplant`（茄子）、`Mushroom`（蘑菇）
* 每类共有超过`60`张的图像，大小固定为`(227, 277, 3)`，每张图像里有一个物体
* 每张图像有一个对应的`xml`文件，格式和`PASCAL VOC`数据集一致，包含图像信息以及标注信息

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   show_img.py
@time:   2020-01-18
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xmltodict

from matplotlib.font_manager import _rebuild

_rebuild()  # reload一下

plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def draw_rect(img_path, xml_path):
    img = cv2.imread(img_path)
    xml_data = xmltodict.parse(open(xml_path, 'rb'))

    bndbox = xml_data['annotation']['object']['bndbox']
    bndbox = np.array([int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])])
    x_min, y_min, x_max, y_max = bndbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)

    return img


def load_img():
    root_dir = './data/image-localization-dataset/training_images/'
    img_cucumber = os.path.join(root_dir, 'cucumber_1.jpg')
    img_eggplant = os.path.join(root_dir, 'eggplant_1.jpg')
    img_mushroom = os.path.join(root_dir, 'mushroom_1.jpg')

    xml_cucumber = os.path.join(root_dir, 'cucumber_1.xml')
    xml_eggplant = os.path.join(root_dir, 'eggplant_1.xml')
    xml_mushroom = os.path.join(root_dir, 'mushroom_1.xml')

    img_cucumber = draw_rect(img_cucumber, xml_cucumber)
    img_eggplant = draw_rect(img_eggplant, xml_eggplant)
    img_mushroom = draw_rect(img_mushroom, xml_mushroom)

    return img_cucumber, img_eggplant, img_mushroom


if __name__ == '__main__':
    img_cucumber, img_eggplant, img_mushroom = load_img()

    plt.style.use('dark_background')

    plt.figure(figsize=(10, 5))  # 设置窗口大小
    plt.suptitle('图像定位数据集')  # 图片名称

    plt.subplot(1, 3, 1)
    plt.title('cucumber')
    plt.imshow(img_cucumber), plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('eggplant')
    plt.imshow(img_eggplant), plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('mushroom')
    plt.imshow(img_mushroom), plt.axis('off')

    plt.show()
```

![](/imgs/dataset-localization/img_location.png)

## sklearn加载

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   localization_test.py
@time:   2020-01-18
"""

import cv2
import glob
import numpy as np
import xmltodict
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

input_dim = 227


def load_image():
    image_paths = glob.glob('./data/image-localization-dataset/training_images/*.jpg')
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        # 缩放图像到固定大小
        img = cv2.resize(img, (input_dim, input_dim))
        # 缩放像素值到[0,1]
        images.append(img / 255.0)
    return images


def load_labels():
    bboxes = []
    classes_raw = []
    annotations_paths = glob.glob('./data/image-localization-dataset/training_images/*.xml')
    for xmlfile in annotations_paths:
        x = xmltodict.parse(open(xmlfile, 'rb'))
        bndbox = x['annotation']['object']['bndbox']
        bndbox = np.array([int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])])
        # 同等比例缩放边界框坐标
        bboxes.append(bndbox * (input_dim / float(x['annotation']['size']['width'])))
        classes_raw.append(x['annotation']['object']['name'])
    return bboxes, classes_raw


def load_data():
    images = load_image()
    bboxes, classes_raw = load_labels()

    # 标签信息自定义
    # 当前等于 标注信息 + one-hot编码
    boxes = np.array(bboxes)
    encoder = LabelBinarizer()
    classes_onehot = encoder.fit_transform(classes_raw)

    Y = np.concatenate([boxes, classes_onehot], axis=1)
    X = np.array(images)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
# 输出
(167, 227, 227, 3)
(167, 7)
(19, 227, 227, 3)
(19, 7)
```

## pytorch加载

参考[[torchvision]自定义数据集和预处理操作](https://zj-image-processing.readthedocs.io/zh_CN/latest/pytorch/[torchvision]%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86%E5%92%8C%E9%A2%84%E5%A4%84%E7%90%86%E6%93%8D%E4%BD%9C/)实现自定义数据集

继承自类`torch.utils.data.Dataset`，重写了函数`__getitem__`和`__len__`。如果是训练部分，加载编号前`50`个图像；如果是测试部分，加载`50`之后的图像

```
class LocationDataSet(Dataset):

    def __init__(self, root_dir, train=True, transform=None, input_dim=1):
        """
        自定义数据集类，加载定位数据集
        1. 训练部分，加载编码前50图像和标记数据
        2. 测试部分，加载编码50之后图像和标记数据
        :param root_dir:
        :param train:
        :param transform:
        """
        cates = ['cucumber', 'eggplant', 'mushroom']
        class_binary_label = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.train = train
        self.transform = transform

        self.imgs = []
        self.bboxes = []
        self.classes = []

        for cate_idx in range(3):
            if self.train:
                for i in range(1, 51):
                    img, bndbox, class_name = self._get_item(root_dir, cates[cate_idx], i)
                    bndbox = bndbox / input_dim

                    self.imgs.append(img)
                    self.bboxes.append(np.hstack((bndbox, class_binary_label[cate_idx])))
                    self.classes.append(class_name)
            else:
                for i in range(51, 61):
                    img, bndbox, class_name = self._get_item(root_dir, cates[cate_idx], i)
                    bndbox = bndbox / input_dim

                    self.imgs.append(img)
                    self.bboxes.append(np.hstack((bndbox, class_binary_label[cate_idx])))
                    self.classes.append(class_name)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            sample = self.transform(img)
        else:
            sample = img
        return sample, torch.Tensor(self.bboxes[idx]).float()

    def __len__(self):
        return len(self.imgs)

    def _get_item(self, root_dir, cate, i):
        img_path = os.path.join(root_dir, '%s_%d.jpg' % (cate, i))
        img = cv2.imread(img_path)

        xml_path = os.path.join(root_dir, '%s_%d.xml' % (cate, i))
        x = xmltodict.parse(open(xml_path, 'rb'))
        bndbox = x['annotation']['object']['bndbox']
        bndbox = np.array(
            [float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(bndbox['ymax'])])

        return img, bndbox, x['annotation']['object']['name']
```

实现自定义类后，通过加载器进行数据处理

```
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def load_data():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_dim),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    root_dir = './data/image-localization-dataset/training_images/'
    train_dataset = LocationDataSet(root_dir, train=True, transform=transform, input_dim=input_dim)
    test_dataset = LocationDataSet(root_dir, train=False, transform=transform, input_dim=input_dim)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader

if __name__ == '__main__':
    train_dataloader, test_dataloader = load_data()
```