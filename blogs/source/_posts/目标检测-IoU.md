---
title: "[目标检测]IoU"
abbrlink: 796ebd4e
date: 2020-01-12 15:20:50
categories:
- [算法, 评价标准]
tags:
- IoU
---

参考：

[Intersection over Union (IoU) for object detection](https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)

[一分钟的CNN：如何理解IoU(Intersection over Union)和NMS(non-maximum suppression)算法？](https://zhuanlan.zhihu.com/p/36303642)

`IoU(Intersection over union, 交集并集比)`是目标检测领域常用的评价标准，通过比较真值边界框（`the ground-truth bounding box`，手动标记）和预测边界框（`the predicted bounding box`）的重合度来判定算法检测性能

![](/imgs/iou/450px-Intersection_over_Union_-_object_detection_bounding_boxes.jpg)

## 原理解析

在上图中，绿色边框是真值边界框，红色边框是算法检测得到的预测边界框，`IoU`需要计算两个边界框的重叠面积和并集面积的比率

$$
IoU = \frac {Area\ of\ Overlap}{Area\ of\ Union}
$$

![](/imgs/iou/Intersection_over_Union_-_visual_equation.png)

更多情况下，分母计算的是由预测边界框和真值边界框所包围的区域

`IoU`的取值范围在`[0,1]`之间，当`IoU>0.5`时通常认为是好的预测

![](/imgs/iou/Intersection_over_Union_-_poor,_good_and_excellent_score.png)

## python实现

分别使用`numpy`和`pytorch`计算两个边界框的`IoU`，分为`3`种情况：

1. 完全重叠
2. 部分重叠
3. 没有重叠

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   iou-compute.py
@time:   2020-01-19
"""

import numpy as np
import torch
import cv2


def numpy_iou(target_boxes, pred_boxes):
    """
    target_boxes和pred_boxes大小相同:[N, 4]，其中N表示边框数目，4表示保存的是[xmin, ymin, xmax, ymax]
    """
    xA = np.maximum(target_boxes[:, 0], pred_boxes[:, 0])
    yA = np.maximum(target_boxes[:, 1], pred_boxes[:, 1])
    xB = np.minimum(target_boxes[:, 2], pred_boxes[:, 2])
    yB = np.minimum(target_boxes[:, 3], pred_boxes[:, 3])
    # 计算交集面积
    intersection = np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA)
    # 计算两个边界框面积
    boxAArea = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    boxBArea = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

    iou = intersection / (boxAArea + boxBArea - intersection)
    return iou


def pytorch_iou(target_boxes, pred_boxes):
    x_min = torch.max(target_boxes[:, 0], pred_boxes[:, 0])
    y_min = torch.max(target_boxes[:, 1], pred_boxes[:, 1])
    x_max = torch.min(target_boxes[:, 2], pred_boxes[:, 2])
    y_max = torch.min(target_boxes[:, 3], pred_boxes[:, 3])
    # 计算交集面积
    intersection = torch.max(torch.zeros(x_max.shape), x_max - x_min) \
                   * torch.max(torch.zeros(y_max.shape), y_max - y_min)

    # 计算两个边界框面积
    boxAArea = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    boxBArea = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])

    iou = intersection / (boxAArea + boxBArea - intersection)
    return iou


if __name__ == '__main__':
    target_boxes = np.array([[10, 10, 50, 50], [40, 270, 100, 380], [450, 300, 500, 500]])
    pred_boxes = np.array([[20, 20, 40, 40], [30, 280, 200, 300], [400, 200, 450, 250]])

    iou1 = numpy_iou(target_boxes, pred_boxes)
    iou2 = pytorch_iou(torch.Tensor(target_boxes), torch.Tensor(pred_boxes))

    print('numpy:', iou1)
    print('pytorch:', iou2)

    img = np.ones((650, 650, 3)) * 255
    for item in target_boxes:
        xmin, ymin, xmax, ymax = item
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
    for item in pred_boxes:
        xmin, ymin, xmax, ymax = item
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
```

![](/imgs/iou/IoU.png)

计算结果如下：

```
numpy: [0.25       0.13636364 0.        ]
pytorch: tensor([0.2500, 0.1364, 0.0000])
```