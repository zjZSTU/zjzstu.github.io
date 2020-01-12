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

在上图中，绿色边框是真值边界框，红色边框是算法检测得到的预测边界框，IoU需要计算两个边界框的重叠面积和并集面积的比率

$$
IoU = \frac {Area\ of\ Overlap}{Area\ of\ Union}
$$

![](/imgs/iou/Intersection_over_Union_-_visual_equation.png)

更多情况下，分母计算的是由预测边界框和真值边界框所包围的区域

`IoU`的取值范围在`[0,1]`之间，当`IoU>0.5`时通常认为是好的预测

![](/imgs/iou/Intersection_over_Union_-_poor,_good_and_excellent_score.png)