---
title: PASCAL VOC 2007数据集解析
categories: 
- [数据集]
- [编程]
tags: 
- pascal voc
- python
abbrlink: 5a56cd45
date: 2019-12-06 16:42:02
---

参考：[PASCAL-VOC数据集解析](https://www.zhujian.tech/posts/28b6703d.html)

## 简介

参考：[The PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)

`PASCAL VOC 2007`数据集基于`4`个大类别，共包含了`20`个目标类：

* `Person: person`
* `Animal: bird, cat, cow（奶牛）, dog, horse, sheep（绵羊）`
* `Vehicle（交通工具）: aeroplane（飞机）, bicycle, boat（小船）, bus（公共汽车）, car（轿车）, motorbike（摩托车）, train（火车）`
* `Indoor（室内）: bottle（瓶子）, chair（椅子）, dining table（餐桌）, potted plant（盆栽植物）, sofa, tv/monitor（电视/显示器）`

`PASCAL VOC 2007`数据集主要用于分类/测试任务，同时也提供了分割和人体部件检测的数据。示例如下：

* [分类/测试示例](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/examples/index.html)
* [分割示例](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/segexamples/index.html)
* [人体部件检测示例](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/layoutexamples/index.html)

## 数据

通过标注文件的方式提供了训练/验证/测试集的数据。整个数据集分为`50%`的训练/验证集以及`50%`的测试集。总共有`9963`幅图像，包含`24640`个标注对象，具体信息如下

1. 标注准则参考[Annotation Guidelines](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/guidelines.html)
2. 详细的训练/验证数据集的个数参考[Database Statistics](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/dbstats.html)

## 下载

训练相关

* 训练/验证数据集下载：[training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
* 工具包代码（`Matlab`版本）及开发文档：[development kit code and documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar)
* 单独下载的开发文档：[PDF documentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/devkit_doc_07-Jun-2007.pdf)

测试相关

* 测试数据集（含标注）下载：[annotated test data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
* 单独的标注信息下载：[annotation only](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar)

## 解析标注数据

参考：[[python]读取XML文件](https://zj-image-processing.readthedocs.io/zh_CN/latest/python/[python]%E8%AF%BB%E5%8F%96XML%E6%96%87%E4%BB%B6.html)

`VOC`数据集的图像保存在文件夹`JPEGImages`中，标注数据保存在`Annotations`中

编写如下代码解析标注数据，将训练/验证/测试数据从原图像中提取出来

```
# -*- coding: utf-8 -*-

"""
@author: zj
@file:   batch_xml.py
@time:   2019-12-07
"""

import cv2
import os
import xml.etree.cElementTree as ET

train_xml_dir = '/home/zj/data/PASCAL-VOC/2007/train/Annotations'
train_jpeg_dir = '/home/zj/data/PASCAL-VOC/2007/train/JPEGImages'

test_xml_dir = '/home/zj/data/PASCAL-VOC/2007/test/Annotations'
test_jpeg_dir = '/home/zj/data/PASCAL-VOC/2007/test/JPEGImages'

# 标注图像保存路径
train_imgs_dir = '/home/zj/data/PASCAL-VOC/2007/train_imgs'
test_imgs_dir = '/home/zj/data/PASCAL-VOC/2007/test_imgs'


def parse_xml(xml_path):
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()

    img_name = ''
    obj_list = list()
    bndbox_list = list()

    # 遍历根节点下所有节点，查询文件名和目标坐标
    for child_node in root:
        if 'filename'.__eq__(child_node.tag):
            img_name = child_node.text
        if 'object'.__eq__(child_node.tag):
            obj_name = ''
            for obj_node in child_node:
                if 'name'.__eq__(obj_node.tag):
                    obj_name = obj_node.text
                if 'bndbox'.__eq__(obj_node.tag):
                    node_bndbox = obj_node

                    node_xmin = node_bndbox[0]
                    node_ymin = node_bndbox[1]
                    node_xmax = node_bndbox[2]
                    node_ymax = node_bndbox[3]

                    obj_list.append(obj_name)
                    bndbox_list.append((
                        int(node_xmin.text), int(node_ymin.text), int(node_xmax.text), int(node_ymax.text)))

    return img_name, obj_list, bndbox_list


def batch_parse(xml_dir, jpeg_dir, imgs_dir):
    xml_list = os.listdir(xml_dir)
    jepg_list = os.listdir(jpeg_dir)

    for xml_name in xml_list:
        xml_path = os.path.join(xml_dir, xml_name)
        img_name, obj_list, bndbox_list = parse_xml(xml_path)
        print(img_name, obj_list, bndbox_list)

        if img_name in jepg_list:
            img_path = os.path.join(jpeg_dir, img_name)
            src = cv2.imread(img_path)
            for i in range(len(obj_list)):
                obj_name = obj_list[i]
                bndbox = bndbox_list[i]

                obj_dir = os.path.join(imgs_dir, obj_name)
                if not os.path.exists(obj_dir):
                    os.mkdir(obj_dir)
                obj_path = os.path.join(obj_dir, '%s-%s-%d-%d-%d-%d.png' % (
                    img_name, obj_name, bndbox[0], bndbox[1], bndbox[2], bndbox[3]))

                res = src[bndbox[1]:bndbox[3], bndbox[0]:bndbox[2]]
                cv2.imwrite(obj_path, res)


if __name__ == '__main__':
    batch_parse(train_xml_dir, train_jpeg_dir, train_imgs_dir)
    batch_parse(test_xml_dir, test_jpeg_dir, test_imgs_dir)
```

通过解析`XML`文件，获取图像名以及标注的目标名和边界框数据；通过`OpenCV`读取图像，截取图像后保存在指定类别文件夹

## 引用

如果利用了`VOC 2007`数据，可以引用（`citation`）以下参考信息：

```
@misc{pascal-voc-2007,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2007 {(VOC2007)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html"}	
```