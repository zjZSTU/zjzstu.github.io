---
title: PASCAL VOC 2007数据集解析
categories: 数据集
tags: pascal voc
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

## 引用

如果利用了`VOC 2007`数据，可以引用（`citation`）以下参考信息：

```
@misc{pascal-voc-2007,
	author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
	title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2007 {(VOC2007)} {R}esults",
	howpublished = "http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html"}	
```