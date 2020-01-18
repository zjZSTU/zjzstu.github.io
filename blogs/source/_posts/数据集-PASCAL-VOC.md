---
title: '[数据集]PASCAL-VOC'
categories: 
- [数据, 数据集]
tags: pascal voc
abbrlink: 28b6703d
date: 2019-08-05 19:54:41
---

`PASCAL VOC(visual object classes)`提供了图像以及标记数据，可用于目标分类、检测、分割等任务

官网地址：[The PASCAL Visual Object Classes Homepage](http://host.robots.ox.ac.uk/pascal/VOC/)

## 数据集与挑战赛

`PASCAL VOC`每年都会发布一个新的数据集，针对数据集进行不同任务的竞赛，从`2005`年持续到`2012`年

* 从`2007`年开始，数据集类别数固定为`20`类
* 从`2008`年开始，不再公布测试集标注数据
* 从`2009`年开始，数据集在前一年数据集的基础上添加新数据

`2012`年数据集包含`11530`张图片，包含`27450`个`ROI`标注对象和`6929`个分割

## 数据集

目前常用的是`VOC 2007`（包含测试集标注信息）和`VOC 2012`数据集（最新数据集）

可分别去各个竞赛主页下载：

* [The PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
* [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

## 排行榜

`PASCAL VOC`提供了一个[评估服务器](http://host.robots.ox.ac.uk:8080/)，可以上传测试结果进行成绩排行，具体标准参考：[Best Practice](http://host.robots.ox.ac.uk/pascal/VOC/#bestpractice)，分两个参与类型，一是仅使用了官方提供的训练/验证数据；二是使用了除测试数据外的综合数据集

看了上面的排行榜，目前仍有不少人员参与其中：[Leaderboards for the Evaluations on PASCAL VOC Data](http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php)

不过要使用评估服务器之前需要先注册，注册要求之一是要提供一个机构邮箱，为的是防止重复上传的刷榜行为，所以独立开发者没办法参与其中了