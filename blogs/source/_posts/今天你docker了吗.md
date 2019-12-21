---
title: 今天你docker了吗?
abbrlink: 5c6c610b
date: 2019-09-23 09:33:08
categories:
- [随笔]
- [工具]
tags:
- docker
---

![](/imgs/docker/docker2.jpeg)

`Docker`是近`10`年来最火的工具之一，从一接触`Docker`开始就被它的概念所吸引，小结`Docker`概念、使用以及相关工具

## 简介

参考：[[译]docker概述](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/basic/[%E8%AF%91]docker%E6%A6%82%E8%BF%B0/)

>`Docker`是一个开发、发布和运行应用程序的开放平台，提供了在松散隔离的环境（称为容器）中打包和运行应用程序的能力。利用`Docker`可以将应用程序与基础架构分离，能够统一应用程序运行环境，保证快速的发布、测试和部署

相比于`VMWare`独立运行完整的操作系统，`Docker`容器共享主机内核，其实现占用更少的内存，不过因此在`Linux`系统上的`Docker`容器无法运行`Windows`系统

## 用途

`Docker`提供了在多平台（`Linux/Windows/IOS`）下的运行程序，但是最主要的还是基于`Linux`系统的操作。我在`Ubuntu`系统上面主要利用`Docker`进行两个方面的使用

1. 运行`GUI`应用
2. 统一开发环境

### 运行GUI应用

通过`Docker`安装`GUI`应用，能够隔离各个运行环境，避免依赖冲突和依赖爆炸，并且有利于快速移植和部署

当前已实现的`Docker GUI`应用，包括`wechat/qq/wps/thunder`等等

* [可视化运行](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/gui/[Docker]GUI%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5/)
* [Containerization-Automation/dockerfiles/](https://github.com/zjZSTU/Containerization-Automation/tree/master/dockerfiles)

### 统一开发环境

通过`Docker`配置开发环境，能够保证开发、测试和发布的一致性，并且能够加速产品的移植和部署

## 管理

越来越多的容器运行在系统上，除了通过`docker-cli`进行直接管理外，还有一些工具可以进行容器编排

* `Docker Compose`：可以将多个`Docker`容器组成一个应用
* `Docker Swarm`：`Docker`官方提供的容器集群管理工具，其主要作用是把若干台`Docker`主机抽象为一个整体，并且通过一个入口统一管理这些`Docker`主机上的各种`Docker`资源
* `K8S`：基于容器的集群管理平台，全称是`kubernetes`

## 小结

使用`Docker`快`1`个半月了，花费了不少时间学习`Docker`容器的制作，后续还需要继续了解容器编排工具的使用

随着对`Docker`学习的深入，更加坚信这项工具对于软件开发的用处。未来的`Docker`会成为基础工具，类似`Linux`系统一样，加速信息服务在各个领域的应用

* [docker使用指南](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/basic/[%E8%AF%91]docker%E6%A6%82%E8%BF%B0/)
* [zjZSTU/Containerization-Automation](https://github.com/zjZSTU/Containerization-Automation)