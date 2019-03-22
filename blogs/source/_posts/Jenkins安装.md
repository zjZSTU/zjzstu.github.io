---
title: Jenkins安装
categories: 构建
tags: 自动化
abbrlink: 5d15ec84
date: 2019-03-22 13:35:53
---

参考：

[Jenkins User Documentation](https://jenkins.io/doc/#about-this-documentation)

## 什么是`Jenkins`？

[Jenkins]()是一个独立开源的自动化服务器，支持几乎所有语言，支持所有自动化任务，包括构建、测试、交付和部署

### 先决条件

硬件上

1. 至少`256MB`内存，推荐`512MB`
2. `10GB`硬盘空间（用于`Jenkins`和`Docker`镜像）

软件上

1. `Java 8`，参考：[[Ubuntu 16.04]Java安装](https://zj-linux-guide.readthedocs.io/zh_CN/latest/tools/[Ubuntu%2016.04]Java%E5%AE%89%E8%A3%85.html)
2. `Docker`，参考：[[Ubuntu 16.04]安装](https://docker-guide.readthedocs.io/zh_CN/latest/basic/[Ubuntu%2016.04]%E5%AE%89%E8%A3%85.html)

## 安装

下载`Jetkins`: http://mirrors.jenkins.io/war-stable/latest/jenkins.war

运行 

    $ java -jar jenkins.war --httpPort=8080

如果是本地安装，登录

    http://localhost:8080

如果是远程安装，登录

    http://远程ip:8080

输入密码，在命令行信息中会有提示

    Jenkins initial setup is required. An admin user has been created and a password generated.
    Please use the following password to proceed to installation:

    ae096c33c43148c3a65a518b5dbaxxxx

    This may also be found at: /home/ubuntu/.jenkins/secrets/initialAdminPassword

*`Jetkins`需要安装额外插件，选择官方建议的插件进行安装*
