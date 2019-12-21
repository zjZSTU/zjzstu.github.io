---
title: '[ubuntu 18.04]重装系统小结'
categories:
  - 随笔
abbrlink: e70eeac0
date: 2019-12-05 14:53:49
tags:
---

原先笔记本自带的是`Win10`系统，想着日常开发中更常用的是`Linux`环境，所以重装了`Ubuntu`。之前用的是`16.04`版本，用了也快一年了，电脑里面的东西攒的挺多的，而且现在也都快`2020`了，所以打算重装`Ubuntu 18.04`版本，小结重装`Ubuntu`系统后相关环境配置

![](/imgs/重装系统小结/screen.png)

## 系统配置

### 镜像源

首先是替换阿里云镜像源，参考：[[Ali mirror]更换国内源](https://zj-linux-guide.readthedocs.io/zh_CN/latest/tool-install-configure/[Ali%20mirror]]%E6%9B%B4%E6%8D%A2%E5%9B%BD%E5%86%85%E6%BA%90/)

调用`apt`命令更新并安装常用的软件包和库

```
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install gcc g++ build_essential cmake git vim
```

### 输入法

系统安装设置中文时自动安装了输入法，不过效果不咋的，重装了`Google-Pinyin`

### Nvidia驱动

之前比较推崇手动安装`Nvidia`驱动，不过这一次发现可以通过工具安装，更加方便快捷，参考[[Ubuntu 18.04]PPA方式安装Nvidia驱动](https://zj-linux-guide.readthedocs.io/zh_CN/latest/tool-install-configure/[Ubuntu%2018.04]PPA%E6%96%B9%E5%BC%8F%E5%AE%89%E8%A3%85Nvidia%E9%A9%B1%E5%8A%A8/)

### 系统美化

`Ubuntu 18.04`默认使用`GNome`桌面，主要美化`4`个部分：

1. 主题
2. 图标
3. 任务栏
4. 壁纸

参考：[[Ubuntu 18.04]桌面美化](https://zj-linux-guide.readthedocs.io/zh_CN/latest/tool-install-configure/[Ubuntu%2018.04]%E6%A1%8C%E9%9D%A2%E7%BE%8E%E5%8C%96/)

## 开发环境

主要完成日常编辑、`Python、C++、NodeJS、JAVA`环境配置

### Python环境

* 安装`Anaconda`工具包，替换国内镜像源能够加速安装，参考：[配置国内镜像源](https://zj-image-processing.readthedocs.io/zh_CN/latest/anaconda/%E9%85%8D%E7%BD%AE%E5%9B%BD%E5%86%85%E9%95%9C%E5%83%8F%E6%BA%90.html)

* 安装`PyCharm`

### C++环境

* 安装`CMake`（`apt`安装的`cmake`版本过低，需要下源码重新安装，参考[CMake安装](https://zj-linux-guide.readthedocs.io/zh_CN/latest/tool-install-configure/CMake%E5%AE%89%E8%A3%85/)
* 安装`CLion`

### JAVA环境

* 配置`JDK`，参考[Java安装](https://zj-linux-guide.readthedocs.io/zh_CN/latest/tool-install-configure/Java%E5%AE%89%E8%A3%85/)
* 安装`IntelliJ IDEA`

### JS环境

* 安装`NodeJS`

### 其他工具

* `Chrome`：虽然系统自带了`FireFox`，不过更常用的还是`Chrome`
* `VSCode`：最方便的编辑工具
* `MindMaster`：思维导图。官网：[MindMaster](https://www.edrawsoft.cn/mindmaster/)
* `EDraw`：国产绘图神器，没想到还有`Linux`版本。官网：[亿图图示](https://www.edrawsoft.cn/lp/edraw.html)

## 虚拟机和容器

### VMWare

虽然在`Linux`环境下开发，但是有时候还是需要在`Windows`下操作

`VMWare`的安装还挺多坑的，参考[[Ubuntu 18.04]VMware安装](https://zj-linux-guide.readthedocs.io/zh_CN/latest/tool-install-configure/[Ubuntu%2018.04]VMware%E5%AE%89%E8%A3%85/)

### Docker

`Docker`是目前最热门的容器工具，几乎所有软件都可以通过`Docker`安装和配置

* `Docker`安装：[安装](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/basic/%E5%AE%89%E8%A3%85/)
* 阿里云加速：[[Aliyun]镜像加速](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/basic/[Aliyun]%E9%95%9C%E5%83%8F%E5%8A%A0%E9%80%9F/)
* 非`root`登录/开机自启动：[可选设置](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/basic/%E5%8F%AF%E9%80%89%E8%AE%BE%E7%BD%AE/)

之前也学习了通过`Docker`配置软件，当前使用的有：
  
1. `WeChat`：[[Docker][deepin-wine]微信运行](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/gui/[Docker][deepin-wine]%E5%BE%AE%E4%BF%A1%E8%BF%90%E8%A1%8C/)
2. `Tim`：[[Docker][deepin-wine]TIM运行](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/gui/[Docker][deepin-wine]TIM%E8%BF%90%E8%A1%8C/)
3. `Thunder`：[[Docker][deepin-wine]迅雷运行](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/gui/[Docker][deepin-wine]%E8%BF%85%E9%9B%B7%E8%BF%90%E8%A1%8C/)
4. `WPS`：包含了`word/excel/ppt/pdf`，完美替换`libreoffice`，参考[[Docker][Ubuntu]WPS运行](https://containerization-automation.readthedocs.io/zh_CN/latest/docker/gui/[Docker][Ubuntu]WPS%E8%BF%90%E8%A1%8C/)
5. `Jenkins`：[在Docker中运行Jenkins](https://www.zhujian.tech/posts/202ee452.html)
6. `GitLab`：[[Docker]GitLab使用](https://zj-git-guide.readthedocs.io/zh_CN/latest/platform/[Docker]GitLab%E4%BD%BF%E7%94%A8/)
7. `Nginx`：[docker安装nginx](https://zj-network-guide.readthedocs.io/zh_CN/latest/nginx/docker%E5%AE%89%E8%A3%85nginx/)

## 小结

![](/imgs/重装系统小结/Ubuntu&#32;18.04.png)
