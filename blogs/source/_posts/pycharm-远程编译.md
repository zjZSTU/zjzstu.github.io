---
title: '[pycharm]远程编译'
categories:
  - server
  - tool
tags: pycharm
abbrlink: a6c06fb8
date: 2019-03-30 13:44:49
---

参考：

[教程 | 使用 PyCharm 连接服务器进行远程开发和调试](https://zhuanlan.zhihu.com/p/38330654)

[Creating a Remote Server Configuration](https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html)

当前使用笔记本进行开发，想要利用服务器加快编译和运行速度

新建`PyCharm`工程，点击菜单栏`File->Settings`，打开设置窗口

点击左侧窗格`Project->Project Interpreter`

![](/imgs/pycharm-远程编译/project-interpreter.png)

点击右上角的齿轮按钮`->Add`，弹出`Add Pycharm Interpreter`窗口

![](/imgs/pycharm-远程编译/gear-button.png)

选择`SSH Interpreter`，在右侧添加`Host`和`Username`，然后点击`Next`

![](/imgs/pycharm-远程编译/add-interpreter.png)

填入服务器中`python`解释器的位置，默认会将本地`pycharm`工程同步到服务器`tmp`路径下，点击`Finish`按钮，即完成创建

![](/imgs/pycharm-远程编译/select-interpreter.png)

等待一段时间，将本地工程先上传到远程服务器，同时更新解释器，就可以使用远程解释器进行运算了