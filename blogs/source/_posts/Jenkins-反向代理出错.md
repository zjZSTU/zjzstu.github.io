---
title: '[Jenkins]反向代理出错'
categories: 构建
tags: 自动化
abbrlink: adc5ce0c
date: 2019-03-26 09:53:52
---

将`Jenkins`架设在局域网服务器上，使用内网穿透技术映射`Jenkins`端口，打开页面，提示错误

>It appears that your reverse proxy set up is broken

参考：[It appears that your reverse proxy set up is broken解决](https://blog.csdn.net/fxy0325/article/details/88131947)

进入配置系统页面(`Manage Jenkins->Configure System`)，找到`Jenkins Location`小节

![](/imgs/Jenkins-反向代理出错/jenkins-location.png)

修改`Jenkins URL`为当前公网地址即可
