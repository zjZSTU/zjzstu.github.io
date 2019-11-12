---
title: '[Jenkins]手动下载插件'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: 373e88b0
date: 2019-10-19 20:33:37
---

参考：[jenkins安装插件的两种方式](https://www.jianshu.com/p/3b5ebe85c034)

使用`Jenkins`的插件页面进行下载太慢了，所以打算手动下载插件

## 查询

通过官网查找需要的插件：[Plugins Index](https://plugins.jenkins.io)。比如[Gitlab](https://plugins.jenkins.io/gitlab-plugin)

![](/imgs/jenkins-plugin/jenkins-plugin-gitlab.png)

## 下载

进入下载页面

![](/imgs/jenkins-plugin/gitlab-archives.png)

得到最新的插件文件`gitlab-plugin.hpi`

## 配置

进入`Manage Jenkins -> Manage Plugins -> Advanced`，在`Upload Plugin`中上传插件

![](/imgs/jenkins-plugin/upload-plugin.png)
