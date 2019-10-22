---
title: '[Jenkins]Tomcat托管'
categories:
  - - 工具
  - - 自动化
tags: jenkins
abbrlink: bc77c204
date: 2019-10-22 11:34:20
---

参考：[ubuntu16.04使用tomcat安装jenkins](http://www.mamicode.com/info-detail-2535358.html)

通过`Tomcat`托管`Jenkins`

## 实现

将`Jenkins.war`文件放置于`Tomcat webapps`目录下

```
/opt/apache-tomcat-9.0.27/webapps
```

登录地址`localhost:8080/jenkins`，即可启动`Jenkins`

`Tomcat`会在`webapps`目录下自动解压`Jenkins.war`，生成一个`jenkins`文件夹，其相应的配置文件在`/root/.jenkins`目录下

## Jenkins升级

下载新版本的`Jenkins.war`文件后，放置于`webapps`目录下，并删除`webapps/jenkins`文件夹，重新浏览器登录即可

## 复用/home/zj/.jenkins配置

之前已经通过手动命令进行`Jenkins`操作，相应的配置保存在`~/.jenkins`目录下

先清空`/root/jenkins`文件夹，然后将`~/.jenkins`文件复制过来

```
# rm -rf /root/jenkins
# cp -r /home/zj/jenkins .
```

最后还需要删除`Tomcat webapps`目录下的`jenkins`文件夹。重新进行浏览器登录，之前的用户与以及相关的配置依旧存在